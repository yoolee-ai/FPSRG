"""Exact causal-prefix row sharing while retaining canonical skinny A products.

The attention operators still see all B32 x 318 positions. Two rows are shared
only if their entire input-token AND padding-mask prefixes are identical at the
same absolute position; coefficients are uniform across the question batch.
All input positions keep the same injection. Output logits still use the same
unmerged parent, dtypes, temperature, and probability/mass formula.
"""
from contextlib import contextmanager
from types import MethodType

MODES = ('prefix_fused', 'prefix_unfused', 'padding_fused', 'prefix_full_lora',
         'prefix_nomemo', 'padding_nomemo', 'block32_packed', 'block64_packed')

def prefix_rows(ids, masks, mode):
    if len(ids) not in (32, 64, 128) or any(len(row) != 318 for row in ids) or len(masks) != len(ids):
        raise ValueError('canonical_input_shape_required')
    valid = [b*318+s for b, row in enumerate(masks) for s, v in enumerate(row) if v]
    if mode.startswith('padding'):
        return valid, valid, list(range(len(valid)))
    if mode.startswith('block'):
        width = int(mode.split('_')[0][5:])
        blocks = sorted({i // width for i in valid})
        reps = [i for block in blocks for i in range(block*width, min((block+1)*width, len(ids)*318))]
        inverse = {r: i for i, r in enumerate(reps)}
        return reps, valid, [inverse[v] for v in valid]
    transitions, representatives, group_by_state = {}, [], {}
    inverse = []
    for b, (row, mask) in enumerate(zip(ids, masks)):
        state = 0
        for pos, (token, flag) in enumerate(zip(row, mask)):
            key = (state, int(token), int(flag))
            if key not in transitions:
                transitions[key] = len(transitions) + 1
            state = transitions[key]
            if flag:
                if state not in group_by_state:
                    group_by_state[state] = len(representatives)
                    representatives.append(b*318+pos)
                inverse.append(group_by_state[state])
    return representatives, valid, inverse

@contextmanager
def prefix_compaction(joint, batch, mode, verify_prefix=False):
    if mode not in MODES:
        raise ValueError('unknown_prefix_mode')
    t = joint.t
    profiles = getattr(joint, '_prefix_profiles', None)
    if profiles is None:
        profiles = joint._prefix_profiles = {}
    key = (mode, batch, joint.batch_size, getattr(joint, '_batch_order_key', 'canonical'))
    if key not in profiles:
        inputs = joint._batch(batch)[0]
        ids = inputs['input_ids'].cpu().tolist()
        masks = inputs['attention_mask'].cpu().tolist()
        reps, valid, inverse = prefix_rows(ids, masks, mode)
        profiles[key] = dict(batch_size=len(ids), reps=t.tensor(reps, device=joint.device),
            valid=t.tensor(valid, device=joint.device), inverse=t.tensor(inverse, device=joint.device),
            unique_rows=len(reps), valid_rows=len(valid))
    p = profiles[key]
    batch_size = p['batch_size']
    count = p['unique_rows']; allocated = (count+255)//256*256
    missing = object(); changes = []; memo = {}
    statistics = dict(mode=mode, unique_rows=count, valid_rows=p['valid_rows'], allocated=allocated,
                      original_rows=batch_size*318, exact_prefix_checks=0, memo_hits=0)
    use_memo = 'nomemo' not in mode

    def pack(x, cache=True):
        if x.ndim != 3 or tuple(x.shape[:2]) != (batch_size, 318):
            raise RuntimeError('prefix_pack_shape_contract')
        if cache and use_memo and memo.get('source') is x and 'packed' in memo:
            statistics['memo_hits'] += 1
            return memo['packed']
        if cache and memo.get('source') is not x:
            memo.clear(); memo['source'] = x
        flat = x.reshape(-1, x.shape[-1])
        selected = flat.index_select(0, p['reps'])
        if verify_prefix and mode.startswith('prefix'):
            if not t.equal(flat.index_select(0, p['valid']), selected.index_select(0, p['inverse'])):
                raise RuntimeError('causal_prefix_hidden_not_bitwise_identical')
            statistics['exact_prefix_checks'] += 1
        if allocated > count:
            selected = t.cat((selected, selected.new_zeros(allocated-count, selected.shape[-1])), dim=0)
        if cache and use_memo:
            memo['packed'] = selected
        return selected

    def full_float(x, module, dtype):
        if use_memo and memo.get('source') is x and 'float' in memo:
            statistics['memo_hits'] += 1
            return memo['float']
        value = module._cast_input_dtype(x, dtype)
        if use_memo and memo.get('source') is x:
            memo['float'] = value
        return value

    def unpack(x):
        result = x.new_zeros(batch_size*318, x.shape[-1])
        result.index_copy_(0, p['valid'], x[:count].index_select(0, p['inverse']))
        return result.view(batch_size, 318, x.shape[-1])

    def install(module, fn):
        previous = module.__dict__.get('forward', missing)
        module.forward = MethodType(fn, module); changes.append((module, previous))

    def norm(module):
        original = module.forward
        def forward(_module, x):
            return unpack(original(pack(x)))
        install(module, forward)

    def peft(module):
        names = list(module.active_adapters)
        if (len(names) != 1 or module.disable_adapters or module.merged or module.lora_variant
                or module.training or module.base_layer.bias is not None):
            raise RuntimeError('unmerged_vanilla_frozen_parent_required')
        name = names[0]; A = module.lora_A[name]; B = module.lora_B[name]
        if (A.weight.dtype != t.float32 or B.weight.dtype != t.float32 or A.out_features != 16
                or module.scaling[name] != 2 or getattr(module.lora_dropout[name], 'p', 0) != 0):
            raise RuntimeError('unchanged_FP32_rank16_scale2_required')
        base = module.base_layer
        def forward(_module, x, *args, **kwargs):
            if args or kwargs or x.requires_grad:
                raise RuntimeError('prefill_only_inference_contract')
            packed = pack(x)
            result = base(packed)
            if mode.startswith('block'):
                delta = B(A(module._cast_input_dtype(packed, A.weight.dtype)))[:count]
            else:
                a = A(full_float(x, module, A.weight.dtype))
                if mode == 'prefix_full_lora':
                    delta = B(a).reshape(-1, B.out_features).index_select(0, p['reps'])
                else:
                    # A changes the last feature dimension; its original full shape is kept.
                    delta = B(pack(a, cache=False))[:count]
            result = result[:count]
            if mode == 'prefix_unfused':
                result = (result + delta * 2).to(result.dtype)
            else:
                # Scale2 is exact for these finite FP32 values: fused multiply/add
                # retains the original FP32 intermediate rounding before BF16 store.
                result.add_(delta, alpha=2)
            return unpack(result)
        install(module, forward)

    try:
        for layer in joint.engine.layers:
            for name in ('gate_proj', 'up_proj', 'down_proj'):
                peft(getattr(layer.mlp, name))
            if layer.block_type == 'linear_attention':
                for name in ('in_proj_qkv', 'in_proj_z', 'out_proj'):
                    peft(getattr(layer.linear_attn, name))
            else:
                for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    peft(getattr(layer.self_attn, name))
            norm(layer.input_layernorm); norm(layer.post_attention_layernorm)
        yield statistics
    finally:
        memo.clear()
        for module, previous in reversed(changes):
            if previous is missing:
                module.__dict__.pop('forward', None)
            else:
                module.forward = previous

