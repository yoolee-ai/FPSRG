"""Frozen FP32 LoRA and native-HF rounding adapters, extracted from Oct12 code."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MethodType

AXES = tuple('OCEANRTVMS')
LAYERS = dict(zip(AXES, (40, 32, 32, 16, 32, 32, 40, 32, 40, 32)))
COMMIT = 'cb2f77e43b96830130d9ffe2ec93a630aae4ad8892d46a96668317c0603ea0fe'
ADAPTER_TENSORS = 992
ADAPTER_PARAMETERS = 116727808
ARCH = 'Qwen3_5ForConditionalGeneration'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def coefficients(values):
    if isinstance(values, dict):
        if set(values) != set(AXES):
            raise ValueError('exact_ten_axes_required')
        values = [values[a] for a in AXES]
    if len(values) != 10 or any(isinstance(x, bool) for x in values):
        raise ValueError('ten_finite_coefficients_required')
    result = tuple(float(x) for x in values)
    if any(not math.isfinite(x) or x not in (-2., -1., 0., 1., 2.) for x in result):
        raise ValueError('unapproved_coefficient')
    return result


def adapter_target(name):
    """Map HF unmerged LoRA names into existing vLLM stacked projections."""
    name = name.removeprefix('base_model.model.')
    match = re.fullmatch(r'model\.language_model\.layers\.(\d+)\.(.+)\.lora_([AB])(?:\.default)?\.weight', name)
    if not match:
        raise ValueError('unexpected_adapter_key:' + name)
    layer, path, part = int(match[1]), match[2], match[3]
    if not 0 <= layer < 64:
        raise ValueError('adapter_layer_out_of_range')
    group, leaf = path.rsplit('.', 1)
    stacked = {
        'self_attn': {'q_proj': ('qkv_proj', 0), 'k_proj': ('qkv_proj', 1), 'v_proj': ('qkv_proj', 2)},
        'mlp': {'gate_proj': ('gate_up_proj', 0), 'up_proj': ('gate_up_proj', 1)},
        'linear_attn': {'in_proj_qkv': ('in_proj_qkvz', 0), 'in_proj_z': ('in_proj_qkvz', 1),
                        'in_proj_b': ('in_proj_ba', 0), 'in_proj_a': ('in_proj_ba', 1)},
    }
    allowed = {'self_attn': {'q_proj', 'k_proj', 'v_proj', 'o_proj'},
               'mlp': {'gate_proj', 'up_proj', 'down_proj'},
               'linear_attn': {'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj'}}
    if group not in allowed or leaf not in allowed[group]:
        raise ValueError('unexpected_adapter_projection:' + path)
    target, slot = stacked.get(group, {}).get(leaf, (leaf, 0))
    return f'language_model.model.layers.{layer}.{group}.{target}', slot, part


def assemble_delta(torch, vectors, values, layer):
    """Same fixed axis-order FP32 sum as the existing HF grid engine."""
    result = torch.zeros_like(vectors[AXES[0]], dtype=torch.float32)
    for axis, value in zip(AXES, coefficients(values)):
        if LAYERS[axis] == layer and value != 0.:
            result = result + vectors[axis] * value
    return result


def residual_intervention(hidden, residual, delta):
    """Materialize block output BEFORE adding delta; never add into MLP alone."""
    complete = hidden if residual is None else hidden + residual
    return complete + delta.to(complete.dtype), None


def hf_rms_norm(torch, hidden, weight, eps, residual=None):
    """Transformers Qwen3.5 eager rounding, with vLLM residual-pair API."""
    complete = hidden if residual is None else hidden + residual
    value = complete.float()
    value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
    value = (value * (1.0 + weight.float())).to(complete.dtype)
    return value if residual is None else (value, complete)


def hf_gated_norm(torch, hidden, gate, weight, eps):
    """HF explicitly rounds normalized value/weight product before FP32 gate."""
    dtype = hidden.dtype
    value = hidden.float()
    value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
    value = weight * value.to(dtype)
    value = value * torch.nn.functional.silu(gate.float())
    return value.to(dtype)


def hf_silu_mul(torch, packed):
    gate, up = packed.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def _install_hf_same_rounding(model, torch):
    """Opt-in numerical alignment; leaves initial vLLM prototype unchanged by default.

    CPU meta initialization confirmed original HF A_log/norm/dt_bias are BF16,
    and Qwen3_5ForConditionalGeneration has no FP32 module exclusions. The
    official checkpoint A_log itself is F32, so vLLM's explicitly F32 A_log
    otherwise retains additional precision absent from the reference model.
    """
    enabled = os.environ.get('FPSRG_HF_SAME_ROUNDING', '0')
    if enabled not in ('0', '1'):
        raise RuntimeError('FPSRG_HF_SAME_ROUNDING_must_be_0_or_1')
    if enabled == '0':
        return dict(enabled=False, mode='initial_vllm_native_rounding')
    torch.set_float32_matmul_precision('highest')
    norm_count = mlp_count = gated_count = a_count = 0

    def norm_forward(this, hidden, residual=None):
        return hf_rms_norm(torch, hidden, this.weight, this.variance_epsilon, residual)

    def gated_forward(this, hidden, gate=None):
        if gate is None:
            raise RuntimeError('Qwen35_GDN_gate_required')
        return hf_gated_norm(torch, hidden, gate, this.weight, this.eps)

    def silu_forward(this, packed):
        return hf_silu_mul(torch, packed)

    for layer in model.language_model.model.layers:
        norms = [layer.input_layernorm, layer.post_attention_layernorm]
        if layer.layer_type == 'full_attention':
            norms.extend((layer.self_attn.q_norm, layer.self_attn.k_norm))
        for norm in norms:
            norm.forward = MethodType(norm_forward, norm); norm_count += 1
        layer.mlp.act_fn.forward = MethodType(silu_forward, layer.mlp.act_fn); mlp_count += 1
        if layer.layer_type == 'linear_attention':
            gdn = layer.linear_attn
            # Keep the FP32 storage expected by its kernel, but exactly match
            # the values present in the original BF16 HF backbone.
            gdn.A_log.data = gdn.A_log.data.bfloat16().float(); a_count += 1
            gdn.dt_bias.data = gdn.dt_bias.data.bfloat16()
            gdn.norm.weight.data = gdn.norm.weight.data.bfloat16()
            gdn.norm.forward = MethodType(gated_forward, gdn.norm); gated_count += 1
    final_norm = model.language_model.model.norm
    final_norm.forward = MethodType(norm_forward, final_norm); norm_count += 1
    # Reproduce the HF float32 softplus/exp and BF16 sigmoid boundary. The
    # returned (1, compact_tokens, heads) layout is the original vLLM contract.
    from vllm.model_executor.layers.mamba import gdn_linear_attn
    def gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
        if beta != 1.0 or threshold != 20.0:
            raise RuntimeError('unexpected_gating_parameters')
        g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias)
        return g.unsqueeze(0), b.sigmoid().unsqueeze(0)
    gdn_linear_attn.fused_gdn_gating = gating
    return dict(enabled=True, mode='HF_same_rounding', rms_norms=norm_count,
        gated_norms=gated_count, mlp_silu_mul=mlp_count, A_log_BF16_value_rounding=a_count,
        gdn_gating='HF_FP32_exp_softplus_BF16_beta', float32_matmul_precision='highest',
        thresholds_unchanged=True, native_attention_and_FLA_kernels_unchanged=True)


def _load_vectors(torch, device):
    import numpy as np
    path = Path(os.environ['FPSRG_VECTORS_JSON'])
    manifest = json.loads(path.read_text())
    if set(manifest) != set(AXES):
        raise RuntimeError('ten_vector_manifest_required')
    vectors, digests = {}, {}
    for axis in AXES:
        row = manifest[axis]
        p = Path(row['vector_path'])
        if row['layer'] != LAYERS[axis] or sha(p) != row['vector_sha256']:
            raise RuntimeError('vector_provenance_changed:' + axis)
        with np.load(p, allow_pickle=False) as bank:
            value = bank['vector'].copy()
        if value.shape != (5120,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError('invalid_vector:' + axis)
        vectors[axis] = torch.from_numpy(value).to(device=device)
        digests[axis] = row['vector_sha256']
    return vectors, {'manifest_sha256': sha(path), 'files': digests}


def _install_fixed_lora(model, torch):
    from safetensors.torch import load_file
    root = Path(os.environ['FPSRG_ADAPTER_CHECKPOINT'])
    if sha(root / 'commit.json') != COMMIT:
        raise RuntimeError('wrong_parent_commit')
    receipt = json.loads((root / 'commit.json').read_text())
    for name in ('adapter/adapter_model.safetensors', 'adapter/adapter_config.json'):
        if receipt['files'].get(name) != sha(root / name):
            raise RuntimeError('adapter_SHA_mismatch:' + name)
    cfg = json.loads((root / 'adapter/adapter_config.json').read_text())
    if (cfg.get('r') != 16 or cfg.get('lora_alpha') != 32 or cfg.get('lora_dropout') != 0
            or cfg.get('bias') != 'none' or cfg.get('use_dora') or cfg.get('use_rslora')):
        raise RuntimeError('unexpected_adapter_recipe')
    weights = load_file(str(root / 'adapter/adapter_model.safetensors'), device='cpu')
    if (len(weights) != ADAPTER_TENSORS or sum(t.numel() for t in weights.values()) != ADAPTER_PARAMETERS
            or any(t.dtype != torch.float32 for t in weights.values())):
        raise RuntimeError('expected_exact_992_FP32_adapter_tensors')
    groups = {}
    for name, weight in weights.items():
        target, slot, part = adapter_target(name)
        if part in groups.setdefault(target, {}).setdefault(slot, {}):
            raise RuntimeError('duplicate_adapter_weight')
        groups[target][slot][part] = weight
    count = 0
    for target, slots in groups.items():
        module = model.get_submodule(target)
        if set(slots) != set(range(len(slots))):
            raise RuntimeError('missing_fused_projection_slice')
        offset, pieces = 0, []
        for slot in sorted(slots):
            pair = slots[slot]
            if set(pair) != {'A', 'B'} or pair['A'].shape[0] != 16 or pair['B'].shape[1] != 16:
                raise RuntimeError('invalid_rank16_branch')
            a, b = pair['A'], pair['B']
            if a.shape[1] != module.weight.shape[1]:
                raise RuntimeError('adapter_input_shape_mismatch:' + target)
            module.register_buffer(f'_fpsrg_A{slot}', a.to(module.weight.device), persistent=False)
            module.register_buffer(f'_fpsrg_B{slot}', b.to(module.weight.device), persistent=False)
            pieces.append((slot, offset, offset + b.shape[0])); offset += b.shape[0]
            count += 1
        if offset != module.weight.shape[0]:
            raise RuntimeError('fused_adapter_output_shape_mismatch:' + target)
        original = module.forward

        def apply(this, x, *args, _original=original, _pieces=tuple(pieces), **kwargs):
            result = _original(x, *args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 2 or result[1] is not None:
                raise RuntimeError('expected_unbiased_vllm_linear_tuple')
            output = result[0]
            x32 = x.float()
            changed = output.clone()
            for slot, begin, end in _pieces:
                a = getattr(this, f'_fpsrg_A{slot}'); b = getattr(this, f'_fpsrg_B{slot}')
                delta = torch.nn.functional.linear(torch.nn.functional.linear(x32, a), b)
                changed[..., begin:end] = (output[..., begin:end].float() + delta * 2.).to(output.dtype)
            return changed, None

        module.forward = MethodType(apply, module)
    if count != ADAPTER_TENSORS // 2:
        raise RuntimeError('missing_adapter_pairs')
    del weights
    return dict(commit_sha256=COMMIT, tensors=ADAPTER_TENSORS, parameters=ADAPTER_PARAMETERS,
                pairs=count, wrapped_vllm_modules=len(groups), dtype='float32', merged=False,
                implementation='fixed_PyTorch_FP32_A_then_B_branches', scale=2.)

