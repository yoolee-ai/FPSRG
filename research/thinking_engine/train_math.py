"""Pure training contracts and full native-ON trajectory preparation."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

AXES = tuple('ACNRMT')
WORKERS = 16
LAYERS = (8, 16, 24, 32, 40, 48, 56, 60)  # zero-based residual layer indices
from common import SYSTEM
TRAIN_CONFIG = dict(steps=240, lr=.003, thought_weight=.7, final_weight=.3,
                    neutral_kl=.1, neutral_every=4, regularization=.01,
                    ref_scale=.1, checkpoint_every=20, grad_clip=1.,
                    contrast_pairs_per_kind=4, head_chunk=32, seed=2026103101)


def validate_config(config):
    names = {'regularization': 'regularizer', 'ref_scale': 'scale_fraction'}
    for key in ('steps', 'lr', 'thought_weight', 'final_weight', 'neutral_kl',
                'neutral_every', 'regularization', 'ref_scale', 'checkpoint_every', 'seed'):
        if config.get(names.get(key, key)) != TRAIN_CONFIG[key]:
            raise ValueError('Frozen training objective mismatch: ' + key)
    if (tuple(config['axes']) != AXES or tuple(config['layers']) != LAYERS
            or config.get('train_batch_size') != 4 or config.get('system') != SYSTEM
            or config.get('world_size') != WORKERS):
        raise ValueError('Training axis/layer/batch/system mismatch')


def candidates():
    return [(axis, layer) for axis in AXES for layer in LAYERS]


def local_gpu(rank):
    if type(rank) is not int or rank not in range(WORKERS):
        raise ValueError('Rank must be an integer in 0..15')
    return rank % 8


def assigned_candidates(rank):
    local_gpu(rank)
    return candidates()[rank::WORKERS]


def encode_prompt(tokenizer, prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Missing unconditioned job.prompt')
    return list(tokenizer.apply_chat_template(
        [dict(role='system', content=SYSTEM), dict(role='user', content=prompt)],
        tokenize=True, return_dict=False, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort='medium', preserve_thinking=False,
        tools=None))


def section_weights(reason_start, answer_start, length, example_weight=1.):
    if not 0 < reason_start < answer_start < length:
        raise ValueError('Empty native thought/final or invalid target boundary')
    weights = [0.] * length
    nr, nf = answer_start - reason_start, length - answer_start
    weights[reason_start:answer_start] = [example_weight * .7 / nr] * nr
    weights[answer_start:] = [example_weight * .3 / nf] * nf
    return weights


def trajectory(record, tokenizer, eos_ids):
    """Re-encode ONLY job.prompt; never reuse conditioned teacher input_ids."""
    job = record['job']
    if record.get('valid') is not True:
        raise ValueError('Invalid demonstration cannot enter vector fitting')
    if job.get('split', 'train') != 'train':
        raise ValueError('Non-training demonstration')
    generated = record.get('generated_ids')
    if not isinstance(generated, list) or not generated or any(type(x) is not int for x in generated):
        raise ValueError('Missing original generated token sequence')
    if generated[-1] not in eos_ids or any(t in eos_ids for t in generated[:-1]):
        raise ValueError('Missing native terminal EOS or tokens after EOS')
    close = list(tokenizer.encode('</think>', add_special_tokens=False))
    starts = [i for i in range(len(generated)-len(close)+1)
              if generated[i:i+len(close)] == close]
    if len(close) != 1 or len(starts) != 1 or starts[0] == 0:
        raise ValueError('Expected one native thinking close after nonempty thought')
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    if raw.count('</think>') != 1:
        raise ValueError('Native closing text/token mismatch')
    final = raw.split('</think>', 1)[1].strip()
    if final != str(job['target']):
        raise ValueError('Teacher final does not exactly match authored target')
    prefix = encode_prompt(tokenizer, job['prompt'])
    row = dict(input_ids=prefix + generated, reason_start=len(prefix),
               answer_start=len(prefix) + starts[0] + len(close),
               job_id=job['id'], axis=job.get('axis'), kind=job.get('kind'),
               side=job.get('side'), mapping=job.get('mapping'), target=str(job['target']))
    section_weights(row['reason_start'], row['answer_start'], len(row['input_ids']))
    return row


def load_pairs(records, tokenizer, eos_ids):
    pool = defaultdict(dict)
    seen = set()
    invalid = []
    for record in records:
        job = record['job']
        jid = job['id']
        if jid in seen:
            raise ValueError('Duplicate teacher job: ' + jid)
        seen.add(jid)
        if job.get('axis') not in AXES or job.get('kind') not in ('Q', 'B'):
            raise ValueError('Unexpected teacher axis/kind: ' + jid)
        if job.get('side') not in ('high', 'low') or not jid.endswith('/' + job['side']):
            raise ValueError('Teacher side/id mismatch: ' + jid)
        if record.get('valid') is not True:
            invalid.append(jid)
            continue
        # A record marked valid must actually satisfy the native/target contract.
        row = trajectory(record, tokenizer, eos_ids)
        pool[jid.rsplit('/', 1)[0]][job['side']] = (record, row)
    paired = {a: {'Q': [], 'B': []} for a in AXES}
    for pid, sides in sorted(pool.items()):
        if set(sides) != {'high', 'low'}:
            continue
        hi, lo = sides['high'], sides['low']
        hj, lj = hi[0]['job'], lo[0]['job']
        for key in ('axis', 'kind', 'prompt', 'mapping', 'source_id'):
            if hj.get(key) != lj.get(key):
                raise ValueError('Contrast pair mismatch: ' + pid + '/' + key)
        if str(hj['target']) == str(lj['target']):
            raise ValueError('Identical high/low targets: ' + pid)
        paired[hj['axis']][hj['kind']].append((hi[1], lo[1]))
    coverage = {a: {k: len(paired[a][k]) for k in ('Q', 'B')} for a in AXES}
    if any(v['Q'] < 8 or v['B'] < 4 for v in coverage.values()):
        raise ValueError('Insufficient complete paired coverage: ' + str(coverage))
    if any({hi['mapping'] for hi, lo in paired[a][k]} != {0, 1}
           for a in AXES for k in ('Q', 'B')):
        raise ValueError('Both digit mappings / A-B orders must survive in each axis/kind')
    return paired, coverage, invalid


def load_neutral(records, tokenizer, eos_ids):
    rows, seen = [], set()
    for record in records:
        job = record['job']
        if job['id'] in seen:
            raise ValueError('Duplicate neutral job')
        seen.add(job['id'])
        if job.get('kind') != 'neutral':
            raise ValueError('Unexpected neutral kind')
        if record.get('valid') is True:
            rows.append(trajectory(record, tokenizer, eos_ids))
    if len(rows) < 4:
        raise ValueError('Fewer than four valid fresh neutral anchors')
    return rows


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_checkpoint(state, axis, layer, lineage):
    if (state.get('schema') != 1 or state.get('axis') != axis
            or state.get('layer') != layer or state.get('lineage') != lineage
            or state.get('config') != TRAIN_CONFIG):
        raise ValueError('Checkpoint lineage/config/candidate mismatch')
    step = state['step']
    if type(step) is not int or not 0 <= step <= TRAIN_CONFIG['steps']:
        raise ValueError('Invalid checkpoint step')
    if [x['step'] for x in state['history']] != list(range(1, step+1)):
        raise ValueError('Checkpoint history is not contiguous')
    if not math.isfinite(state['scale']) or state['scale'] <= 0:
        raise ValueError('Invalid vector reference scale')
    if not all(k in state for k in ('unit', 'optimizer', 'python_rng', 'sampler_rng',
                                   'torch_rng', 'cuda_rng', 'initial_contrast_norm')):
        raise ValueError('Incomplete training checkpoint')
    return step

