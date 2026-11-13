"""Scientific training functions; host scheduling/launching is omitted."""
from contextlib import contextmanager
import fcntl, os, random, tempfile, time
from pathlib import Path
import common as c
from train_math import TRAIN_CONFIG as T, candidates, digest
_stop_requested = False

def check_stop():
    if _stop_requested:
        raise RuntimeError('Worker_signal_stop')
    c.check_stop()

@contextmanager
def candidate_lock(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with (folder/'candidate.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

def atomic_torch_save(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name+'.')
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def load_latest(folder, axis, layer, lineage):
    import torch
    files = sorted((Path(folder)/'checkpoint_history').glob('step_*.pt'))
    latest = files[-1] if files else Path(folder)/'checkpoint.pt'
    if not latest.exists():
        return None
    receipt_path = latest.with_suffix('.json')
    if not receipt_path.exists():
        raise RuntimeError('Uncommitted_vector_checkpoint_requires_inspection')
    receipt = c.read(receipt_path)
    if (receipt.get('axis') != axis or receipt.get('layer') != layer
            or receipt.get('lineage') != lineage or receipt.get('file') != latest.name
            or receipt.get('sha256') != c.sha(latest)):
        raise RuntimeError('Vector_checkpoint_SHA_or_identity_mismatch')
    # Only our locally authored optimizer/RNG checkpoint, never arbitrary remote input.
    state = torch.load(latest, map_location='cpu', weights_only=False)
    validate_checkpoint(state, axis, layer, lineage)
    if receipt.get('step') != state['step']:
        raise RuntimeError('Vector_checkpoint_step_receipt_mismatch')
    progress = Path(folder)/'training_progress.json'
    if progress.exists() and c.read(progress).get('step', 0) > state['step']:
        c.save(Path(folder)/'resume_history'/f'progress_{time.time_ns()}.json', c.read(progress))
    return state

def save_checkpoint(folder, axis, layer, lineage, step, unit, scale, initial_norm,
                    optimizer, sampler, history):
    import torch
    state = dict(schema=1, axis=axis, layer=layer, lineage=lineage, config=dict(T),
        step=step, unit=unit.detach().cpu().clone(), scale=scale,
        initial_contrast_norm=initial_norm, optimizer=optimizer.state_dict(),
        python_rng=random.getstate(), sampler_rng=sampler.getstate(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
        history=list(history), time=time.time())
    target = Path(folder)/'checkpoint_history'/f'step_{step:04d}.pt'
    if target.exists():
        old = torch.load(target, map_location='cpu', weights_only=False)
        validate_checkpoint(old, axis, layer, lineage)
        if old['step'] != step or not torch.equal(old['unit'], state['unit']) or old['history'] != history:
            raise RuntimeError('Refusing to overwrite a committed vector checkpoint')
        # Restore the latest alias after interruption between immutable save and alias.
        state = old
    else:
        atomic_torch_save(target, state)
    receipt = dict(axis=axis, layer=layer, lineage=lineage, step=step,
                   file=target.name, sha256=c.sha(target))
    commit_path = target.with_suffix('.json')
    if commit_path.exists():
        if c.read(commit_path) != receipt:
            raise RuntimeError('Committed_vector_checkpoint_receipt_changed')
    else:
        c.save(commit_path, receipt)
    atomic_torch_save(Path(folder)/'checkpoint.pt', state)
    c.save(Path(folder)/'checkpoint.json', dict(receipt, file='checkpoint.pt',
           sha256=c.sha(Path(folder)/'checkpoint.pt')))
    c.save(Path(folder)/'training_progress.json', dict(axis=axis, layer=layer, step=step,
        total=T['steps'], history=history, checkpoint_sha=c.sha(target), lineage=lineage,
        time=time.time()))

def completed(folder, axis, layer, lineage):
    marker = Path(folder)/'complete.json'
    if not marker.exists():
        return False
    record = c.read(marker)
    if (record.get('axis') != axis or record.get('layer') != layer
            or record.get('updates') != T['steps'] or record.get('lineage') != lineage
            or record.get('vector_sha') != c.sha(Path(folder)/'vector.pt')
            or record.get('checkpoint_sha') != c.sha(Path(folder)/'checkpoint.pt')):
        raise RuntimeError('Completed candidate provenance mismatch')
    return True

def fresh_contrast(base, axis_pairs, layer):
    import torch
    from vector_ops import model_device
    contrasts, references = [], []
    for kind in ('Q', 'B'):
        for high, low in axis_pairs[kind][:T['contrast_pairs_per_kind']]:
            means = []
            for row in (high, low):
                check_stop()
                captured = []
                def capture(module, args, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    captured.append(hidden[0, row['reason_start']:row['answer_start']]
                                    .detach().float().mean(0))
                handle = base.model.language_model.layers[layer].register_forward_hook(capture)
                try:
                    ids = torch.tensor([row['input_ids']], device=model_device(base))
                    with torch.inference_mode():
                        base.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            position_ids=torch.arange(ids.shape[1], device=ids.device)[None],
                            use_cache=False, return_dict=True)
                finally:
                    handle.remove()
                if len(captured) != 1:
                    raise RuntimeError('Fresh contrast capture count mismatch')
                means.append(captured[0]); references.append(float(captured[0].norm()))
            contrasts.append(means[0]-means[1])
    initial = torch.stack(contrasts).mean(0)
    norm = float(initial.norm())
    scale = T['ref_scale']*sum(references)/len(references)
    if not torch.isfinite(initial).all() or norm < 1e-6 or not 0 < scale < float('inf'):
        raise RuntimeError('Degenerate fresh training contrast')
    return initial/norm, scale, norm

def fit_candidate(rank, base, model, tokenizer, pairs, anchors, coverage, axis, layer, lineage):
    import torch
    import model_loader
    from vector_ops import assert_frozen, batch_loss, inject, neutral_kl, project_unit_ball_
    folder = c.RUN/f'candidates/{axis}_L{layer}'
    job_number = candidates().index((axis, layer))
    seed = T['seed'] + job_number
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    sampler = random.Random(seed)
    resumed = load_latest(folder, axis, layer, lineage)
    model_loader.checkpointing(model, False); model.eval()
    if resumed is None:
        c.status(rank, 'initializing_fresh_training_contrast', axis=axis, layer=layer)
        initial, scale, initial_norm = fresh_contrast(base, pairs[axis], layer)
        unit = torch.nn.Parameter(initial.detach().clone().float())
        step, history = 0, []
    else:
        unit = torch.nn.Parameter(resumed['unit'].to('cuda', dtype=torch.float32))
        scale, initial_norm = resumed['scale'], resumed['initial_contrast_norm']
        step, history = resumed['step'], resumed['history']
    if unit.ndim != 1 or unit.numel() != 5120 or not torch.isfinite(unit).all() or float(unit.detach().norm()) > 1.00001:
        raise RuntimeError('Invalid unit-ball vector state')
    optimizer = torch.optim.Adam([unit], lr=T['lr'])
    if resumed is not None:
        optimizer.load_state_dict(resumed['optimizer'])
        random.setstate(resumed['python_rng']); sampler.setstate(resumed['sampler_rng'])
        torch.set_rng_state(resumed['torch_rng']); torch.cuda.set_rng_state(resumed['cuda_rng'])
        # Reconcile alias after an interruption between immutable history and alias,
        # including a resume already at step240 (the loop below would be skipped).
        save_checkpoint(folder, axis, layer, lineage, step, unit, scale, initial_norm,
                        optimizer, sampler, history)
    else:
        save_checkpoint(folder, axis, layer, lineage, step, unit, scale, initial_norm,
                        optimizer, sampler, history)
    model.train(); model.config.use_cache = False; model_loader.checkpointing(model, True)
    assert_frozen(model)
    versions = [(name, parameter, parameter._version) for name, parameter in model.named_parameters()]
    signs = torch.tensor([1., -1., 1., -1.], device='cuda')
    try:
        while step < T['steps']:
            try:
                check_stop()
            except RuntimeError:
                save_checkpoint(folder, axis, layer, lineage, step, unit, scale, initial_norm,
                                optimizer, sampler, history)
                raise
            next_step = step + 1
            batch = [*sampler.choice(pairs[axis]['Q']), *sampler.choice(pairs[axis]['B'])]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); start = time.time()
            with inject(base, layer, unit*scale, signs):
                supervised = batch_loss(base, batch, tokenizer.pad_token_id, T['head_chunk'])
                supervised.backward()
            klvalue = 0.
            if next_step % T['neutral_every'] == 0:
                row, reference = anchors[(next_step//T['neutral_every']-1) % len(anchors)]
                for sign in (1., -1.):
                    with inject(base, layer, unit*scale, sign):
                        kl = neutral_kl(base, row, reference, T['head_chunk'])
                        (T['neutral_kl']*.5*kl).backward()
                    klvalue += float(kl.detach())*.5
                    del kl
            regularizer = T['regularization']*unit.square().sum()
            regularizer.backward()
            if unit.grad is None or not bool(torch.isfinite(unit.grad).all()):
                raise RuntimeError('Nonfinite/missing vector gradient')
            if not bool(torch.isfinite(supervised)):
                raise RuntimeError('Nonfinite supervised loss')
            assert_frozen(model)
            gradient = float(unit.grad.norm())
            torch.nn.utils.clip_grad_norm_([unit], T['grad_clip'], error_if_nonfinite=True)
            optimizer.step(); project_unit_ball_(unit)
            torch.cuda.synchronize(); step = next_step
            history.append(dict(step=step, ce=float(supervised.detach()), neutral_kl=klvalue,
                regularizer=float(regularizer.detach()), grad_norm=gradient,
                seconds=time.time()-start, unit_norm=float(unit.detach().norm()),
                jobs=[row['job_id'] for row in batch]))
            del supervised, regularizer
            c.status(rank, 'vector_training', axis=axis, layer=layer, step=step, total=T['steps'])
            if step % T['checkpoint_every'] == 0:
                save_checkpoint(folder, axis, layer, lineage, step, unit, scale, initial_norm,
                                optimizer, sampler, history)
        assert_frozen(model)
        if any(parameter._version != version for _, parameter, version in versions):
            raise RuntimeError('Frozen model parameter was modified')
        vector = dict(vector=(unit.detach()*scale).cpu(), unit=unit.detach().cpu(),
            scale=scale, axis=axis, layer=layer, lineage=lineage, injection='full_input_and_decode')
        if (folder/'vector.pt').exists():
            old = torch.load(folder/'vector.pt', map_location='cpu', weights_only=True)
            if old.get('lineage') != lineage or not torch.equal(old['vector'], vector['vector']):
                raise RuntimeError('Existing uncommitted vector differs from resumed checkpoint')
        else:
            atomic_torch_save(folder/'vector.pt', vector)
        c.save(folder/'complete.json', dict(axis=axis, layer=layer, updates=step,
            vector_sha=c.sha(folder/'vector.pt'), checkpoint_sha=c.sha(folder/'checkpoint.pt'),
            lineage=lineage, initial_contrast_norm=initial_norm, scale=scale,
            coverage=coverage[axis], parent_frozen=True, development_run=False, finished=time.time()))
    finally:
        optimizer.zero_grad(set_to_none=True)
        model_loader.checkpointing(model, False); model.eval()

def training_lineage(hashes, scientific_digest):
    return digest(dict(model_commit_sha=c.MODEL_COMMIT_SHA,
        model_lineage=c.MODEL_LINEAGE, model_epoch=c.MODEL_EPOCH,
        teacher=hashes['teacher'], neutral=hashes['neutral'],
        scientific_manifest=scientific_digest, config=T))
