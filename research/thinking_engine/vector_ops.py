"""Differentiable full-input + decode injection and native-ON losses."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from train_math import section_weights


def model_device(base):
    return next(base.parameters()).device


@contextmanager
def inject(base, layer, vector, alpha=1.):
    """Inject one FP32 delta per layer, including every real prefill/decode token.

    A caller combining axes on the same layer supplies their FP32 sum here;
    the hook performs exactly one delta cast before adding to the residual.
    """
    if vector is None or (not torch.is_tensor(alpha) and alpha == 0):
        yield
        return
    attention = [None]

    def capture(module, args, kwargs):
        attention[0] = kwargs.get('attention_mask')
        if attention[0] is None or attention[0].ndim != 2:
            raise ValueError('Explicit 2D attention mask required')

    def hook(module, args, out):
        hidden = out[0] if isinstance(out, tuple) else out
        mask = attention[0][:, -hidden.shape[1]:].bool()
        if mask.shape != hidden.shape[:2] or vector.ndim != 1 or vector.numel() != hidden.shape[-1]:
            raise ValueError('Injection shape mismatch')
        if torch.is_tensor(alpha):
            if alpha.ndim != 1 or len(alpha) != hidden.shape[0]:
                raise ValueError('One signed alpha per batch row required')
            delta = alpha.float()[:, None, None] * vector.float()[None, None, :]
        else:
            delta = vector.float() * float(alpha)
        result = torch.where(mask.unsqueeze(-1), hidden + delta.to(hidden.dtype), hidden)
        return (result, *out[1:]) if isinstance(out, tuple) else result

    handles = [base.model.register_forward_pre_hook(capture, with_kwargs=True)]
    try:
        handles.append(base.model.language_model.layers[layer].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def batch_tensors(base, rows, pad):
    device = model_device(base)
    length = max(len(r['input_ids']) for r in rows)
    ids = torch.full((len(rows), length), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    weights = torch.zeros_like(ids, dtype=torch.float32)
    for i, row in enumerate(rows):
        seq = row['input_ids']; offset = length-len(seq)
        ids[i, offset:] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, offset:] = 1
        weights[i, offset:] = torch.tensor(section_weights(row['reason_start'],
            row['answer_start'], len(seq), 1./len(rows)), device=device)
    return ids, mask, weights


def batch_loss(base, rows, pad, chunk=32):
    """Q-high/Q-low/B-high/B-low => equal Q/B and high/low weighting."""
    ids, mask, weights = batch_tensors(base, rows, pad)
    hidden = base.model(input_ids=ids, attention_mask=mask,
        position_ids=(mask.cumsum(-1)-1).clamp_min(0), use_cache=False,
        return_dict=True).last_hidden_state
    keep = weights[:, 1:] > 0
    hs, labels, ws = hidden[:, :-1][keep], ids[:, 1:][keep], weights[:, 1:][keep]
    loss = hs.sum()*0

    def block(h, y, w):
        return (F.cross_entropy(base.lm_head(h.float()), y, reduction='none')*w).sum()

    for start in range(0, len(labels), chunk):
        loss = loss + checkpoint(block, hs[start:start+chunk], labels[start:start+chunk],
                                 ws[start:start+chunk], use_reentrant=False)
    return loss


def final_hidden(base, row):
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device=model_device(base))
    return base.model(input_ids=ids, attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(ids.shape[1], device=ids.device)[None],
        use_cache=False, return_dict=True).last_hidden_state[0, row['answer_start']-1:-1]


def neutral_reference(base, row, chunk=32):
    """Keep reference distributions on CPU; no neutral output generation here."""
    with torch.no_grad():
        hidden = final_hidden(base, row)
        return torch.cat([base.lm_head(hidden[i:i+chunk].float()).log_softmax(-1).cpu()
                          for i in range(0, len(hidden), chunk)])


def neutral_kl(base, row, reference, chunk=32):
    hidden = final_hidden(base, row)
    if len(hidden) != len(reference):
        raise ValueError('Neutral reference token count mismatch')
    value = hidden.sum()*0

    def block(h, logq):
        logp = base.lm_head(h.float()).log_softmax(-1)
        return (logq.exp()*(logq-logp)).sum(-1).sum()/len(hidden)

    for i in range(0, len(hidden), chunk):
        q = reference[i:i+chunk].to(hidden.device)
        value = value + checkpoint(block, hidden[i:i+chunk], q, use_reentrant=False)
    return value


def project_unit_ball_(unit):
    with torch.no_grad():
        unit.div_(unit.norm().clamp_min(1.))


def assert_frozen(model):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad or parameter.grad is not None:
            raise RuntimeError('Frozen parent trainability/gradient violation: ' + name)

