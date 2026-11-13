"""Explicit injection scopes, without altering the historical answer-only hook.

The active query mask is the rightmost query_length columns of the original
2-D attention mask. Thus a full prefill covers every non-padding system/user
and assistant-prefix token, teacher forcing additionally covers every target
token, and cached decoding covers every valid newly appended token. This mask
is deliberately NOT inferred from the token id (EOS may also be padding).
"""

FULL_SEQUENCE = 'full_sequence_nonpadding'
ANSWER_ONLY = 'answer_only'
SCOPES = (FULL_SEQUENCE, ANSWER_ONLY)


def validate_scope(value):
    if value not in SCOPES:
        raise ValueError('unknown_injection_scope:' + str(value))
    return value


def query_mask(torch, attention_mask, batch_size, query_length, device):
    """Return a [batch, query] boolean mask; reject ambiguous implicit layouts."""
    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError('full_sequence_requires_explicit_2d_attention_mask')
    if (query_length <= 0 or attention_mask.shape[0] != batch_size
            or attention_mask.shape[1] < query_length):
        raise ValueError('full_sequence_query_mask_shape')
    return attention_mask[:, -query_length:].to(device=device, dtype=torch.bool)


def inject(torch, hidden, delta, mask):
    """Out-of-place differentiable injection, preserving padding byte-for-byte."""
    if hidden.ndim != 3 or mask.ndim != 2 or tuple(mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError('full_sequence_hidden_mask_shape')
    if delta.ndim == 1:
        if delta.shape[0] != hidden.shape[-1]:
            raise ValueError('full_sequence_delta_hidden_shape')
        addition = delta.to(hidden.dtype)[None, None, :]
    elif delta.ndim == 2:
        if tuple(delta.shape) != (hidden.shape[0], hidden.shape[-1]):
            raise ValueError('full_sequence_delta_batch_shape')
        addition = delta.to(hidden.dtype)[:, None, :]
    else:
        raise ValueError('full_sequence_delta_rank')
    return torch.where(mask[:, :, None], hidden + addition, hidden)


def install_query_capture(engine):
    """Capture the current model-call mask once, reused by the selected layer."""
    engine._full_sequence_mask = None

    def capture(_module, args, kwargs):
        engine._full_sequence_mask = None
        if engine.injection_scope != FULL_SEQUENCE or engine.delta is None:
            return
        ids = kwargs.get('input_ids')
        embeds = kwargs.get('inputs_embeds')
        query = ids if ids is not None else embeds
        if query is None and args:
            query = args[0]
        if query is None or query.ndim not in (2, 3):
            raise ValueError('full_sequence_missing_query_tokens')
        engine._full_sequence_mask = query_mask(engine.t, kwargs.get('attention_mask'),
            int(query.shape[0]), int(query.shape[1]), query.device)

    return engine.model.register_forward_pre_hook(capture, with_kwargs=True)


def scoped_hook(engine, layer, original):
    """Historical path is called verbatim for controls and all inactive layers."""
    def hook(module, inputs, output):
        if (engine.injection_scope != FULL_SEQUENCE or engine.layer != layer
                or engine.delta is None):
            return original(module, inputs, output)
        h = output[0] if isinstance(output, tuple) else output
        if engine.capture:
            engine.captured[layer] = h[:, -1].detach().float().cpu()
        active = engine.mode == 'all' or engine.calls == 0
        engine.calls += 1
        if not active:
            return output
        mask = engine._full_sequence_mask
        if mask is None:
            raise RuntimeError('full_sequence_query_mask_not_captured')
        y = inject(engine.t, h, engine.delta, mask)
        audit = engine._generation_audit
        if audit is not None and layer == engine._audit_layer:
            audit.setdefault('full_sequence_query_shapes', []).append(list(mask.shape))
        return (y, *output[1:]) if isinstance(output, tuple) else y
    return hook

