"""Unconditional valid-label mass, not the conditional Q/B/S scoring formula."""
def selected_mass(torch, scaled_logits, groups):
    """FP64 positive partitions share their numerator; never clip or filter.

    Inputs are the exact FP32 temperature-scaled logits used by the original
    conditional softmax. This does not rescale logits or replace conditional
    scores/logprobs. Finite-input masses are bounded by construction because
    each denominator is numerator plus its nonnegative vocabulary complement.
    """
    if scaled_logits.ndim!=2 or scaled_logits.dtype!=torch.float32:
        raise ValueError('expected_original_FP32_scaled_logits')
    values=scaled_logits.to(torch.float64)
    weights=(values-values.amax(-1,keepdim=True)).exp()
    masses=[]
    for ids in groups:
        if not ids or len(set(ids))!=len(ids) or any(type(i)is not int or not 0<=i<values.shape[1]for i in ids):
            raise ValueError('invalid_label_partition')
        numerator=weights[:,ids].sum(-1)
        complement=weights.clone()
        complement[:,ids]=0
        denominator=numerator+complement.sum(-1)
        masses.append(numerator/denominator)
    return torch.stack(masses,-1)

