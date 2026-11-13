"""Explicit Qwen3.5-family kernel adapters for the isolated 105 environment.

Call install_qwen35_kernels() before constructing the model. This changes only
the current Python process, never Transformers or model files on disk. The
Official Qwen3.5-27B uses this Qwen3.5-family implementation.
"""

import importlib.metadata
import inspect
import os


KERNEL_CALLS = {"chunk": 0, "recurrent": 0, "conv": 0, "conv_update": 0}
_ORIGINALS = {}
_STATUS = None


def install_qwen35_kernels(conv_backend=None):
    """Enable tested FLA/causal-conv1d functions; retain Liger as opt-in loss.

    FLA's cached decode operation is named fused_recurrent_gated_delta_rule,
    unlike the symbol sought by this installed Transformers release. Explicit
    adapters also prevent unrelated model kwargs leaking into kernel APIs.
    All packed-sequence callers must supply cu_seqlens (the model translates
    cu_seq_lens_q). This project currently uses ordinary padded sequences.
    """
    global _STATUS
    conv_backend = conv_backend or os.environ.get("NLA_CONV_BACKEND", "fla")
    if conv_backend not in ("torch", "fla", "causal"):
        raise ValueError("conv_backend must be torch, fla, or causal")
    if _STATUS is not None:
        if _STATUS["requested_conv_backend"] != conv_backend:
            raise ValueError("Kernel backend is fixed per process; start a fresh process to compare backends")
        return dict(_STATUS)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as model_code
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
             "causal_conv1d_fn", "causal_conv1d_update")
    for name in names:
        _ORIGINALS[name] = inspect.unwrap(getattr(model_code, name))
    conv_version = None
    if conv_backend == "causal":
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        conv_version = importlib.metadata.version("causal-conv1d")
    elif conv_backend == "torch":
        causal_conv1d_fn = _ORIGINALS["causal_conv1d_fn"]
        causal_conv1d_update = _ORIGINALS["causal_conv1d_update"]
    else:
        from fla.modules.convolution import causal_conv1d as fla_conv
        from fla.modules.convolution import causal_conv1d_update as fla_conv_update

        def causal_conv1d_fn(x, weight, bias=None, activation=None):
            output, _ = fla_conv(x.transpose(1, 2), weight, bias, activation=activation, backend="triton")
            return output.transpose(1, 2)

        def causal_conv1d_update(x, state, weight, bias=None, activation=None):
            output, _ = fla_conv_update(x.transpose(1, 2), state, weight=weight, bias=bias, activation=activation)
            return output.transpose(1, 2)


    def chunk(query, key, value, g, beta, chunk_size=64, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        if chunk_size != 64:
            raise ValueError("The tested adapter supports the model's default chunk_size=64")
        KERNEL_CALLS["chunk"] += 1
        return chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=kwargs.get("cu_seqlens"),
        )

    def recurrent(query, key, value, g, beta, initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        KERNEL_CALLS["recurrent"] += 1
        return fused_recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=kwargs.get("cu_seqlens"),
        )

    def conv(hidden_states, weight, bias=None, activation=None, **kwargs):
        KERNEL_CALLS["conv"] += 1
        return causal_conv1d_fn(hidden_states, weight, bias, activation=activation)

    def conv_update(hidden_states, conv_state, weight, bias=None, activation=None):
        KERNEL_CALLS["conv_update"] += 1
        return causal_conv1d_update(hidden_states, conv_state, weight, bias, activation=activation)

    model_code.torch_chunk_gated_delta_rule = chunk
    model_code.torch_recurrent_gated_delta_rule = recurrent
    model_code.causal_conv1d_fn = conv
    model_code.causal_conv1d_update = conv_update
    # Inference-only steering needs FLA but does not need Liger's training CE.
    try:
        liger_version = importlib.metadata.version("liger-kernel")
    except importlib.metadata.PackageNotFoundError:
        liger_version = None
    _STATUS = {
        "dispatch": "explicit process-local Qwen3.5-family kernel adapters",
        "fla_core": importlib.metadata.version("fla-core"),
        "causal_conv1d": conv_version,
        "requested_conv_backend": conv_backend,
        "conv_backend": {"causal":"causal-conv1d CUDA", "fla":"FLA Triton convolution", "torch":"Torch reference fallback"}[conv_backend],
        "liger_kernel": liger_version,
        "liger_note": ("available; no global model patch, opt into fused linear CE explicitly" if liger_version
                       else "not installed; optional training CE is not needed for steering inference"),
    }
    return dict(_STATUS)


def reference_kernel(name):
    """Return unwrapped Torch reference for numerical diagnostic tests only."""
    if name not in _ORIGINALS:
        raise ValueError("Install kernels first and specify a saved kernel name")
    return _ORIGINALS[name]


def kernel_call_counts():
    return dict(KERNEL_CALLS)

