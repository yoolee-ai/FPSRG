"""Checkpointing helper used by the preserved refinement loop."""

def checkpointing(model, enabled):
    if enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.gradient_checkpointing_disable()
    # External injected vectors carry gradients; frozen embeddings must not acquire them.
    if hasattr(model, '_require_grads_hook'):
        model.disable_input_require_grads()

