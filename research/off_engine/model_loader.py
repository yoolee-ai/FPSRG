"""Read-only original 27B and immutable-parent LoRA, never merged into base."""
from pathlib import Path
import re

import torch


EXPECTED_LORA_PARAMETERS = 116727808
EXPECTED_LORA_TENSORS = 992
EXPECTED_TARGETS = (r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)'
                    r'|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)'
                    r'|mlp\.(?:gate_proj|up_proj|down_proj))')


def validate_architecture(config):
    text = getattr(config, 'text_config', None)
    if (getattr(config, 'model_type', None) != 'qwen3_5' or text is None
            or getattr(text, 'num_hidden_layers', None) != 64
            or getattr(text, 'hidden_size', None) != 5120
            or getattr(config, 'quantization_config', None) is not None
            or getattr(text, 'quantization_config', None) is not None
            or getattr(text, 'num_experts', 0) not in (0, None)
            or getattr(text, 'num_local_experts', 0) not in (0, None)
            or getattr(text, 'max_position_embeddings', 0) < 4096
            or getattr(config, 'architectures', None) != ['Qwen3_5ForConditionalGeneration']):
        raise RuntimeError('Expected unquantized dense Qwen3.5-family27B, 64 layers, hidden5120')
    return dict(model_type=config.model_type, layers=64, hidden_size=5120,
                architecture='Qwen3_5ForConditionalGeneration', dense=True)


def validate_adapter_metadata(metadata):
    if (metadata.get('r') != 16 or metadata.get('lora_alpha') != 32
            or metadata.get('lora_dropout', 0) != 0 or metadata.get('bias') != 'none'
            or metadata.get('peft_type') != 'LORA' or metadata.get('task_type') != 'CAUSAL_LM'
            or metadata.get('target_modules') != EXPECTED_TARGETS
            or metadata.get('modules_to_save') or metadata.get('rank_pattern')
            or metadata.get('alpha_pattern') or metadata.get('layers_to_transform')
            or metadata.get('layer_replication') or metadata.get('target_parameters')
            or metadata.get('trainable_token_indices') or metadata.get('lora_bias')
            or metadata.get('use_dora') or metadata.get('use_rslora') or metadata.get('use_qalora')):
        raise RuntimeError('Parent adapter recipe or target inventory mismatch')


def assert_trainable(model):
    layers, count, tensors = set(), 0, 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        match = re.search(r'\.layers\.(\d+)\.', name)
        if 'lora_' not in name or 'visual' in name or match is None:
            raise RuntimeError('Immutable base or unexpected parameter trainable: ' + name)
        if parameter.dtype != torch.float32:
            raise RuntimeError('LoRA must remain FP32: ' + name)
        layers.add(int(match.group(1))); count += parameter.numel(); tensors += 1
    if layers != set(range(64)):
        raise RuntimeError('Expected trainable LoRA in all 64 language layers')
    if count != EXPECTED_LORA_PARAMETERS or tensors != EXPECTED_LORA_TENSORS:
        raise RuntimeError(f'Expected exact parent LoRA inventory: got {count} parameters, {tensors} tensors')
    return dict(layers=sorted(layers), parameters=count, tensors=tensors)


def freeze_base(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('lora_' in name and 'visual' not in name)
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return assert_trainable(model)


def verify_checkpoint(checkpoint):
    from runtime import read, sha
    checkpoint = Path(checkpoint)
    record = read(checkpoint / 'commit.json')
    if not record.get('files'):
        raise RuntimeError('Empty checkpoint commit')
    for name, digest in record['files'].items():
        # Inference releases retain the adapter but omit optimizer/RNG state.
        # Vector fitting freezes this adapter, so optimizer state is not loaded.
        if name == 'training_state.pt':
            continue
        target = checkpoint / name
        if target.resolve().is_relative_to(checkpoint.resolve()) is False:
            raise RuntimeError('Unsafe checkpoint member')
        if sha(target) != digest:
            raise RuntimeError('Checkpoint SHA mismatch: ' + str(target))
    for name in ('adapter/adapter_model.safetensors', 'adapter/adapter_config.json'):
        if name not in record['files']:
            raise RuntimeError('Missing committed adapter member: ' + name)
    return record


def load_adapter(model, checkpoint):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    verify_checkpoint(checkpoint)
    result = set_peft_model_state_dict(model, load_file(str(Path(checkpoint) / 'adapter/adapter_model.safetensors')))
    if result.unexpected_keys or any('lora_' in name for name in result.missing_keys):
        raise RuntimeError('Adapter key mismatch')
    assert_trainable(model)


def load_model(adapter=None, trainable=True):
    """Returns (base, model, tokenizer, audit); base shares live PEFT modules."""
    from runtime import MODEL
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration
    import acceleration
    kernels = acceleration.install_qwen35_kernels('fla')
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    architecture = validate_architecture(config)
    base, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, config=config, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': 0}, local_files_only=True, trust_remote_code=False,
        output_loading_info=True)
    if (info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs')
            or any(not name.startswith('mtp.') for name in info.get('unexpected_keys', []))):
        raise RuntimeError('Original model loading audit failed')
    base.requires_grad_(False)
    base.lm_head.float()
    base.lm_head.register_forward_pre_hook(lambda module, args: (args[0].float(), *args[1:]))
    base.eval(); base.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = base
    if adapter is not None:
        from peft import PeftModel
        from runtime import read
        verify_checkpoint(adapter)
        metadata = read(Path(adapter) / 'adapter/adapter_config.json')
        validate_adapter_metadata(metadata)
        model = PeftModel.from_pretrained(base, str(Path(adapter) / 'adapter'),
                                          is_trainable=True, autocast_adapter_dtype=True)
        scope = freeze_base(model)
        if not trainable:
            model.requires_grad_(False)
    else:
        scope = {'layers': [], 'parameters': 0}
    return base, model, tokenizer, {'kernels': kernels, 'scope': scope, 'architecture': architecture,
        'precision': 'BF16 immutable base; FP32 lm_head and LoRA',
        'quantization': False, 'attention': 'sdpa', 'merged_adapter': False}

