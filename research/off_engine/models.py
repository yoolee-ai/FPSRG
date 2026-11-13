"""Keep the exact original official unmerged parent loader and precision."""
def load_av(model_path,*,rank=0,**kwargs):
    if rank!=0:raise RuntimeError('No backbone/LoRA optimization authorized')
    from common import PARENT,RUN,CONFIG,sha
    from model_loader import load_model
    commit_sha=sha(PARENT/'commit.json')
    if commit_sha!=CONFIG['expected_parent_commit_sha256']:raise RuntimeError('wrong_latest_persononly_parent')
    base,wrapped,tok,audit=load_model(PARENT,trainable=False)
    base.requires_grad_(False);base.eval()
    trainable=sum(p.numel() for p in base.parameters() if p.requires_grad)
    lora=[p for n,p in base.named_parameters() if 'lora_' in n]
    if trainable or len(lora)!=992 or any(str(p.dtype)!='torch.float32' for p in lora):raise RuntimeError('parent_freeze_or_precision_failure')
    base._nla_load_report=dict(layers=64,hidden_size=5120,rank=16,lora_modules=[n for n,m in base.named_modules() if hasattr(m,'lora_A')],trainable_parameters=trainable,unmerged_parent=str(PARENT),parent_commit_sha256=commit_sha,model_version=CONFIG['parent_model_version'],lora_tensors=len(lora),loader_audit=audit)
    return base
def get_text_layers(model):return model.model.language_model.layers


