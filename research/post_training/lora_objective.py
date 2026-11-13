"""Unchanged proven objective: full native target CE, no packing or truncation."""
import random

def schedule(rows,seed):
    # Length grouping limits idle ranks; every target appears exactly once.
    rng=random.Random(seed);order=list(range(len(rows)));rng.shuffle(order)
    order.sort(key=lambda i:rows[i]['sequence_tokens'])
    batches=[order[i:i+8] for i in range(0,len(order),8)];rng.shuffle(batches)
    for batch in batches:rng.shuffle(batch)
    if len(batches)!=125 or any(len(b)!=8 for b in batches):raise ValueError('Expected125fullbatches')
    if sorted(i for b in batches for i in b)!=list(range(1000)):raise ValueError('Coverage_mismatch')
    return batches

def lr(step,config):
    if step<=config['warmup_steps']:return config['lr']*step/config['warmup_steps']
    x=(step-config['warmup_steps'])/(config['steps']-config['warmup_steps'])
    return config['lr']*(1-x*(1-config['final_lr_fraction']))

def token_loss(base,row,device='cuda:0',chunk=32):
    import torch
    from torch.utils.checkpoint import checkpoint
    import torch.nn.functional as F
    ids=torch.tensor([row['input_ids']],device=device,dtype=torch.long)
    labels=torch.tensor(row['labels'][1:],device=device,dtype=torch.long)
    mask=labels.ne(-100)
    hidden=base.model(input_ids=ids,attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(ids.shape[1],device=device).unsqueeze(0),
        use_cache=False,return_dict=True).last_hidden_state[0,:-1]
    hs=hidden[mask];targets=labels[mask];loss=hidden.sum()*0
    def block(h,y):return F.cross_entropy(base.lm_head(h.float()),y,reduction='sum')
    for start in range(0,len(targets),chunk):
        h,y=hs[start:start+chunk],targets[start:start+chunk]
        loss=loss+checkpoint(block,h,y,use_reentrant=False)
    return loss,len(targets)

def reduce_gradients(parameters,dist):
    import torch
    buckets=[];size=0
    def flush():
        flat=torch.cat([p.grad.reshape(-1) for p in buckets]);dist.all_reduce(flat,op=dist.ReduceOp.SUM)
        pos=0
        for p in buckets:p.grad.copy_(flat[pos:pos+p.numel()].view_as(p));pos+=p.numel()
    for p in parameters:
        if p.grad is None:raise RuntimeError('unused_trainable_parameter')
        if buckets and size+p.numel()>4*1024*1024:flush();buckets=[];size=0
        buckets.append(p);size+=p.numel()
    if buckets:flush()

