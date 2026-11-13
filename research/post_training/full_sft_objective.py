"""Exact full-SFT data/objective kernels; callers own model/FSDP lifecycle."""
from types import SimpleNamespace
c = SimpleNamespace(PLAN=dict(max_sequence=8192,total_updates=188,warmup_steps=10,lr=5e-6,final_lr_fraction=.1))

def validate_data(rows, schedule, *, expected_rows=1500, epochs=2, global_batch=16):
    if len(rows) != expected_rows or len({r['id'] for r in rows}) != expected_rows:
        raise RuntimeError('Training_row_identity_or_count')
    for row in rows:
        ids, labels = row['input_ids'], row['labels']
        if (not 1 < len(ids) <= c.PLAN['max_sequence'] or len(ids) != len(labels)
                or row['sequence_tokens'] != len(ids) or labels[0] != -100
                or any(type(v) is not int or v < 0 for v in ids)
                or any(type(v) is not int for v in labels)):
            raise RuntimeError('Invalid_encoded_sequence:' + row['id'])
        selected = [i for i, value in enumerate(labels) if value != -100]
        if (not selected or selected != list(range(selected[0], len(ids)))
                or any(labels[i] != ids[i] for i in selected)
                or row.get('prompt_tokens') != selected[0]
                or not row.get('task_complete') or not row.get('training_authorized')):
            raise RuntimeError('Incomplete_native_target:' + row['id'])
        sections = row.get('sections', [])
        if (len(sections) != len(ids) or not {'reasoning', 'thought_boundary', 'final', 'turn_end'}
                <= {sections[i] for i in selected}):
            raise RuntimeError('Missing_reasoning_final_EOS:' + row['id'])
    if len(schedule) != epochs:
        raise RuntimeError('Epoch_count')
    steps = (expected_rows + global_batch - 1) // global_batch
    for epoch in schedule:
        if len(epoch) != steps or any(len(batch) != global_batch for batch in epoch):
            raise RuntimeError('Schedule_dimensions')
        if any(type(i) is not int or i < -1 or i >= expected_rows for batch in epoch for i in batch):
            raise RuntimeError('Invalid_schedule_index')
        if sorted(i for batch in epoch for i in batch if i >= 0) != list(range(expected_rows)):
            raise RuntimeError('Each_real_row_requires_exactly_one_exposure_per_epoch')
        if (any(i == -1 for batch in epoch[:-1] for i in batch)
                or sum(i == -1 for i in epoch[-1]) != steps * global_batch - expected_rows):
            raise RuntimeError('Only_epoch_tail_may_have_zero_weight_placeholders')
    return dict(unique_rows=expected_rows, epochs=epochs, steps_per_epoch=steps,
                exposures=expected_rows * epochs)

def learning_rate(step):
    if not 1 <= step <= c.PLAN['total_updates']:
        raise ValueError('Update_out_of_range')
    if step <= c.PLAN['warmup_steps']:
        return c.PLAN['lr'] * step / c.PLAN['warmup_steps']
    fraction = (step - c.PLAN['warmup_steps']) / (c.PLAN['total_updates'] - c.PLAN['warmup_steps'])
    return c.PLAN['lr'] * (1 - fraction * (1 - c.PLAN['final_lr_fraction']))

def batch_target_tokens(rows, batch):
    return sum(sum(v != -100 for v in rows[i]['labels'][1:]) for i in batch if i >= 0)

def objective_scale(real, denominator, world_size=8):
    if denominator <= 0:
        raise ValueError('Empty_real_batch')
    # FSDP divides gradients by WORLD_SIZE. Compensate to get the SUM of
    # real-rank token losses divided by the global real-token count.
    return world_size / denominator if real else 0.

def rank_indices(batch, rank):
    """Two synchronized microsteps preserve the frozen 16-example batch."""
    if len(batch) != 16 or rank not in range(8):
        raise RuntimeError('Expected_16_slots_mapped_to_8_training_ranks')
    return (batch[rank], batch[rank + 8])

def make_loss_module(base, chunk=128):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    class NativeTokenLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base

        def forward(self, input_ids, labels, reference=False):
            hidden = self.base.model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                position_ids=torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0),
                use_cache=False, return_dict=True).last_hidden_state[0, :-1]
            shifted = labels[0, 1:]
            selected = shifted.ne(-100)
            hs, targets = hidden[selected], shifted[selected]
            loss = hidden.sum(dtype=torch.float32) * 0.

            def block(h, y):
                # The root FSDP wrapper gathers LM head parameters before this
                # forward and before backward checkpoint recomputation.
                return F.cross_entropy(self.base.lm_head(h).float(), y, reduction='sum')

            for offset in range(0, len(targets), chunk):
                loss = loss + checkpoint(block, hs[offset:offset + chunk], targets[offset:offset + chunk],
                                         use_reentrant=False)
            if reference:
                return loss, F.cross_entropy(self.base.lm_head(hs).float(), targets, reduction='sum')
            return loss

    return NativeTokenLoss()

