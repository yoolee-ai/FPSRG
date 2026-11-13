"""Same forward math; retain two shallow banks across grid chunks.

Each retained bank contains ONE prompt batch and ONE upstream coefficient key.
The complete (not last-token) hidden sequence remains on its original device.
Every boundary deeper than the first two is discarded after each prompt batch.
The original A routing retains exactly L16/L32 and leaves L40 transient.
"""
from joint_engine import JointEngine, coefficients, grid_coefficients

CAPACITY = 'L16_L32_one_upstream_key_per_each_of_20_B32_batches_L40_transient'
ORIGINAL_CAPACITY = 'one_B32_batch_one_upstream_key_per_boundary'


def retained_boundaries(boundaries):
    if (not boundaries or tuple(sorted(set(boundaries))) != tuple(boundaries)
            or any(type(layer) is not int for layer in boundaries)):
        raise ValueError('sorted_distinct_boundaries_required')
    return tuple(boundaries[:2])


def cache_capacity(batch, boundaries=(16, 32, 40)):
    if batch not in (32, 128):
        raise ValueError('unsupported_batch')
    shallow = retained_boundaries(boundaries)
    retained = '_'.join(f'L{layer}' for layer in shallow)
    transient = '_'.join(f'L{layer}' for layer in boundaries[len(shallow):]) or 'none'
    return f'{retained}_one_upstream_key_per_each_of_{640//batch}_B{batch}_batches_{transient}_transient'


class PersistentEngine(JointEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.jobs) != 640 or self.batch_size != 32:
            raise RuntimeError('persistent_cache_requires_exact_existing_layout')
        self.retained = {layer: {} for layer in retained_boundaries(self.boundaries)}

    def enter_batch(self, batch_index):
        if type(batch_index) is not int or not 0 <= batch_index < len(self.jobs)//self.batch_size:
            raise ValueError('invalid_persistent_batch')
        self.banks.clear()
        for layer, batches in self.retained.items():
            if batch_index in batches:
                bank = batches[batch_index]
                if set(bank['chunks']) != {batch_index}:
                    raise RuntimeError('retained_bank_must_have_one_exact_batch')
                self.banks[layer] = bank

    def leave_batch(self, batch_index):
        for layer, batches in self.retained.items():
            if layer in self.banks:
                bank = self.banks[layer]
                if set(bank['chunks']) != {batch_index}:
                    raise RuntimeError('unexpected_cross_batch_cache_alias')
                batches[batch_index] = bank
        self.banks.clear()

    def clear_cache(self):
        super().clear_cache()
        for batches in getattr(self, 'retained', {}).values():
            batches.clear()

    def persistent_bytes(self):
        return sum(value.numel() * value.element_size()
                   for batches in self.retained.values()
                   for bank in batches.values() for value in bank['chunks'].values())

    def probe_chunk(self, nodes, on_batch=None):
        if self.protocol is None:
            raise RuntimeError('validate_before_formal_scoring')
        values = [grid_coefficients(row['coefficients'], self.config) for row in nodes]
        result = [[] for _ in nodes]
        with self.t.inference_mode():
            for batch_index in range(len(self.jobs)//self.batch_size):
                self.enter_batch(batch_index)
                try:
                    for index, value in enumerate(values):
                        logits = self._run(batch_index, value, True)
                        result[index].extend(self._measure(logits, batch_index))
                finally:
                    self.leave_batch(batch_index)
                if on_batch is not None:
                    on_batch(batch_index + 1)
        return result

    def validate_persistent(self):
        """Bounded cross-batch, upstream-key, reuse, and immutability check."""
        self.clear_cache()
        first = (2., -2., 1., -1., 2., -2., 1., 2., -1., 2.)
        downstream = list(first); downstream[0] = -2.; downstream[6] = -1.
        upstream = list(first); upstream[3] = 1.
        if self.boundaries != (16, 32, 40):
            # For new layer routings, explicitly perturb one last-boundary axis
            # and one first-boundary axis. Preserve the old A admission sequence.
            axes = tuple('OCEANRTVMS')
            deep = next(i for i, axis in enumerate(axes)
                        if self.layerdict[axis] == self.boundaries[-1])
            early = next(i for i, axis in enumerate(axes)
                         if self.layerdict[axis] == self.boundaries[0])
            downstream = list(first); downstream[deep] = -first[deep]
            upstream = list(first); upstream[early] = -first[early]
        cases = [(0, first), (16, first), (0, downstream),
                 (16, upstream), (0, upstream), (0, first)]
        checks = []
        with self.t.inference_mode():
            for batch, values in cases:
                reference = self._run(batch, values, False)
                self.enter_batch(batch)
                candidate = self._run(batch, values, True)
                saved = {layer: bank['chunks'][batch].clone()
                         for layer, bank in self.banks.items()}
                warm = self._run(batch, values, True)
                if not self.t.equal(candidate, warm) or any(
                    not self.t.equal(value, self.banks[layer]['chunks'][batch])
                    for layer, value in saved.items()
                ):
                    raise RuntimeError('persistent_warm_cache_mutation')
                checks.append(dict(batch=batch, coefficients=list(values),
                                   **self._comparison(reference, candidate, batch)))
                self.leave_batch(batch)
                del reference, candidate, warm, saved
        self.clear_cache()
        return dict(passed=True, comparisons=checks, cross_batch_and_upstream_keys=True,
                    warm_cache_immutable=True, retained_boundaries=list(self.retained),
                    transient_boundaries=list(self.boundaries[len(self.retained):]),
                    maximum_retained_batches=len(self.retained)*(len(self.jobs)//self.batch_size))

    def validate(self):
        result = super().validate()
        result['persistent_cache'] = self.validate_persistent()
        for output_name, field in (('max_selected_logprob_error', 'selected_logprob_error'),
                                   ('max_probability_error', 'probability_error')):
            result[output_name] = max(result[output_name], max(
                row[field] for row in result['persistent_cache']['comparisons']))
        self.protocol['cache_capacity'] = cache_capacity(self.batch_size, self.boundaries)
        result['protocol'] = self.protocol
        return result

