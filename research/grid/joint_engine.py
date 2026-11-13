"""Frozen Qwen3.5 joint, full-nonpadding-sequence probability engine.

Every cached tensor is the *entire* post-block/pre-intervention sequence.
One lazy bank per boundary is keyed by exactly its upstream coefficients;
changing a downstream coefficient cannot invalidate or alter that bank.
Autoregressive generation uses the identical intervention, without these banks.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import time
from types import MethodType

AXES = tuple('OCEANRTVMS')
LEVELS = (-2., -1., 0., 1., 2.)
APPROVED_LAYERS = (8, 16, 24, 32, 40, 48, 56, 60)
THRESHOLDS = dict(selected_logprob=1e-4, probability=1e-5)
VALIDATE_BATCH = 32


def coefficients(values):
    if isinstance(values, dict):
        if set(values) != set(AXES):
            raise ValueError('exactly_ten_named_coefficients_required')
        values = [values[axis] for axis in AXES]
    if len(values) != len(AXES):
        raise ValueError('ten_coefficients_required')
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value not in LEVELS for value in result):
        raise ValueError('only_frozen_grid_or_curve_levels_allowed')
    return result


def grid_coefficients(values, config):
    """Formal nodes cannot accidentally import old +/-2 validation conditions."""
    result = coefficients(values)
    if any(value not in config['levels'] for value in result):
        raise ValueError('formal_node_outside_approved_range_one_grid')
    return result


def routing(layerdict):
    if (set(layerdict) != set(AXES) or any(type(layerdict[a]) is not int
            or layerdict[a] not in APPROVED_LAYERS for a in AXES)):
        raise ValueError('ten_axes_require_approved_integer_layers')
    return {axis: layerdict[axis] for axis in AXES}


def prefix_key(values, layerdict, boundary):
    values = coefficients(values)
    return tuple((axis, values[index]) for index, axis in enumerate(AXES)
                 if layerdict[axis] < boundary)


def validate_configuration(config):
    batch = config.get('batch_size', VALIDATE_BATCH)
    if type(batch) is not int or batch != VALIDATE_BATCH:
        raise ValueError('fixed_batch_must_be_32_no_automatic_tuning')
    if config.get('temperature', .7) != .7:
        raise ValueError('frozen_temperature_is_0.7')
    if tuple(config.get('levels', ())) != (-1,0,1):
        raise ValueError('frozen_grid_levels_changed')
    if config.get('validation_reference_mode') not in ('saved', 'runtime'):
        raise ValueError('explicit_saved_or_runtime_reference_mode_required')
    if config.get('thresholds', THRESHOLDS) != THRESHOLDS:
        raise ValueError('numerical_thresholds_may_not_be_relaxed')
    if (config.get('logprob_tolerance', 1e-4) != 1e-4
            or config.get('probability_tolerance', 1e-5) != 1e-5):
        raise ValueError('frozen_numerical_tolerances_changed')
    return batch


def query_mask(torch, attention, hidden):
    if (attention is None or attention.ndim != 2 or hidden.ndim != 3
            or attention.shape[0] != hidden.shape[0]
            or attention.shape[1] < hidden.shape[1]):
        raise ValueError('explicit_full_sequence_attention_mask_required')
    return attention[:, -hidden.shape[1]:].to(device=hidden.device, dtype=torch.bool)


def inject(torch, hidden, delta, mask):
    if (hidden.ndim != 3 or delta.ndim != 1 or delta.shape[0] != hidden.shape[2]
            or tuple(mask.shape) != tuple(hidden.shape[:2])):
        raise ValueError('joint_injection_shape_mismatch')
    # The FP32 sum is cast exactly once, before the one BF16 addition.
    return torch.where(mask[:, :, None], hidden + delta.to(hidden.dtype)[None, None, :], hidden)


class Captured(Exception):
    pass


@contextmanager
def bypass_through(engine, hidden, boundary):
    """Re-enter Qwen's normal forward at one full-sequence decoder output."""
    if not 0 <= boundary < len(engine.layers) or hidden.ndim != 3:
        raise ValueError('invalid_full_sequence_cache_boundary')
    missing = object()
    previous = [(layer, layer.__dict__.get('forward', missing))
                for layer in engine.layers[:boundary + 1]]
    counts = [0] * (boundary + 1)

    def forward(index):
        def apply(_module, value, *args, **kwargs):
            if kwargs.get('use_cache') or kwargs.get('past_key_values') is not None:
                raise ValueError('staged_prefill_does_not_support_autoregressive_cache')
            counts[index] += 1
            if index != boundary:
                return value
            if (hidden.shape != value.shape or hidden.dtype != value.dtype
                    or hidden.device != value.device):
                raise ValueError('full_sequence_cache_shape_dtype_device_changed')
            return hidden
        return apply

    try:
        for index, (layer, _) in enumerate(previous):
            layer.forward = MethodType(forward(index), layer)
        yield counts
    finally:
        for layer, value in previous:
            if value is missing:
                layer.__dict__.pop('forward', None)
            else:
                layer.forward = value


class JointEngine:
    def __init__(self, gpu, jobs, vectors, layerdict, config):
        import numpy as np
        from engine import Engine
        from common import budget
        self.np, self.config, self.gpu = np, deepcopy(config), gpu
        self.batch_size = validate_configuration(config)
        self.layerdict = routing(layerdict)
        if config.get('layer_by_trait', self.layerdict) != self.layerdict:
            raise ValueError('resolved_vector_routing_differs_from_frozen_config')
        self.boundaries = tuple(sorted(set(self.layerdict.values())))
        self.jobs = deepcopy(list(jobs))
        if (len(self.jobs) != 640 or len({row['id'] for row in self.jobs}) != 640
                or sum(row['kind'] == 'likert' for row in self.jobs) != 480
                or sum(row['kind'] == 'behavior_ab' for row in self.jobs) != 160):
            raise ValueError('expected_480_likert_and_160_behavior_unique_jobs')
        for row in self.jobs:
            expected = list('12345') if row['kind'] == 'likert' else ['A', 'B']
            if row['labels'] != expected or not isinstance(row['prompt'], str):
                raise ValueError('measurement_label_or_prompt_contract_changed')
            if row.get('system', config['system']) != config['system']:
                raise ValueError('job_system_differs_from_frozen_engine_system')
        if set(vectors) != set(AXES):
            raise ValueError('exactly_ten_frozen_vectors_required')
        arrays = {}
        for axis in AXES:
            value = np.asarray(vectors[axis])
            if value.shape != (5120,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('frozen_vectors_must_be_finite_5120_FP32')
            arrays[axis] = np.ascontiguousarray(value).copy()
        self.budget = budget
        began = time.perf_counter()
        self.engine = Engine(gpu)
        self.t = self.engine.t
        self.engine.set(None, None, 0., 'all')
        self.engine.capture = False
        self.engine._generation_audit = None
        self.inputs = self.engine.inputs([row['prompt'] for row in self.jobs])
        self.device = self.inputs['input_ids'].device
        self.vectors = {axis: self.t.from_numpy(arrays[axis]).to(self.device) for axis in AXES}
        self.vector_sha256 = {axis: hashlib.sha256(arrays[axis].tobytes()).hexdigest() for axis in AXES}
        self.label_ids = {label: self.engine.tok.encode(label, add_special_tokens=False)
                          for label in list('12345') + ['A', 'B']}
        if any(len(value) != 1 for value in self.label_ids.values()):
            raise ValueError('all_measurement_labels_must_be_single_tokens')
        self.label_ids = {label: value[0] for label, value in self.label_ids.items()}
        self.all_label_ids = [self.label_ids[label] for label in list('12345') + ['A', 'B']]
        if len(set(self.all_label_ids)) != 7:
            raise ValueError('measurement_labels_must_be_seven_distinct_tokens')
        self.binding = hashlib.sha256(json.dumps(dict(jobs=self.jobs, layerdict=self.layerdict,
            vectors=self.vector_sha256, batch_size=self.batch_size,
            width=int(self.inputs['input_ids'].shape[1])), sort_keys=True).encode()).hexdigest()
        self.banks = {}
        self.cache_stats = dict(captures=0, replacements=0, hits=0)
        self._context_active = False
        self.protocol = None
        self._check_frozen()
        self._sync()
        self.load_seconds = time.perf_counter() - began

    def _check_frozen(self):
        if len(self.engine.layers) != 64:
            raise ValueError('frozen_backbone_requires_64_layers')
        if any(parameter.requires_grad or parameter.grad is not None
               for parameter in self.engine.model.parameters()):
            raise ValueError('all_backbone_and_LoRA_weights_must_remain_frozen')
        lora = [p for name, p in self.engine.model.named_parameters() if 'lora_' in name]
        if len(lora) != 992 or any(p.dtype != self.t.float32 for p in lora):
            raise ValueError('expected_992_frozen_FP32_LoRA_parameters')
        if self.engine.model.get_output_embeddings().weight.dtype != self.t.float32:
            raise ValueError('frozen_vocabulary_head_must_be_FP32')

    def _sync(self):
        if self.device.type == 'cuda':
            self.t.cuda.synchronize()

    def _deltas(self, values):
        values = coefficients(values)
        result = {}
        for layer in self.boundaries:
            active = [(axis, values[index]) for index, axis in enumerate(AXES)
                      if self.layerdict[axis] == layer and values[index] != 0.]
            if active:
                total = self.t.zeros_like(self.vectors[active[0][0]], dtype=self.t.float32)
                for axis, value in active:
                    total = total + self.vectors[axis] * value
                result[layer] = total
        return result

    @contextmanager
    def intervention(self, values, minimum_layer=None, _deltas=None):
        """Continuous input+decode hooks; safe around native model.generate.

        Caller supplies the model's explicit attention mask. Returned audit
        records every model forward and each active layer's query shape.
        """
        if self._context_active:
            raise RuntimeError('joint_intervention_contexts_may_not_be_nested')
        values = coefficients(values)
        deltas = self._deltas(values) if _deltas is None else _deltas
        self.engine.set(None, None, 0., 'all')
        self.engine.capture = False
        self.engine._generation_audit = None
        self._context_active = True
        attention = [None]
        audit = dict(forward_calls=0, layers={str(layer): [] for layer in deltas
                     if minimum_layer is None or layer >= minimum_layer},
                     injection_scope='full_sequence_nonpadding', coefficients=list(values))
        handles = []

        def capture_attention(_module, args, kwargs):
            attention[0] = kwargs.get('attention_mask')
            if attention[0] is None:
                raise ValueError('joint_hooks_require_explicit_attention_mask')
            audit['forward_calls'] += 1

        try:
            handles.append(self.engine.model.register_forward_pre_hook(capture_attention, with_kwargs=True))
            for layer, delta in deltas.items():
                if minimum_layer is not None and layer < minimum_layer:
                    continue

                def hook(_module, _args, output, layer=layer, delta=delta):
                    hidden = output[0] if isinstance(output, tuple) else output
                    mask = query_mask(self.t, attention[0], hidden)
                    changed = inject(self.t, hidden, delta, mask)
                    audit['layers'][str(layer)].append([int(hidden.shape[0]), int(hidden.shape[1])])
                    return (changed, *output[1:]) if isinstance(output, tuple) else changed

                handles.append(self.engine.layers[layer].register_forward_hook(hook))
            yield audit
        finally:
            for handle in reversed(handles):
                handle.remove()
            self.engine.set(None, None, 0., 'all')
            self._context_active = False

    def _batch(self, batch_index):
        start = batch_index * self.batch_size
        rows = list(range(start, min(start + self.batch_size, len(self.jobs))))
        if not rows:
            raise ValueError('invalid_batch_index')
        active = len(rows)
        rows += [rows[0]] * (self.batch_size - active)
        index = self.t.tensor(rows, dtype=self.t.long, device=self.device)
        return {name: value.index_select(0, index) for name, value in self.inputs.items()}, rows, active

    def _execute(self, batch_index, values, cached_hidden=None, cached_boundary=None,
                 capture_boundary=None):
        self.budget()
        inputs, _, _ = self._batch(batch_index)
        captured, handle = [], None

        def capture(_module, _args, output):
            if not self.t.is_tensor(output) or output.ndim != 3:
                raise ValueError('expected_full_sequence_tensor_decoder_output')
            captured.append(output.detach().clone())
            raise Captured()

        if capture_boundary is not None:
            # Registered BEFORE intervention hooks: cache excludes this layer's delta.
            handle = self.engine.layers[capture_boundary].register_forward_hook(capture)
        try:
            with self.intervention(values, minimum_layer=cached_boundary):
                try:
                    if cached_hidden is None:
                        result = self.engine.model(**inputs, use_cache=False, logits_to_keep=1)
                    else:
                        with bypass_through(self.engine, cached_hidden.clone(), cached_boundary) as counts:
                            result = self.engine.model(**inputs, use_cache=False, logits_to_keep=1)
                        if counts != [1] * (cached_boundary + 1):
                            raise RuntimeError('staged_bypass_invocation_mismatch')
                except Captured:
                    if capture_boundary is None:
                        raise
                    result = None
            if capture_boundary is not None:
                if len(captured) != 1 or result is not None:
                    raise RuntimeError('full_sequence_boundary_capture_missing')
                return captured[0]
            return result.logits[:, -1].float()
        finally:
            if handle is not None:
                handle.remove()

    def _ensure_bank(self, boundary_index, values, batch_index):
        layer = self.boundaries[boundary_index]
        key = (self.binding, prefix_key(values, self.layerdict, layer))
        bank = self.banks.get(layer)
        if bank is None or bank['key'] != key:
            if bank is not None:
                self.cache_stats['replacements'] += 1
            bank = self.banks[layer] = dict(key=key, chunks={})
        if batch_index not in bank['chunks']:
            if boundary_index == 0:
                hidden = self._execute(batch_index, (0.,) * len(AXES), capture_boundary=layer)
            else:
                previous = self._ensure_bank(boundary_index - 1, values, batch_index)
                hidden = self._execute(batch_index, values, previous,
                    self.boundaries[boundary_index - 1], capture_boundary=layer)
            bank['chunks'][batch_index] = hidden
            self.cache_stats['captures'] += 1
        else:
            self.cache_stats['hits'] += 1
        return bank['chunks'][batch_index]

    def _run(self, batch_index, values, use_cache):
        if use_cache:
            hidden = self._ensure_bank(len(self.boundaries) - 1, values, batch_index)
            return self._execute(batch_index, values, hidden, self.boundaries[-1])
        return self._execute(batch_index, values)

    def _measure(self, logits, batch_index):
        from stable_mass import selected_mass
        _, rows, active = self._batch(batch_index)
        scaled = logits / .7
        log_normalizer = scaled.logsumexp(-1)
        selected_all = scaled[:, self.all_label_ids]
        conditional = self.t.cat((selected_all[:, :5].softmax(-1),
                                  selected_all[:, 5:].softmax(-1)), dim=-1).cpu().tolist()
        legacy_masses = self.t.stack(((selected_all[:, :5].logsumexp(-1) - log_normalizer).exp(),
                              (selected_all[:, 5:].logsumexp(-1) - log_normalizer).exp()),
                             dim=-1).cpu().tolist()
        masses = selected_mass(self.t,scaled,[self.all_label_ids[:5],self.all_label_ids[5:]]).cpu().tolist()
        selected_logprobs = (selected_all - log_normalizer[:, None]).cpu().tolist()
        output = []
        for position, job_index in enumerate(rows[:active]):
            row = self.jobs[job_index]
            ids = [self.label_ids[label] for label in row['labels']]
            part = slice(0, 5) if row['kind'] == 'likert' else slice(5, 7)
            output.append(dict(row, probabilities=conditional[position][part],
                token_mass=masses[position][0 if row['kind'] == 'likert' else 1],
                legacy_token_mass=legacy_masses[position][0 if row['kind'] == 'likert' else 1],
                selected_logprobs=selected_logprobs[position][part], token_ids=ids))
            current = output[-1]
            if (any(not math.isfinite(value) for value in current['probabilities']
                    + current['selected_logprobs'] + [current['token_mass']])
                    or abs(sum(current['probabilities']) - 1.) > 1e-5
                    or not 0. <= current['token_mass'] <= 1.000001):
                from measure_diagnostics import diagnose_logits
                from common import RUN, save
                detail=diagnose_logits(logits,self.all_label_ids,[self.jobs[i]for i in rows[:active]],
                    context=dict(batch_index=batch_index,worker=self.gpu,stable_masses=masses))
                save(RUN/'numerical_failures'/f'gpu{self.gpu}_{time.time_ns()}.json',detail)
                raise RuntimeError('nonfinite_or_unnormalized_measurement_probabilities')
        return output

    def probe(self, values, use_cache=True):
        """Return 640 job records with conditional probabilities and token_mass."""
        if self.protocol is None:
            raise RuntimeError('validate_before_formal_scoring')
        values = coefficients(values)
        output = []
        with self.t.inference_mode():
            for batch_index in range((len(self.jobs) + self.batch_size - 1) // self.batch_size):
                logits = self._run(batch_index, values, use_cache)
                output.extend(dict(row, coefficients=list(values)) for row in self._measure(logits, batch_index))
        return output

    def probe_chunk(self, nodes, on_batch=None):
        """Exact B32 math; batch-major traversal bounds all prefix banks to B32.

        Node order affects only cache hits, not model numerics. Never retain
        640 full sequences at every injection boundary in GPU memory.
        """
        if self.protocol is None:
            raise RuntimeError('validate_before_formal_scoring')
        values=[grid_coefficients(row['coefficients'],self.config) for row in nodes]
        result=[[] for _ in nodes]
        with self.t.inference_mode():
            for batch_index in range((len(self.jobs)+self.batch_size-1)//self.batch_size):
                self.clear_cache()
                for index,value in enumerate(values):
                    logits=self._run(batch_index,value,True)
                    result[index].extend(self._measure(logits,batch_index))
                if on_batch:on_batch(batch_index+1)
                self.clear_cache()
        return result

    def _comparison(self, reference, candidate, batch_index):
        ref = (reference / .7).log_softmax(-1)[:, self.all_label_ids]
        alt = (candidate / .7).log_softmax(-1)[:, self.all_label_ids]
        log_error = float((ref - alt).abs().max())
        a, b = self._measure(reference, batch_index), self._measure(candidate, batch_index)
        prob_error = max(abs(x - y) for r, s in zip(a, b)
                         for x, y in zip(r['probabilities'] + [r['token_mass']],
                                         s['probabilities'] + [s['token_mass']]))
        if (not math.isfinite(log_error) or not math.isfinite(prob_error)
                or log_error > THRESHOLDS['selected_logprob']
                or prob_error > THRESHOLDS['probability']):
            raise RuntimeError(f'joint_full_staged_numerical_mismatch:{log_error}:{prob_error}')
        return dict(selected_logprob_error=log_error, probability_error=prob_error)

    def _hook_audit(self, batch_index):
        values = (1., -1., 2., 1., -2., -1., 2., -2., 1., 2.)
        expected = self._deltas(values)
        inputs, _, _ = self._batch(batch_index)
        before, observed, handles = {}, {}, []
        for layer in expected:
            def capture(_module, _args, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                before[layer] = hidden.detach().clone()
            handles.append(self.engine.layers[layer].register_forward_hook(capture))
        try:
            with self.intervention(values):
                for layer in expected:
                    def inspect(_module, _args, output, layer=layer):
                        hidden = output[0] if isinstance(output, tuple) else output
                        mask = inputs['attention_mask'].bool()
                        # Independent indexed reference, without production inject().
                        wanted = before[layer].clone()
                        addition = expected[layer].to(wanted.dtype).expand_as(wanted)
                        wanted[mask] += addition[mask]
                        observed[layer] = dict(exact=bool(self.t.equal(hidden, wanted)),
                            padding_unchanged=bool(self.t.equal(hidden[~mask], before[layer][~mask])),
                            nonpadding_positions=int(mask.sum()), padding_positions=int((~mask).sum()))
                        del before[layer]
                    handles.append(self.engine.layers[layer].register_forward_hook(inspect))
                self.engine.model(**inputs, use_cache=False, logits_to_keep=1)
        finally:
            for handle in reversed(handles):
                handle.remove()
        if (set(observed) != set(expected) or any(not row['exact'] or not row['padding_unchanged']
                or row['nonpadding_positions'] <= self.batch_size for row in observed.values())):
            raise RuntimeError('full_sequence_joint_mask_or_same_layer_sum_failed')
        # A synthetic cancelling pair at a genuine layer must preserve every bit.
        layer = self.boundaries[-1]
        cancel = self.vectors['O'] + (-self.vectors['O'])
        self.engine.set(None, None, 0., 'all')
        bare = self.engine.model(**inputs, use_cache=False, logits_to_keep=1).logits
        with self.intervention((0.,) * len(AXES), _deltas={layer: cancel}):
            cancelled = self.engine.model(**inputs, use_cache=False, logits_to_keep=1).logits
        if not self.t.equal(bare, cancelled):
            raise RuntimeError('joint_exact_cancellation_changed_logits')
        return dict(layers={str(key): value for key, value in observed.items()}, cancellation_exact=True)

    def _decode_audit(self, values):
        inputs, _, _ = self._batch(0)
        full = self._run(0, values, False)
        generation = deepcopy(self.engine.model.generation_config)
        generation.max_new_tokens = generation.min_new_tokens = 2
        generation.do_sample = False
        generation.return_dict_in_generate = True
        generation.output_scores = False
        generation.output_logits = True
        generation.use_cache = True
        with self.intervention(values) as audit:
            native = self.engine.model.generate(**inputs, generation_config=generation, logits_to_keep=1)
        width = int(inputs['input_ids'].shape[1])
        if (len(native.logits) != 2 or audit['forward_calls'] != 2
                or any(shapes != [[self.batch_size, width], [self.batch_size, 1]]
                       for shapes in audit['layers'].values())):
            raise RuntimeError('joint_native_cached_decode_hook_continuity_failed')
        comparison = self._comparison(full, native.logits[0].float(), 0)
        return dict(passed=True, generated_tokens=2, audit=audit, prefill_comparison=comparison)

    def validate(self):
        """Bounded admission; no tolerance search, retraining, or hidden retries."""
        if self.protocol is not None:
            raise RuntimeError('validation_is_once_per_worker')
        self._check_frozen()
        began = time.perf_counter()
        zero, checks, anchors = (0.,) * len(AXES), [], []
        # Include both measurement families while retaining production B/width.
        batch_indices = sorted({0, next(i for i, row in enumerate(self.jobs)
                                        if row['kind'] == 'behavior_ab') // self.batch_size})
        with self.t.inference_mode():
            for batch_index in batch_indices:
                inputs, _, _ = self._batch(batch_index)
                self.engine.set(None, None, 0., 'all')
                bare = self.engine.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                hooked_zero = self._run(batch_index, zero, False)
                if not self.t.equal(bare, hooked_zero):
                    raise RuntimeError('joint_zero_differs_from_disabled_model')
                checks.append(self._comparison(bare, self._run(batch_index, zero, True), batch_index))
                anchors.extend((bare / .7).log_softmax(-1)[0, self.all_label_ids].cpu().tolist())
            for axis_index, axis in enumerate(AXES):
                row = [0.] * len(AXES)
                row[axis_index] = -1. if axis_index % 2 else 1.
                joint = self._run(0, row, False)
                # Independent historical full-sequence single-vector hooks.
                self.engine.set(self.layerdict[axis], self.vectors[axis], row[axis_index], 'all')
                inputs, _, _ = self._batch(0)
                single = self.engine.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                self.engine.set(None, None, 0., 'all')
                checks.append(self._comparison(single, joint, 0))
                checks.append(self._comparison(joint, self._run(0, row, True), 0))
            mixed = (2., -2., 1., -1., 2., -2., 1., 2., -1., 2.)
            for batch_index in batch_indices:
                full = self._run(batch_index, mixed, False)
                staged = self._run(batch_index, mixed, True)
                checks.append(self._comparison(full, staged, batch_index))
                anchors.extend((full / .7).log_softmax(-1)[0, self.all_label_ids].cpu().tolist())
            hooks = self._hook_audit(batch_indices[-1])
            decode = self._decode_audit(mixed)
            # Warm calls cannot mutate cached full-sequence tensors. A-B-A
            # changes all upstream axes and must rebuild the correct banks.
            first = self._run(0, mixed, True)
            saved = {layer: bank['chunks'][0].clone() for layer, bank in self.banks.items()
                     if 0 in bank['chunks']}
            warm = self._run(0, mixed, True)
            if not self.t.equal(first, warm) or any(not self.t.equal(value, self.banks[layer]['chunks'][0])
                                                   for layer, value in saved.items()):
                raise RuntimeError('warm_probe_mutated_full_sequence_bank')
            changed = tuple(-value for value in mixed)
            keys_before = {layer: bank['key'] for layer, bank in self.banks.items()}
            self._run(0, changed, True)
            for layer, bank in self.banks.items():
                should_change = prefix_key(mixed, self.layerdict, layer) != prefix_key(changed, self.layerdict, layer)
                if (bank['key'] != keys_before[layer]) != should_change:
                    raise RuntimeError('upstream_cache_key_invalidation_failed')
            repeated = self._run(0, mixed, True)
            if not self.t.equal(first, repeated):
                raise RuntimeError('cache_A_B_A_logits_changed')
            if any(not self.t.equal(value, self.banks[layer]['chunks'][0]) for layer, value in saved.items()):
                raise RuntimeError('cache_A_B_A_hidden_sequences_changed')
            self._check_frozen()
        self._sync()
        self.protocol = dict(strategy='joint_full_sequence_staged', batch_size=self.batch_size,
            axis_order=list(AXES), layer_by_trait=dict(self.layerdict), levels=list(self.config['levels']), curve_levels=list(LEVELS),
            temperature=.7, thresholds=dict(THRESHOLDS), jobs_per_node=len(self.jobs),
            prompt_width=int(self.inputs['input_ids'].shape[1]), injection_scope='full_sequence_nonpadding',
            cache_boundaries=list(self.boundaries), cache_capacity='one_B32_batch_one_upstream_key_per_boundary',
            cache_representation='entire_post_block_pre_injection_sequence',
            sum_rule='axis_order_FP32_sum_then_one_BF16_cast_add_nonpadding',
            measurement_numerics='mass_FP64_positive_partition; conditional_and_logprobs_original_FP32',
            backbone_dtype='bfloat16', lora_dtype='float32', head_dtype='float32',
            vector_sha256=dict(self.vector_sha256), binding=self.binding)
        return dict(passed=True, protocol=self.protocol, anchor=anchors,
            anchor_token_ids=self.all_label_ids * (2 * len(batch_indices)),
            anchor_batch_indices=batch_indices, anchor_conditions=['zero', 'mixed'],
            zero_disabled_logits_bitwise_equal=True, single_vector_equivalence=True,
            full_sequence_hook_audit=hooks, cache_A_B_A_bitwise_equal=True,
            continuous_native_decode=decode,
            warm_cache_immutable=True, upstream_cache_invalidation=True,
            max_selected_logprob_error=max(row['selected_logprob_error'] for row in checks),
            max_probability_error=max(row['probability_error'] for row in checks),
            comparison_count=len(checks), model_load_seconds=self.load_seconds,
            validation_seconds=time.perf_counter() - began, cache_stats=dict(self.cache_stats))

    def clear_cache(self):
        self.banks.clear()

