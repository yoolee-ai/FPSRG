"""Exact-prefix throughput successor, admitted against original B32 scores.

Reference validation uses the unchanged original engine and batch. Production
uses a separately recorded B128 layout; token order/width/scores never change.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import time
from persistent_engine import PersistentEngine, cache_capacity
from joint_engine import coefficients, grid_coefficients, THRESHOLDS


def specification(kernel, batch):
    if kernel not in ('prefix', 'packed_mlp') or batch not in (32, 128):
        raise ValueError('unreviewed_acceleration')
    return dict(mode='prefix_fused', kernel=kernel, packed_alignment=256,
        production_batch_size=batch, job_order='original_canonical',
        attention_width=318, LoRA_A='original_full_batch_shape_FP32',
        full_nonpadding_input_injection_unchanged=True, all_real_tokens_retained=True,
        weights_and_dtypes_unchanged=True, conditional_scores_and_masses_unchanged=True,
        optimization_prefill_only=True,
        prefix_equivalence='entire_token_and_padding_mask_prefix_same_absolute_position')


def layout_binding(reference, batch, kernel):
    return hashlib.sha256(json.dumps(dict(reference_binding=reference,
        acceleration=specification(kernel,batch)),sort_keys=True).encode()).hexdigest()


def capacity(batch, boundaries=(16, 32, 40)):
    return cache_capacity(batch, boundaries)


def reference_conditions(mode):
    """Old A references are unchanged; B references use only B's own vectors."""
    if mode not in ('saved', 'runtime'):
        raise ValueError('explicit_saved_or_runtime_reference_mode_required')
    return ((0.,)*10, ((-2.,) if mode == 'saved' else (-1.,))*10)


def validate_reference_rows(references, jobs, mode):
    if (len(references) != 2 or
            [tuple(row['coefficients']) for row in references] != list(reference_conditions(mode))):
        raise RuntimeError('canonical_zero_and_negative_references_required')
    for reference in references:
        for name in ('probabilities', 'selected_logprobs', 'token_mass'):
            if len(reference[name]) != len(jobs):
                raise RuntimeError('reference_must_cover_all_canonical_jobs')
        for i, job in enumerate(jobs):
            if (len(reference['probabilities'][i]) != len(job['labels']) or
                    len(reference['selected_logprobs'][i]) != len(job['labels'])):
                raise RuntimeError('reference_label_shape_mismatch')
            values = (reference['probabilities'][i] + reference['selected_logprobs'][i]
                      + [reference['token_mass'][i]])
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError('nonfinite_reference')


class AcceleratedEngine(PersistentEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel=self.config['acceleration_kernel']
        self.production_batch=self.config['acceleration_batch_size']
        self.reference_mode=self.config['validation_reference_mode']
        self.acceleration_specification=specification(self.kernel,self.production_batch)
        self.reference_binding=self.binding
        self.batch_count=20  # During canonical reference validation only.

    @contextmanager
    def optimized(self,batch,verify=False):
        if self.kernel=='packed_mlp':
            from fused_mlp_ops import prefix_compaction
        else:
            from batch_prefix_ops import prefix_compaction
        with prefix_compaction(self,batch,'prefix_fused',verify_prefix=verify) as record:
            yield record

    def _set_production_layout(self):
        self.clear_cache()
        self.batch_size=self.production_batch
        self.batch_count=640//self.batch_size
        self.binding=layout_binding(self.reference_binding,self.batch_size,self.kernel)
        self._batch_order_key='canonical'

    def _runtime_references(self):
        """Unoptimized canonical B32 forwards using THIS run's resolved vectors.

        This never reads A's saved nonzero answers. References are held only as
        small CPU probability arrays; whole hidden states are not retained.
        """
        if self.batch_size != 32:
            raise RuntimeError('runtime_reference_requires_original_B32_layout')
        references=[]
        with self.t.inference_mode():
            for values in reference_conditions('runtime'):
                records=[]
                for batch in range(len(self.jobs)//self.batch_size):
                    self.clear_cache()
                    logits=self._run(batch,values,False)
                    records.extend(self._measure(logits,batch))
                    del logits
                references.append(dict(coefficients=list(values),
                    probabilities=[row['probabilities'] for row in records],
                    token_mass=[row['token_mass'] for row in records],
                    selected_logprobs=[row['selected_logprobs'] for row in records],
                    source_kind='runtime_current_vectors_unoptimized_B32_full_forward',
                    reference_binding=self.reference_binding,
                    vector_sha256=dict(self.vector_sha256)))
        self.clear_cache()
        return references

    def validate(self):
        from common import RUN,read,sha
        began=time.perf_counter()
        # Full canonical admission retains historical mask/cache/single-vector
        # and native decode checks. It is not claimed to validate the optimizer.
        result=super().validate()
        original_protocol=dict(result['protocol'])
        original_anchor=list(result['anchor'])
        if self.reference_mode=='runtime':
            references=self._runtime_references()
        else:
            references=read(RUN/'acceleration_references.json')
        validate_reference_rows(references,self.jobs,self.reference_mode)
        self._set_production_layout()
        zero=(0.,)*10
        mixed=(2.,-2.,1.,-1.,2.,-2.,1.,2.,-1.,2.)
        checks=[]; anchors=[]
        with self.t.inference_mode():
            for reference in references:
                if (self.reference_mode=='saved' and
                        sha(reference['source_path'])!=reference['source_sha256']):
                    raise RuntimeError('saved_reference_changed')
                for batch in range(self.batch_count):
                    self.clear_cache()
                    with self.optimized(batch,verify=True) as packing:
                        self.enter_batch(batch)
                        candidate=self._run(batch,reference['coefficients'],True)
                        output=self._measure(candidate,batch)
                        self.leave_batch(batch)
                    rows=self._batch(batch)[1]
                    pe=max(abs(a-b) for i,row in zip(rows,output)
                        for a,b in zip(row['probabilities']+[row['token_mass']],
                            reference['probabilities'][i]+[reference['token_mass'][i]]))
                    le=max(abs(a-b) for i,row in zip(rows,output)
                        for a,b in zip(row['selected_logprobs'],reference['selected_logprobs'][i]))
                    if (not math.isfinite(pe) or not math.isfinite(le)
                            or pe>THRESHOLDS['probability'] or le>THRESHOLDS['selected_logprob']):
                        raise RuntimeError(f'canonical_saved_score_mismatch:{le}:{pe}')
                    checks.append(dict(batch=batch,coefficients=reference['coefficients'],
                        selected_logprob_error=le,probability_error=pe,packing=packing))
                    del candidate,output
            # Actual optimized anchors for the SAME global questions as the
            # original B32 anchor: job0 and job480, zero then mixed, seven labels.
            self.clear_cache()
            cache_checks=[]
            for values in (zero,mixed):
                for job in (0,480):
                    batch=job//self.batch_size; position=job%self.batch_size
                    with self.optimized(batch,verify=True):
                        self.enter_batch(batch)
                        candidate=self._run(batch,values,True)
                        saved={l:b['chunks'][batch].clone() for l,b in self.banks.items()}
                        warm=self._run(batch,values,True)
                        if not self.t.equal(candidate,warm) or any(not self.t.equal(v,
                                self.banks[l]['chunks'][batch]) for l,v in saved.items()):
                            raise RuntimeError('optimized_warm_cache_mutation')
                        cache_checks.append(dict(batch=batch,values=list(values),immutable=True))
                        anchors.extend((candidate/.7).log_softmax(-1)[position,self.all_label_ids].cpu().tolist())
                        self.leave_batch(batch)
                    del candidate,warm,saved
            self.clear_cache()
            # Full-input hook scope and exact cancellation on actual production B.
            with self.optimized(480//self.batch_size):
                hook_audit=self._hook_audit(480//self.batch_size)
            self.clear_cache()
            # Allocate every retained bank under the real B128 layout; verifies
            # memory feasibility and cross-batch identity before any formal node.
            for batch in range(self.batch_count):
                with self.optimized(batch):
                    self.enter_batch(batch)
                    self._run(batch,mixed,True)
                    self.leave_batch(batch)
            if [len(b) for b in self.retained.values()]!=[self.batch_count]*len(self.retained):
                raise RuntimeError('retained_production_bank_count')
            retained_bytes=self.persistent_bytes()
        self.clear_cache(); self._check_frozen()
        error=max(abs(a-b) for a,b in zip(anchors,original_anchor))
        if len(anchors)!=len(original_anchor) or not math.isfinite(error) or error>1e-4:
            raise RuntimeError('optimized_canonical_anchor_mismatch')
        self.protocol=dict(original_protocol,batch_size=self.batch_size,
            binding=self.binding,reference_binding=self.reference_binding,
            cache_capacity=capacity(self.batch_size,self.boundaries),
            validation_reference_mode=self.reference_mode,
            tokenwise_acceleration=self.acceleration_specification)
        result.update(protocol=self.protocol,anchor=anchors,
            # Legacy fields describe identity of reference anchor questions,
            # NOT production batch indices. Actual layout is explicit below.
            anchor_global_job_indices=[0,480],
            accelerated_comparisons=dict(passed=True,comparisons=checks,
                original_canonical_reference=True,canonical_anchor_error=error,
                reference_mode=self.reference_mode,
                reference_conditions=[list(values) for values in reference_conditions(self.reference_mode)],
                reference_vector_sha256=dict(self.vector_sha256),
                runtime_reference_sha256=(hashlib.sha256(json.dumps(references,sort_keys=True,
                    allow_nan=False,separators=(',',':')).encode()).hexdigest()
                    if self.reference_mode=='runtime' else None),
                production_batch_size=self.batch_size,prompt_width=318,
                full_input_hook_audit=hook_audit,warm_cache=cache_checks,
                retained_bytes=retained_bytes,maximum_retained_batches=len(self.retained)*self.batch_count,
                peak_reserved_bytes=self.t.cuda.max_memory_reserved()),
            validation_seconds=time.perf_counter()-began)
        result['max_selected_logprob_error']=max(result['max_selected_logprob_error'],error,
            max(r['selected_logprob_error'] for r in checks))
        result['max_probability_error']=max(result['max_probability_error'],
            max(r['probability_error'] for r in checks))
        return result

    def probe_chunk(self,nodes,on_batch=None):
        if self.protocol is None or 'tokenwise_acceleration' not in self.protocol:
            raise RuntimeError('optimized_admission_required')
        values=[grid_coefficients(node['coefficients'],self.config) for node in nodes]
        result=[[] for _ in nodes]
        with self.t.inference_mode():
            for batch in range(self.batch_count):
                with self.optimized(batch):
                    self.enter_batch(batch)
                    try:
                        for i,value in enumerate(values):
                            result[i].extend(self._measure(self._run(batch,value,True),batch))
                    finally:
                        self.leave_batch(batch)
                if on_batch is not None:on_batch(batch+1)
        return result

