"""Vector-only car recipe on the latest frozen, unmerged person-only LoRA.

The historical loader name `load_av` does not imply AVAR. The base27B and its
person-only step125 LoRA remain frozen; only one5120D vector is optimized for
each ten-axis/layer/injection-mode candidate.
"""
from pathlib import Path
import hashlib
import importlib.util
import math
import time

from common import RUN, PARENT, CONFIG, read, save, sha, budget, status, CandidateAuditFailed
import protocol
from full_sequence import FULL_SEQUENCE, validate_scope, install_query_capture, scoped_hook

_parent_path = Path(__file__).with_name('historical_engine.py')
if not _parent_path.is_file():
    _parent_path = Path(__file__).resolve().parents[1] / 'attribute_wtp' / 'engine.py'
_spec = importlib.util.spec_from_file_location('bigfive_car_training_engine', _parent_path)
parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)

SAMPLING = dict(temperature=.7, top_p=1., top_k=0, repetition_penalty=1.,
                batch_size=16, max_new_tokens=128)


def trait_of(record):
    return record.get('trait', record.get('feature'))


def condition_id(condition):
    return condition.get('condition_id', condition.get('id'))


def expected_score(probabilities, mapping, key):
    """Five conditional token probabilities -> signed normalized trait score."""
    if mapping not in ('normal', 'reversed') or key not in ('+', '-', 1, -1):
        raise ValueError('Unknown Likert mapping or item polarity')
    if len(probabilities) != 5 or any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError('Expected five finite probabilities')
    if abs(sum(probabilities) - 1.) > 1e-5:
        raise ValueError('Conditional digit probabilities must sum to one')
    value = 0.
    for digit, probability in enumerate(probabilities, 1):
        rating = digit if mapping == 'normal' else 6 - digit
        score = rating if key in ('+', 1) else 6 - rating
        value += probability * (score - 1) / 4
    return value


def sampling_seed(phase, trait, batch_id):
    # Alpha is intentionally absent: matched cells use common random numbers.
    value = f'participant_bigfive_20260929:{phase}:{trait}:{batch_id}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big') % (2**63 - 1)


def valid_answer(row):
    return (row.get('format_valid') is True and row.get('reason_present') is True
            and row.get('terminated') is True and not row.get('generated_thinking_tag',False))


def validate_saved(saved, expected, condition, vector_sha, eos_ids):
    if len(saved) != len(expected):
        raise ValueError('Saved response count differs from frozen jobs')
    for index, (actual, job) in enumerate(zip(saved, expected)):
        if any(actual.get(k) != v for k, v in job.items()):
            raise ValueError('Saved response prompt or item metadata changed')
        if actual.get('condition') != condition or actual.get('vector_sha256') != vector_sha:
            raise ValueError('Saved intervention or vector identity changed')
        if actual.get('sampling') != SAMPLING:
            raise ValueError('Saved sampling configuration changed')
        phase = ('test_vector' if condition['method'] == 'vector' else
                 'test_control_' + (condition.get('role') or 'bare'))
        batch_id = expected[index // SAMPLING['batch_size'] * SAMPLING['batch_size']]['id']
        if (actual.get('sampling_phase') != phase or actual.get('batch_seed_job_id') != batch_id
                or actual.get('seed') != sampling_seed(phase, trait_of(condition), batch_id)):
            raise ValueError('Saved batch sampling identity changed')
        tokens = actual.get('tokens')
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
            raise ValueError('Missing generated tokens')
        if len(tokens) > SAMPLING['max_new_tokens']:
            raise ValueError('Generation exceeds its frozen cap')
        if actual.get('terminated') != (tokens[-1] in eos_ids) or any(t in eos_ids for t in tokens[:-1]):
            raise ValueError('EOS metadata does not match actual tokens')
        parsed = protocol.parse_likert(actual['response'], job)
        if any(actual.get(k) != v for k, v in parsed.items() if k != 'valid'):
            raise ValueError('Stored Likert parser output differs from recomputation')
        audit = actual.get('injection_audit', {})
        enabled = condition['method'] == 'vector' and bool(condition['alpha'])
        forwards = audit.get('forward_calls', 0)
        if forwards < len(tokens) or audit.get('passed') is not True:
            raise ValueError('Missing actual generation hook history')
        if audit.get('applied_calls') != (forwards if enabled else 0):
            raise ValueError('Continuous intervention count mismatch')
        if actual.get('answer_effective_injection_calls') != (len(tokens) if enabled else 0):
            raise ValueError('Per-answer effective intervention count mismatch')
        lengths = audit.get('sequence_lengths', [])
        if len(lengths) != forwards or lengths[1:] != [1] * (forwards - 1):
            raise ValueError('Unexpected cache/decode sequence history')
        if actual.get('valid') != valid_answer(actual):
            raise ValueError('Stored validity flag changed')
    return saved


class Engine(parent.Engine):
    @property
    def injection_scope(self):
        return self._injection_scope

    @injection_scope.setter
    def injection_scope(self, value):
        self._injection_scope = validate_scope(value)

    def __init__(self, gpu):
        self.injection_scope = CONFIG.get('injection_scope', FULL_SEQUENCE)
        super().__init__(gpu)
        self._generation_audit = None
        self._audit_layer = None
        self._initializations = {}
        self._dev_bare = {}
        self.digits = [self.tok.encode(str(i), add_special_tokens=False) for i in range(1, 6)]
        if any(len(v) != 1 for v in self.digits):
            raise ValueError('Likert digit labels must each be one token')
        self.digits = [v[0] for v in self.digits]
        self._query_capture_handle = install_query_capture(self)
        # Keep the exact old hook for answer-only controls. Only new-scope calls
        # replace its positional mask, independently of the loss/evaluation API.
        for layer, handle in zip(protocol.LAYERS, self.handles):
            module = self.layers[layer]
            original = scoped_hook(self, layer, module._forward_hooks[handle.id])
            def audited(mod, inputs, output, layer=layer, original=original):
                if self._generation_audit is not None and layer == self._audit_layer:
                    self._generation_audit['injection_scope'] = self.injection_scope
                    h = output[0] if isinstance(output, tuple) else output
                    self._generation_audit['forward_calls'] += 1
                    self._generation_audit['sequence_lengths'].append(int(h.shape[1]))
                    if self.layer == layer and self.delta is not None and (self.mode == 'all' or self.calls == 0):
                        self._generation_audit['applied_calls'] += 1
                return original(mod, inputs, output)
            module._forward_hooks[handle.id] = audited
        runtime = read(RUN / self.who / 'runtime.json')
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Parent_model_or_LoRA_must_be_frozen')
        lora_parameters=[p for n,p in self.model.named_parameters() if 'lora_' in n]
        if len(lora_parameters)!=992 or any(p.dtype!=self.t.float32 for p in lora_parameters):
            raise RuntimeError('Frozen_unmerged_FP32_parent_LoRA_inventory_changed')
        runtime.update(experiment=CONFIG['study'], uses_avar=False,
            unmerged_lora_parent=CONFIG['adapter_checkpoint'],parent_commit_sha256=sha(PARENT/'commit.json'),
            system=CONFIG['system'],thinking=False,native_on_template=CONFIG['native_template_on'],
            uses_lora=True, vector_only_training=True, candidate_layers=list(protocol.LAYERS),
            inference_mode='all', digit_token_ids=self.digits,
            injection_scope=self.injection_scope,
            full_sequence_mask='attention_mask_rightmost_query_length; padding_never_injected',
            positive_direction={'O':'openness','C':'conscientiousness','E':'extraversion',
                                'A':'agreeableness','N':'neuroticism_not_emotional_stability',
                                'R':'risk_preference','T':'future_consequences',
                                'V':'prosocial_allocation','M':'material_values',
                                'S':'sensory_overstimulation'})
        save(RUN / self.who / 'runtime.json', runtime)

    def audit(self, layer, mode, initial, feature):
        try:
            return super().audit(layer, mode, initial, feature)
        except AssertionError as exc:
            # A failed numerical/causal audit is not a poor behavioral score.
            raise CandidateAuditFailed(str(exc)) from exc

    def dev_probe(self, rows, layer=None, vector=None, alpha=0.):
        output = []
        for start in range(0, len(rows), 16):
            budget()
            batch = rows[start:start + 16]
            with self.t.inference_mode():
                logits = self.logits([r['prompt'] for r in batch], layer, vector, alpha, 'all') / .7
                probs = logits[:, self.digits].softmax(-1).cpu().tolist()
                from stable_mass import selected_mass
                mass = selected_mass(self.t, logits, [self.digits])[:, 0].cpu().tolist()
            for row, probabilities, number_mass in zip(batch, probs, mass):
                output.append(dict(row, alpha=alpha, layer=layer, inference_mode='all',
                    injection_scope=self.injection_scope,
                    expected_score01=expected_score(probabilities, row['mapping'], row['key']),
                    probabilities=probabilities, number_mass=number_mass,
                    measurement='conditional_five_digit_probability_not_generated_response'))
        self.cleanup()
        return output

    def fit_candidate(self, condition):
        trait = trait_of(condition)
        cid = condition_id(condition) or condition.get('candidate_id')
        out = RUN / 'candidates' / cid
        out.mkdir(parents=True, exist_ok=True)
        done = out / 'complete.json'
        if done.exists():
            record = read(done)
            if record.get('condition') != condition:
                raise ValueError('Candidate identity changed on resume')
            if record.get('status') == 'complete':
                for name, digest in record['artifact_sha256'].items():
                    if sha(out / name) != digest:
                        raise ValueError('Committed candidate artifact changed')
            return record
        began = time.time()
        layer, mode = condition['layer'], condition['train_mode']
        if trait not in self._initializations:
            arrays, refs = self.initialize(trait)
            self._initializations[trait] = ({k: arrays[k].copy() for k in arrays}, refs)
            if hasattr(arrays, 'close'):
                arrays.close()
        arrays, refs = self._initializations[trait]
        vector = self.train(layer, mode, arrays[f'L{layer:02d}'], refs[str(layer)], trait)
        historical = RUN / 'vectors' / f'{trait}_L{layer:02d}_{mode}'
        # A copy keeps the selected scientific vector self-contained; source
        # checkpoints and Adam histories remain under vectors/ for exact resume.
        import shutil
        for name in ('vector.npz', 'vector_metadata.json'):
            shutil.copy2(historical / name, out / name)
        rows = read(RUN / 'data' / trait / 'dev_likert.json')
        if trait not in self._dev_bare:
            self._dev_bare[trait] = self.dev_probe(rows)
            save(RUN / self.who / f'{trait}_dev_bare.json', self._dev_bare[trait])
        scores = []
        for alpha in protocol.DEV_ALPHAS:
            if alpha == 0:
                # Identical unsteered forward path; no baseline estimated using
                # another layer or candidate and no activation injected at zero.
                current = [dict(r, layer=layer, alpha=0.) for r in self._dev_bare[trait]]
            else:
                current = self.dev_probe(rows, layer, vector, alpha)
            scores.extend(dict(row, candidate_id=cid, train_mode=mode) for row in current)
        save(out / 'dev_scores.json', scores)
        record = dict(status='complete', condition=condition, candidate_id=cid,
            trait=trait, feature=trait, layer=layer, train_mode=mode,
            inference_mode='all', completed_at=time.time(), seconds=time.time()-began,
            artifact_sha256={name:sha(out/name) for name in ('vector.npz','vector_metadata.json','dev_scores.json')})
        save(done, record)
        status(self.who, 'candidate_complete', candidate=cid, feature=trait,
               seconds=record['seconds'])
        self.cleanup()
        return record

    def generate_likert(self, rows, dest, condition, layer, vector, vector_sha):
        if dest.exists():
            return validate_saved(read(dest), rows, condition, vector_sha, self.eos)
        output = []
        trait = trait_of(condition)
        method = condition['method']
        phase = 'test_vector' if method == 'vector' else 'test_control_' + (condition.get('role') or 'bare')
        alpha = condition['alpha'] if method == 'vector' else 0.
        enabled = method == 'vector' and bool(alpha)
        for start in range(0, len(rows), SAMPLING['batch_size']):
            budget()
            batch = rows[start:start + SAMPLING['batch_size']]
            path = dest.parent / 'chunks' / dest.stem / f'{start:06d}.json'
            if path.exists():
                output.extend(validate_saved(read(path), batch, condition, vector_sha, self.eos))
                continue
            seed = sampling_seed(phase, trait, batch[0]['id'])
            self.t.manual_seed(seed)
            self.t.cuda.manual_seed_all(seed)
            self.set(layer, vector if enabled else None, alpha, 'all')
            self._audit_layer = layer
            self._generation_audit = dict(forward_calls=0, applied_calls=0, sequence_lengths=[])
            inputs = self.inputs([r['prompt'] for r in batch])
            began = time.time()
            with self.t.inference_mode():
                sequences = self.model.generate(**inputs, max_new_tokens=128, do_sample=True,
                    temperature=.7, top_p=1., top_k=0, repetition_penalty=1.,
                    eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id,
                    use_cache=True, logits_to_keep=1)
            elapsed = time.time() - began
            sequences = sequences[:, inputs['input_ids'].shape[1]:].tolist()
            audit = dict(self._generation_audit)
            audit['passed'] = (audit['forward_calls'] == len(sequences[0])
                and audit['applied_calls'] == (len(sequences[0]) if enabled else 0)
                and audit['sequence_lengths'][1:] == [1] * (len(sequences[0])-1))
            self._generation_audit = None
            if not audit['passed']:
                raise RuntimeError('Actual continuous-generation injection audit failed')
            saved = []
            for row, tokens in zip(batch, sequences):
                stop = next((i for i, token in enumerate(tokens) if token in self.eos), None)
                tokens = tokens if stop is None else tokens[:stop+1]
                response = self.tok.decode(tokens, skip_special_tokens=True).strip()
                raw_response = self.tok.decode(tokens, skip_special_tokens=False)
                parsed = protocol.parse_likert(response, row)
                result = dict(row, **parsed, condition_id=condition_id(condition), condition=condition,
                    method=method, alpha=condition['alpha'], layer=layer,
                    inference_mode='all' if enabled else 'none', vector_sha256=vector_sha,
                    response=response, raw_response=raw_response, tokens=tokens, terminated=stop is not None,
                    native_thinking_enabled=False,generated_thinking_tag=('<think>' in raw_response or '</think>' in raw_response),
                    seed=seed, sampling_phase=phase, batch_seed_job_id=batch[0]['id'],
                    sampling=SAMPLING, batch_seconds=elapsed, injection_audit=audit,
                    answer_effective_injection_calls=len(tokens) if enabled else 0,
                    measurement='actual_sampled_likert_response')
                result['valid'] = valid_answer(result)
                saved.append(result)
            validate_saved(saved, batch, condition, vector_sha, self.eos)
            save(path, saved)
            output.extend(saved)
            status(self.who, 'final_generating', trait=trait, condition_id=condition_id(condition),
                method=method, alpha=condition['alpha'], completed=len(output), total=len(rows),
                last_batch_seconds=elapsed)
        save(dest, output)
        self.cleanup()
        return output

    def evaluate_selected(self, selected):
        trait = trait_of(selected)
        candidate = selected.get('candidate')
        plan = read(RUN / 'plan.json')
        all_conditions = [c for tr in protocol.TRAITS for c in plan['final_conditions'][tr]]
        conditions = [condition for index, condition in enumerate(all_conditions)
                      if index % CONFIG['workers'] == self.gpu and condition['trait']==trait]
        layer, vector, digest = protocol.LAYERS[0], None, None
        if candidate is not None:
            cid = condition_id(candidate) or candidate['candidate_id']
            root = RUN / 'candidates' / cid
            committed = read(root / 'complete.json')
            digest = sha(root / 'vector.npz')
            if committed['condition'] != candidate or committed['artifact_sha256']['vector.npz'] != digest:
                raise ValueError('Selected vector differs from the committed trained candidate')
            vector = self.t.tensor(self.np.load(root / 'vector.npz')['vector'], device='cuda:0')
            layer = candidate['layer']
        artifacts, counts, skipped = {}, {}, []
        for condition in conditions:
            if candidate is None and condition['method'] == 'vector':
                skipped.append(condition_id(condition))
                continue
            rows = read(RUN / condition['input_data'])
            if len(rows) != condition['rows_expected']:
                raise ValueError('Final frozen job count differs from its condition')
            dest = RUN / 'final' / trait / 'generations' / condition_id(condition) / 'answers.json'
            # Role controls never carry or apply a trained vector.
            active_digest = digest if condition['method'] == 'vector' else None
            answers = self.generate_likert(rows, dest, condition, layer, vector, active_digest)
            key = str(dest.relative_to(RUN / 'final' / trait))
            artifacts[key], counts[key] = sha(dest), len(answers)
        record = dict(status='complete', trait=trait, feature=trait, selection=selected,
            physical_gpu=self.gpu, condition_ids=[condition_id(c) for c in conditions],
            qualified=selected.get('qualified', False), selected_vector_sha256=digest,
            completed_at=time.time(), total_answers=sum(counts.values()), counts=counts,
            artifact_sha256=artifacts, skipped_vector_conditions=skipped,
            interpretation='Finished scheduled evaluation; not a guarantee of trait controllability')
        save(RUN / 'final' / trait / 'shards' / f'gpu{self.gpu}.json', record)
        return record

