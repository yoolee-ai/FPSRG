"""Approved V3 losses, now with input+answer non-padding injection throughout.

The frozen historical hook is preserved for answer-only controls. The Engine
wrapper supplies the same full-sequence mask to every new training/dev API.
"""
from pathlib import Path
import hashlib
import fcntl
import json
import math
import os
import shutil
import time

from common import CONFIG, RUN, PARENT, budget, read, save, sha, status
from engine import Engine, parent as historical
import protocol
from behavior_input_v3 import SYSTEM, input_messages, prefix_sha256, render_behavior_input
from full_sequence import FULL_SEQUENCE

STEPS = 240
AXES = ''.join(CONFIG['axes'])


def step_spec(step):
    if not isinstance(step, int) or not 0 <= step < STEPS:
        raise ValueError('invalid_training_step')
    amplitude = (.5, 1.)[(step // 16) % 2]
    return dict(step=step, questionnaire_index=step % 30,
                behavior_index=step % 16, amplitude=amplitude,
                row_alphas=[amplitude, -amplitude],
                questionnaire_alpha=protocol.TRAIN_ALPHAS[step % 4],
                neutral_alpha=historical.neutral_alpha_for_step(step))


def validate_behavior_row(row):
    messages = input_messages(row)
    if CONFIG['system'] != SYSTEM or row.get('input_messages') != messages:
        raise ValueError('behavior_complete_messages_changed')
    if row.get('input_text') != render_behavior_input(row):
        raise ValueError('behavior_complete_user_input_changed')
    if row.get('input_sha256') != prefix_sha256(row):
        raise ValueError('behavior_input_hash_changed')
    for side in ('high', 'low'):
        if not isinstance(row.get(side + '_response'), str) or not row[side + '_response'].strip():
            raise ValueError('empty_behavior_target')
    return messages


def encode_behavior_pair(tokenizer, row):
    validate_behavior_row(row)
    examples = [historical.encode_answer(tokenizer, row['input_text'], row[side + '_response'])
                for side in ('high', 'low')]
    a, b = examples
    if a['prefix'] != b['prefix'] or a['ids'][:a['prefix']] != b['ids'][:b['prefix']]:
        raise ValueError('positive_negative_token_prefix_differs')
    return examples


def signed_rows(torch, vector, row_alphas, count):
    """Different signs, one differentiable vector; never detach/copy through NumPy."""
    if vector.ndim != 1 or len(row_alphas) != count:
        raise ValueError('row_delta_shape')
    if any(not math.isfinite(float(a)) for a in row_alphas):
        raise ValueError('nonfinite_row_alpha')
    coefficients = torch.as_tensor(row_alphas, device=vector.device, dtype=vector.dtype)
    return coefficients[:, None] * vector[None, :]


def mean_target_losses(torch, logits, targets, counts):
    if not counts or any(c <= 0 for c in counts) or sum(counts) != len(targets):
        raise ValueError('target_count_mismatch')
    token_losses = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    return torch.stack([part.mean() for part in token_losses.split(counts)])


class KnowledgeActionEngine(Engine):
    def __init__(self, gpu):
        super().__init__(gpu)
        if int(CONFIG.get('train_steps', STEPS)) != STEPS:
            raise ValueError('approved_steps_changed')
        self.behavior = read(RUN / 'data/behavior.json')
        if len(self.behavior) != 140:
            raise ValueError('behavior_bank_size')
        for row in self.behavior:
            validate_behavior_row(row)
        self._behavior_encoded = {}
        self._neutral_examples = None
        self._preflight_result = None

    def inputs(self, prompts, thinking=False):
        rendered = [self.tok.apply_chat_template(
            [dict(role='system', content=CONFIG['system']), dict(role='user', content=x)],
            tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking)) for x in prompts]
        encoded = self.tok(rendered, padding=True, add_special_tokens=False,
                           return_tensors='pt').to('cuda:0')
        encoded['position_ids'] = historical.text_position_ids(encoded['attention_mask'])
        return encoded

    def initialize(self, feature):
        if feature not in AXES:
            raise ValueError('only_new_axes_may_be_initialized')
        path = RUN / 'initialization' / f'shared_{feature}.npz'
        metadata = path.with_suffix('.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        binding = dict(parent_commit_sha256=sha(PARENT/'commit.json'),
            questionnaire_sha256=sha(RUN/'data/catalog.json'), feature=feature,
            historical_engine_sha256=sha(Path(__file__).with_name('historical_engine.py')),
            protocol_sha256=sha(Path(__file__).with_name('protocol.py')),
            train_item_ids=[r['id'] for r in protocol.scenarios('train', feature)])
        with (path.parent / f'{feature}.lock').open('a') as lock:
            while True:
                budget()
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    status(self.who, 'waiting_axis_initialization', feature=feature)
                    time.sleep(1)
            if metadata.exists():
                info = read(metadata)
                if sha(path) != info['sha256'] or info.get('binding') != binding:
                    raise ValueError('shared_initialization_hash_or_binding')
                return self.np.load(path), info['refs']
            # Reuse the historical train-only, high-minus-low role extraction
            # unchanged. No prior OCEAN vector is used to initialize new axes.
            arrays, refs = historical.Engine.initialize(self, feature)
            if hasattr(arrays, 'close'):
                arrays.close()
            source = path.parent / f'{feature}_gpu{self.gpu}.npz'
            pending = path.with_suffix('.pending')
            shutil.copy2(source, pending)
            pending.replace(path)
            save(metadata, dict(refs=refs, sha256=sha(path), binding=binding,
                source=str(source), normalized_to='0.1 * median train bare activation norm',
                extraction='high_minus_low_role_activation_mean_train_questions_only',
                newly_extracted=True, worker=self.gpu))
            return self.np.load(path), refs

    def signed_full_logits(self, examples, layer=None, vector=None, row_alphas=None):
        """Full CE+EOS, row-specific signs; scope is supplied by the model wrapper."""
        ids, attention, positions, targets = historical.layout(examples, self.tok.pad_token_id)
        device = next(self.model.parameters()).device
        ids = self.t.tensor(ids, device=device)
        attention = self.t.tensor(attention, device=device)
        positions = self.t.tensor(positions, device=device)
        mask = self.t.tensor(targets, device=device, dtype=self.t.bool)[:, 1:]
        if vector is None:
            if row_alphas is not None and any(float(a) != 0 for a in row_alphas):
                raise ValueError('nonzero_alpha_without_vector')
            self.set()
        else:
            if row_alphas is None:
                raise ValueError('missing_row_alphas')
            delta = signed_rows(self.t, vector, row_alphas, len(examples))
            if all(float(a) == 0 for a in row_alphas):
                self.set()
            else:
                # Full-sequence scope ignores target-start positions for injection;
                # positions/target masks continue to identify supervised logits only.
                self.set(layer, delta, 1., 'all', positions)
        out = self.model(input_ids=ids, attention_mask=attention,
                         position_ids=historical.text_position_ids(attention),
                         use_cache=False, logits_to_keep=0)
        return out.logits[:, :-1][mask].float(), ids[:, 1:][mask], mask.sum(-1).tolist()

    def signed_target_losses(self, examples, layer=None, vector=None, row_alphas=None):
        return mean_target_losses(self.t, *self.signed_full_logits(examples, layer, vector, row_alphas))

    def _examples(self, row):
        if row['id'] not in self._behavior_encoded:
            self._behavior_encoded[row['id']] = encode_behavior_pair(self.tok, row)
        return self._behavior_encoded[row['id']]

    def preflight(self):
        """One small fixed check per replica; no benchmark or test-set access."""
        if self._preflight_result is not None:
            return self._preflight_result
        if self.injection_scope != FULL_SEQUENCE:
            raise RuntimeError('new_training_preflight_requires_full_sequence')
        row = sorted((r for r in self.behavior if r['axis'] == AXES[0] and r['split'] == 'train'),
                     key=lambda r:r['id'])[0]
        examples = self._examples(row)
        if len(examples[0]['ids']) == len(examples[1]['ids']):
            for candidate in sorted((r for r in self.behavior if r['axis'] == AXES[0]
                                     and r['split'] == 'train'), key=lambda r:r['id']):
                pair = self._examples(candidate)
                if len(pair[0]['ids']) != len(pair[1]['ids']):
                    row, examples = candidate, pair
                    break
            else:
                raise RuntimeError('preflight_requires_one_unequal_target_length_pair_for_padding_check')
        layer = protocol.LAYERS[0]
        vector = self.t.nn.Parameter(self.t.linspace(-.02, .02, 5120, device='cuda:0'))
        on = self.tok.apply_chat_template(input_messages(row), tokenize=False,
                  add_generation_prompt=True, enable_thinking=True)
        off = self.tok.apply_chat_template(input_messages(row), tokenize=False,
                   add_generation_prompt=True, enable_thinking=False)
        if not on.rstrip().endswith('<think>') or on == off:
            raise RuntimeError('native_thinking_template_failure')
        with self.t.no_grad():
            bare = self.logits([row['input_text']], mode='all')
            zero = self.logits([row['input_text']], layer, vector, 0., 'all')
            if not self.t.equal(bare, zero):
                raise RuntimeError('zero_vector_prefix_changed')
            full_bare, _, _ = self.signed_full_logits(examples)
            full_zero, _, _ = self.signed_full_logits(examples, layer, vector, [0., 0.])
            if not self.t.equal(full_bare, full_zero):
                raise RuntimeError('zero_vector_full_changed')
            bare_anchor = bare[0].log_softmax(-1)[self.ab+self.digits].cpu().tolist()
            del full_bare, full_zero, bare, zero
            steered = self.logits([row['input_text']], layer, vector, 1., 'all')
            full_sequence_anchor = steered[0].log_softmax(-1)[self.ab+self.digits].cpu().tolist()
            del steered
            # Both halves participate in the existing cross-worker equality gate.
            anchor = bare_anchor + full_sequence_anchor
        observed = {}
        module = self.layers[layer]
        expected_attention = self.t.tensor(historical.layout(examples, self.tok.pad_token_id)[1],
                                           dtype=self.t.bool, device='cuda:0')
        def before(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            observed['before'] = h.detach().clone()
        def after(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            with self.t.no_grad():
                original = observed.pop('before')
                mask = self._full_sequence_mask
                observed['query_mask_matches_teacher_layout'] = bool(self.t.equal(mask, expected_attention))
                # Independent reference, not the production inject helper.
                reference_delta = self.delta.to(h.dtype)[:, None, :].expand_as(h)
                expected = original.clone()
                expected[expected_attention] += reference_delta[expected_attention]
                observed['hook_exact'] = bool(self.t.equal(expected, h.detach()))
                observed['opposite_signs'] = bool(self.t.equal(self.delta[0], -self.delta[1]))
                observed['padding_untouched'] = bool(self.t.equal(original[~mask], h.detach()[~mask]))
                supervised_start = self.positions[:, None]
                sequence_positions = self.t.arange(h.shape[1], device=h.device)[None, :]
                observed['valid_prompt_positions'] = int((mask & (sequence_positions < supervised_start)).sum())
                observed['valid_answer_positions'] = int((mask & (sequence_positions >= supervised_start)).sum())
                observed['padding_positions'] = int((~mask).sum())
        handles = [module.register_forward_hook(before, prepend=True), module.register_forward_hook(after)]
        try:
            logits, labels, counts = self.signed_full_logits(examples, layer, vector, [1., -1.])
            losses = self.t.nn.functional.cross_entropy(logits, labels, reduction='none')
            tails = [part[5:] for part in losses.split(counts)]
            if any(len(part) == 0 for part in tails):
                raise RuntimeError('no_target_tail')
            tail = self.t.stack([part.mean() for part in tails]).mean()
            gradient = self.t.autograd.grad(tail, vector)[0]
            grad_norm = float(gradient.norm())
            if not bool(self.t.isfinite(gradient).all()) or grad_norm <= 1e-8:
                raise RuntimeError('full_target_tail_gradient_missing')
            if (not observed.get('hook_exact') or not observed.get('opposite_signs')
                    or not observed.get('padding_untouched')
                    or not observed.get('query_mask_matches_teacher_layout')
                    or observed.get('valid_prompt_positions', 0) <= 0
                    or observed.get('valid_answer_positions', 0) <= 0):
                raise RuntimeError('row_signed_continuous_hook_failed')
            first = logits[0].detach().softmax(-1)
            del logits, labels, losses, tails, tail, gradient
        finally:
            for handle in handles:
                handle.remove()
        with self.t.no_grad():
            changed = [dict(ex, ids=ex['ids'][:ex['prefix']] +
                        [self.ab[1]] * (len(ex['ids']) - ex['prefix'])) for ex in examples]
            mutated, _, _ = self.signed_full_logits(changed, layer, vector, [1., -1.])
            causal_tv = float((mutated[0].softmax(-1) - first).abs().sum() / 2)
            del mutated, first
        if causal_tv >= 1e-5 or any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('causality_or_parent_freeze_failure')
        # Same batch/length/position layout: request cached versus uncached
        # prefill. Subsequent decode is additionally checked against native
        # generate in evaluation.sampler_preflight, for ON and OFF.
        cache_probes = [row['input_text'], '我刚走进房间。'] * 2
        x = self.inputs(cache_probes)
        with self.t.no_grad():
            self.set(layer, vector, 1., 'all')
            uncached = self.model(**x, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            reference = uncached.log_softmax(-1)[:, self.ab+self.digits]
            del uncached
            self.set(layer, vector, 1., 'all')
            cached = self.model(**x, use_cache=True, logits_to_keep=1)
            comparison = cached.logits[:, -1].float().log_softmax(-1)[:, self.ab+self.digits]
            cache_error = float((reference-comparison).abs().max())
            if not math.isfinite(cache_error) or cache_error > 1e-4:
                raise RuntimeError('full_sequence_same_shape_cache_prefill_mismatch:' + str(cache_error))
            del reference, comparison, cached, x
        result = dict(passed=True, native_off=True, native_on_template=True,
            zero_equal=True, row_hook_exact=True, opposite_row_signs=True,
            injection_scope=self.injection_scope, padding_untouched=observed['padding_untouched'],
            query_mask_matches_teacher_layout=observed['query_mask_matches_teacher_layout'],
            valid_prompt_positions=observed['valid_prompt_positions'],
            valid_answer_positions=observed['valid_answer_positions'],
            padding_positions=observed['padding_positions'],
            same_shape_cached_prefill_max_selected_logprob_error=cache_error,
            full_tail_gradient=grad_norm, future_token_causality_tv=causal_tv,
            parent_frozen=True, anchor_token_ids=(self.ab+self.digits)*2,
            anchor_conditions=['bare']*7+['full_sequence_alpha1']*7,
            anchor=anchor, bare_anchor=bare_anchor, full_sequence_anchor=full_sequence_anchor,
            parent_commit_sha256=sha(PARENT/'commit.json'),
            behavior_sha256=sha(RUN/'data/behavior.json'))
        self._preflight_result = result
        self.cleanup()
        return result

    def _binding(self, condition):
        files = {name:sha(Path(__file__).with_name(name)) for name in
                 ('ka_engine.py', 'engine.py', 'full_sequence.py', 'historical_engine.py',
                  'protocol.py', 'behavior_input_v3.py')}
        obj = dict(condition=condition, parent_commit_sha256=sha(PARENT/'commit.json'),
                   behavior_sha256=sha(RUN/'data/behavior.json'), source_sha256=files,
                   questionnaire_sha256=sha(RUN/'data/catalog.json'),
                   injection_scope=self.injection_scope,
                   initialization_sha256=sha(RUN/'initialization'/f"shared_{condition['trait']}.npz"),
                   initialization_metadata_sha256=sha(RUN/'initialization'/f"shared_{condition['trait']}.json"))
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

    def _save_checkpoint(self, out, step, u, optimizer, history, binding):
        path = out / f'step-{step:04d}.pt'
        pending = path.with_suffix('.pending')
        self.t.save(dict(step=step, u=u.detach(), optimizer=optimizer.state_dict(), history=history,
                        binding=binding, cpu_rng=self.t.get_rng_state(),
                        cuda_rng=self.t.cuda.get_rng_state()), pending)
        with pending.open('rb') as handle:
            os.fsync(handle.fileno())
        pending.replace(path)
        save(out/'latest.json', dict(step=step, path=path.name, sha256=sha(path), binding=binding))

    def _fit_vector(self, condition, out, initial, ref, binding):
        layer, trait = condition['layer'], condition['trait']
        metadata_path = out/'vector_metadata.json'
        if metadata_path.exists():
            metadata = read(metadata_path)
            if metadata['binding'] != binding or sha(out/'vector.npz') != metadata['sha256']:
                raise ValueError('saved_vector_binding_changed')
            return self.t.tensor(self.np.load(out/'vector.npz')['vector'], device='cuda:0')
        scenarios = sorted(protocol.scenarios('train', feature=trait), key=lambda r:r['id'])
        behavior = sorted((r for r in self.behavior if r['axis']==trait and r['split']=='train'), key=lambda r:r['id'])
        if len(scenarios)!=30 or len(behavior)!=16:
            raise ValueError('approved_training_split_changed')
        base_rows = self.probe(protocol.jobs('train', feature=trait),
                    RUN/self.who/f'{trait}_train_bare_probe.json', f'{trait}_train_bare')
        probabilities = {(r['scenario_id'], r['mapping']):r['p_upgrade'] for r in base_rows}
        encoded = {(s['id'], mapping):[historical.encode_answer(self.tok, protocol.choice_prompt(s, 0., mapping), answer)
                   for answer in protocol.answer_pair(s, 0., mapping)] for s in scenarios for mapping in ('A','B')}
        if self._neutral_examples is None:
            self._neutral_examples = [historical.encode_answer(self.tok, r['prompt'], r['response']) for r in protocol.NEUTRAL]
        u = self.t.nn.Parameter(self.t.tensor(initial, dtype=self.t.float32, device='cuda:0')/(ref*.1))
        optimizer = self.t.optim.Adam([u], lr=.003)
        history, start = [], 0
        if (out/'latest.json').exists():
            latest = read(out/'latest.json')
            if latest['binding']!=binding or sha(out/latest['path'])!=latest['sha256']:
                raise ValueError('checkpoint_binding_changed')
            ck = self.t.load(out/latest['path'], map_location='cuda:0', weights_only=False)
            if ck['binding']!=binding:
                raise ValueError('checkpoint_content_binding_changed')
            with self.t.no_grad():u.copy_(ck['u'])
            optimizer.load_state_dict(ck['optimizer']); start=ck['step']; history=ck['history']
            self.t.set_rng_state(ck['cpu_rng'].cpu()); self.t.cuda.set_rng_state(ck['cuda_rng'].cpu())
        for step in range(start, STEPS):
            budget(); began=time.time(); spec=step_spec(step)
            s=scenarios[spec['questionnaire_index']]; row=behavior[spec['behavior_index']]
            optimizer.zero_grad(set_to_none=True)
            choice_value, self_value = 0., 0.
            alpha=spec['questionnaire_alpha']
            for mapping in ('A','B'):
                target=historical.shifted_target(probabilities[(s['id'],mapping)], alpha, 1.)
                logits=self.logits([protocol.choice_prompt(s,0.,mapping)],layer,u*ref*.1,alpha,'all')
                ab=(logits[0,self.ab]/historical.TEMPERATURE).log_softmax(-1)
                p_a=target if mapping=='A' else 1-target
                ce=-(p_a*ab[0]+(1-p_a)*ab[1]); (.25*ce).backward()
                choice_value+=float(ce.detach())*.5
                full=self.weighted_full_ce(encoded[(s['id'],mapping)],target,layer,u*ref*.1,alpha,'all')
                (.05*full).backward(); self_value+=float(full.detach())*.5
                del logits,ab,ce,full
            action_losses=self.signed_target_losses(self._examples(row),layer,u*ref*.1,spec['row_alphas'])
            action=action_losses.mean(); (.5*action).backward()
            action_value=float(action.detach()); del action_losses,action
            neutral_value=0.
            if spec['neutral_alpha'] is not None:
                ex=self._neutral_examples[(step//4)%len(self._neutral_examples)]
                with self.t.no_grad():
                    reference,_,_=self.full_logits([ex]); logref=reference.log_softmax(-1)
                    probs=logref.exp(); del reference
                moved,_,_=self.full_logits([ex],layer,u*ref*.1,spec['neutral_alpha'],'all')
                kl=(probs*(logref-moved.log_softmax(-1))).sum(-1).mean()
                (.1*kl).backward(); neutral_value=float(kl.detach())
                del moved,probs,logref,kl
            (.01*u.square().sum()).backward()
            grad=float(u.grad.norm())
            if not math.isfinite(grad) or not bool(self.t.isfinite(u.grad).all()):
                raise RuntimeError('nonfinite_vector_gradient')
            if any(p.grad is not None for p in self.model.parameters()):
                raise RuntimeError('parent_received_gradient')
            self.t.nn.utils.clip_grad_norm_([u],1.);optimizer.step()
            with self.t.no_grad():u.mul_(min(1.,1./max(float(u.norm()),1e-12)))
            history.append(dict(step=step+1,questionnaire_id=s['id'],behavior_id=row['id'],
                questionnaire_alpha=alpha,paired_alphas=spec['row_alphas'],choice_ce=choice_value,
                self_full_ce=self_value,action_full_ce=action_value,neutral_kl=neutral_value,
                neutral_alpha=spec['neutral_alpha'],grad_norm=grad,unit_norm=float(u.norm()),seconds=time.time()-began))
            if (step+1)%5==0:
                save(out/'history.json',history)
                status(self.who,'training',candidate=condition['id'],feature=trait,step=step+1,
                       total_steps=STEPS,**{k:history[-1][k] for k in ('seconds','choice_ce','self_full_ce','action_full_ce')})
            if (step+1)%40==0:
                self._save_checkpoint(out,step+1,u,optimizer,history,binding)
        vector=u.detach().cpu().numpy()*ref*.1
        pending=out/'vector.pending'
        with pending.open('wb') as handle:
            self.np.savez(handle,vector=vector);handle.flush();os.fsync(handle.fileno())
        pending.replace(out/'vector.npz')
        save(metadata_path,dict(layer=layer,mode='all',feature=trait,trait=trait,steps=STEPS,
            ref=ref,norm=float(self.np.linalg.norm(vector)),binding=binding,
            parent_commit_sha256=sha(PARENT/'commit.json'),sha256=sha(out/'vector.npz'),
            objective='0.5*(AB_mean+0.2*self_full_mean)+0.5*paired_action_full_mean+0.1*neutral_KL_every4+0.01*norm2',
            native_thinking=False,continuous_target_injection=True,
            injection_scope=self.injection_scope, full_input_injection=True,
            padding_injected=False))
        return self.t.tensor(vector,device='cuda:0')

    def behavior_development(self, trait, layer, vector):
        rows=sorted((r for r in self.behavior if r['axis']==trait and r['split']=='dev'),key=lambda r:r['id'])
        if len(rows)!=4:raise ValueError('development_scene_count')
        records=[]
        with self.t.inference_mode():
            for alpha in protocol.DEV_ALPHAS:
                for row in rows:
                    budget()
                    losses=self.signed_target_losses(self._examples(row),layer,vector,[alpha,alpha])
                    high,low=(-losses).cpu().tolist()
                    records.append(dict(id=row['id'],scene_id=row['id'],trait=trait,split='dev',alpha=alpha,
                        high_mean_logprob=high,low_mean_logprob=low,preference_logprob_delta=high-low,
                        input_sha256=row['input_sha256'],layer=layer,inference_mode='all',
                        injection_scope=self.injection_scope,
                        measurement='length_normalized_target_loglikelihood_difference_not_behavior_probability'))
        return records

    def fit_candidate(self, condition):
        if self.injection_scope != FULL_SEQUENCE:
            raise RuntimeError('candidate_training_requires_full_sequence_scope')
        if condition.get('trait') not in AXES or condition.get('train_mode')!='all' or condition.get('layer') not in protocol.LAYERS:
            raise ValueError('unapproved_candidate')
        cid=condition['id']
        if Path(cid).name!=cid:raise ValueError('unsafe_candidate_id')
        out=RUN/'candidates'/cid;out.mkdir(parents=True,exist_ok=True)
        trait=condition['trait']
        if trait not in self._initializations:
            arrays,refs=self.initialize(trait)
            self._initializations[trait]=({k:arrays[k].copy() for k in arrays},refs)
            if hasattr(arrays,'close'):arrays.close()
        binding=self._binding(condition)
        if (out/'complete.json').exists():
            record=read(out/'complete.json')
            if record['condition']!=condition or record['binding']!=binding:
                raise ValueError('completed_candidate_binding_changed')
            for name,digest in record['artifact_sha256'].items():
                if sha(out/name)!=digest:raise ValueError('completed_candidate_artifact_changed')
            return record
        began=time.time();trait=condition['trait'];layer=condition['layer']
        arrays,refs=self._initializations[trait]
        vector=self._fit_vector(condition,out,arrays[f'L{layer:02d}'],refs[str(layer)],binding)
        dev=read(RUN/'data'/trait/'dev_likert.json')
        if trait not in self._dev_bare:self._dev_bare[trait]=self.dev_probe(dev)
        scores=[]
        for alpha in protocol.DEV_ALPHAS:
            current=([dict(r,alpha=0.,layer=layer) for r in self._dev_bare[trait]] if alpha==0
                     else self.dev_probe(dev,layer,vector,alpha))
            scores.extend(dict(r,candidate_id=cid,train_mode='all') for r in current)
        save(out/'dev_scores.json',scores)
        save(out/'dev_behavior.json',self.behavior_development(trait,layer,vector))
        names=('vector.npz','vector_metadata.json','dev_scores.json','dev_behavior.json')
        record=dict(status='complete',condition=condition,candidate_id=cid,trait=trait,feature=trait,
            layer=layer,train_mode='all',inference_mode='all',injection_scope=self.injection_scope,
            binding=binding,completed_at=time.time(),
            seconds=time.time()-began,artifact_sha256={name:sha(out/name) for name in names})
        save(out/'complete.json',record);self.cleanup()
        status(self.who,'candidate_complete',candidate=cid,feature=trait,seconds=record['seconds'])
        return record
