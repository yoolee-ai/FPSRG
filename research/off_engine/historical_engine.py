"""Frozen Qwen backbone and vector-only, multi-dose WTP pilot.

The economic target is a controlled *change* in choice log odds relative to
each bare prompt, not a claim that alpha already has units of money.
"""
import gc
import hashlib
import json
import math
import os
import os
import random
import re
import time

from common import BASE, RUN, CONFIG, budget, read, save, sha, status
from protocol import LAYERS, TRAIN_ALPHAS, NEUTRAL, answer_pair, jobs

TEMPERATURE = 0.7
STEPS = 240
MODES = ["prefill", "all"]


def validate_cached_rows(saved, expected, method, alpha, layer, mode):
    if len(saved) != len(expected):
        raise ValueError("Cached output count differs from frozen jobs")
    for actual, job in zip(saved, expected):
        for key in ("id", "split", "scenario_id", "premium", "mapping", "role", "feature",
                    "zero_gain", "label_variant", "source_axis", "evaluation_suite", "prompt"):
            if actual.get(key) != job.get(key):
                raise ValueError(f"Cached output job mismatch: {key}")
        for key, value in (("method", method), ("alpha", alpha), ("layer", layer), ("mode", mode)):
            if actual.get(key) != value:
                raise ValueError(f"Cached output intervention mismatch: {key}")
    return saved


def shifted_target(p, alpha, benefit_units):
    p = min(1 - 1e-5, max(1e-5, float(p)))
    z = math.log(p / (1 - p)) + 2.0 * alpha * benefit_units
    return 1.0 / (1.0 + math.exp(-z))


def neutral_alpha_for_step(step):
    # Independent schedule: otherwise step % 4 == 0 protects only alpha=-1.
    return TRAIN_ALPHAS[(step // 4) % len(TRAIN_ALPHAS)] if step % 4 == 0 else None


def text_position_ids(attention):
    """Match GenerationMixin's text-only left-padding position convention."""
    return (attention.long().cumsum(-1) - 1).masked_fill(attention == 0, 0)


def encode_answer(tok, prompt, answer):
    turns = [dict(role="system", content=CONFIG['system']),dict(role="user", content=prompt)]
    kw = dict(tokenize=True, return_dict=False, enable_thinking=False)
    prefix = tok.apply_chat_template(turns, add_generation_prompt=True, **kw)
    full = tok.apply_chat_template(turns + [dict(role="assistant", content=answer)],
                                   add_generation_prompt=False, **kw)
    assert full[:len(prefix)] == prefix
    stop = full.index(tok.eos_token_id, len(prefix))
    return dict(ids=full[:stop + 1], prefix=len(prefix))


def layout(examples, pad_id):
    width = max(len(e["ids"]) for e in examples)
    ids, attention, positions, targets = [], [], [], []
    for e in examples:
        pad = width - len(e["ids"])
        ids.append([pad_id] * pad + e["ids"])
        attention.append([0] * pad + [1] * len(e["ids"]))
        positions.append(pad + e["prefix"] - 1)
        targets.append([False] * (pad + e["prefix"]) + [True] * (len(e["ids"]) - e["prefix"]))
    return ids, attention, positions, targets


class Engine:
    def __init__(self, gpu):
        import numpy as np
        import torch
        from transformers import AutoTokenizer
        from models import load_av, get_text_layers
        from acceleration import install_qwen35_kernels
        self.t, self.np, self.gpu, self.who = torch, np, gpu, f"gpu{gpu}"
        torch.set_num_threads(4)
        acceleration = install_qwen35_kernels()
        status(self.who, "loading_original_qwen27b")
        # Historical loader name only. rank=0 loads the full original text LM.
        self.model = load_av(str(BASE), device="cuda:0", dtype=torch.bfloat16,
                             rank=0, gradient_checkpointing=False, attention="sdpa").eval()
        self.model.requires_grad_(False)
        # Full-answer vs prefix BF16 GEMM layouts produced ~3% TV even when bare.
        # Use FP32 for the frozen vocabulary head in ALL execution paths.
        head = self.model.get_output_embeddings()
        head.float()
        self.head_cast_handle = head.register_forward_pre_hook(
            lambda module, args: (args[0].float(), *args[1:]))
        report = self.model._nla_load_report
        assert report["layers"] == 64 and report["hidden_size"] == 5120
        assert report["rank"] == 16 and len(report["lora_modules"]) == 496 and not report["trainable_parameters"]
        self.tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.ab = [self.tok.encode(x, add_special_tokens=False) for x in ("A", "B")]
        assert all(len(x) == 1 for x in self.ab)
        self.ab = [x[0] for x in self.ab]
        eos = self.model.generation_config.eos_token_id
        self.eos = sorted(set(eos if isinstance(eos, list) else [eos]) | {self.tok.eos_token_id})
        self.layers = get_text_layers(self.model)
        self.layer = self.delta = self.positions = None
        self.mode, self.calls, self.capture, self.captured = "prefill", 0, False, {}
        self.handles = []
        for layer in LAYERS:
            def hook(_module, _inputs, output, layer=layer):
                h = output[0] if isinstance(output, tuple) else output
                if self.capture:
                    self.captured[layer] = h[:, -1].detach().float().cpu()
                if self.layer != layer or self.delta is None:
                    return output
                active = self.mode == "all" or self.calls == 0
                self.calls += 1
                if not active:
                    return output
                y = h.clone()
                delta = self.delta.to(h.dtype)
                if self.positions is None:
                    y[:, -1] = y[:, -1] + delta
                elif self.mode == "prefill":
                    y[self.t.arange(h.shape[0], device=h.device), self.positions] += delta
                else:
                    mask = self.t.arange(h.shape[1], device=h.device)[None, :] >= self.positions[:, None]
                    y = y + mask[:, :, None].to(h.dtype) * delta[:, None, :]
                return (y, *output[1:]) if isinstance(output, tuple) else y
            self.handles.append(self.layers[layer].register_forward_hook(hook))
        save(RUN / self.who / "runtime.json", dict(model=report, uses_avar=False, uses_lora=True,
             backbone_precision="bfloat16", output_head_precision="float32", explicit_text_position_ids=True,
             accelerator=acceleration, eos_ids=self.eos, sampling_temperature=TEMPERATURE,
             top_p=1.0, top_k=0, max_new_tokens=128))

    def set(self, layer=None, vector=None, alpha=0., mode="prefill", positions=None):
        assert mode in MODES
        if vector is not None:
            assert layer in LAYERS, "Intervention layer has no registered hook"
            assert vector.shape[-1] == 5120, "Unexpected vector dimension"
        self.layer, self.mode, self.calls, self.positions = layer, mode, 0, positions
        self.delta = None if vector is None or alpha == 0 else vector * alpha

    def inputs(self, prompts):
        rendered = [self.tok.apply_chat_template([dict(role="system", content=CONFIG['system']),dict(role="user", content=x)], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for x in prompts]
        encoded = self.tok(rendered, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda:0")
        encoded["position_ids"] = text_position_ids(encoded["attention_mask"])
        return encoded

    def logits(self, prompts, layer=None, vector=None, alpha=0., mode="prefill"):
        self.set(layer, vector, alpha, mode)
        return self.model(**self.inputs(prompts), use_cache=False, logits_to_keep=1).logits[:, -1].float()

    def probe(self, rows, dest, method, alpha=0., layer=None, vector=None, mode="prefill"):
        if dest.exists():
            return validate_cached_rows(read(dest), rows, method, alpha, layer, mode)
        output = []
        for start in range(0, len(rows), 16):
            budget()
            chunk = dest.parent / "chunks" / dest.stem / f"{start:06d}.json"
            if chunk.exists():
                output.extend(validate_cached_rows(read(chunk), rows[start:start + 16], method, alpha, layer, mode)); continue
            batch = rows[start:start + 16]
            with self.t.inference_mode():
                logits = self.logits([r["prompt"] for r in batch], layer, vector, alpha, mode) / TEMPERATURE
                ab = logits[:, self.ab].softmax(-1).cpu().tolist()
                mass = (logits[:, self.ab].logsumexp(-1) - logits.logsumexp(-1)).exp().cpu().tolist()
            new = []
            for row, probs, ab_mass in zip(batch, ab, mass):
                new.append(dict(row, method=method, alpha=alpha, layer=layer, mode=mode,
                                p_upgrade=probs[0 if row["mapping"] == "A" else 1], ab_mass=ab_mass,
                                measurement="conditional_first_token_probability_not_sample"))
            save(chunk, new); output.extend(new)
            status(self.who, "probing", method=method, alpha=alpha, completed=len(output), total=len(rows))
        save(dest, output)
        return output

    def initialize(self, feature):
        path = RUN / "initialization" / f"{feature}_gpu{self.gpu}.npz"
        info_path = path.with_suffix(".json")
        if path.exists() and info_path.exists():
            info = read(info_path); assert sha(path) == info["sha256"]
            return self.np.load(path), info["refs"]
        base_rows = jobs("train", premiums=[.2, .6], feature=feature)
        from protocol import role_prefix
        diffs, norms = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
        self.capture = True
        for start in range(0, len(base_rows), 8):
            batch = base_rows[start:start + 8]
            budget()
            with self.t.no_grad():
                self.logits([role_prefix("high", feature=feature) + r["prompt"] for r in batch])
                hi = dict(self.captured)
                self.logits([role_prefix("low", feature=feature) + r["prompt"] for r in batch])
                lo = dict(self.captured)
                self.logits([r["prompt"] for r in batch])
                for l in LAYERS:
                    diffs[l].extend((hi[l] - lo[l]).numpy())
                    norms[l].extend(self.captured[l].norm(dim=-1).tolist())
            status(self.who, "initializing", completed=min(start + 8, len(base_rows)), total=len(base_rows))
        self.capture = False
        arrays, refs = {}, {}
        for l in LAYERS:
            d = self.np.mean(diffs[l], axis=0); ref = float(self.np.median(norms[l]))
            assert self.np.isfinite(d).all() and self.np.linalg.norm(d) > 1e-7
            arrays[f"L{l:02d}"] = d / self.np.linalg.norm(d) * ref * .1
            refs[str(l)] = ref
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".pending")
        with pending.open("wb") as f:
            self.np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
        pending.replace(path)
        save(info_path, dict(refs=refs, sha256=sha(path), normalization="0.1 median bare norm",
                            source="high-minus-low role activation mean; train prompts only"))
        return arrays, refs

    def full_logits(self, examples, layer=None, vector=None, alpha=0., mode="prefill"):
        ids, attention, positions, targets = layout(examples, self.tok.pad_token_id)
        ids = self.t.tensor(ids, device="cuda:0")
        attention = self.t.tensor(attention, device="cuda:0")
        positions = self.t.tensor(positions, device="cuda:0")
        mask = self.t.tensor(targets, device="cuda:0", dtype=self.t.bool)[:, 1:]
        v = None if vector is None else vector[None, :].expand(len(examples), -1)
        self.set(layer, v, alpha, mode, positions)
        out = self.model(input_ids=ids, attention_mask=attention, position_ids=text_position_ids(attention),
                         use_cache=False, logits_to_keep=0)
        return out.logits[:, :-1][mask].float(), ids[:, 1:][mask], mask.sum(-1).tolist()

    def weighted_full_ce(self, examples, probability, layer, vector, alpha, mode):
        logits, targets, counts = self.full_logits(examples, layer, vector, alpha, mode)
        losses = self.t.nn.functional.cross_entropy(logits, targets, reduction="none")
        a, b = [x.mean() for x in losses.split(counts)]
        return probability * a + (1 - probability) * b

    def audit(self, layer, mode, initial, feature):
        from protocol import scenarios, choice_prompt
        s = scenarios("train", feature=feature)[0]
        prompt = choice_prompt(s, .5, "A")
        examples = [encode_answer(self.tok, prompt, x) for x in answer_pair(s, .5, "A")]
        with self.t.no_grad():
            bare = self.logits([prompt])
            zero = self.logits([prompt], layer, initial, 0., mode)
        assert self.t.equal(bare, zero), "Zero intervention changed model"
        u = self.t.nn.Parameter(initial.clone())
        logits, targets, counts = self.full_logits(examples, layer, u, 1., mode)
        losses = self.t.nn.functional.cross_entropy(logits, targets, reduction="none")
        tail = self.t.cat([x[5:] for x in losses.split(counts)]).mean()
        grad = self.t.autograd.grad(tail, u)[0]
        assert self.t.isfinite(grad).all() and float(grad.norm()) > 1e-8
        first = logits[0].detach()
        with self.t.no_grad():
            prefix = self.logits([prompt], layer, u.detach(), 1., mode)[0]
        tv = float((first.softmax(-1) - prefix.softmax(-1)).abs().sum() / 2)
        first_top, prefix_top = first.argmax().item(), prefix.argmax().item()
        audit_values = dict(prefix_tv=tv, full_first_argmax=first_top,
                            prefix_argmax=prefix_top, later_token_grad=float(grad.norm()),
                            passed=tv < .02 and first_top == prefix_top)
        with self.t.no_grad():
            bare_full, _, _ = self.full_logits(examples)
            audit_values['bare_prefix_tv'] = float((bare_full[0].softmax(-1)-bare[0].softmax(-1)).abs().sum()/2)
            del bare_full
            changed = [dict(ex, ids=ex['ids'][:ex['prefix']] +
                            [self.ab[1]] * (len(ex['ids'])-ex['prefix'])) for ex in examples]
            changed_logits, _, _ = self.full_logits(changed, layer, u.detach(), 1., mode)
            audit_values['future_token_causality_tv'] = float((changed_logits[0].softmax(-1)-first.softmax(-1)).abs().sum()/2)
            del changed_logits
        audit_values['passed'] = audit_values['passed'] and audit_values['future_token_causality_tv'] < 1e-5
        save(RUN / "vectors" / f"{feature}_L{layer:02d}_{mode}" / "audit_check.json", audit_values)
        assert audit_values["passed"], f"Full-answer/prefix consistency failed: {audit_values}"
        return dict(zero_equal=True, later_token_grad=float(grad.norm()), prefix_tv=tv,
                    base_model_gradients=any(p.grad is not None for p in self.model.parameters()))

    def train(self, layer, mode, initial, ref, feature):
        from protocol import scenarios, choice_prompt
        candidate = f"{feature}_L{layer:02d}_{mode}"; out = RUN / "vectors" / candidate
        out.mkdir(parents=True, exist_ok=True)
        if (out / "vector.npz").exists() and (out / "vector_metadata.json").exists():
            metadata = read(out / "vector_metadata.json")
            assert metadata["layer"] == layer and metadata["mode"] == mode and metadata["steps"] == STEPS
            assert sha(out / "vector.npz") == metadata["sha256"]
            return self.t.tensor(self.np.load(out / "vector.npz")["vector"], device="cuda:0")
        initial = self.t.tensor(initial, device="cuda:0", dtype=self.t.float32)
        if not (out / "audit.json").exists():
            save(out / "audit.json", self.audit(layer, mode, initial, feature))
        base_rows = self.probe(jobs("train", feature=feature), RUN / self.who / f"{feature}_train_bare_probe.json", f"{feature}_train_bare")
        probabilities = {(r["scenario_id"], r["premium"], r["mapping"]): r["p_upgrade"] for r in base_rows}
        encoded = {}
        by_id = {s["id"]: s for s in scenarios("train", feature=feature)}
        for row in base_rows:
            s = by_id[row["scenario_id"]]
            encoded[(s["id"], row["premium"], row["mapping"])] = [encode_answer(self.tok, row["prompt"], a)
                                                       for a in answer_pair(s, row["premium"], row["mapping"])]
        neutral = [encode_answer(self.tok, r["prompt"], r["response"]) for r in NEUTRAL]
        u = self.t.nn.Parameter(initial / (ref * .1)); opt = self.t.optim.Adam([u], lr=.003)
        start, history = 0, []
        protocol_hash = sha(RUN / "source" / "protocol.py")
        if (out / "latest.json").exists():
            ckinfo = read(out / "latest.json"); assert sha(out / ckinfo["path"]) == ckinfo["sha256"]
            ck = self.t.load(out / ckinfo["path"], map_location="cuda:0", weights_only=False)
            assert ck["layer"] == layer and ck["mode"] == mode and ck["protocol_sha256"] == protocol_hash
            with self.t.no_grad(): u.copy_(ck["u"])
            opt.load_state_dict(ck["optimizer"]); start, history = ck["step"], ck["history"]
            self.t.set_rng_state(ck['cpu_rng'].cpu());self.t.cuda.set_rng_state(ck['cuda_rng'].cpu())
        base_a = [r for r in base_rows if r["mapping"] == "A"]
        for step in range(start, STEPS):
            budget(); began = time.time(); rng = random.Random(202609120100 + step)
            row = rng.choice(base_a); alpha = TRAIN_ALPHAS[step % len(TRAIN_ALPHAS)]
            s = by_id[row["scenario_id"]]; opt.zero_grad(set_to_none=True)
            choice_value, full_value, kl_value = 0., 0., 0.
            for mapping in ["A", "B"]:
                key = (s["id"], row["premium"], mapping)
                target = shifted_target(probabilities[key], alpha, s["benefit_units"])
                logits = self.logits([choice_prompt(s, row["premium"], mapping)], layer, u * ref * .1, alpha, mode)
                ab_log = (logits[0, self.ab] / TEMPERATURE).log_softmax(-1)
                p_a = target if mapping == "A" else 1 - target
                ce = -(p_a * ab_log[0] + (1 - p_a) * ab_log[1])
                if not bool(self.t.isfinite(ce)):raise RuntimeError('Nonfinite choice loss')
                (.5 * ce).backward(); choice_value += float(ce.detach()) * .5
                full = self.weighted_full_ce(encoded[key], target, layer, u * ref * .1, alpha, mode)
                if not bool(self.t.isfinite(full)):raise RuntimeError('Nonfinite full-answer loss')
                (.1 * full).backward(); full_value += float(full.detach()) * .5
                del logits, ab_log, ce, full
            # Exact schedule is documented: soft neutral KL every fourth step.
            neutral_alpha = neutral_alpha_for_step(step)
            if neutral_alpha is not None:
                ex = rng.choice(neutral)
                with self.t.no_grad():
                    ref_logits, _, _ = self.full_logits([ex]); logref = ref_logits.log_softmax(-1)
                    ref_probs = logref.exp(); del ref_logits
                moved, _, _ = self.full_logits([ex], layer, u * ref * .1, neutral_alpha, mode)
                kl = (ref_probs * (logref - moved.log_softmax(-1))).sum(-1).mean()
                (.1 * kl).backward(); kl_value = float(kl.detach())
                del moved, ref_probs, logref, kl
            (.01 * u.square().sum()).backward()
            grad = float(u.grad.norm()); assert math.isfinite(grad)
            assert not any(p.grad is not None for p in self.model.parameters())
            self.t.nn.utils.clip_grad_norm_([u], 1.); opt.step()
            with self.t.no_grad(): u.mul_(min(1., 1. / max(float(u.norm()), 1e-12)))
            history.append(dict(step=step + 1, alpha=alpha, choice_ce=choice_value, full_ce=full_value,
                                neutral_kl=kl_value, neutral_alpha=neutral_alpha, grad_norm=grad, unit_norm=float(u.norm()),
                                seconds=time.time() - began))
            if (step + 1) % 5 == 0:
                save(out / "history.json", history)
                status(self.who, "training", candidate=candidate, step=step + 1, total_steps=STEPS,
                       seconds=history[-1]["seconds"], choice_ce=choice_value, full_ce=full_value)
            if step+1 in (1,5) or (step + 1) % 40 == 0:
                path = out / f"step-{step + 1:04d}.pt"; pending = path.with_suffix(".pending")
                self.t.save(dict(step=step + 1, u=u.detach(), optimizer=opt.state_dict(), history=history,
                                 layer=layer, mode=mode, protocol_sha256=protocol_hash,
                                 cpu_rng=self.t.get_rng_state(),cuda_rng=self.t.cuda.get_rng_state()), pending)
                with pending.open("rb") as f: os.fsync(f.fileno())
                pending.replace(path); save(out / "latest.json", dict(step=step + 1, path=path.name, sha256=sha(path)))
        v = u.detach().cpu().numpy() * ref * .1
        pending = out / "vector.pending"
        with pending.open("wb") as f:
            self.np.savez(f, vector=v); f.flush(); os.fsync(f.fileno())
        pending.replace(out / "vector.npz")
        save(out / "vector_metadata.json", dict(layer=layer, mode=mode, steps=STEPS, ref=ref,
             initial_cosine=float(self.np.dot(v, initial.cpu().numpy()) / (self.np.linalg.norm(v) * float(initial.norm()))),
             feature=feature, protocol_sha256=protocol_hash,
             norm=float(self.np.linalg.norm(v)), sha256=sha(out / "vector.npz")))
        self.cleanup()
        return self.t.tensor(v, device="cuda:0")

    def generate(self, rows, dest, method, alpha=0., layer=None, vector=None, mode="prefill", free=False):
        if dest.exists(): return validate_cached_rows(read(dest), rows, method, alpha, layer, mode)
        output = []
        for start in range(0, len(rows), 16):
            budget(); chunk = dest.parent / "chunks" / dest.stem / f"{start:06d}.json"
            if chunk.exists():
                output.extend(validate_cached_rows(read(chunk), rows[start:start + 16], method, alpha, layer, mode)); continue
            batch = rows[start:start + 16]; inp = self.inputs([r["prompt"] for r in batch])
            seed = int.from_bytes(hashlib.sha256(f'attributes_v1:{free}:{batch[0]["split"]}:{start}'.encode()).digest()[:8], "big") % (2**63 - 1)
            self.t.manual_seed(seed); self.t.cuda.manual_seed_all(seed)
            self.set(layer, vector, alpha, mode); began = time.time()
            with self.t.inference_mode():
                sequences = self.model.generate(**inp, max_new_tokens=128, do_sample=True,
                    temperature=TEMPERATURE, top_p=1., top_k=0, repetition_penalty=1.,
                    eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id, use_cache=True, logits_to_keep=1)
            new = []
            for r, seq in zip(batch, sequences[:, inp["input_ids"].shape[1]:].tolist()):
                end = next((i for i, t in enumerate(seq) if t in self.eos), None)
                tokens = seq if end is None else seq[:end + 1]
                text = self.tok.decode(tokens, skip_special_tokens=True).strip()
                m = re.match(r"^(?:选择\s*)?([AB])(?=\s|[。:：，,.！!？?、；;]|$)", text)
                choice = m.group(1) if m else None
                reason = text[m.end():].strip() if m else ""
                parsed = None; valid_json = False
                if free:
                    try:
                        obj = json.loads(text)
                        valid_json = (isinstance(obj, dict) and obj.get("model_id") in ("base", "upgrade")
                                      and isinstance(obj.get("reason"), str))
                        if valid_json: parsed = obj["model_id"] == "upgrade"; reason = obj["reason"]
                    except (ValueError, TypeError): pass
                elif choice is not None:
                    parsed = choice == r["mapping"]
                new.append(dict(r, method=method, alpha=alpha, layer=layer, mode=mode, seed=seed,
                    response=text, tokens=tokens, terminated=end is not None, choice=choice,
                    parsed_upgrade=parsed, reason_present=len(re.findall(r"[\u4e00-\u9fff]", reason)) >= 6,
                    json_valid=valid_json if free else None, measurement="sampled_json" if free else "sampled_choice"))
            save(chunk, new); output.extend(new)
            status(self.who, "generating", method=method, alpha=alpha, free=free,
                   completed=len(output), total=len(rows), last_batch_seconds=time.time() - began)
        save(dest, output); self.cleanup(); return output

    def cleanup(self):
        self.capture = False
        self.set(); self.model.zero_grad(set_to_none=True); gc.collect(); self.t.cuda.empty_cache()

