"""Pure-CPU compilation and scoring for the frozen mixed 640-probe battery.

The first 480 prompts comprise 240 unchanged IPIP and 240 new-axis test jobs.
The remaining 160 prompts use approved test responses as counterbalanced choices. Scores
are conditional first-token expectations, not generated behavioral ratings or
a validated combined psychometric scale. N always means neuroticism.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math


AXES = tuple("OCEANRTVMS")
MAPPINGS = ("normal", "reversed")
ORDERS = ("high_first", "low_first")
SYSTEM = "你是一个人，根据场景事实回答。"
RESPONSE_INSTRUCTION = "结合当前场景和对方的话，以第一人称直接呈现你此刻会说的话或会采取的行动，也可以同时包含两者。"
CHOICE_INSTRUCTION = (
    "结合当前场景和对方的话，从下面两个选项中选择你此刻更可能说的话或采取的行动。"
    "只输出所选选项的一个大写字母A或B，第一个字符必须是该字母，不要解释。"
)
CHANNELS = {"face_to_face": "当面交谈", "text": "文字私聊",
            "voice": "语音对话", "group_chat": "群聊"}
SCHEMA = "tenaxis_knowledge_action_measurement_v1"


def stable_hash(value):
    """SHA256 of canonical JSON; this is not a source-file byte hash."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing_or_empty_text:{key}")
    return value


def _messages(prompt):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}]


def _message_hash(messages):
    # Match the approved V3 input_sha256 convention exactly (no key sorting).
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _context(row):
    for key in ("scene", "interlocutor", "interaction_channel", "last_utterance"):
        _text(row, key)
    if row["interaction_channel"] not in CHANNELS:
        raise ValueError("unknown_interaction_channel")
    return (f'【场景事实】\n{row["scene"]}\n\n'
            f'【正在与你交谈的人】\n{row["interlocutor"]}\n\n'
            f'【交流方式】\n{CHANNELS[row["interaction_channel"]]}\n\n'
            f'【对方刚刚说】\n{row["last_utterance"]}\n\n')


def score_digit(digit, mapping, key):
    """Return the existing correctly keyed Likert score on 0..100."""
    if type(digit) is not int or digit not in range(1, 6):
        raise ValueError("invalid_digit")
    if mapping not in MAPPINGS or key not in ("+", "-"):
        raise ValueError("invalid_scoring_key")
    score = 25.0 * (digit - 1)
    if mapping == "reversed":
        score = 100.0 - score
    return 100.0 - score if key == "-" else score


def _validate_questionnaire(rows):
    if len(rows) != 480:
        raise ValueError("questionnaire_requires_480_test_jobs")
    seen = set()
    items = defaultdict(dict)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("questionnaire_row_not_dict")
        ident, item = _text(row, "id"), _text(row, "item_id")
        trait, mapping = row.get("trait"), row.get("mapping")
        if ident in seen:
            raise ValueError("duplicate_questionnaire_job")
        seen.add(ident)
        if row.get("split") != "test":
            raise ValueError("questionnaire_train_dev_leakage")
        prefix=trait if trait in 'OCEAN' else f'KAXQ_{trait}_'
        if trait not in AXES or not item.startswith(prefix):
            raise ValueError("questionnaire_trait_mismatch")
        if mapping not in MAPPINGS or row.get("key") not in ("+", "-"):
            raise ValueError("invalid_questionnaire_scoring")
        if row.get("role") is not None or row.get("repeat", 0) != 0:
            raise ValueError("questionnaire_role_or_repeat_not_allowed")
        if "system" in row and row["system"] != SYSTEM:
            raise ValueError("questionnaire_system_changed")
        prompt = _text(row, "prompt")
        if "input_messages" in row and row["input_messages"] != _messages(prompt):
            raise ValueError("questionnaire_input_messages_changed")
        if mapping in items[(trait, item)]:
            raise ValueError("duplicate_questionnaire_item_mapping")
        items[(trait, item)][mapping] = row
    if Counter(trait for trait, _ in items) != Counter({a: 24 for a in AXES}):
        raise ValueError("questionnaire_requires_24_items_per_axis")
    for pair in items.values():
        if set(pair) != set(MAPPINGS):
            raise ValueError("questionnaire_missing_mapping")
        normal, reversed_row = pair["normal"], pair["reversed"]
        for field in ("key", "facet", "family_id", "en", "zh", "split"):
            if normal.get(field) != reversed_row.get(field):
                raise ValueError(f"questionnaire_pair_metadata_changed:{field}")


def _behavior_test_rows(rows):
    if len(rows) not in (80, 280):
        raise ValueError("behavior_requires_80_test_or_280_all_rows")
    seen = set()
    split_counts = Counter()
    versions = defaultdict(set)
    test = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("behavior_row_not_dict")
        ident = _text(row, "id")
        axis, split = row.get("axis"), row.get("split")
        if ident in seen:
            raise ValueError("duplicate_behavior_id")
        seen.add(ident)
        if axis not in AXES or split not in ("train", "dev", "test"):
            raise ValueError("invalid_behavior_axis_or_split")
        prefix, count = {"train": ("TR", 16), "dev": ("DV", 4), "test": ("TE", 8)}[split]
        source_prefixes = (axis, axis + "4") if axis in ("O", "E") else (("KAXB_"+axis,) if axis in "RTVMS" else (axis,))
        if ident not in {f"{p}_{prefix}{i:02d}" for p in source_prefixes for i in range(1, count + 1)}:
            raise ValueError("behavior_id_split_mismatch")
        versions[axis].add(ident.split("_")[0])
        split_counts[(axis, split)] += 1
        if split == "test":
            test.append(row)
    expected = {(a, "test"): 8 for a in AXES}
    if len(rows) == 280:
        expected.update({(a, "train"): 16 for a in AXES})
        expected.update({(a, "dev"): 4 for a in AXES})
    if split_counts != Counter(expected):
        raise ValueError("behavior_split_coverage_changed")
    if any(len(values) != 1 for values in versions.values()):
        raise ValueError("mixed_behavior_versions_within_axis")
    return test


def compile_jobs(questionnaire_rows, behavior_rows):
    """Return 480 questionnaire jobs followed by 160 choice jobs.

    Inputs combine old/new test_probability.json rows and either
    all 280 approved behavior rows or their exact 80-row test subset. Latest
    O4/E4 action anchors can be mixed with the original V3 C/A/N rows. Input
    objects are never mutated. All high/low labels, keys, evidence and trait
    fields remain metadata; only the scene and literal responses enter prompts.
    The outer run manifest must separately bind the source-file byte hashes.
    """
    questionnaire_rows, behavior_rows = list(questionnaire_rows), list(behavior_rows)
    _validate_questionnaire(questionnaire_rows)
    test_rows = _behavior_test_rows(behavior_rows)
    source_q = stable_hash(questionnaire_rows)
    source_b = stable_hash(test_rows)
    jobs = []
    for source in questionnaire_rows:
        row = deepcopy(source)
        row.update(kind="likert", system=SYSTEM, labels=list("12345"), tokens=list("12345"),
                   high_label="5" if score_digit(5, row["mapping"], row["key"]) == 100 else "1",
                   source_row_sha256=stable_hash(source), source_family_sha256=source_q,
                   source_family="IPIP" if row['trait'] in 'OCEAN' else 'original_theory_informed_selfreport', schema=SCHEMA)
        row["input_messages"] = _messages(row["prompt"])
        row["input_sha256"] = _message_hash(row["input_messages"])
        row["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
        jobs.append(row)
    for source in test_rows:
        context = _context(source)
        if source.get("response_instruction") != RESPONSE_INSTRUCTION:
            raise ValueError("behavior_response_instruction_changed")
        original_prompt = context + RESPONSE_INSTRUCTION
        original_messages = _messages(original_prompt)
        source_input_hash = _message_hash(original_messages)
        for field, expected in (("input_text", original_prompt),
                                ("input_messages", original_messages),
                                ("input_sha256", source_input_hash)):
            if field in source and source[field] != expected:
                raise ValueError("behavior_approved_input_changed")
        high, low = _text(source, "high_response"), _text(source, "low_response")
        if high == low:
            raise ValueError("identical_behavior_options")
        for order in ORDERS:
            first, second = (high, low) if order == "high_first" else (low, high)
            prompt = context + CHOICE_INSTRUCTION + f"\n\nA：{first}\n\nB：{second}"
            ident = f'behavior_ab:{source["id"]}:{order}'
            messages = _messages(prompt)
            jobs.append(dict(id=ident, job_id=ident, item_id=source["id"],
                scenario_id=source["id"], trait=source["axis"], feature=source["axis"],
                kind="behavior_ab", split="test", facet=source["facet"],
                mapping=order, order=order, key="high_response",
                high_label="A" if order == "high_first" else "B",
                labels=["A", "B"], tokens=["A", "B"], options={"A": first, "B": second},
                prompt=prompt, system=SYSTEM, input_messages=messages,
                input_sha256=_message_hash(messages),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                source_row_sha256=stable_hash(source), source_family_sha256=source_b,
                source_input_sha256=source_input_hash,
                source_high_response_sha256=hashlib.sha256(high.encode()).hexdigest(),
                source_low_response_sha256=hashlib.sha256(low.encode()).hexdigest(),
                source_family="approved_behavior_test",
                source_data_version=("tenaxis_ka_original_20261010_v1" if source["axis"] in "RTVMS" else
                    "action_anchors_v4" if source["id"].split("_")[0] in ("O4", "E4") else "V3"),
                schema=SCHEMA))
    _validate_jobs(jobs)
    return jobs


def _validate_jobs(jobs):
    if len(jobs) != 640 or len({j.get("id") for j in jobs}) != 640:
        raise ValueError("requires_640_unique_jobs")
    pairs = defaultdict(dict)
    for job in jobs:
        kind, trait, mapping = job.get("kind"), job.get("trait"), job.get("mapping")
        if trait not in AXES or job.get("split") != "test":
            raise ValueError("non_test_or_unknown_trait_job")
        labels = list("12345") if kind == "likert" else ["A", "B"]
        allowed = MAPPINGS if kind == "likert" else ORDERS
        if kind not in ("likert", "behavior_ab") or mapping not in allowed:
            raise ValueError("unknown_kind_or_mapping")
        if job.get("labels") != labels or job.get("tokens") != labels:
            raise ValueError("unexpected_token_labels")
        high = ("5" if score_digit(5, mapping, job.get("key")) == 100 else "1") if kind == "likert" else ("A" if mapping == "high_first" else "B")
        if job.get("high_label") != high:
            raise ValueError("high_label_scoring_mismatch")
        if kind == "behavior_ab" and (job.get("key") != "high_response" or job.get("order") != mapping):
            raise ValueError("behavior_order_scoring_mismatch")
        pair = pairs[(trait, kind, _text(job, "item_id"))]
        if mapping in pair:
            raise ValueError("duplicate_item_mapping")
        pair[mapping] = job
    counts = Counter((a, kind) for a, kind, _ in pairs)
    expected = {(a, kind): 24 if kind == "likert" else 8
                for a in AXES for kind in ("likert", "behavior_ab")}
    if counts != Counter(expected):
        raise ValueError("mixed_battery_item_coverage_changed")
    for (_, kind, _), pair in pairs.items():
        expected_mappings = MAPPINGS if kind == "likert" else ORDERS
        if set(pair) != set(expected_mappings):
            raise ValueError("incomplete_item_pair")
        if kind == "likert" and len({r["key"] for r in pair.values()}) != 1:
            raise ValueError("paired_item_key_mismatch")


def _probability(value, name):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"invalid_probability:{name}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_probability:{name}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"invalid_probability:{name}")
    return number


def _mean(values):
    return math.fsum(values) / len(values)


def _stats(values):
    return {"mean": _mean(values), "min": min(values), "max": max(values),
            "count": len(values), "zero_count": sum(v == 0.0 for v in values)}


def summarize_node(jobrows, probabilities, token_mass):
    """Score one complete node without filtering or clipping any observations.

    Each probability row follows that job's labels, conditional on the valid
    tokens, and must sum to one within 1e-5. Floating-point sum error within
    that bound is normalized explicitly; raw arrays remain the caller's output.
    token_mass is the corresponding unconditional selected-token probability
    mass at the same temperature. Zero mass is retained as a diagnostic.
    Mapping/order gaps are signed first-minus-second; absolute gaps and paired
    item gaps are also returned so opposing item biases cannot cancel unseen.
    """
    jobs, probabilities, token_mass = list(jobrows), list(probabilities), list(token_mass)
    _validate_jobs(jobs)
    if len(probabilities) != len(jobs) or len(token_mass) != len(jobs):
        raise ValueError("probability_or_mass_length_mismatch")
    scores = defaultdict(dict)
    mass = defaultdict(list)
    all_mass = []
    sum_error = 0.0
    for job, raw, raw_mass in zip(jobs, probabilities, token_mass):
        if len(raw) != len(job["labels"]):
            raise ValueError("probability_label_count_mismatch")
        p = [_probability(v, "conditional") for v in raw]
        total = math.fsum(p)
        if abs(total - 1.0) > 1e-5:
            raise ValueError("conditional_probabilities_not_normalized")
        sum_error = max(sum_error, abs(total - 1.0))
        p = [v / total for v in p]
        valid_mass = _probability(raw_mass, "token_mass")
        kind, trait, mapping = job["kind"], job["trait"], job["mapping"]
        if kind == "likert":
            score = math.fsum(p[i] * score_digit(i + 1, mapping, job["key"]) for i in range(5))
        else:
            score = 100.0 * p[job["labels"].index(job["high_label"])]
        scores[(trait, kind, job["item_id"])][mapping] = score
        mass[(trait, kind)].append(valid_mass)
        all_mass.append(valid_mass)
    result = {"Q": {}, "B": {}, "S": {}, "Q_by_mapping": {}, "B_by_order": {},
              "mapping_gap": {}, "mapping_gap_abs": {}, "order_gap": {}, "order_gap_abs": {},
              "mapping_item_gap": {}, "order_item_gap": {}, "counts": {},
              "token_mass": {"overall": _stats(all_mass), "by_axis": {}},
              "conditional_probability_sum_error_max": sum_error,
              "family_weights": {"Q": 0.5, "B": 0.5}, "schema": SCHEMA,
              "measurement": "native_OFF_conditional_first_token_expectation"}
    for trait in AXES:
        result["token_mass"]["by_axis"][trait] = {}
        for kind, family, mappings, by_name, gap_name, item_gap_name in (
                ("likert", "Q", MAPPINGS, "Q_by_mapping", "mapping_gap", "mapping_item_gap"),
                ("behavior_ab", "B", ORDERS, "B_by_order", "order_gap", "order_item_gap")):
            pairs = [pair for (a, k, _), pair in scores.items() if a == trait and k == kind]
            by_mapping = {m: _mean([pair[m] for pair in pairs]) for m in mappings}
            result[by_name][trait] = by_mapping
            result[family][trait] = _mean([_mean(list(pair.values())) for pair in pairs])
            gap = by_mapping[mappings[0]] - by_mapping[mappings[1]]
            result[gap_name][trait] = gap
            result[gap_name + "_abs"][trait] = abs(gap)
            paired_gaps = [abs(pair[mappings[0]] - pair[mappings[1]]) for pair in pairs]
            result[item_gap_name][trait] = {"mean_abs": _mean(paired_gaps), "max_abs": max(paired_gaps)}
            result["token_mass"]["by_axis"][trait][family] = _stats(mass[(trait, kind)])
        result["S"][trait] = 0.5 * result["Q"][trait] + 0.5 * result["B"][trait]
        result["counts"][trait] = {"Q_items": 24, "Q_probes": 48, "B_scenes": 8, "B_probes": 16}
    return result

