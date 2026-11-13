"""Development-only maximin percentile selection; no test-derived threshold.

Both endpoints retain their pretrained sign. Failure to achieve two positive
development effects is exposed as diagnostic, not hidden by dropping an axis.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile

from study import TRAITS, LAYERS, DEV_ALPHAS, MAPPINGS, candidates


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _number(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a development score")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite development value")
    return number


def _effects(curve):
    scores = [point["value"] for point in curve]
    return dict(positive_effect=scores[-1] - scores[2], negative_effect=scores[2] - scores[0],
                bidirectional_min_effect=min(scores[-1] - scores[2], scores[2] - scores[0]),
                endpoint_span=scores[-1] - scores[0])


def questionnaire_metrics(rows, trait, frozen_dev):
    """Equal-weight mapping mean, retaining every mapping-specific curve."""
    expected = {(str(r["item_id"]), r["mapping"]): r for r in frozen_dev}
    if len(expected) != 12 or len({k[0] for k in expected}) != 6:
        raise ValueError("frozen questionnaire dev must contain six items/two mappings")
    if any(r.get("split") != "dev" or r.get("trait", r.get("feature")) != trait for r in frozen_dev):
        raise ValueError("nondev or wrong-axis frozen questionnaire input")
    grouped, seen = defaultdict(list), set()
    for row in rows:
        if row.get("split", "dev") != "dev" or row.get("trait", row.get("feature")) != trait:
            raise ValueError("selection received nondev or wrong-axis questionnaire score")
        alpha, mapping = _number(row["alpha"]), row["mapping"]
        key = (str(row["item_id"]), mapping)
        identity = (key, alpha)
        if alpha not in DEV_ALPHAS or mapping not in MAPPINGS or key not in expected or identity in seen:
            raise ValueError("unexpected or duplicate questionnaire development cell")
        seen.add(identity)
        if "key" in row and row["key"] != expected[key]["key"]:
            raise ValueError("questionnaire scoring key changed")
        score = _number(row["expected_score01"])
        if not 0 <= score <= 1:
            raise ValueError("questionnaire score outside [0,1]")
        mass = _number(row["number_mass"]) if "number_mass" in row else None
        if mass is not None and not 0 <= mass <= 1 + 1e-7:
            raise ValueError("invalid numeric token mass")
        grouped[(mapping, alpha)].append((score, mass))
    if seen != {(key, alpha) for key in expected for alpha in DEV_ALPHAS}:
        raise ValueError("questionnaire development incomplete")
    curves = {mapping: [dict(alpha=alpha, value=mean(x[0] for x in grouped[(mapping, alpha)]),
                            n=len(grouped[(mapping, alpha)]),
                            mean_number_mass=(mean(x[1] for x in grouped[(mapping, alpha)])
                                if all(x[1] is not None for x in grouped[(mapping, alpha)]) else None))
                       for alpha in DEV_ALPHAS] for mapping in MAPPINGS}
    pooled = [dict(alpha=a, value=mean(curves[m][i]["value"] for m in MAPPINGS))
              for i, a in enumerate(DEV_ALPHAS)]
    gaps = [abs(curves["normal"][i]["value"] - curves["reversed"][i]["value"]) for i in range(5)]
    return dict(**_effects(pooled), curves=curves, pooled_curve=pooled,
                mapping_effects={m: _effects(curves[m]) for m in MAPPINGS},
                baseline_mapping_gap=gaps[2], maximum_mapping_gap=max(gaps),
                mapping_drift=max(0.0, max(gaps) - gaps[2]), n_rows=len(rows))


def behavior_metrics(rows, trait, frozen_dev):
    expected = {r["id"]: r for r in frozen_dev}
    if len(expected) != 4 or any(r.get("split") != "dev" or r.get("axis", r.get("trait")) != trait for r in frozen_dev):
        raise ValueError("frozen behavior dev must contain four own-axis dev scenes")
    grouped, seen, scene_values = defaultdict(list), set(), defaultdict(dict)
    for row in rows:
        ident = str(row.get("scene_id", row.get("id", "")))
        alpha = _number(row["alpha"])
        if row.get("split") != "dev" or row.get("trait") != trait:
            raise ValueError("selection received nondev or wrong-axis behavior score")
        if ident not in expected or alpha not in DEV_ALPHAS or (ident, alpha) in seen:
            raise ValueError("unexpected or duplicate behavior development cell")
        seen.add((ident, alpha))
        if row.get("input_sha256") != expected[ident]["input_sha256"]:
            raise ValueError("behavior full-prefix hash differs from frozen V3 input")
        high, low = _number(row["high_mean_logprob"]), _number(row["low_mean_logprob"])
        delta = _number(row["preference_logprob_delta"])
        if high > 1e-6 or low > 1e-6 or abs(delta - (high - low)) > 1e-6:
            raise ValueError("invalid target log-likelihood or inconsistent delta")
        grouped[alpha].append(delta)
        scene_values[ident][str(alpha)] = dict(high_mean_logprob=high, low_mean_logprob=low, delta=delta)
    if seen != {(ident, alpha) for ident in expected for alpha in DEV_ALPHAS}:
        raise ValueError("behavior development incomplete")
    curve = [dict(alpha=alpha, value=mean(grouped[alpha]), n=len(grouped[alpha])) for alpha in DEV_ALPHAS]
    differences = [b["value"] - a["value"] for a, b in zip(curve, curve[1:])]
    monotonicity = mean(1.0 if d > 0 else 0.5 if d == 0 else 0.0 for d in differences)
    return dict(**_effects(curve), curve=curve, scene_values=dict(scene_values),
                monotonicity=monotonicity, maximum_adjacent_reversal=max(0.0, *(-d for d in differences)),
                n_rows=len(rows), measurement="length_normalized_target_loglikelihood_difference_not_behavior_probability")


def percentile_ranks(values):
    """Higher is better; tied midranks scaled by n-1, all ties yield 0.5."""
    values = [_number(v) for v in values]
    if len(values) < 2:
        raise ValueError("rank needs at least two candidates")
    return [(sum(x < v for x in values) + 0.5 * (sum(x == v for x in values) - 1)) / (len(values) - 1)
            for v in values]


def rank_entries(entries):
    if len(entries) != 8 or {e["candidate"]["layer"] for e in entries} != set(LAYERS):
        raise ValueError("selection requires complete eight-layer development evidence")
    qr = percentile_ranks([e["questionnaire"]["bidirectional_min_effect"] for e in entries])
    br = percentile_ranks([e["behavior"]["bidirectional_min_effect"] for e in entries])
    for entry, q, b in zip(entries, qr, br):
        entry["ranks"] = dict(questionnaire=q, behavior=b, minimum=min(q, b), average=(q + b) / 2)
        entry["both_positive"] = (entry["questionnaire"]["bidirectional_min_effect"] > 0
                                  and entry["behavior"]["bidirectional_min_effect"] > 0)
    return sorted(entries, key=lambda e: (-e["ranks"]["minimum"], -e["ranks"]["average"],
        -e["behavior"]["monotonicity"], e["questionnaire"]["mapping_drift"],
        e["candidate"]["layer"], e["candidate"]["id"]))


def select_candidates(run):
    """Read ONLY candidate artifacts and frozen dev inputs; return axis mapping.

    Incomplete/corrupt data are errors, not evidence of weak effect. Weak finite
    effects remain eligible and diagnostic. selection_metrics.json records all
    40 candidates. The controller commits the returned mapping exactly once as
    selection.json; this function never publishes that admission artifact.
    """
    run = Path(run)
    behavior = _read(run / "data" / "behavior.json")
    frozen_behavior = {a: [r for r in behavior if r.get("axis", r.get("trait")) == a and r["split"] == "dev"] for a in TRAITS}
    grouped = defaultdict(list)
    for candidate in candidates():
        axis, cid = candidate["trait"], candidate["id"]
        folder = run / "candidates" / cid
        complete = _read(folder / "complete.json")
        if complete.get("status") != "complete" or complete.get("candidate_id") != cid:
            raise ValueError(f"candidate not complete: {cid}")
        if complete.get("condition") != candidate:
            raise ValueError(f"candidate condition differs from frozen schedule: {cid}")
        artifacts = complete.get("artifact_sha256", {})
        required = ("vector.npz", "vector_metadata.json", "dev_scores.json", "dev_behavior.json")
        for name in required:
            if name not in artifacts or _sha(folder / name) != artifacts[name]:
                raise ValueError(f"candidate artifact unsealed or changed: {cid}/{name}")
        metadata = _read(folder / "vector_metadata.json")
        if metadata.get("layer") != candidate["layer"] or metadata.get("mode", metadata.get("train_mode")) != "all":
            raise ValueError("vector metadata differs from candidate")
        if metadata.get("sha256") != artifacts["vector.npz"]:
            raise ValueError("vector metadata content hash differs")
        q = questionnaire_metrics(_read(folder / "dev_scores.json"), axis,
                                  _read(run / "data" / axis / "dev_likert.json"))
        b = behavior_metrics(_read(folder / "dev_behavior.json"), axis, frozen_behavior[axis])
        grouped[axis].append(dict(candidate=candidate, questionnaire=q, behavior=b,
            artifact_sha256=artifacts, vector_metadata=metadata,
            vector_file=str(folder / "vector.npz"), vector_relative_path=f"candidates/{cid}/vector.npz"))
    selected, ranked = {}, {}
    for axis in TRAITS:
        ordered = rank_entries(grouped[axis])
        ranked[axis] = ordered
        best = ordered[0]
        any_positive = any(e["both_positive"] for e in ordered)
        selected[axis] = dict(trait=axis, feature=axis, candidate_id=best["candidate"]["id"],
            id=best["candidate"]["id"], candidate=best["candidate"], layer=best["candidate"]["layer"],
            train_mode="all", inference_mode="all", vector_file=best["vector_file"],
            vector_path=best["vector_file"], vector_relative_path=best["vector_relative_path"],
            vector_sha256=best["artifact_sha256"]["vector.npz"], vector_metadata=best["vector_metadata"],
            artifact_sha256=best["artifact_sha256"], ranks=best["ranks"],
            questionnaire=best["questionnaire"], behavior=best["behavior"], both_positive=best["both_positive"],
            diagnostic_fallback=not best["both_positive"], any_candidate_both_positive=any_positive,
            selection_status="positive_both_metrics" if best["both_positive"] else "diagnostic_fallback",
            sign_flipped=False)
    _write(run / "selection_metrics.json", dict(schema_version=1, source="development_only",
        rule="max_min_percentile_then_mean_percentile_then_behavior_adjacent_monotonicity_then_added_mapping_drift_then_layer",
        rank="tied_midranks_divided_by_n_minus_one", questionnaire_mapping_pooling="equal_weight_mean",
        endpoint_effect="min(value(+1)-value(0),value(0)-value(-1))",
        no_effect_gate=True, no_sign_flip=True, candidates=ranked))
    return selected

