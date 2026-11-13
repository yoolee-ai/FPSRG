"""Complete-final rating and choice scoring from the six-axis experiment."""
import math, re
from collections import defaultdict
from statistics import mean

def score(case, result):
    """Score exact final strings, never search raw thinking for an answer."""
    if result.get("complete") is not True:
        return None
    # Complete is the shared runtime's native-EOS/closure/nonempty-final flag.
    # Explicit contradictory native flags remain invalid, if supplied.
    if any(result.get(k) is False for k in ("ended_with_eos", "thinking_closed")):
        return None
    final = result.get("final")
    if not isinstance(final, str):
        return None
    if case["kind"] == "neutral":
        if re.fullmatch(r"-?(0|[1-9][0-9]*)", final) is None:
            return None
        return 100.0 if final == str(case["target"]) else 0.0
    if case["kind"] == "Q":
        if final not in ("1", "2", "3", "4", "5"):
            return None
        high = str(case["targets"]["high"])
        if high not in ("1", "5"):
            raise ValueError("Invalid_questionnaire_target")
        value = 25.0 * (int(final) - 1)
        return value if high == "5" else 100.0 - value
    if case["kind"] == "B":
        if final not in ("A", "B"):
            return None
        high = str(case["targets"]["high"])
        if high not in ("A", "B"):
            raise ValueError("Invalid_behavior_target")
        return 100.0 if final == high else 0.0
    raise ValueError("Unknown_case_kind:" + str(case["kind"]))

def _stats(values):
    valid = [v for v in values if v is not None]
    return {"n": len(values), "valid_n": len(valid), "invalid_n": len(values) - len(valid),
            "valid_rate": len(valid) / len(values) if values else None,
            "score_valid_only": mean(valid) if valid else None}

def summary(rows):
    """Return denominators and valid-only means; caller selects a condition.

    Input is iterable[{case, result, optional repeat/alpha/node/...}]. Per-axis
    Q and B means are equally weighted as S, not pooled by unequal item counts.
    Mapping/repeat groups are retained. No invalid-value imputation is applied.
    """
    groups, all_values = defaultdict(list), []
    for row in rows:
        case, result = row["case"], row["result"]
        value = score(case, result)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 100):
            raise ValueError("Invalid_score")
        all_values.append(value)
        axis, kind = case["axis"], case["kind"]
        groups[(axis, kind)].append(value)
        if "mapping" in case:
            groups[(axis, kind, "mapping", str(case["mapping"]))].append(value)
        repeat = row.get("repeat", result.get("repeat"))
        if repeat is not None:
            groups[(axis, kind, "repeat", str(repeat))].append(value)
            if "mapping" in case:
                groups[(axis, kind, "mapping_repeat", str(case["mapping"]), str(repeat))].append(value)
    by_axis = {}
    for axis in sorted({key[0] for key in groups}):
        kinds = {key[1]: _stats(values) for key, values in groups.items() if len(key) == 2 and key[0] == axis}
        q, b = kinds.get("Q", {}).get("score_valid_only"), kinds.get("B", {}).get("score_valid_only")
        by_axis[axis] = {**kinds, "S": (q + b) / 2 if q is not None and b is not None else None}
    return {"overall": _stats(all_values), "by_axis": by_axis,
            "by_mapping": {"/".join(key): _stats(values) for key, values in sorted(groups.items()) if len(key) > 2 and key[2] == "mapping"},
            "by_repeat": {"/".join(key): _stats(values) for key, values in sorted(groups.items()) if len(key) > 2 and key[2] == "repeat"},
            "by_mapping_repeat": {"/".join(key): _stats(values) for key, values in sorted(groups.items()) if len(key) > 2 and key[2] == "mapping_repeat"},
            "note": "Valid-only means; invalids kept in denominators, never low-trait zero. S=(Q+B)/2. Overall is a pooled diagnostic, not the equal-task trait composite."}

