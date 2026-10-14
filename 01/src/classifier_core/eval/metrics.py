from collections import Counter
from statistics import mean
from ..taxonomy import PRIORITY_MARKERS, ACTION_MARKERS


def ratio(n, d):
    return n / d if d else None


def match(case, result):
    gold = case.expected
    fields = {
        "category": gold.primary_category == result.primary_category,
        "secondary": set(gold.secondary_categories) == set(result.secondary_categories),
        "priority": gold.priority == result.priority,
        "next_step": gold.next_step == result.next_step_raw,
        "priority_markers": set(gold.priority_markers) == set(result.priority_markers),
        "action_markers": set(gold.action_markers) == set(result.action_markers),
    }
    fields["case_pass"] = all(fields[k] for k in ("category", "priority", "next_step"))
    fields["strict_all_fields"] = all(fields.values())
    return fields


def prf(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def percentile(values, q):
    if not values:
        return None
    data = sorted(values)
    index = (len(data) - 1) * q
    low = int(index)
    return data[low] + (data[min(low + 1, len(data) - 1)] - data[low]) * (index - low)


def metrics(cases, results):
    if len(cases) != len(results):
        raise ValueError("Mismatched case/result counts")
    n = len(cases)
    if not n:
        return {"count": 0}
    matches = [match(c, r) for c, r in zip(cases, results)]
    avg = lambda key: mean(m[key] for m in matches)
    out = {
        "count": n,
        "category_accuracy": avg("category"),
        "secondary_exact": avg("secondary"),
        "secondary_lenient": mean(
            set(c.expected.secondary_categories) <= set(r.secondary_categories)
            for c, r in zip(cases, results)
        ),
        "priority_accuracy": avg("priority"),
        "priority_within_1": mean(
            abs(int(c.expected.priority[1]) - int(r.priority[1])) <= 1
            for c, r in zip(cases, results)
        ),
        "next_step_accuracy": avg("next_step"),
        "case_pass": avg("case_pass"),
        "overall_accuracy": avg("case_pass"),
        "strict_all_fields": avg("strict_all_fields"),
        "under_triage_rate": mean(
            r.priority > c.expected.priority for c, r in zip(cases, results)
        ),
        "over_triage_rate": mean(
            r.priority < c.expected.priority for c, r in zip(cases, results)
        ),
    }
    for kind, names in [
        ("priority_markers", PRIORITY_MARKERS),
        ("action_markers", ACTION_MARKERS),
    ]:
        per = {}
        for marker in names:
            pairs = [
                (marker in getattr(c.expected, kind), marker in getattr(r, kind))
                for c, r in zip(cases, results)
            ]
            per[marker] = prf(
                sum(g and p for g, p in pairs),
                sum(not g and p for g, p in pairs),
                sum(g and not p for g, p in pairs),
            )
        out[kind] = {
            "per_marker": per,
            "micro": prf(
                *(sum(v[k] for v in per.values()) for k in ("tp", "fp", "fn"))
            ),
        }
    pairs = [
        (c.expected.priority == "P0", r.priority == "P0")
        for c, r in zip(cases, results)
    ]
    safety = prf(
        sum(g and p for g, p in pairs),
        sum(not g and p for g, p in pairs),
        sum(g and not p for g, p in pairs),
    )
    out.update(
        P0_recall=safety["recall"],
        P0_precision=safety["precision"],
        P0_counts=safety,
        self_harm_recall=out["priority_markers"]["per_marker"]["self_harm"]["recall"],
    )
    invalid = {
        "invalid_json",
        "schema_violation",
        "truncated",
        "empty",
        "schema_mismatch",
    }
    for key, errors in [
        ("invalid_output_rate", invalid),
        ("timeout_rate", {"timeout", "deadline"}),
        ("rate_limit_rate", {"rate_limit"}),
    ]:
        out[key] = mean(
            any(a.error_type in errors for a in r.attempts) for r in results
        )
    out["invalid_output_rate"] = mean(
        r.status == "recovered" or any(a.error_type in invalid for a in r.attempts)
        for r in results
    )
    out["repair_rate"] = mean(
        r.status == "repaired"
        or any(a.error_type in invalid for a in r.attempts)
        and len(r.attempts) > 1
        for r in results
    )
    out["fallback_rate"] = mean(r.status == "fallback_model" for r in results)
    out["refusal_rate"] = mean(r.status == "provider_refusal" for r in results)
    lat = [r.latency_ms for r in results if not r.cache_hit]
    out["latency_ms"] = {
        "mean": mean(lat) if lat else None,
        "p50": percentile(lat, 0.5),
        "p95": percentile(lat, 0.95),
        "count": len(lat),
    }
    out["cache_hit_rate"] = mean(r.cache_hit for r in results)
    for key in (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
    ):
        values = [getattr(r.usage, key) for r in results]
        out[key] = {
            "total": sum(values) if all(v is not None for v in values) else None,
            "mean": mean(values) if all(v is not None for v in values) else None,
        }
    costs = [r.cost_usd for r in results]
    out["cost_per_ticket"] = mean(costs) if all(v is not None for v in costs) else None
    for key in ("input_cost_usd", "output_cost_usd"):
        vals = [getattr(a, key) for r in results for a in r.attempts]
        out[key] = sum(vals) / n if vals and all(v is not None for v in vals) else None
    out["calibration"] = []
    for lo, hi in [
        (0, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0),
    ]:
        indices = [
            i
            for i, r in enumerate(results)
            if lo <= r.confidence and (r.confidence < hi or hi == 1.0)
        ]
        out["calibration"].append(
            {
                "low": lo,
                "high": hi,
                "count": len(indices),
                "accuracy": (
                    mean(matches[i]["case_pass"] for i in indices) if indices else None
                ),
            }
        )
    out["hitl"] = []
    for threshold in (0.5, 0.7, 0.8, 0.9):
        indices = [
            i
            for i, r in enumerate(results)
            if r.confidence >= threshold
            and r.status == "ok"
            and not any(reason != "low_confidence" for reason in r.review_reasons)
        ]
        out["hitl"].append(
            {
                "threshold": threshold,
                "coverage": len(indices) / n,
                "accuracy": (
                    mean(matches[i]["case_pass"] for i in indices) if indices else None
                ),
            }
        )
    accepted = [i for i, r in enumerate(results) if not r.needs_review]
    out["coverage"] = len(accepted) / n
    out["accepted_accuracy"] = (
        mean(matches[i]["case_pass"] for i in accepted) if accepted else None
    )
    out["confusion_matrix"] = dict(
        Counter(
            f"{c.expected.primary_category} -> {r.primary_category}"
            for c, r in zip(cases, results)
        )
    )
    return out


def sliced_metrics(cases, results):
    out = {"overall": metrics(cases, results), "slices": {}}
    for dimension in ("difficulty", "language", "category"):
        key = lambda c: (
            c.expected.primary_category
            if dimension == "category"
            else getattr(c, dimension)
        )
        out["slices"][dimension] = {}
        for label in sorted({key(c) for c in cases}):
            subset = [(c, r) for c, r in zip(cases, results) if key(c) == label]
            out["slices"][dimension][label] = metrics(*zip(*subset))
    return out


def stability(repetitions):
    if not repetitions:
        return {"flip_rate": None, "accuracy_mean": None, "accuracy_range": None}
    ids = set(repetitions[0])
    if any(set(rep) != ids for rep in repetitions):
        raise ValueError("Repetitions must cover identical cases")
    tuples = lambda r: (
        r.primary_category,
        tuple(sorted(r.secondary_categories)),
        tuple(sorted(r.priority_markers)),
        tuple(sorted(r.action_markers)),
        r.priority,
        r.next_step_raw,
        r.next_step_final,
        r.needs_review,
    )
    return {
        "flip_rate": ratio(
            sum(len({tuples(rep[i]) for rep in repetitions}) > 1 for i in ids), len(ids)
        )
    }
