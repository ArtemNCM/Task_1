from collections import defaultdict
import json
from statistics import mean
from pathlib import Path
from ..schema import Case, ClassifierResult
from .metrics import stability, metrics


def read_predictions(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        r["case"]["id"]: (
            Case.model_validate(r["case"]),
            ClassifierResult.model_validate(r["result"]),
        )
        for r in rows
    }


def read_runs(roots, include_mock=False):
    runs = []
    for root in roots:
        for path in Path(root).glob("*/meta.json"):
            meta = json.loads(path.read_text(encoding="utf-8"))
            if not meta.get("complete") or (meta["is_mock"] and not include_mock):
                continue
            summary = json.loads(
                (path.parent / "summary.json").read_text(encoding="utf-8")
            )
            runs.append(
                (meta, summary, read_predictions(path.parent / "predictions.jsonl"))
            )
    return runs


def display(value):
    return "price missing" if value is None else f"{value:.6f}"


def comparison(runs):
    lines = [
        "# Model comparison",
        "",
        "Mock runs are infrastructure evidence only. Full-set results are secondary; dev/holdout are separate. Latency excludes cache hits.",
        "",
        "| Classifier | Prompt | Split | Runs | Accuracy mean [min,max] | P0 recall TP/gold | P0 precision TP/pred | Self-harm recall | p50/p95 ms | Cost/ticket | Invalid/repair/fallback/refusal/timeout/rate-limit | Flip rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups = defaultdict(list)
    for run in runs:
        m = run[0]
        groups[
            (
                m["classifier"],
                m["prompt"],
                m["split"],
                m["dataset_hash"],
                m["config_hash"],
                m["prompt_hash"],
            )
        ].append(run)
    for key, group in groups.items():
        results = [{i: r for i, (_, r) in g[2].items()} for g in group]
        cases = [c for c, _ in group[0][2].values()]
        # Aggregate repeats as measured samples; do not pretend independent safety cases.
        all_cases = [c for g in group for c, _ in g[2].values()]
        all_results = [r for g in group for _, r in g[2].values()]
        s = metrics(all_cases, all_results)
        p = s["P0_counts"]
        acc = [g[1]["overall"]["case_pass"] for g in group]
        flip = stability(results)["flip_rate"]
        rates = "/".join(
            f"{s[k]:.3f}"
            for k in (
                "invalid_output_rate",
                "repair_rate",
                "fallback_rate",
                "refusal_rate",
                "timeout_rate",
                "rate_limit_rate",
            )
        )
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(group)} | {mean(acc):.4f} [{min(acc):.4f},{max(acc):.4f}] | {s['P0_recall']} ({p['tp']}/{p['tp']+p['fn']}) | {s['P0_precision']} ({p['tp']}/{p['tp']+p['fp']}) | {s['self_harm_recall']} | {s['latency_ms']['p50']}/{s['latency_ms']['p95']} | {display(s['cost_per_ticket'])} | {rates} | {flip} |"
        )
    if not groups:
        lines += ["", "NOT RUN YET — no eligible live runs."]
    lines += [
        "",
        "Only 3 P0 cases exist in the full gold dataset; uncertainty is wide. Counts across repetitions repeat the same cases.",
        "",
        "TODO(human): argued final model choice after live measurements.",
    ]
    return "\n".join(lines) + "\n"


def simulate(cheap, strong):
    if set(cheap) != set(strong):
        raise ValueError("Cascade requires identical case ids")
    cases = []
    results = []
    costs = []
    escalations = 0
    for key, (case, a) in cheap.items():
        other, b = strong[key]
        if case.model_dump() != other.model_dump():
            raise ValueError("Cascade case contents/gold differ")
        if a.cache_hit or b.cache_hit:
            raise ValueError(
                "Cascade cost requires uncached predictions; rerun with --no-cache"
            )
        take = a.needs_review
        escalations += take
        cases.append(case)
        results.append(b if take else a)
        costs.append(
            None
            if a.cost_usd is None or (take and b.cost_usd is None)
            else a.cost_usd + (b.cost_usd if take else 0)
        )
    s = metrics(cases, results)
    s["cost_per_ticket"] = mean(costs) if all(c is not None for c in costs) else None
    s["escalation_rate"] = escalations / len(cases)
    s["latency_ms"] = {
        "note": "Offline simulation: sequential live latency NOT MEASURED"
    }
    s["simulation"] = True
    return s


def projections(runs):
    lines = [
        "# Cost projection: 10,000 tickets/month",
        "",
        "All real-model results: NOT RUN YET unless listed below. Mock token counts are synthetic and excluded from projections.",
        "",
        "| Run | Mean input/output tokens | Single model USD/month | Batch assumption | Cached input tokens | Measured provider cache savings USD/month |",
        "|---|---|---|---|---|---|",
    ]
    for meta, summary, pred in runs:
        if meta["is_mock"]:
            continue
        results = [r for _, r in pred.values()]
        # Exclude local cache hits: zero-bill reruns are not workload measurements.
        noncached = [(c, r) for c, r in pred.values() if not r.cache_hit]
        if not noncached:
            continue
        s = metrics(*zip(*noncached))
        cost = s["cost_per_ticket"]
        batch = "not applicable"
        if meta["classifier"] in ("haiku-5-5", "sonnet-5-5"):
            batch = (
                display(cost * 10000 * 0.5 if cost is not None else None)
                + " USD (hypothetical direct Anthropic Batch)"
            )
        savings = []
        for _, result in noncached:
            ticket_savings = 0.0
            for attempt in result.attempts:
                model = next(
                    (m for m in meta["params"] if m["id"] == attempt.model_id), None
                )
                price = meta["prices"].get(model["price_key"]) if model else None
                cached = attempt.usage.cached_input_tokens
                if (
                    cached is None
                    or not price
                    or price.get("cached_input_per_mtok") is None
                    or price.get("input_per_mtok") is None
                ):
                    ticket_savings = None
                    break
                ticket_savings += (
                    cached
                    * (price["input_per_mtok"] - price["cached_input_per_mtok"])
                    / 1e6
                )
            savings.append(ticket_savings)
        saving = mean(savings) * 10000 if all(v is not None for v in savings) else None
        lines.append(
            f"| {meta['classifier']} / {meta['prompt']} / {meta['split']} | {s['input_tokens']['mean']}/{s['output_tokens']['mean']} | {display(cost*10000 if cost is not None else None)} | {batch} | {s['cached_input_tokens']['mean']} | {display(saving)} |"
        )
    lines += [
        "",
        "Assumptions: measured API usage only, including retries. Null usage or pricing produces price missing; never assume zero. Seed prices are UNVERIFIED, dated 2026-10-12. The hypothetical direct Anthropic Batch scenario assumes the same measured tokens and prices, with the documented 50% discount; it is not an OpenRouter billing claim. No Batch adapter is implemented. Source: https://platform.claude.com/docs/en/build-with-claude/batch-processing. Provider prompt-cache savings require reported cached usage and a configured cache-read price.",
        "",
        "## Offline cascade scenarios",
        "",
    ]
    pairs = 0
    for cheap in runs:
        if cheap[0]["is_mock"]:
            continue
        for strong in runs:
            if strong[0]["classifier"] != "sonnet-5-5" or strong[0]["is_mock"]:
                continue
            if cheap[0]["classifier"] == strong[0]["classifier"]:
                continue
            if any(
                cheap[0][key] != strong[0][key]
                for key in ("dataset_hash", "split_hash", "split")
            ):
                continue
            sim = simulate(cheap[2], strong[2])
            cost = sim["cost_per_ticket"]
            pairs += 1
            lines.append(
                f"- {cheap[0]['classifier']} → {strong[0]['classifier']}: escalation {sim['escalation_rate']:.4f}; monthly {display(cost*10000 if cost is not None else None)} USD (offline paired predictions)."
            )
    if not pairs:
        lines.append("NOT RUN YET — matched cheap/reference predictions required.")
    return "\n".join(lines) + "\n"
