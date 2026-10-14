import json
from .metrics import match


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_table(cases, results, summary):
    lines = [
        "# Evaluation facts",
        "",
        "P0 metrics have wide uncertainty: the full gold set contains only 3 P0 cases. Raw counts accompany percentages in summary.json.",
        "",
        "| # | lang | diff | input (truncated 80 chars) | expected (cat / P / next) | actual (cat / P / next) | marker diff | pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    failures = []
    for c, r in zip(cases, results):
        g = c.expected
        m = match(c, r)
        diff = {
            key: {
                "missing": sorted(set(getattr(g, key)) - set(getattr(r, key))),
                "extra": sorted(set(getattr(r, key)) - set(getattr(g, key))),
            }
            for key in ("priority_markers", "action_markers")
            if set(getattr(g, key)) != set(getattr(r, key))
        }
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    [
                        c.id,
                        c.language,
                        c.difficulty,
                        c.ticket[:80],
                        f"{g.primary_category} / {g.priority} / {g.next_step}",
                        f"{r.primary_category} / {r.priority} / {r.next_step_raw}",
                        json.dumps(diff),
                        m["case_pass"],
                    ],
                )
            )
            + " |"
        )
        if not m["strict_all_fields"]:
            failures.append(
                f"## Case {c.id}\n\nFailed fields: "
                + ", ".join(k for k, v in m.items() if not v)
                + "\n\nExpected: "
                + json.dumps(g.model_dump(), ensure_ascii=False)
                + "\n\nActual: "
                + json.dumps(
                    r.model_dump(
                        include={
                            "primary_category",
                            "secondary_categories",
                            "priority_markers",
                            "action_markers",
                            "priority",
                            "next_step_raw",
                        }
                    ),
                    ensure_ascii=False,
                )
            )
    lines += [
        "",
        "## Failures",
        "",
        *failures,
        "",
        "## Confusion matrix",
        "",
        "```json",
        json.dumps(summary["overall"]["confusion_matrix"], indent=2),
        "```",
        "",
        "## Per-slice metrics",
        "",
        "| Dimension | Slice | N | Case pass | P0 recall (TP/gold) |",
        "|---|---|---|---|---|",
    ]
    for dim, groups in summary["slices"].items():
        for label, s in groups.items():
            p = s["P0_counts"]
            lines.append(
                f"| {dim} | {label} | {s['count']} | {s['case_pass']:.4f} | {s['P0_recall']} ({p['tp']}/{p['tp']+p['fn']}) |"
            )
    return "\n".join(lines) + "\n"
