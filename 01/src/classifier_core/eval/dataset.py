import json
from pathlib import Path
from ..schema import Case, Decision
from ..rules import derive


def load_dataset(*paths):
    cases = []
    for path in paths:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        data = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else json.loads(text)
        )
        rows = data["cases"] if isinstance(data, dict) else data
        for row in rows:
            if "human" in row:
                row = row["human"]
            cases.append(Case.model_validate(row))
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dataset ids")
    return cases


def gold_decision(case):
    gold = case.expected.model_dump(exclude={"priority", "next_step"})
    # Gold has no routing hint: explicit evaluation-only adapter for marketing labels.
    return Decision(
        **gold,
        routing_hint=(
            "marketing"
            if case.expected.next_step == "route_product_marketing"
            else None
        ),
        confidence=1.0,
        language=case.language,
    )


def validate_rules(cases):
    mismatches = []
    for case in cases:
        actual = derive(gold_decision(case))
        if (actual["priority"], actual["next_step_raw"]) != (
            case.expected.priority,
            case.expected.next_step,
        ):
            mismatches.append(
                {
                    "id": case.id,
                    "expected": case.expected.model_dump(),
                    "actual": actual,
                }
            )
    return mismatches
