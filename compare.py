"""Print a compact comparison of saved evaluation runs. No API calls."""

import argparse
import json
from pathlib import Path

from bootstrap import ROOT
from classifier_core.eval.comparison import read_predictions
from classifier_core.eval.metrics import match


def summarize(folder: Path, reference: dict | None) -> tuple[list[str], dict]:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    rows = read_predictions(folder / "predictions.jsonl")
    if not meta["complete"] or len(rows) != 70 or meta["is_mock"]:
        raise ValueError(f"Expected a complete live 70-ticket run: {folder}")

    dataset = {case_id: case.model_dump() for case_id, (case, _) in rows.items()}
    if reference is not None and dataset != reference:
        raise ValueError("Runs must use the same tickets and gold labels")

    main = sum(match(case, result)["case_pass"] for case, result in rows.values())
    strict = sum(
        match(case, result)["strict_all_fields"] for case, result in rows.values()
    )
    attempts = sum(len(result.attempts) for _, result in rows.values())
    line = [
        str(meta["classifier"]),
        str(meta["prompt"]),
        folder.name,
        f"{main}/70",
        f"{strict}/70",
        str(attempts),
    ]
    return line, dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="evaluation run folders")
    parser.add_argument("--output", type=Path, help="optional Markdown output file")
    args = parser.parse_args()

    reference = None
    rows = []
    for folder in args.runs:
        row, reference = summarize(folder, reference)
        rows.append(row)

    table = [
        "| Model | Prompt | Run | Main | Strict | Attempts |",
        "|---|---|---|---:|---:|---:|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]
    report = "\n".join(table) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from None
