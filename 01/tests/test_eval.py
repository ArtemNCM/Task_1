import json
import pytest
from classifier_core.eval.dataset import load_dataset, gold_decision
from classifier_core.eval.split import get_split, digest
from classifier_core.eval.metrics import metrics
from classifier_core.rules import derive
from classifier_core.schema import ClassifierResult
from classifier_core.taxonomy import ROOT


def test_split(tmp_path):
    cases = load_dataset(ROOT / "data/obrio_task1_70_tickets.json")
    path = tmp_path / "split.json"
    first = get_split(path, cases)
    sha = digest(path)
    assert get_split(path, cases) == first and digest(path) == sha
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        get_split(path, cases)


def test_duplicate_ids(tmp_path):
    row = load_dataset(ROOT / "data/obrio_task1_70_tickets.json")[0].model_dump()
    path = tmp_path / "data.json"
    path.write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_toy_metrics():
    cases = load_dataset(ROOT / "data/obrio_task1_70_tickets.json")
    cases = [cases[0], cases[49]]
    results = []
    for c in cases:
        d = gold_decision(c)
        results.append(
            ClassifierResult(
                **d.model_dump(),
                **derive(d),
                status="ok",
                model_id="toy",
                prompt_version="v1",
            )
        )
    results[1].priority = "P3"
    results[1].priority_markers = []
    results[1].next_step_raw = "route_support"
    m = metrics(cases, results)
    assert (
        m["case_pass"] == 0.5
        and m["under_triage_rate"] == 0.5
        and m["over_triage_rate"] == 0
    )
    assert m["P0_recall"] == 0 and m["P0_counts"]["fn"] == 1
    assert m["priority_markers"]["per_marker"]["self_harm"]["recall"] == 0
    assert (
        m["calibration"][-1]["count"] == 2 and m["calibration"][-1]["accuracy"] == 0.5
    )
    assert m["cost_per_ticket"] is None
