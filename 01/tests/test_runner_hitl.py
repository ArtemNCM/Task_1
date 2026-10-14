import asyncio
import json
import pytest
from classifier_core.eval.runner import arguments, run_evaluation
from classifier_core.eval.dataset import load_dataset, gold_decision
from classifier_core.eval.comparison import read_predictions, simulate
from classifier_core.classifier import LLMClassifier
from classifier_core.providers.mock import MockProvider
from classifier_core.taxonomy import ROOT, load_config
from classifier_core.hitl import save_correction


def test_holdout_guards():
    for split in ("all", "holdout"):
        with pytest.raises(SystemExit):
            arguments(["--split", split])
    assert arguments(["--split", "holdout", "--i-confirm-frozen"]).i_confirm_frozen


def test_persistence_and_feedback(tmp_path):
    cases = [load_dataset(ROOT / "data/obrio_task1_70_tickets.json")[49]]
    model = load_config("models.yaml")["models"]["mock-oracle"]
    classifier = LLMClassifier(
        [model],
        {"mock-oracle": MockProvider({c.ticket: gold_decision(c) for c in cases})},
        load_config("settings.yaml"),
        {},
    )
    out = tmp_path / "run"
    queue = tmp_path / "review.jsonl"
    asyncio.run(
        run_evaluation(
            cases, classifier, out, {"is_mock": True, "split": "unit"}, queue=queue
        )
    )
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["complete"]
    pred = read_predictions(out / "predictions.jsonl")
    assert len(pred) == 1
    assert simulate(pred, pred)["case_pass"] == 1
    item = json.loads(queue.read_text(encoding="utf-8"))
    corrections = tmp_path / "corrections.jsonl"
    save_correction(
        corrections, item, cases[0].expected.model_dump(), "accepted unit fixture"
    )
    assert load_dataset(corrections)[0].expected == cases[0].expected
