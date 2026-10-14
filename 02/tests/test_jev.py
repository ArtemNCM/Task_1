import asyncio
from pathlib import Path
import httpx
import pytest
import yaml
from classifier_core.taxonomy import ROOT, TAXONOMY, load_config
from classifier_core.eval.dataset import load_dataset, gold_decision
from classifier_core.providers.base import Response
from classifier_jev.questions import build_questions
from classifier_jev.jev_client import JevClient, mock_transport
from classifier_jev.jev_classifier import JevClassifier

CONFIG = yaml.safe_load(
    (Path(__file__).parents[1] / "config/jev.yaml").read_text(encoding="utf-8")
)


def test_questions():
    questions = build_questions()
    assert len(questions) == 24 and len(questions[0].options) == 7
    assert set(TAXONOMY["markers"]) <= {q.key for q in questions}


def test_question_versions_preserve_wire_contract():
    original = build_questions("q1")
    tuned = build_questions("q2")
    assert [(q.key, q.kind, q.options) for q in original] == [
        (q.key, q.kind, q.options) for q in tuned
    ]
    assert sum(a.proposition != b.proposition for a, b in zip(original, tuned)) == 9
    assert build_questions() == original
    revised = build_questions("q3")
    assert [(q.key, q.kind, q.options) for q in revised] == [
        (q.key, q.kind, q.options) for q in tuned
    ]
    assert {
        a.key for a, b in zip(tuned, revised) if a.proposition != b.proposition
    } == {"primary_category", "insufficient_info"}
    next_version = build_questions("q4")
    assert {a.key for a, b in zip(revised, next_version) if a != b} == {
        "insufficient_info"
    }
    with pytest.raises(ValueError):
        build_questions("unknown")


@pytest.mark.parametrize(
    "scenario,status,count",
    [
        ("ok", "ok", 1),
        ("server_error_then_ok", "ok", 2),
        ("schema_mismatch", "unclassified_human", 2),
        ("always_fail", "unclassified_human", 4),
    ],
)
def test_http_shared_pipeline(scenario, status, count):
    case = load_dataset(ROOT / "data/obrio_task1_70_tickets.json")[0]
    qs = build_questions()
    settings = load_config("settings.yaml")
    settings["failure"].update(backoff_base=0, jitter=0)
    model = {"id": "jev", "slug": "jev-1.13.0", "params": {}, "price_key": "jev-1.13.0"}

    async def run():
        async with httpx.AsyncClient(
            transport=mock_transport({case.ticket: gold_decision(case)}, qs, scenario)
        ) as client:
            provider = JevClient(client, CONFIG, "fixture", qs)
            classifier = JevClassifier(
                [model],
                {"jev": provider},
                settings,
                load_config("prices.yaml"),
                jev_config=CONFIG,
            )
            result = await classifier.classify(case.ticket)
            assert result.status == status and len(result.attempts) == count
            if scenario == "ok":
                assert (
                    result.primary_category == case.expected.primary_category
                    and result.question_count == 24
                    and result.usage.input_tokens == 150
                )

    asyncio.run(run())


def test_low_p0_threshold_and_review():
    qs = build_questions()
    probs = {q.key: 0.0 for q in qs}
    probs["primary_category"] = {
        k: 1.0 if k == "other" else 0.0 for k in TAXONOMY["categories"]
    }
    probs["self_harm"] = 0.35
    response = Response("{}", metadata={"probabilities": probs})
    classifier = JevClassifier(
        [], {}, load_config("settings.yaml"), {}, jev_config=CONFIG
    )
    decision, _ = classifier.parser(response)
    assert "self_harm" in decision.priority_markers and decision.confidence == 0.65
    assert classifier.additional_review(response) == ["uncertain_safety_probability"]
