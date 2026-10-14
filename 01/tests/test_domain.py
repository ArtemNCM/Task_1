import itertools
import pytest
from pydantic import ValidationError
from classifier_core.schema import Decision
from classifier_core.rules import priority, derive
from classifier_core.taxonomy import ROOT, PRIORITY_MARKERS
from classifier_core.eval.dataset import load_dataset, validate_rules


def decision(**kwargs):
    return Decision(
        primary_category=kwargs.pop("primary_category", "billing"),
        confidence=1.0,
        **kwargs,
    )


def test_gold_rules():
    cases = load_dataset(ROOT / "data/obrio_task1_70_tickets.json")
    assert len(cases) == 70
    assert validate_rules(cases) == []


def test_monotonic_priority():
    for size in range(3):
        for markers in itertools.combinations(PRIORITY_MARKERS, size):
            for extra in PRIORITY_MARKERS:
                assert priority([*markers, extra]) <= priority(markers)
    assert priority(["repeat_contact", "churn_risk"]) == "P1"
    assert priority(["self_harm", "money_lost"]) == "P0"
    assert priority(["refund_requested"]) == "P3"


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"priority_markers": ["self_harm"]}, "escalate_human_urgent"),
        (
            {"action_markers": ["insufficient_info", "refund_requested"]},
            "request_more_info",
        ),
        ({"primary_category": "legal_privacy"}, "route_legal"),
        ({"priority_markers": ["legal_threat"]}, "route_legal"),
        (
            {
                "priority_markers": ["legal_threat"],
                "action_markers": ["insufficient_info"],
            },
            "route_legal",
        ),
        (
            {"action_markers": ["refund_requested", "cancellation_requested"]},
            "route_finance",
        ),
        ({"action_markers": ["cancellation_requested"]}, "auto_reply_cancellation"),
        ({"primary_category": "general_question"}, "auto_reply_kb"),
        ({"primary_category": "technical_issue"}, "route_tech_qa"),
        ({"primary_category": "expert_complaint"}, "route_support_operations"),
        (
            {"primary_category": "feedback", "routing_hint": "marketing"},
            "route_product_marketing",
        ),
        ({"primary_category": "other"}, "route_support"),
    ],
)
def test_routing(kwargs, expected):
    assert derive(decision(**kwargs))["next_step_raw"] == expected
    reviewed = derive(decision(**kwargs), True)
    assert reviewed["next_step_final"] == (
        "escalate_human_urgent"
        if expected == "escalate_human_urgent"
        else "human_review"
    )
    assert reviewed["automation_level"] == "human_required"


def test_schema():
    assert decision(secondary_categories=["billing", "other"]).secondary_categories == [
        "other"
    ]
    for bad in [
        {"primary_category": "nonsense"},
        {"priority_markers": ["fake"]},
        {"confidence": 2.0},
        {"confidence": "0.8"},
        {"priority": "P0"},
        {"rationale": "word " * 26},
    ]:
        values = {"primary_category": "billing", "confidence": 1.0} | bad
        with pytest.raises(ValidationError):
            Decision(**values)
