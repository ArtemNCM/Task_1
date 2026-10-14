"""Check public entrypoints without a network or paid inference."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
import pytest
import classify
from bootstrap import ROOT
from classifier_core.taxonomy import ROOT as CORE_ROOT


def test_resources_are_inside_delivery():
    assert CORE_ROOT == ROOT / "01"
    import classifier_core

    assert Path(classifier_core.__file__).resolve().is_relative_to(ROOT)


@pytest.mark.parametrize(
    "command", [["classify.py", "--text", "Hello"], ["evaluate.py", "--model", "all"]]
)
def test_live_requires_opt_in(command):
    result = subprocess.run(
        [sys.executable, *command], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "--allow-live" in result.stderr


def test_empty_ticket_rejected_before_network():
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(classify.classify_ticket("   "))


def test_missing_key_rejected_before_network(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        asyncio.run(classify.classify_ticket("Cancel my subscription"))


def test_single_ticket_routes_safety_to_review(monkeypatch, tmp_path):
    import json
    from classifier_core.providers.base import Response

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    monkeypatch.setattr(classify, "ROOT", tmp_path)
    (tmp_path / "01").mkdir()

    async def complete(self, **kwargs):
        return Response(
            json.dumps(
                {
                    "primary_category": "other",
                    "secondary_categories": [],
                    "priority_markers": ["self_harm"],
                    "action_markers": [],
                    "routing_hint": None,
                    "language": "en",
                    "confidence": 0.99,
                    "rationale": "Immediate safety concern.",
                }
            )
        )

    monkeypatch.setattr(classify.OpenRouterProvider, "complete", complete)
    result = asyncio.run(classify.classify_ticket("Synthetic safety fixture"))
    assert result.priority == "P0"
    assert result.next_step_final == "escalate_human_urgent"
    assert result.automation_level == "human_required"
    assert (tmp_path / "01/review_queue.jsonl").exists()
