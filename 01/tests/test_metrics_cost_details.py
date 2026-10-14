import asyncio
import pytest
from classifier_core.schema import ClassifierResult, Usage, Attempt
from classifier_core.eval.dataset import load_dataset, gold_decision
from classifier_core.eval.metrics import metrics
from classifier_core.taxonomy import ROOT
from classifier_core.rules import derive
from classifier_core.cost import price_usage, total_usage
from classifier_core.providers.base import ProviderError
from classifier_core.providers.openrouter import OpenRouterProvider
from classifier_core.taxonomy import load_config
import httpx


def test_hand_computed_marker_precision_recall():
    base = load_dataset(ROOT / "data/obrio_task1_70_tickets.json")[0]
    cases = []
    results = []
    for index, (gold, pred) in enumerate([(True, True), (True, False), (False, True)]):
        case = base.model_copy(deep=True)
        case.id = index
        case.expected.priority_markers = ["money_lost"] if gold else []
        case.expected.priority = "P1" if gold else "P3"
        d = gold_decision(case)
        d.priority_markers = ["money_lost"] if pred else []
        r = ClassifierResult(
            **d.model_dump(),
            **derive(d),
            status="ok",
            model_id="toy",
            prompt_version="v1",
        )
        cases.append(case)
        results.append(r)
    m = metrics(cases, results)
    marker = m["priority_markers"]["per_marker"]["money_lost"]
    assert marker == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert m["under_triage_rate"] == pytest.approx(1 / 3) and m[
        "over_triage_rate"
    ] == pytest.approx(1 / 3)


def test_unknown_usage_never_estimated():
    assert (
        total_usage(
            [
                Attempt(model_id="x", latency_ms=0.0, usage=Usage(input_tokens=20)),
                Attempt(model_id="x", latency_ms=0.0),
            ]
        ).input_tokens
        is None
    )
    assert price_usage(
        Usage(input_tokens=10, output_tokens=5, cached_input_tokens=2),
        {"input_per_mtok": 1, "output_per_mtok": 2},
    ) == (None, None, None)


def test_malformed_envelope_retains_reported_usage():
    secret = "private-fixture"

    async def run():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [],
                    "usage": {"prompt_tokens": 33, "completion_tokens": 4},
                    "echo": secret,
                },
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ProviderError) as exc:
                await OpenRouterProvider(
                    client, "https://fixture.invalid", secret
                ).complete(
                    ticket="x",
                    model=load_config("models.yaml")["models"]["haiku-5-5"],
                    system="s",
                    user="u",
                )
            assert exc.value.response.usage.input_tokens == 33
            assert secret not in exc.value.response.text

    asyncio.run(run())
