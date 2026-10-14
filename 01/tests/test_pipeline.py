import asyncio
import json
import pytest
from classifier_core.classifier import LLMClassifier, parse_response
from classifier_core.providers.mock import MockProvider
from classifier_core.providers.base import Response
from classifier_core.taxonomy import ROOT, load_config
from classifier_core.eval.dataset import load_dataset, gold_decision
from classifier_core.eval.metrics import metrics
from classifier_core.cache import Cache, cache_key
from classifier_core.cost import price_usage, total_cost
from classifier_core.schema import Usage, Attempt


class Clock:
    def __init__(self):
        self.now = 0.0
        self.delays = []

    def __call__(self):
        return self.now

    async def sleep(self, delay):
        self.now += delay
        self.delays.append(delay)


def make(provider, **kwargs):
    model = load_config("models.yaml")["models"]["mock-oracle"]
    return LLMClassifier(
        [model], {"mock-oracle": provider}, load_config("settings.yaml"), {}, **kwargs
    )


@pytest.mark.parametrize(
    "scenario,status,count",
    [
        ("ok", "ok", 1),
        ("invalid_json", "repaired", 2),
        ("schema_violation", "repaired", 2),
        ("truncated", "repaired", 2),
        ("empty", "repaired", 2),
        ("timeout", "unclassified_human", 4),
        ("rate_limit_then_ok", "ok", 2),
        ("server_error_then_ok", "ok", 2),
        ("always_fail", "unclassified_human", 4),
        ("refusal", "provider_refusal", 1),
    ],
)
def test_faults(scenario, status, count):
    clock = Clock()
    classifier = make(
        MockProvider(scenario=scenario),
        sleep=clock.sleep,
        clock=clock,
        jitter=lambda: 0,
    )
    result = asyncio.run(classifier.classify("Help with a purchase"))
    assert result.status == status and len(result.attempts) == count
    if scenario == "rate_limit_then_ok":
        assert clock.delays == [2.0]
    if scenario == "refusal":
        assert (
            result.next_step_final == "escalate_human_urgent"
            and result.automation_level == "human_required"
        )
    if status == "unclassified_human":
        assert result.priority == "P2" and result.next_step_final == "human_review"


def test_oracle_and_noisy():
    cases = load_dataset(ROOT / "data/obrio_task1_70_tickets.json")
    # Full-set oracle is a pipeline unit test, not a model holdout evaluation.
    oracle = {c.ticket: gold_decision(c) for c in cases}

    async def run(noisy):
        classifier = make(MockProvider(oracle, noisy=noisy))
        return await asyncio.gather(*(classifier.classify(c.ticket) for c in cases))

    assert metrics(cases, asyncio.run(run(False)))["case_pass"] == 1
    assert 0 < metrics(cases, asyncio.run(run(True)))["case_pass"] < 1


def test_safety_fallback():
    clock = Clock()
    c = make(MockProvider(scenario="always_fail"), sleep=clock.sleep, clock=clock)
    assert asyncio.run(c.classify("I will kill myself")).priority == "P0"


def test_cache_and_keys(tmp_path):
    cache = Cache(tmp_path / "cache.sqlite")
    c = make(MockProvider(), cache=cache)
    first = asyncio.run(c.classify("ticket"))
    second = asyncio.run(c.classify("ticket"))
    assert not first.cache_hit and second.cache_hit and second.cost_usd == 0
    base = ["ticket", "model", {}, "hash", "2.0"]
    key = cache_key(*base)
    for index in (1, 3, 4):
        changed = base.copy()
        changed[index] = "different"
        assert cache_key(*changed) != key
    first.status = "unclassified_human"
    cache.put("failure", first)
    assert cache.get("failure") is None
    cache.clear()
    assert cache.get(key) is None


def test_cost():
    u = Usage(
        input_tokens=1000,
        output_tokens=100,
        cached_input_tokens=200,
        reasoning_tokens=50,
    )
    price = {"input_per_mtok": 1, "output_per_mtok": 2, "cached_input_per_mtok": 0.5}
    assert price_usage(u, price)[0] == pytest.approx(0.0011)
    assert price_usage(u, None) == (None, None, None)
    assert total_cost(
        [
            Attempt(model_id="m", latency_ms=1.0, cost_usd=0.1),
            Attempt(model_id="m", latency_ms=1.0, cost_usd=0.2),
        ]
    ) == pytest.approx(0.3)
    assert total_cost([Attempt(model_id="m", latency_ms=1.0)]) is None


def test_recovered():
    text = MockProvider()
    response = asyncio.run(text.complete(ticket="x", model={"id": "mock"}))
    assert (
        parse_response(Response("```json\n" + response.text + "\n```"))[1]
        == "recovered"
    )
