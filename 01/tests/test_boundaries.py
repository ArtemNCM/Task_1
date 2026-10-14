import asyncio
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime
import json
import httpx
import pytest
from classifier_core.classifier import LLMClassifier, parse_response
from classifier_core.providers.base import ProviderError, Response
from classifier_core.providers.mock import MockProvider
from classifier_core.providers.openrouter import OpenRouterProvider, retry_after
from classifier_core.schema import Decision
from classifier_core.taxonomy import ROOT, load_config
from classifier_core.eval.dataset import load_dataset, gold_decision
from classifier_core.eval.runner import arguments, select_cases
from classifier_core.eval.split import get_split, digest
from classifier_core.eval.metrics import stability


def setup(provider, settings=None, **kwargs):
    model = load_config("models.yaml")["models"]["mock-oracle"]
    return LLMClassifier(
        [model],
        {model["id"]: provider},
        settings or load_config("settings.yaml"),
        {},
        **kwargs,
    )


def test_fallback_records_all_attempts():
    config = load_config("models.yaml")
    a = config["models"]["mock-oracle"]
    b = config["models"]["mock-noisy"]
    settings = load_config("settings.yaml")
    settings["failure"].update(backoff_base=0, jitter=0)
    classifier = LLMClassifier(
        [a, b],
        {a["id"]: MockProvider(scenario="always_fail"), b["id"]: MockProvider()},
        settings,
        {},
    )
    r = asyncio.run(classifier.classify("Some ticket"))
    assert r.status == "fallback_model" and len(r.attempts) == 5 and r.needs_review
    assert [attempt.model_id for attempt in r.attempts] == [a["id"]] * 4 + [b["id"]]


def test_ticket_deadline_real_timeout():
    class Slow:
        async def complete(self, **kwargs):
            await asyncio.sleep(0.1)

    settings = load_config("settings.yaml")
    settings["failure"].update(
        timeout_seconds=0.01, deadline_seconds=0.025, backoff_base=0, jitter=0
    )
    r = asyncio.run(setup(Slow(), settings).classify("I need help"))
    assert (
        r.status == "unclassified_human"
        and 1 <= len(r.attempts) <= 4
        and r.latency_ms < 500
    )


def test_refusal_configurable():
    settings = load_config("settings.yaml")
    settings["failure"]["refusal_policy"] = "human_review"
    r = asyncio.run(
        setup(MockProvider(scenario="refusal"), settings).classify("ordinary ticket")
    )
    assert r.status == "provider_refusal" and r.next_step_final == "human_review"


def test_always_on_safety_only_raises():
    d = Decision(
        primary_category="billing", priority_markers=["money_lost"], confidence=0.99
    )
    settings = load_config("settings.yaml")
    settings["safety_net_always_on"] = True
    r = asyncio.run(
        setup(MockProvider({"I will kill myself": d}), settings).classify(
            "I will kill myself"
        )
    )
    assert r.priority == "P0" and "money_lost" in r.priority_markers and r.needs_review


def test_config_and_hash_guard(tmp_path):
    original = ROOT / "data/obrio_task1_70_tickets.json"
    cases = load_dataset(original)
    split = get_split(ROOT / "data/split.json", cases)
    assert (
        digest(ROOT / "data/split.json")
        == "bda15bcf19374e2762a621c181e40e30522f79b9cfeb028168b5cbc0271b597d"
    )
    assert (
        digest(original)
        == "4cd2e629ab1998a3a5c24ac98ead06f1386b39c9179c58331db0c672264474f2"
    )
    held = next(c for c in cases if c.id in split["holdout"])
    row = held.model_dump()
    row["id"] = 999
    custom = tmp_path / "copied.json"
    custom.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="No cases"):
        select_cases(arguments(["--dataset", str(custom), "--split", "dev"]))


def test_missing_fields_are_schema_failure():
    with pytest.raises(ProviderError) as exc:
        parse_response(Response('{"primary_category":"billing","confidence":0.8}'))
    assert exc.value.kind == "schema_violation"


def test_stability_with_flip():
    r = asyncio.run(setup(MockProvider()).classify("one"))
    changed = r.model_copy(deep=True)
    changed.primary_category = "billing"
    assert stability([{1: r}, {1: changed}])["flip_rate"] == 1


@pytest.mark.parametrize(
    "code,error", [(429, "rate_limit"), (503, "server_error"), (401, "http_error")]
)
def test_http_failure_types_do_not_leak_keys(code, error):
    secret = "fake-secret-for-test"
    model = load_config("models.yaml")["models"]["haiku-5-5"]

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    code, text=secret, headers={"Retry-After": "3"}
                )
            )
        ) as client:
            with pytest.raises(ProviderError) as exc:
                await OpenRouterProvider(
                    client, "https://fixture.invalid", secret
                ).complete(ticket="x", model=model, system="s", user="u")
            assert str(exc.value) == error and secret not in str(exc.value)
            if code == 429:
                assert exc.value.retry_after == 3

    asyncio.run(run())


def test_retry_after_date():
    date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
    assert 28 < retry_after(date) <= 30
    assert retry_after("bad") is None
