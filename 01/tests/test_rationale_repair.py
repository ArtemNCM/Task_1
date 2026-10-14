import asyncio
import json
import httpx
import pytest
from classifier_core.classifier import LLMClassifier, parse_response
from classifier_core.providers.base import Response, ProviderError
from classifier_core.providers.openrouter import OpenRouterProvider
from classifier_core.schema import Decision
from classifier_core.taxonomy import load_config
from classifier_core.prompt_builder import (
    load_prompt,
    generate_prompts,
    RATIONALE_INSTRUCTION,
)


def test_word_limit_boundary_and_safe_feedback():
    data = Decision(primary_category="billing", confidence=0.9).model_dump()
    data["rationale"] = " ".join(["private-value"] * 25)
    assert parse_response(Response(json.dumps(data)))[1] == "ok"
    data["rationale"] += " private-value"
    with pytest.raises(ProviderError) as error:
        parse_response(Response(json.dumps(data)))
    feedback = error.value.validation_error
    assert "received 26" in feedback and "maximum is 25" in feedback
    assert "private-value" not in feedback


def test_specific_feedback_reaches_http_repair():
    data = Decision(primary_category="billing", confidence=0.9).model_dump()
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        response = data | {
            "rationale": (
                "word " * 26
                if len(calls) == 1
                else "Billing question about a subscription."
            )
        }
        if len(calls) == 2:
            assert "received 26" in body["messages"][-1]["content"]
            assert "maximum is 25" in body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps(response)},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run():
        model = load_config("models.yaml")["models"]["haiku-5-5"] | {
            "slug": "fixture/model"
        }
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenRouterProvider(
                client, "https://fixture.invalid", "fixture-secret"
            )
            classifier = LLMClassifier(
                [model], {model["id"]: provider}, load_config("settings.yaml"), {}
            )
            result = await classifier.classify("A subscription question")
            assert result.status == "repaired" and len(result.attempts) == 2
            assert "maximum is 25" in result.attempts[0].validation_error
            assert result.primary_category == "billing"

    asyncio.run(run())


def test_saved_and_generated_prompts_explain_limit():
    generated = generate_prompts()
    for version in ("v2", "v3"):
        assert RATIONALE_INSTRUCTION in load_prompt(version)[0]
        assert RATIONALE_INSTRUCTION in generated[version]


def test_versioned_prompt_paths_cannot_escape_directory():
    for version in ("../v2", "v2/other", "v0", "v2.md"):
        with pytest.raises(ValueError):
            load_prompt(version)
