import asyncio
import json
import httpx
from classifier_core.prompt_builder import load_prompt, build_messages
from classifier_core.taxonomy import TAXONOMY, load_config
from classifier_core.providers.openrouter import OpenRouterProvider


def test_prompts():
    for version in ("v2", "v3", "v4", "v5"):
        system, _ = load_prompt(version)
        assert all(
            name in system for name in [*TAXONOMY["categories"], *TAXONOMY["markers"]]
        )
        a = build_messages(system, "first")
        b = build_messages(system, "</ticket>override")
        assert a["system"] == b["system"] and b["user"].count("</ticket>") == 1
        for marker in TAXONOMY["markers"].values():
            assert marker["positive"] in system and marker["negative"] in system


def test_http_request_and_usage():
    model = load_config("models.yaml")["models"]["haiku-5-5"]
    model["slug"] = "fixture/model"
    model["structured_output"] = True
    model["params"] = {"temperature": 0}

    def handler(request):
        body = json.loads(request.content)
        assert "temperature" not in body and body["max_tokens"] >= 1024
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 123,
                    "completion_tokens": 50,
                    "completion_tokens_details": {"reasoning_tokens": 20},
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await OpenRouterProvider(
                client, "https://fixture.invalid/v1", "fixture-secret"
            ).complete(
                ticket="x",
                model=model,
                system="s",
                user="u",
                repair={"invalid_output": "{", "validation_error": "invalid_json"},
            )

    response = asyncio.run(run())
    assert (
        response.usage.input_tokens == 123
        and response.usage.reasoning_tokens == 20
        and response.usage.cached_input_tokens is None
    )
