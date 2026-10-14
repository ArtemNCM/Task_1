"""Only module aware of the TypeSafe System One HTTP wire format.

Sources: https://docs.typesafe.ai/api and OpenRouter's TypeSafe SDK guide.
"""

import json
import math
import httpx
from classifier_core.providers.base import Response, ProviderError
from classifier_core.providers.openrouter import retry_after
from classifier_core.schema import Usage


def probability(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Invalid probability")
    return float(value)


class JevClient:
    def __init__(self, client, config, key, questions):
        self.client, self.config, self._key, self.questions = (
            client,
            config,
            key,
            questions,
        )

    def request_body(self, ticket, model):
        questions = {}
        for q in self.questions:
            entry = {"type": q.kind, "instructions": q.proposition}
            if q.options:
                entry["criteria"] = q.options
            elif q.positive:
                entry["criteria"] = {"true": q.positive, "false": q.negative}
            questions[q.key] = entry
        return {"model": model["slug"], "state": ticket, "questions": questions}

    async def complete(self, *, ticket, model, system="", user="", repair=None):
        if self.config["auth_scheme"] != "bearer":
            raise ProviderError("unsupported_auth")
        try:
            response = await self.client.post(
                self.config["base_url"].rstrip("/") + self.config["endpoint_path"],
                headers={"Authorization": "Bearer " + self._key},
                json=self.request_body(ticket, model),
            )
        except httpx.TimeoutException:
            raise ProviderError("timeout") from None
        except httpx.RequestError:
            raise ProviderError("connection") from None
        if response.status_code == 429:
            raise ProviderError(
                "rate_limit", retry_after(response.headers.get("Retry-After"))
            )
        if response.status_code >= 500:
            raise ProviderError("server_error")
        if response.status_code >= 400:
            raise ProviderError("http_error")
        raw = (
            response.text.replace(self._key, "[REDACTED]")
            if self._key
            else response.text
        )
        result = Response(raw)
        try:
            data = response.json()
            usage = data.get("usage") or {}
            result.usage = Usage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            )
            answers = data["answers"]
            normalized = {}
            for q in self.questions:
                answer = answers[q.key]
                if answer["type"] != q.kind:
                    raise ValueError("Wrong answer type")
                if q.kind == "choice":
                    probs = {
                        k: probability(v) for k, v in answer["probabilities"].items()
                    }
                    if set(probs) != set(q.options) or not math.isclose(
                        sum(probs.values()), 1.0, abs_tol=0.01
                    ):
                        raise ValueError("Invalid distribution")
                    if answer["choice"] not in probs or probs[answer["choice"]] != max(
                        probs.values()
                    ):
                        raise ValueError("Invalid choice")
                    probability(answer["confidence"])
                    normalized[q.key] = probs
                else:
                    normalized[q.key] = probability(answer["noul"])
            result.metadata = {
                "probabilities": normalized,
                "question_count": len(self.questions),
                "resolved_model": data["model"],
                "usage_scope": "request",
            }
            return result
        except (KeyError, TypeError, ValueError):
            raise ProviderError("schema_mismatch", response=result) from None


def mock_transport(oracle, questions, scenario="ok"):
    """Local HTTP fixture: synthetic probabilities/usage; never model evidence."""
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if scenario == "server_error_then_ok" and count == 1:
            return httpx.Response(503)
        if scenario == "always_fail":
            raise httpx.ConnectError("fixture", request=request)
        if scenario == "schema_mismatch":
            return httpx.Response(
                200,
                json={
                    "answers": {},
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                },
            )
        body = json.loads(request.content)
        decision = oracle[body["state"]]
        answers = {}
        for q in questions:
            if q.kind == "choice":
                answers[q.key] = {
                    "type": "choice",
                    "choice": decision.primary_category,
                    "probabilities": {
                        k: 0.94 if k == decision.primary_category else 0.01
                        for k in q.options
                    },
                    "confidence": 0.9,
                }
            else:
                yes = (
                    q.key in decision.priority_markers + decision.action_markers
                    or q.key == "marketing"
                    and decision.routing_hint == "marketing"
                    or q.key.startswith("secondary_")
                    and q.key[10:] in decision.secondary_categories
                )
                answers[q.key] = {"type": "noul", "noul": 0.99 if yes else 0.01}
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 150, "output_tokens": 0},
            },
        )

    return httpx.MockTransport(handler)
