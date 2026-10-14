import hashlib
from .base import Response, ProviderError
from ..schema import Decision, Usage


class MockProvider:
    """Oracle explicitly reads gold labels: infrastructure check, never model evidence."""

    def __init__(self, oracle=None, scenario="ok", noisy=False):
        self.oracle = oracle or {}
        self.scenario = scenario
        self.noisy = noisy
        self.calls = {}

    async def complete(self, *, ticket, model, system="", user="", repair=None):
        key = (ticket, model["id"])
        count = self.calls.get(key, 0)
        self.calls[key] = count + 1
        scenario = self.scenario
        if scenario in ("timeout", "always_fail"):
            raise ProviderError("timeout" if scenario == "timeout" else "connection")
        if scenario == "rate_limit_then_ok" and count == 0:
            raise ProviderError("rate_limit", retry_after=2.0)
        if scenario == "server_error_then_ok" and count == 0:
            raise ProviderError("server_error")
        usage = Usage(
            input_tokens=100,
            output_tokens=40,
            cached_input_tokens=0,
            reasoning_tokens=0,
        )
        if scenario == "refusal":
            return Response("", usage, "content_filter")
        if count == 0:
            if scenario == "invalid_json":
                return Response("{broken", usage)
            if scenario == "schema_violation":
                return Response('{"primary_category":"fake"}', usage)
            if scenario == "truncated":
                return Response('{"primary_category":', usage, "max_tokens")
            if scenario == "empty":
                return Response("", usage)
        decision = self.oracle.get(
            ticket,
            Decision(
                primary_category="other",
                confidence=0.5,
                action_markers=["insufficient_info"],
            ),
        )
        decision = decision.model_copy(deep=True)
        if (
            self.noisy
            and int(hashlib.sha256(ticket.encode("utf-8")).hexdigest(), 16) % 4 == 0
        ):
            decision.primary_category = (
                "other" if decision.primary_category != "other" else "general_question"
            )
            decision.priority_markers = []
            decision.action_markers = []
            decision.routing_hint = None
            decision.confidence = 0.85
        return Response(decision.model_dump_json(), usage, "stop")
