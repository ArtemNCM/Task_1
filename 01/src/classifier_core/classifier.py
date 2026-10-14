import json
from typing import Protocol
from pydantic import ValidationError
from .schema import Decision, ClassifierResult
from .providers.base import ProviderError
from .resilience import execute
from .rules import derive
from .safety_net import detect
from .cost import total_cost, total_usage
from .cache import cache_key
from .taxonomy import TAXONOMY
from .prompt_builder import build_messages


class Classifier(Protocol):
    async def classify(self, ticket: str) -> ClassifierResult: ...


def parse_response(response):
    text = response.text.strip()
    status = "ok"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            start = text.index("{")
            data, _ = json.JSONDecoder().raw_decode(text[start:])
            status = "recovered"
        except (ValueError, json.JSONDecodeError):
            raise ProviderError("invalid_json", response=response) from None
    try:
        if not isinstance(data, dict) or set(data) != set(Decision.model_fields):
            raise ProviderError("schema_violation", response=response)
        return Decision.model_validate(data), status
    except ValidationError as exc:
        # Build feedback from known schema fields, never echo invalid values,
        # arbitrary exception text, or user-controlled extra field names.
        details = []
        for issue in exc.errors(
            include_input=False, include_context=False, include_url=False
        ):
            field = issue["loc"][0] if issue["loc"] else None
            if (
                field == "rationale"
                and isinstance(data.get("rationale"), str)
                and len(data["rationale"].split()) > 25
            ):
                count = len(data["rationale"].split())
                details.append(
                    f"rationale: received {count} whitespace-separated words; maximum is 25. Shorten rationale to 25 words or fewer (aim for 10–15), or use an empty string. Do not change classification fields merely to shorten the note."
                )
            elif field in Decision.model_fields:
                details.append(
                    f"{field}: invalid value; follow the supplied JSON schema."
                )
            else:
                details.append("Output violates the supplied JSON schema.")
        raise ProviderError(
            "schema_violation", response=response, validation_error=" ".join(details)
        ) from None


class LLMClassifier:
    def __init__(
        self,
        models,
        providers,
        settings,
        prices,
        prompt_version="v2",
        system="",
        prompt_hash="",
        cache=None,
        **execution_options,
    ):
        self.models, self.providers, self.settings, self.prices = (
            models,
            providers,
            settings,
            prices,
        )
        self.prompt_version, self.system, self.prompt_hash, self.cache = (
            prompt_version,
            system,
            prompt_hash,
            cache,
        )
        self.execution_options = execution_options

    def messages(self, ticket):
        return {"ticket": ticket, **build_messages(self.system, ticket)}

    def parser(self, response):
        return parse_response(response)

    def additional_review(self, response):
        return []

    async def classify(self, ticket):
        key = cache_key(
            ticket,
            self.models[0]["id"],
            {"models": self.models, "settings": self.settings},
            self.prompt_hash,
            TAXONOMY["version"],
        )
        if self.cache:
            cached = self.cache.get(key)
            if cached:
                return cached
        outcome = await execute(
            self.models,
            self.providers,
            self.messages(ticket),
            self.parser,
            self.settings,
            self.prices,
            **self.execution_options,
        )
        decision = outcome.value
        fallback = decision is None
        if fallback:
            decision = Decision(
                primary_category="other",
                priority_markers=detect(ticket),
                confidence=0.0,
                rationale="No validated classification; human review required.",
            )
        reasons = []
        if outcome.status != "ok":
            reasons.append(outcome.status)
        if decision.confidence < self.settings["review"]["min_confidence"]:
            reasons.append("low_confidence")
        reasons.extend(self.additional_review(outcome.response))
        if self.settings.get("safety_net_always_on"):
            hits = detect(ticket)
            if hits:
                decision.priority_markers = list(
                    dict.fromkeys(decision.priority_markers + hits)
                )
                reasons.append("keyword_safety_net")
        urgent = (
            outcome.status == "provider_refusal"
            and self.settings["failure"]["refusal_policy"] == "escalate_urgent"
        )
        derived = derive(
            decision,
            bool(reasons),
            self.settings["fallback"]["default_priority"] if fallback else None,
            urgent,
        )
        result = ClassifierResult(
            **decision.model_dump(),
            **derived,
            status=outcome.status,
            model_id=outcome.model_id,
            prompt_version=self.prompt_version,
            usage=total_usage(outcome.attempts),
            latency_ms=outcome.latency_ms,
            attempts=outcome.attempts,
            cost_usd=total_cost(outcome.attempts),
            raw_response=outcome.response.text if outcome.response else None,
            review_reasons=reasons,
            question_count=(
                outcome.response.metadata.get("question_count")
                if outcome.response
                else None
            ),
        )
        if self.cache:
            self.cache.put(key, result)
        return result
