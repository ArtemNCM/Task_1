"""Bounded retry orchestration, independent of transport and domain parsing."""

import asyncio
import random
import time
from dataclasses import dataclass, field
from .providers.base import ProviderError
from .schema import Attempt
from .cost import price_usage


@dataclass
class Outcome:
    value: object = None
    status: str = "unclassified_human"
    attempts: list = field(default_factory=list)
    model_id: str = ""
    latency_ms: float = 0.0
    response: object = None


async def execute(
    models,
    providers,
    request,
    parser,
    settings,
    prices,
    sleep=asyncio.sleep,
    clock=time.monotonic,
    jitter=random.random,
):
    policy = settings["failure"]
    started = clock()
    deadline = started + policy["deadline_seconds"]
    attempts = []
    for model_index, model in enumerate(models):
        retries = 0
        repair = None
        repaired = False
        while clock() < deadline:
            began = clock()
            response = None
            error = None
            retry_after = None
            validation_error = None
            try:
                async with asyncio.timeout(
                    min(policy["timeout_seconds"], max(0.001, deadline - clock()))
                ):
                    response = await providers[model["id"]].complete(
                        **request, model=model, repair=repair
                    )
                if response.finish_reason in (
                    "content_filter",
                    "refusal",
                    "safety",
                    "blocked",
                ):
                    raise ProviderError("refusal", response=response)
                if response.finish_reason in ("max_tokens", "length"):
                    raise ProviderError("truncated", response=response)
                if not response.text.strip():
                    raise ProviderError("empty", response=response)
                value, parse_status = parser(response)
            except (TimeoutError, asyncio.TimeoutError):
                error = "timeout"
            except ProviderError as exc:
                error = exc.kind
                retry_after = exc.retry_after
                response = exc.response or response
                validation_error = exc.validation_error
            except Exception:
                # No exception text: an arbitrary transport exception may contain credentials.
                error = "unexpected_error"
            from .schema import Usage

            usage = response.usage if response else Usage()
            cost, inp, out = price_usage(usage, prices.get(model["price_key"]))
            attempts.append(
                Attempt(
                    model_id=model["id"],
                    latency_ms=(clock() - began) * 1000,
                    usage=usage,
                    cost_usd=cost,
                    input_cost_usd=inp,
                    output_cost_usd=out,
                    error_type=error,
                    validation_error=validation_error,
                    raw_response=response.text if response else None,
                )
            )
            if error is None:
                status = (
                    "fallback_model"
                    if model_index
                    else ("repaired" if repaired else parse_status)
                )
                return Outcome(
                    value,
                    status,
                    attempts,
                    model["id"],
                    (clock() - started) * 1000,
                    response,
                )
            if error == "refusal":
                return Outcome(
                    None,
                    "provider_refusal",
                    attempts,
                    model["id"],
                    (clock() - started) * 1000,
                    response,
                )
            if (
                error
                in (
                    "invalid_json",
                    "schema_violation",
                    "schema_mismatch",
                    "empty",
                    "truncated",
                )
                and not repaired
            ):
                repaired = True
                repair = {
                    "invalid_output": response.text if response else "",
                    "validation_error": validation_error or error,
                }
                continue
            if (
                error in ("timeout", "rate_limit", "server_error", "connection")
                and retries < policy["max_retries"]
            ):
                delay = max(
                    policy["backoff_base"] * policy["backoff_multiplier"] ** retries
                    + policy["jitter"] * jitter(),
                    retry_after or 0,
                )
                retries += 1
                if clock() + delay >= deadline:
                    break
                await sleep(delay)
                continue
            break
    if clock() >= deadline and not attempts:
        attempts.append(
            Attempt(model_id=models[0]["id"], latency_ms=0.0, error_type="deadline")
        )
    return Outcome(
        None,
        "unclassified_human",
        attempts,
        attempts[-1].model_id if attempts else models[0]["id"],
        (clock() - started) * 1000,
    )
