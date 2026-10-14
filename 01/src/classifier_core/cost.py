from .schema import Usage


def price_usage(usage, price):
    if (
        not price
        or price.get("input_per_mtok") is None
        or price.get("output_per_mtok") is None
        or usage.input_tokens is None
        or usage.output_tokens is None
    ):
        return None, None, None
    cached = usage.cached_input_tokens or 0
    if cached > usage.input_tokens:
        return None, None, None
    if cached and price.get("cached_input_per_mtok") is None:
        return None, None, None
    input_cost = (
        (usage.input_tokens - cached) * price["input_per_mtok"]
        + cached * (price.get("cached_input_per_mtok") or 0)
    ) / 1e6
    # Reasoning tokens are included in completion tokens, not added twice.
    output_cost = usage.output_tokens * price["output_per_mtok"] / 1e6
    return input_cost + output_cost, input_cost, output_cost


def total_cost(attempts):
    return (
        sum(a.cost_usd for a in attempts)
        if attempts and all(a.cost_usd is not None for a in attempts)
        else None
    )


def total_usage(attempts):
    values = {}
    for field in Usage.model_fields:
        tokens = [getattr(a.usage, field) for a in attempts]
        values[field] = (
            sum(tokens) if tokens and all(v is not None for v in tokens) else None
        )
    return Usage(**values)
