"""Pure business rules. Taxonomy is the only source of marker tiers/routes."""

from .taxonomy import TAXONOMY


def priority(markers):
    tiers = [
        TAXONOMY["markers"][m]["tier"]
        for m in set(markers)
        if TAXONOMY["markers"][m]["tier"]
    ]
    if "P0" in tiers:
        return "P0"
    if "P1" in tiers or tiers.count("P2") >= 2:
        return "P1"
    return "P2" if tiers else "P3"


def next_step(decision, level, needs_review=False):
    cat, pm, am = (
        decision.primary_category,
        decision.priority_markers,
        decision.action_markers,
    )
    if level == "P0":
        return "escalate_human_urgent"
    if needs_review:
        return "human_review"
    # A legal threat must reach Legal even if the customer omitted case details.
    if cat == "legal_privacy" or "legal_threat" in pm:
        return "route_legal"
    if "insufficient_info" in am:
        return "request_more_info"
    if cat == "billing" and "refund_requested" in am:
        return "route_finance"
    if cat == "billing" and "cancellation_requested" in am and not pm:
        return "auto_reply_cancellation"
    if cat == "general_question" and level == "P3":
        return "auto_reply_kb"
    if cat == "feedback" and decision.routing_hint == "marketing":
        return "route_product_marketing"
    return TAXONOMY["routes"][cat]


def derive(decision, needs_review=False, priority_floor=None, refusal_urgent=False):
    level = priority(decision.priority_markers)
    if priority_floor:
        level = min(level, priority_floor)
    raw = next_step(decision, level)
    final = (
        "escalate_human_urgent"
        if refusal_urgent
        else next_step(decision, level, needs_review)
    )
    human = (
        level == "P0"
        or needs_review
        or refusal_urgent
        or bool(set(decision.priority_markers) & set(TAXONOMY["human_markers"]))
    )
    restricted = (
        "refund_requested" in decision.action_markers
        or "privacy_request" in decision.priority_markers
    )
    auto = (
        "human_required"
        if human
        else (
            "auto_reply_allowed"
            if not restricted and final in ("auto_reply_kb", "auto_reply_cancellation")
            else "route_only"
        )
    )
    return dict(
        priority=level,
        next_step_raw=raw,
        next_step_final=final,
        automation_level=auto,
        needs_review=needs_review,
    )
