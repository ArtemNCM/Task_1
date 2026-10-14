from classifier_core.classifier import LLMClassifier
from classifier_core.schema import Decision
from classifier_core.taxonomy import TAXONOMY


class JevClassifier(LLMClassifier):
    def __init__(self, *args, jev_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.jev_config = jev_config
        self.questions_version = jev_config["questions_version"]

    def parser(self, response):
        probabilities = response.metadata["probabilities"]
        categories = probabilities["primary_category"]
        primary = max(categories, key=categories.get)
        pm = []
        am = []
        for name, info in TAXONOMY["markers"].items():
            threshold = (
                self.jev_config["p0_threshold"]
                if info["tier"] == "P0"
                else self.jev_config["threshold"]
            )
            if probabilities[name] >= threshold:
                (pm if info["tier"] else am).append(name)
        secondary = [
            cat
            for cat in TAXONOMY["precedence"]
            if cat != primary
            and probabilities["secondary_" + cat]
            >= self.jev_config["secondary_threshold"]
        ][:2]
        safety = [
            probabilities[k]
            for k, v in TAXONOMY["markers"].items()
            if v["tier"] in ("P0", "P1")
        ]
        # Explicit definition of unspecified uncertainty: max min(p, 1-p).
        uncertainty = max(min(p, 1 - p) for p in safety)
        decision = Decision(
            primary_category=primary,
            secondary_categories=secondary,
            priority_markers=pm,
            action_markers=am,
            routing_hint=(
                "marketing"
                if primary == "feedback"
                and probabilities["marketing"] >= self.jev_config["threshold"]
                else None
            ),
            language=None,
            confidence=min(max(categories.values()), 1 - uncertainty),
        )
        return decision, "ok"

    def additional_review(self, response):
        if not response or "probabilities" not in response.metadata:
            return []
        lo, hi = self.settings["review"]["uncertain_band"]
        probs = response.metadata["probabilities"]
        return (
            ["uncertain_safety_probability"]
            if any(
                lo <= probs[k] <= hi
                for k, v in TAXONOMY["markers"].items()
                if v["tier"] in ("P0", "P1")
            )
            else []
        )
