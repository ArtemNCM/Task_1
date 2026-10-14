from dataclasses import dataclass, field
from pathlib import Path
import yaml
from classifier_core.taxonomy import TAXONOMY


@dataclass
class Question:
    key: str
    kind: str
    proposition: str
    options: dict = field(default_factory=dict)
    positive: str | None = None
    negative: str | None = None


def build_questions(version="q1"):
    if version not in ("q1", "q2", "q3", "q4"):
        raise ValueError("Unknown Jev question version")
    categories = TAXONOMY["categories"]
    questions = [
        Question(
            "primary_category",
            "choice",
            "Select the primary topical category. " + TAXONOMY["boundary_rules"],
            {k: v["definition"] for k, v in categories.items()},
        )
    ]
    for name, info in TAXONOMY["markers"].items():
        questions.append(
            Question(
                name,
                "noul",
                "Does this ticket satisfy this single marker definition? "
                + info["definition"],
                positive=info["positive"],
                negative=info["negative"],
            )
        )
    for name, info in categories.items():
        questions.append(
            Question(
                "secondary_" + name,
                "noul",
                "The ticket contains this topic: " + info["definition"],
            )
        )
    questions.append(
        Question(
            "marketing",
            "noul",
            "The ticket concerns advertisements or promotional mailings.",
        )
    )
    if version in ("q2", "q3", "q4"):
        overrides = yaml.safe_load(
            (
                Path(__file__).resolve().parents[2] / "config/questions_q2.yaml"
            ).read_text(encoding="utf-8")
        )
        if version in ("q3", "q4"):
            overrides.update(
                yaml.safe_load(
                    (
                        Path(__file__).resolve().parents[2] / "config/questions_q3.yaml"
                    ).read_text(encoding="utf-8")
                )
            )
        if version == "q4":
            overrides[
                "insufficient_info"
            ] += " Positive feedback, praise, thanks, and product suggestions are complete feedback even without a problem, question, or requested action. Do not request more information merely because no support problem needs solving."
        known = {q.key for q in questions}
        if set(overrides) - known:
            raise ValueError("Unknown question override")
        for question in questions:
            if question.key in overrides:
                question.proposition = overrides[question.key]
    return questions
