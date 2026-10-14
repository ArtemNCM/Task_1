from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .taxonomy import CATEGORIES, PRIORITY_MARKERS, ACTION_MARKERS, NEXT_STEPS

Category = Literal[*CATEGORIES]
PriorityMarker = Literal[*PRIORITY_MARKERS]
ActionMarker = Literal[*ACTION_MARKERS]
Priority = Literal["P0", "P1", "P2", "P3"]
NextStep = Literal[*NEXT_STEPS]
Status = Literal[
    "ok",
    "recovered",
    "repaired",
    "fallback_model",
    "provider_refusal",
    "unclassified_human",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Decision(StrictModel):
    primary_category: Category
    secondary_categories: list[Category] = Field(default_factory=list, max_length=2)
    priority_markers: list[PriorityMarker] = Field(default_factory=list)
    action_markers: list[ActionMarker] = Field(default_factory=list)
    routing_hint: Literal["marketing"] | None = None
    language: str | None = None
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(
        default="",
        description="Brief debug note: at most 25 whitespace-separated words. Aim for 10–15 words; an empty string is allowed.",
    )

    @field_validator("rationale")
    @classmethod
    def short_rationale(cls, value):
        if len(value.split()) > 25:
            raise ValueError(
                f"rationale must contain at most 25 whitespace-separated words; received {len(value.split())}"
            )
        return value

    @model_validator(mode="after")
    def normalize(self):
        self.secondary_categories = list(
            dict.fromkeys(
                c for c in self.secondary_categories if c != self.primary_category
            )
        )
        self.priority_markers = list(dict.fromkeys(self.priority_markers))
        self.action_markers = list(dict.fromkeys(self.action_markers))
        if self.routing_hint and self.primary_category != "feedback":
            raise ValueError("marketing requires feedback")
        return self


class Usage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class Attempt(StrictModel):
    model_id: str
    latency_ms: float
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float | None = None
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    error_type: str | None = None
    validation_error: str | None = None
    raw_response: str | None = None


class ClassifierResult(Decision):
    priority: Priority
    next_step_raw: NextStep
    next_step_final: NextStep
    automation_level: Literal["human_required", "auto_reply_allowed", "route_only"]
    needs_review: bool
    status: Status
    model_id: str
    prompt_version: str
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    attempts: list[Attempt] = Field(default_factory=list)
    cost_usd: float | None = None
    raw_response: str | None = None
    cache_hit: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    question_count: int | None = None


class Expected(StrictModel):
    primary_category: Category
    secondary_categories: list[Category] = Field(default_factory=list, max_length=2)
    priority_markers: list[PriorityMarker] = Field(default_factory=list)
    action_markers: list[ActionMarker] = Field(default_factory=list)
    priority: Priority
    next_step: NextStep


class Case(StrictModel):
    id: int
    difficulty: Literal["easy", "medium", "hard"]
    language: str
    ticket: str
    expected: Expected
