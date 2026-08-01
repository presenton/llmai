from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ReasoningEffortValue(str, Enum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"
    DEFAULT = "default"


class ReasoningSummary(str, Enum):
    AUTO = "auto"
    CONCISE = "concise"
    DETAILED = "detailed"


class ReasoningEffort(BaseModel):
    effort: ReasoningEffortValue | None = None
    tokens: int | None = Field(default=None, ge=0)
    summary: ReasoningSummary | None = None
    include_trace: bool | None = None


class ReasoningTraceMode(str, Enum):
    AUTO = "auto"
    SUMMARY = "summary"
    FULL = "full"
    NONE = "none"


class ReasoningHistoryMode(str, Enum):
    PROVIDER_DEFAULT = "provider_default"
    DISABLED = "disabled"
    INTERLEAVED = "interleaved"
    PRESERVED = "preserved"


class ReasoningConfig(BaseModel):
    """Provider-neutral controls for thinking/reasoning models.

    ``enabled=None`` deliberately means "let the provider/model decide".  This
    is different from ``False``, which asks llmai to disable optional thinking.
    """

    enabled: bool | None = None
    effort: ReasoningEffortValue | None = None
    budget_tokens: int | None = Field(default=None, ge=0)
    trace: ReasoningTraceMode = ReasoningTraceMode.AUTO
    history: ReasoningHistoryMode = ReasoningHistoryMode.PROVIDER_DEFAULT

    @model_validator(mode="after")
    def _validate_controls(self) -> "ReasoningConfig":
        if self.enabled is False:
            if self.budget_tokens not in (None, 0):
                raise ValueError(
                    "reasoning budget cannot be set when reasoning is disabled"
                )
            if self.effort not in (
                None,
                ReasoningEffortValue.NONE,
                ReasoningEffortValue.DEFAULT,
            ):
                raise ValueError(
                    "reasoning effort cannot be enabled when reasoning is disabled"
                )
        if self.enabled is True and self.effort == ReasoningEffortValue.NONE:
            raise ValueError("reasoning effort 'none' conflicts with enabled=True")
        return self


ReasoningUsageSource = Literal[
    "provider", "provider_details", "estimated_visible", "unavailable"
]


class ReasoningUsage(BaseModel):
    """Normalized reasoning token accounting.

    Provider-billed reasoning and locally visible trace estimates are kept
    separate because hidden reasoning cannot be reconstructed from text.
    """

    billed_tokens: int | None = None
    visible_tokens: int | None = None
    billed_estimated: bool = False
    visible_estimated: bool = False
    source: ReasoningUsageSource = "unavailable"
