from enum import Enum

from pydantic import BaseModel, Field


class ReasoningEffortValue(str, Enum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class ReasoningSummary(str, Enum):
    AUTO = "auto"
    CONCISE = "concise"
    DETAILED = "detailed"


class ReasoningEffort(BaseModel):
    effort: ReasoningEffortValue | None = None
    tokens: int | None = Field(default=None, ge=0)
    summary: ReasoningSummary | None = None
