from typing import Literal

from pydantic import BaseModel, Field


DEFAULT_MODEL_CONTEXT_WINDOW = 4_000
ModelTokenLimitsSource = Literal["provider", "default"]


class ModelTokenLimits(BaseModel):
    context_window: int = Field(
        default=DEFAULT_MODEL_CONTEXT_WINDOW,
        gt=0,
    )
    max_input_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    source: ModelTokenLimitsSource = "default"


class ModelInfo(BaseModel):
    id: str
    provider: str
    display_name: str | None = None
    token_limits: ModelTokenLimits


def model_token_limits(
    *,
    context_window: object = None,
    max_input_tokens: object = None,
    max_output_tokens: object = None,
) -> ModelTokenLimits:
    context = _positive_int(context_window)
    max_input = _positive_int(max_input_tokens)
    max_output = _positive_int(max_output_tokens)
    resolved_context = context or max_input
    if resolved_context is None:
        return ModelTokenLimits()

    return ModelTokenLimits(
        context_window=resolved_context,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        source="provider",
    )


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None

    return parsed if parsed > 0 else None
