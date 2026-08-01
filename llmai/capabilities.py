from __future__ import annotations

from typing import Any

from llmai.models import ModelLookupError, get_model_metadata
from llmai.shared.generation import (
    CapabilitySource,
    CapabilityStatus,
    CapabilityValue,
    ModelCapabilities,
    ToolCallCapabilities,
)


def _capability(value: Any, *, source: CapabilitySource) -> CapabilityValue:
    if isinstance(value, bool):
        return CapabilityValue(
            status=(
                CapabilityStatus.SUPPORTED if value else CapabilityStatus.UNSUPPORTED
            ),
            value=value,
            source=source,
            fresh=False,
        )
    if value is not None:
        return CapabilityValue(
            status=CapabilityStatus.SUPPORTED,
            value=value,
            source=source,
            fresh=False,
        )
    return CapabilityValue()


def _override_for(
    model: str,
    provider: str | None,
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not overrides:
        return {}
    candidates = [model]
    if provider:
        candidates = [f"{provider}:{model}", f"{provider}/{model}", *candidates]
    for candidate in candidates:
        if candidate in overrides:
            return overrides[candidate]
    return {}


def get_model_capabilities(
    model: str,
    *,
    provider: str | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> ModelCapabilities:
    """Return tri-state capabilities with their metadata provenance."""

    metadata: dict[str, Any] = {}
    candidates = [model]
    if model.casefold().startswith("models/"):
        candidates.append(model.split("/", 1)[1])
    for candidate in candidates:
        try:
            metadata = get_model_metadata(candidate, provider=provider)
            break
        except ModelLookupError:
            continue

    override = _override_for(model, provider, overrides)
    source = CapabilitySource.BUNDLED if metadata else CapabilitySource.UNKNOWN
    reasoning_options = metadata.get("reasoning_options")
    if not isinstance(reasoning_options, list):
        reasoning_options = []
    levels: list[str] = []
    budget: dict[str, int] | None = None
    for option in reasoning_options:
        if not isinstance(option, dict):
            continue
        if option.get("type") == "effort" and isinstance(option.get("values"), list):
            levels.extend(str(value) for value in option["values"])
        if option.get("type") == "budget_tokens":
            budget = {
                key: int(option[key])
                for key in ("min", "max")
                if isinstance(option.get(key), int)
            }
    levels_are_inferred = False
    if budget is not None and not levels:
        levels = ["minimal", "low", "medium", "high", "xhigh", "max"]
        levels_are_inferred = True

    def selected(key: str, fallback: Any = None) -> tuple[Any, CapabilitySource]:
        if key in override:
            return override[key], CapabilitySource.APPLICATION
        return metadata.get(key, fallback), source

    reasoning_value, reasoning_source = selected("reasoning")
    tool_value, tool_source = selected("tool_call")
    max_value, max_source = selected("max_output_tokens")
    levels_value, levels_source = selected("reasoning_levels", levels or None)
    if levels_are_inferred and "reasoning_levels" not in override:
        levels_source = CapabilitySource.INFERRED
    budget_value, budget_source = selected("reasoning_budget", budget)
    interleaved_value, interleaved_source = selected(
        "reasoning_interleaved", metadata.get("interleaved")
    )
    parallel_value, parallel_source = selected("parallel_tool_calls")
    streaming_value, streaming_source = selected("streaming_tool_calls")

    return ModelCapabilities(
        provider=provider or metadata.get("provider_id"),
        model=model,
        max_output_tokens=_capability(max_value, source=max_source),
        reasoning=_capability(reasoning_value, source=reasoning_source),
        reasoning_levels=_capability(levels_value, source=levels_source),
        reasoning_budget=_capability(budget_value, source=budget_source),
        reasoning_interleaved=_capability(interleaved_value, source=interleaved_source),
        tool_call=ToolCallCapabilities(
            support=_capability(tool_value, source=tool_source),
            parallel=_capability(parallel_value, source=parallel_source),
            streaming=_capability(streaming_value, source=streaming_source),
        ),
        raw=metadata,
    )


def supports_tool_call(model: str, *, provider: str | None = None) -> bool | None:
    return get_model_capabilities(model, provider=provider).tool_call.support.supported


def supports_thinking(model: str, *, provider: str | None = None) -> bool | None:
    return get_model_capabilities(model, provider=provider).reasoning.supported


def get_reasoning_levels(model: str, *, provider: str | None = None) -> list[str]:
    value = get_model_capabilities(model, provider=provider).reasoning_levels.value
    return list(value) if isinstance(value, list) else []


def require_tool_call_support(model: str, *, provider: str | None = None) -> None:
    supported = supports_tool_call(model, provider=provider)
    if supported is False:
        raise ValueError(f"Model {model!r} does not support tool calls")
    if supported is None:
        raise ValueError(f"Tool-call support for model {model!r} is unknown")
