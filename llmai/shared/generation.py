from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from llmai.shared.reasoning import (
    ReasoningConfig,
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
    ReasoningTraceMode,
)


class GenerationProfile(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"
    MODEL_MAX = "model_max"


class ValidationMode(str, Enum):
    ADAPTIVE = "adaptive"
    STRICT = "strict"
    OFF = "off"


class CapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class CapabilitySource(str, Enum):
    APPLICATION = "application"
    LIVE = "live"
    BUNDLED = "bundled"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class CapabilityValue(BaseModel):
    status: CapabilityStatus = CapabilityStatus.UNKNOWN
    value: Any = None
    source: CapabilitySource = CapabilitySource.UNKNOWN
    fresh: bool = False

    @property
    def supported(self) -> bool | None:
        if self.status == CapabilityStatus.SUPPORTED:
            return True
        if self.status == CapabilityStatus.UNSUPPORTED:
            return False
        return None


class ToolCallCapabilities(BaseModel):
    support: CapabilityValue = Field(default_factory=CapabilityValue)
    parallel: CapabilityValue = Field(default_factory=CapabilityValue)
    streaming: CapabilityValue = Field(default_factory=CapabilityValue)


class ModelCapabilities(BaseModel):
    provider: str | None = None
    model: str
    max_output_tokens: CapabilityValue = Field(default_factory=CapabilityValue)
    reasoning: CapabilityValue = Field(default_factory=CapabilityValue)
    reasoning_levels: CapabilityValue = Field(default_factory=CapabilityValue)
    reasoning_budget: CapabilityValue = Field(default_factory=CapabilityValue)
    reasoning_interleaved: CapabilityValue = Field(default_factory=CapabilityValue)
    tool_call: ToolCallCapabilities = Field(default_factory=ToolCallCapabilities)
    raw: dict[str, Any] = Field(default_factory=dict)


class GenerationWarning(BaseModel):
    code: str
    message: str
    provider: str | None = None
    model: str | None = None
    parameter: str | None = None


class GenerationDefaults(BaseModel):
    """Application-wide defaults applied to every llmai provider client."""

    profile: GenerationProfile = GenerationProfile.BALANCED
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens_cap: int | None = Field(default=None, gt=0)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    validation: ValidationMode = ValidationMode.ADAPTIVE
    discover_capabilities: bool = True
    discovery_success_ttl_seconds: int = Field(default=3600, ge=0)
    discovery_failure_ttl_seconds: int = Field(default=300, ge=0)
    bundled_metadata_max_age_days: int = Field(default=7, ge=0)
    capability_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PreparedGeneration(BaseModel):
    model: str
    provider: str | None = None
    max_output_tokens: int
    reasoning: ReasoningConfig
    capabilities: ModelCapabilities
    warnings: list[GenerationWarning] = Field(default_factory=list)
    max_output_tokens_explicit: bool = False
    reasoning_explicit: bool = False
    reasoning_summary: ReasoningSummary | None = None
    include_trace: bool | None = None

    @property
    def reasoning_effort(self) -> ReasoningEffort | None:
        config = self.reasoning
        if (
            config.enabled is None
            and config.effort is None
            and config.budget_tokens is None
            and config.trace == ReasoningTraceMode.AUTO
        ):
            return None
        effort = config.effort
        if config.enabled is False:
            effort = ReasoningEffortValue.NONE
        return ReasoningEffort(
            effort=effort,
            tokens=config.budget_tokens,
            summary=self.reasoning_summary,
            include_trace=self.include_trace,
        )


class GenerationRequest(BaseModel):
    profile: GenerationProfile | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    reasoning: ReasoningConfig | None = None

    @model_validator(mode="after")
    def _reasoning_consistency(self) -> GenerationRequest:
        if (
            self.reasoning is not None
            and self.reasoning.enabled is False
            and self.reasoning.budget_tokens not in (None, 0)
        ):
            raise ValueError(
                "reasoning budget cannot be set when reasoning is disabled"
            )
        return self


def _profile_limit(profile: GenerationProfile, advertised: int | None) -> int:
    if profile == GenerationProfile.FAST:
        return min(advertised, 4_096) if advertised else 4_096
    if profile == GenerationProfile.DEEP:
        return min(advertised, 65_536) if advertised else 32_768
    if profile == GenerationProfile.MODEL_MAX:
        return advertised or 32_768
    return min(advertised, 32_768) if advertised else 8_192


def _merge_reasoning(
    base: ReasoningConfig,
    explicit: ReasoningConfig | None,
    legacy: ReasoningEffort | None,
    profile: GenerationProfile,
) -> ReasoningConfig:
    result = base.model_copy(deep=True)
    if profile == GenerationProfile.FAST and result.enabled is None:
        result.enabled = False
    elif profile == GenerationProfile.DEEP and result.enabled is None:
        result.enabled = True
        result.effort = result.effort or ReasoningEffortValue.HIGH

    if legacy is not None:
        if legacy.effort is not None:
            result.effort = legacy.effort
            result.enabled = legacy.effort != ReasoningEffortValue.NONE
        if legacy.tokens is not None:
            result.budget_tokens = legacy.tokens
            result.enabled = legacy.tokens > 0
    if explicit is not None:
        for field_name in explicit.model_fields_set:
            setattr(result, field_name, getattr(explicit, field_name))
    return result


def prepare_generation(
    *,
    model: str,
    provider: str | None = None,
    defaults: GenerationDefaults | None = None,
    profile: GenerationProfile | str | None = None,
    max_tokens: int | None = None,
    max_output_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    tools_requested: bool = False,
    capabilities: ModelCapabilities | None = None,
) -> PreparedGeneration:
    """Resolve provider-neutral generation settings using one precedence chain."""

    from llmai.capabilities import get_model_capabilities

    if max_tokens is not None and max_output_tokens is not None:
        raise ValueError("Pass only one of max_tokens or max_output_tokens")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    defaults = defaults or GenerationDefaults()
    selected_profile = GenerationProfile(profile or defaults.profile)
    capabilities = capabilities or get_model_capabilities(
        model, provider=provider, overrides=defaults.capability_overrides
    )
    advertised = capabilities.max_output_tokens.value
    if not isinstance(advertised, int) or advertised <= 0:
        advertised = None

    explicit_max = max_output_tokens if max_output_tokens is not None else max_tokens
    resolved_max = explicit_max or defaults.max_output_tokens
    if resolved_max is None:
        resolved_max = _profile_limit(selected_profile, advertised)
    if defaults.max_output_tokens_cap is not None:
        resolved_max = min(resolved_max, defaults.max_output_tokens_cap)

    warnings: list[GenerationWarning] = []
    if advertised and resolved_max > advertised:
        message = (
            f"Requested {resolved_max} output tokens but bundled metadata advertises "
            f"a maximum of {advertised}."
        )
        if defaults.validation == ValidationMode.STRICT:
            raise ValueError(message)
        if defaults.validation == ValidationMode.ADAPTIVE:
            warnings.append(
                GenerationWarning(
                    code="output_limit_exceeds_metadata",
                    message=message,
                    provider=provider,
                    model=model,
                    parameter="max_output_tokens",
                )
            )

    resolved_reasoning = _merge_reasoning(
        defaults.reasoning,
        reasoning,
        reasoning_effort,
        selected_profile,
    )
    reasoning_explicit = (
        reasoning is not None and bool(reasoning.model_fields_set)
    ) or reasoning_effort is not None
    supported_levels = capabilities.reasoning_levels.value
    if not isinstance(supported_levels, list):
        supported_levels = []
    supported_levels = [str(level) for level in supported_levels]
    requested_effort = resolved_reasoning.effort
    if requested_effort == ReasoningEffortValue.DEFAULT:
        resolved_reasoning.effort = None
    elif (
        requested_effort is not None
        and supported_levels
        and not (
            resolved_reasoning.enabled is False
            and requested_effort == ReasoningEffortValue.NONE
        )
    ):
        effort_value = requested_effort.value
        if effort_value not in supported_levels:
            message = (
                f"Reasoning effort {effort_value!r} is not advertised for "
                f"{model!r}; supported levels are {supported_levels}."
            )
            if defaults.validation == ValidationMode.STRICT:
                raise ValueError(message)
            if defaults.validation == ValidationMode.ADAPTIVE:
                rank = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
                requested_rank = rank.index(effort_value)
                nearest = min(
                    supported_levels,
                    key=lambda level: abs(
                        rank.index(level) - requested_rank
                        if level in rank
                        else len(rank)
                    ),
                )
                resolved_reasoning.effort = ReasoningEffortValue(nearest)
                warnings.append(
                    GenerationWarning(
                        code="reasoning_effort_adapted",
                        message=f"{message} Using {nearest!r}.",
                        provider=provider,
                        model=model,
                        parameter="reasoning.effort",
                    )
                )

    budget_limits = capabilities.reasoning_budget.value
    if not isinstance(budget_limits, dict):
        budget_limits = None
    supports_toggle = any(
        isinstance(option, dict) and option.get("type") == "toggle"
        for option in capabilities.raw.get("reasoning_options", [])
    )
    mandatory_reasoning = bool(
        capabilities.reasoning.supported
        and not supports_toggle
        and (
            (supported_levels and "none" not in supported_levels)
            or (budget_limits and int(budget_limits.get("min", 0)) > 0)
        )
    )
    if resolved_reasoning.enabled is False and mandatory_reasoning:
        message = f"Model {model!r} does not advertise a way to disable reasoning."
        if defaults.validation == ValidationMode.STRICT and reasoning_explicit:
            raise ValueError(message)
        if defaults.validation == ValidationMode.ADAPTIVE:
            resolved_reasoning.enabled = True
            if supported_levels:
                resolved_reasoning.effort = ReasoningEffortValue(supported_levels[0])
            elif budget_limits:
                resolved_reasoning.budget_tokens = int(budget_limits.get("min", 1024))
            warnings.append(
                GenerationWarning(
                    code="reasoning_cannot_be_disabled",
                    message=message,
                    provider=provider,
                    model=model,
                    parameter="reasoning.enabled",
                )
            )

    if (
        resolved_reasoning.enabled is True
        and resolved_reasoning.budget_tokens is None
        and resolved_reasoning.effort is None
        and budget_limits
    ):
        desired_budget = 16_384 if selected_profile == GenerationProfile.DEEP else 4_096
        minimum = int(budget_limits.get("min", 0))
        maximum = int(budget_limits.get("max", desired_budget))
        resolved_reasoning.budget_tokens = max(minimum, min(desired_budget, maximum))

    if resolved_reasoning.enabled is True and capabilities.reasoning.supported is False:
        message = f"Model {model!r} is marked as not supporting reasoning."
        if defaults.validation == ValidationMode.STRICT:
            raise ValueError(message)
        if defaults.validation == ValidationMode.ADAPTIVE:
            warnings.append(
                GenerationWarning(
                    code="reasoning_not_supported",
                    message=message,
                    provider=provider,
                    model=model,
                    parameter="reasoning",
                )
            )

    if tools_requested and capabilities.tool_call.support.supported is False:
        message = f"Model {model!r} is marked as not supporting tool calls."
        if defaults.validation == ValidationMode.STRICT:
            raise ValueError(message)
        if defaults.validation == ValidationMode.ADAPTIVE:
            warnings.append(
                GenerationWarning(
                    code="tool_call_not_supported",
                    message=message,
                    provider=provider,
                    model=model,
                    parameter="tools",
                )
            )

    # Manual-budget APIs generally require room for a visible answer.
    if resolved_reasoning.budget_tokens is not None:
        maximum_budget = max(0, resolved_max - 1_024)
        if resolved_reasoning.budget_tokens > maximum_budget:
            message = (
                "reasoning budget must leave at least 1024 tokens for visible output"
            )
            if reasoning_explicit or defaults.validation == ValidationMode.STRICT:
                raise ValueError(message)
            resolved_reasoning.budget_tokens = maximum_budget
            warnings.append(
                GenerationWarning(
                    code="reasoning_budget_clamped",
                    message=message,
                    provider=provider,
                    model=model,
                    parameter="reasoning.budget_tokens",
                )
            )

    return PreparedGeneration(
        model=model,
        provider=provider,
        max_output_tokens=resolved_max,
        reasoning=resolved_reasoning,
        capabilities=capabilities,
        warnings=warnings,
        max_output_tokens_explicit=explicit_max is not None,
        reasoning_explicit=reasoning_explicit,
        reasoning_summary=(
            reasoning_effort.summary
            if reasoning_effort and reasoning_effort.summary is not None
            else {
                ReasoningTraceMode.AUTO: ReasoningSummary.AUTO,
                ReasoningTraceMode.SUMMARY: ReasoningSummary.CONCISE,
                ReasoningTraceMode.FULL: ReasoningSummary.DETAILED,
            }.get(resolved_reasoning.trace)
        ),
        include_trace=(
            reasoning_effort.include_trace
            if reasoning_effort and reasoning_effort.include_trace is not None
            else resolved_reasoning.trace != ReasoningTraceMode.NONE
        ),
    )
