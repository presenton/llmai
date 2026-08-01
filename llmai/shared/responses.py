from collections.abc import AsyncIterator, Awaitable, Generator
from math import ceil
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from llmai.shared.messages import (
    AssistantToolCall,
    Message,
    ReasoningState,
    ReasoningTrace,
    ThinkingContent,
    flatten_thinking_content,
)
from llmai.shared.reasoning import ReasoningUsage


class ResponseUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    reasoning: ReasoningUsage | None = None

    @property
    def thinking_tokens(self) -> int | None:
        """Best available thinking count: provider-billed first, visible second."""

        if self.reasoning is None:
            return None
        if self.reasoning.billed_tokens is not None:
            return self.reasoning.billed_tokens
        return self.reasoning.visible_tokens


class ResponseDiagnostic(BaseModel):
    code: str
    message: str
    provider: str | None = None
    model: str | None = None
    parameter: str | None = None
    retry_performed: bool = False


def estimate_visible_reasoning_tokens(thinking: ThinkingContent) -> int | None:
    text = "\n".join(flatten_thinking_content(thinking))
    return ceil(len(text.encode("utf-8")) / 4) if text else None


def _native_reasoning_tokens(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in (
            "reasoning_tokens",
            "thoughts_token_count",
            "thinking_tokens",
            "reasoning_token_count",
        ):
            candidate = value.get(key)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
        for nested in value.values():
            found = _native_reasoning_tokens(nested)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _native_reasoning_tokens(nested)
            if found is not None:
                return found
    return None


def normalize_reasoning_usage(
    usage: ResponseUsage | None,
    thinking: ThinkingContent,
) -> ResponseUsage | None:
    visible = estimate_visible_reasoning_tokens(thinking)
    if usage is None and visible is None:
        return None
    usage = usage or ResponseUsage()
    if usage.reasoning is not None and usage.reasoning.billed_tokens is not None:
        if usage.reasoning.visible_tokens is None:
            usage.reasoning.visible_tokens = visible
            usage.reasoning.visible_estimated = visible is not None
        return usage
    native = _native_reasoning_tokens(usage.details)
    if native is not None:
        usage.reasoning = ReasoningUsage(
            billed_tokens=native,
            visible_tokens=visible,
            visible_estimated=visible is not None,
            source="provider_details",
        )
    elif visible is not None:
        usage.reasoning = ReasoningUsage(
            visible_tokens=visible,
            visible_estimated=True,
            source="estimated_visible",
        )
    return usage


class ResponseContent(BaseModel):
    type: Literal["content"] = "content"
    content: Any = None
    thinking: ThinkingContent = None
    reasoning_trace: ReasoningTrace | None = None
    reasoning_state: ReasoningState | None = None
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[AssistantToolCall] = Field(default_factory=list)
    usage: ResponseUsage | None = None
    duration_seconds: float | None = None
    diagnostics: list[ResponseDiagnostic] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_reasoning(self) -> "ResponseContent":
        if self.reasoning_trace is None and self.thinking:
            self.reasoning_trace = ReasoningTrace(items=self.thinking)
        if self.reasoning_state is None and self.thinking:
            opaque = [
                item
                for item in self.thinking
                if item.encrypted_content
                or item.signature
                or item.redacted_content
                or item.raw
            ]
            if opaque:
                provider = next(
                    (item.provider for item in opaque if item.provider), "unknown"
                )
                self.reasoning_state = ReasoningState(
                    provider=provider,
                    blocks=[
                        {
                            "id": item.id,
                            "encrypted_content": item.encrypted_content,
                            "signature": item.signature,
                            "redacted_content": item.redacted_content,
                            "raw": item.raw,
                        }
                        for item in opaque
                    ],
                )
        if self.reasoning_state is None:
            signatures = [
                tool.thought_signature
                for tool in self.tool_calls
                if tool.thought_signature is not None
            ]
            if signatures:
                self.reasoning_state = ReasoningState(
                    provider="google", signatures=signatures
                )
        self.usage = normalize_reasoning_usage(self.usage, self.thinking)
        return self

    def safe_model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize user-visible output without opaque continuation state."""

        exclude = kwargs.pop("exclude", None) or set()
        if isinstance(exclude, set):
            exclude = {*exclude, "reasoning_state"}
        elif isinstance(exclude, dict):
            exclude = {**exclude, "reasoning_state": True}
        return self.model_dump(exclude=exclude, **kwargs)

    def lossless_model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize output including provider continuation state."""

        return self.model_dump(**kwargs)


ResponseStreamChunkType = Literal["content", "thinking", "tool"]
ResponseStreamChunkEvent = Literal["start", "end"]


class BaseResponseStreamChunk(BaseModel):
    pass


class ResponseStreamChunk(BaseResponseStreamChunk):
    type: Literal["event"] = "event"
    chunk_type: ResponseStreamChunkType
    event: ResponseStreamChunkEvent
    tool: str | None = None


class ResponseStreamContentChunk(BaseResponseStreamChunk):
    type: Literal["content"] = "content"
    chunk: str


class ResponseStreamThinkingChunk(BaseResponseStreamChunk):
    type: Literal["thinking"] = "thinking"
    chunk: str
    thinking_tokens: int | None = None

    @model_validator(mode="after")
    def _estimate_tokens(self) -> "ResponseStreamThinkingChunk":
        if self.thinking_tokens is None and self.chunk:
            self.thinking_tokens = ceil(len(self.chunk.encode("utf-8")) / 4)
        return self


class ResponseStreamToolChunk(BaseResponseStreamChunk):
    id: str
    type: Literal["tool"] = "tool"
    tool: str | None = None
    chunk: str


class ResponseStreamToolCompleteChunk(BaseResponseStreamChunk):
    id: str
    type: Literal["tool_complete"] = "tool_complete"
    tool: str | None = None
    arguments: str | None = None


class ResponseStreamCompletionChunk(BaseResponseStreamChunk):
    type: Literal["completion"] = "completion"
    content: Any = None
    thinking: ThinkingContent = None
    reasoning_trace: ReasoningTrace | None = None
    reasoning_state: ReasoningState | None = None
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[AssistantToolCall] = Field(default_factory=list)
    usage: ResponseUsage | None = None
    duration_seconds: float | None = None
    diagnostics: list[ResponseDiagnostic] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_reasoning(self) -> "ResponseStreamCompletionChunk":
        if self.reasoning_trace is None and self.thinking:
            self.reasoning_trace = ReasoningTrace(items=self.thinking)
        if self.reasoning_state is None and self.thinking:
            opaque = [
                item
                for item in self.thinking
                if item.encrypted_content
                or item.signature
                or item.redacted_content
                or item.raw
            ]
            if opaque:
                provider = next(
                    (item.provider for item in opaque if item.provider), "unknown"
                )
                self.reasoning_state = ReasoningState(
                    provider=provider,
                    blocks=[
                        {
                            "id": item.id,
                            "encrypted_content": item.encrypted_content,
                            "signature": item.signature,
                            "redacted_content": item.redacted_content,
                            "raw": item.raw,
                        }
                        for item in opaque
                    ],
                )
        if self.reasoning_state is None:
            signatures = [
                tool.thought_signature
                for tool in self.tool_calls
                if tool.thought_signature is not None
            ]
            if signatures:
                self.reasoning_state = ReasoningState(
                    provider="google", signatures=signatures
                )
        self.usage = normalize_reasoning_usage(self.usage, self.thinking)
        return self

    def safe_model_dump(self, **kwargs: Any) -> dict[str, Any]:
        exclude = kwargs.pop("exclude", None) or set()
        if isinstance(exclude, set):
            exclude = {*exclude, "reasoning_state"}
        elif isinstance(exclude, dict):
            exclude = {**exclude, "reasoning_state": True}
        return self.model_dump(exclude=exclude, **kwargs)

    def lossless_model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return self.model_dump(**kwargs)


ResponseStreamEvent = (
    ResponseStreamChunk
    | ResponseStreamContentChunk
    | ResponseStreamThinkingChunk
    | ResponseStreamToolChunk
    | ResponseStreamToolCompleteChunk
    | ResponseStreamCompletionChunk
)
ResponseResult = ResponseContent | Generator[ResponseStreamEvent, None, None]
AsyncResponseResult = Awaitable[ResponseContent] | AsyncIterator[ResponseStreamEvent]
