from collections.abc import Generator
from typing import Any, Literal

from pydantic import BaseModel, Field

from llmai.shared.messages import AssistantToolCall, Message, ThinkingContent


class ResponseUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ResponseContent(BaseModel):
    type: Literal["content"] = "content"
    content: Any = None
    thinking: ThinkingContent = None
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[AssistantToolCall] = Field(default_factory=list)
    usage: ResponseUsage | None = None
    duration_seconds: float | None = None


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
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[AssistantToolCall] = Field(default_factory=list)
    usage: ResponseUsage | None = None
    duration_seconds: float | None = None


ResponseStreamEvent = (
    ResponseStreamChunk
    | ResponseStreamContentChunk
    | ResponseStreamThinkingChunk
    | ResponseStreamToolChunk
    | ResponseStreamToolCompleteChunk
    | ResponseStreamCompletionChunk
)
ResponseResult = ResponseContent | Generator[ResponseStreamEvent, None, None]
