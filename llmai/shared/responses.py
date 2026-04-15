from typing import Any, Literal

from pydantic import BaseModel, Field

from llmai.shared.messages import AssistantToolCall, Message


class ResponseContent(BaseModel):
    type: Literal["content"] = "content"
    content: Any = None
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[AssistantToolCall] = Field(default_factory=list)


class ResponseStreamChunk(BaseModel):
    id: str


class ResponseStreamContentChunk(ResponseStreamChunk):
    type: Literal["stream_content"] = "stream_content"
    source: Literal["direct", "tool"]
    tool: str | None = None
    chunk: str


class ResponseStreamCompletionChunk(ResponseStreamChunk):
    type: Literal["stream_completion"] = "stream_completion"
    content: Any = None
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[AssistantToolCall] = Field(default_factory=list)
