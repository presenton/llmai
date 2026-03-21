from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    pass


class UserMessage(Message):
    role: Literal["user"] = "user"
    content: str


class SystemMessage(Message):
    role: Literal["system"] = "system"
    content: str


class AssistantToolCall(BaseModel):
    id: str
    name: str
    arguments: str | None = None


class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[AssistantToolCall] = Field(default_factory=list)


class ToolResponseMessage(Message):
    role: Literal["tool"] = "tool"
    id: str
    content: str | None = None
