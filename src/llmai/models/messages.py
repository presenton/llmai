from typing import List, Literal, Optional
from pydantic import BaseModel



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
    arguments: str | None

class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: Optional[List[AssistantToolCall]] = None

class ToolResponse(BaseModel):
    id: str
    content: str | None

class ToolResponseMessage(Message):
    role: Literal["tool"] = "tool"
    responses: List[ToolResponse]