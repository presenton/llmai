from typing import List, Literal, Optional
from pydantic import BaseModel
from google.genai.types import Content as GoogleContent

from models.llm_tool_call import AnthropicToolCall


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

class ToolResponseMessage(Message):
    role: Literal["tool"] = "tool"
    id: str
    content: str | None = None   


class GoogleAssistantMessage(LLMMessage):
    role: Literal["assistant"] = "assistant"
    content: GoogleContent


class AnthropicAssistantMessage(LLMMessage):
    role: Literal["assistant"] = "assistant"
    content: List[AnthropicToolCall]


class AnthropicToolCallMessage(LLMMessage):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str


class AnthropicUserMessage(LLMMessage):
    role: Literal["user"] = "user"
    content: List[AnthropicToolCallMessage]


class OpenAIToolCallMessage(LLMMessage):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


class GoogleToolCallMessage(LLMMessage):
    role: Literal["tool"] = "tool"
    id: Optional[str] = None
    name: str
    response: dict
