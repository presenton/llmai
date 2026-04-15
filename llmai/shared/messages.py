from __future__ import annotations

from typing import List, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator


class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContentPart(BaseModel):
    type: Literal["image"] = "image"
    url: str | None = None
    data: bytes | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "ImageContentPart":
        has_url = self.url is not None
        has_data = self.data is not None

        if has_url == has_data:
            raise ValueError("Image content requires exactly one of url or data")

        if has_data and not self.mime_type:
            raise ValueError("mime_type is required when image data is provided")

        return self


ContentPart: TypeAlias = TextContentPart | ImageContentPart
MessageContent: TypeAlias = List[ContentPart]
AssistantContent: TypeAlias = List[ContentPart] | None
TextMessageContent: TypeAlias = List[TextContentPart]


def normalize_content_parts(content: MessageContent | None) -> list[ContentPart]:
    return [] if content is None else list(content)


def collapse_content_parts(parts: list[ContentPart]) -> AssistantContent:
    return list(parts) or None


def content_from_text(text: str | None) -> AssistantContent:
    if text is None:
        return None

    return [TextContentPart(text=text)]


def content_has_images(content: AssistantContent) -> bool:
    return any(
        isinstance(part, ImageContentPart) for part in normalize_content_parts(content)
    )


class Message(BaseModel):
    pass


class UserMessage(Message):
    role: Literal["user"] = "user"
    content: MessageContent


class SystemMessage(Message):
    role: Literal["system"] = "system"
    content: TextMessageContent


class AssistantToolCall(BaseModel):
    id: str
    name: str
    arguments: str | None = None


class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"
    content: AssistantContent = None
    thinking: str | None = None
    tool_calls: list[AssistantToolCall] = Field(default_factory=list)


class ToolResponseMessage(Message):
    role: Literal["tool"] = "tool"
    id: str
    content: TextMessageContent | None = None
