from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

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


ContentPart: TypeAlias = Annotated[
    TextContentPart | ImageContentPart,
    Field(discriminator="type"),
]
MessageContent: TypeAlias = str | list[ContentPart]
AssistantContent: TypeAlias = MessageContent | None


def normalize_content_parts(content: MessageContent | None) -> list[ContentPart]:
    if content is None:
        return []

    if isinstance(content, str):
        return [TextContentPart(text=content)]

    return list(content)


def collapse_content_parts(parts: list[ContentPart]) -> AssistantContent:
    if not parts:
        return None

    if all(isinstance(part, TextContentPart) for part in parts):
        return "".join(part.text for part in parts if isinstance(part, TextContentPart))

    return list(parts)


def content_has_images(content: AssistantContent) -> bool:
    return any(
        isinstance(part, ImageContentPart)
        for part in normalize_content_parts(content)
    )


class Message(BaseModel):
    pass


class UserMessage(Message):
    role: Literal["user"] = "user"
    content: MessageContent


class SystemMessage(Message):
    role: Literal["system"] = "system"
    content: str


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
    content: str | None = None
