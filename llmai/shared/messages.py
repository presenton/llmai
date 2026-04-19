from __future__ import annotations

from typing import List, Literal, Sequence, TypeAlias

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
MessageContentPart: TypeAlias = ContentPart | str
TextMessageContentPart: TypeAlias = TextContentPart | str
MessageContent: TypeAlias = List[MessageContentPart] | str
AssistantContent: TypeAlias = List[MessageContentPart] | None
TextMessageContent: TypeAlias = List[TextMessageContentPart]
ThinkingContent: TypeAlias = list[str] | None


def normalize_content_parts(
    content: Sequence[MessageContentPart] | str | None,
) -> list[ContentPart]:
    if content is None:
        return []

    if isinstance(content, str):
        return [TextContentPart(text=content)]

    normalized: list[ContentPart] = []
    for part in content:
        if isinstance(part, str):
            normalized.append(TextContentPart(text=part))
            continue
        normalized.append(part)

    return normalized


def collapse_content_parts(parts: list[ContentPart]) -> AssistantContent:
    return list(parts) or None


def content_from_text(text: str | None) -> AssistantContent:
    if text is None:
        return None

    return [TextContentPart(text=text)]


def collapse_thinking_blocks(blocks: list[str]) -> ThinkingContent:
    return blocks or None


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
    thinking: ThinkingContent = None
    tool_calls: list[AssistantToolCall] = Field(default_factory=list)


class ToolResponseMessage(Message):
    role: Literal["tool"] = "tool"
    id: str
    content: TextMessageContent | None = None
