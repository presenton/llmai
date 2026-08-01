from __future__ import annotations

from typing import Any, List, Literal, Sequence, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    content: str


class AssistantToolCall(BaseModel):
    model_config = ConfigDict(
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    id: str
    name: str
    arguments: str | None = None
    thought_signature: bytes | None = None


class AssistantReasoningItem(BaseModel):
    model_config = ConfigDict(
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    id: str | None = None
    summary: list[str] = Field(default_factory=list)
    encrypted_content: str | None = None
    signature: str | None = None
    redacted_content: str | bytes | None = None
    provider: str | None = None
    raw: dict[str, Any] | None = Field(default=None, exclude=True)


class ReasoningTrace(BaseModel):
    """Human-readable reasoning returned by a provider.

    This is intentionally distinct from ``ReasoningState``: applications may
    display or log a trace, while state should only be replayed to the provider.
    """

    items: list[AssistantReasoningItem] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(part for item in self.items for part in item.summary if part)


class ReasoningState(BaseModel):
    """Opaque continuation state such as signatures or encrypted blocks."""

    model_config = ConfigDict(
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    provider: str
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    signatures: list[bytes] = Field(default_factory=list)


ThinkingContent: TypeAlias = list[AssistantReasoningItem] | None


def collapse_thinking_blocks(blocks: list[str]) -> ThinkingContent:
    return [AssistantReasoningItem(summary=[block]) for block in blocks] or None


def flatten_thinking_content(thinking: ThinkingContent) -> list[str]:
    if thinking is None:
        return []

    flattened: list[str] = []
    for item in thinking:
        flattened.extend(item.summary)

    return flattened


class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"
    id: str | None = None
    content: AssistantContent = None
    thinking: ThinkingContent = None
    tool_calls: list[AssistantToolCall] = Field(default_factory=list)


class ToolResponseMessage(Message):
    role: Literal["tool"] = "tool"
    id: str
    content: TextMessageContent | None = None
