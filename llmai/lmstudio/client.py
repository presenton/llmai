from __future__ import annotations

from logging import Logger

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import LMStudioClientConfig, OpenAIClientConfig
from llmai.shared.messages import (
    AssistantMessage,
    AssistantReasoningItem,
    flatten_thinking_content,
)
from llmai.shared.schema import process_schema


class LMStudioClient(OpenAIClient):
    PROVIDER_NAME = "lmstudio"
    PROVIDER_LABEL = "LM Studio"
    DEFAULT_BASE_URL = "http://localhost:1234/v1"
    DEFAULT_API_KEY = "lm-studio"
    SUPPORTED_SCHEMA_FIELDS = [
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "description",
        "enum",
        "items",
        "maxItems",
        "minItems",
        "properties",
        "required",
        "type",
    ]

    def __init__(
        self,
        *,
        config: LMStudioClientConfig,
        logger: Logger | None = None,
    ):
        super().__init__(
            config=OpenAIClientConfig(
                api_key=config.api_key or self.DEFAULT_API_KEY,
                base_url=self._base_url(config.base_url),
                api_type=OpenAIApiType.COMPLETIONS,
            ),
            logger=logger,
        )

    def _base_url(self, base_url: str | None) -> str:
        resolved = base_url or self.DEFAULT_BASE_URL
        if resolved.rstrip("/").endswith("/v1"):
            return resolved

        return f"{resolved.rstrip('/')}/v1"

    def _openai_schema(
        self,
        schema: dict,
        *,
        strict: bool,
    ) -> dict:
        del strict
        return process_schema(
            schema,
            flatten_allof=True,
            supported_schema_fields=self.SUPPORTED_SCHEMA_FIELDS,
        )

    def _chat_completion_message_to_thinking_items(
        self,
        message: object,
    ) -> list[AssistantReasoningItem]:
        reasoning_content = getattr(message, "reasoning_content", None)
        if not reasoning_content:
            return []

        return [AssistantReasoningItem(summary=[reasoning_content])]

    def _chat_completion_delta_to_thinking_text(self, delta: object) -> str | None:
        reasoning_content = getattr(delta, "reasoning_content", None)
        return reasoning_content or None

    def _assistant_message_to_chat_completion_assistant_message_param(
        self,
        message: AssistantMessage,
    ):
        lmstudio_message = (
            super()._assistant_message_to_chat_completion_assistant_message_param(message)
        )
        reasoning_content = "\n".join(flatten_thinking_content(message.thinking))
        if reasoning_content:
            lmstudio_message["reasoning_content"] = reasoning_content

        return lmstudio_message
