from __future__ import annotations

from logging import Logger

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import FireworksClientConfig, OpenAIClientConfig
from llmai.shared.messages import (
    AssistantMessage,
    AssistantReasoningItem,
    flatten_thinking_content,
)
from llmai.shared.response_formats import (
    JSONSchemaResponse,
    ResponseFormat,
    get_response_format_name,
    get_response_format_strict,
    get_response_schema,
)
from llmai.shared.schema import process_schema


class FireworksClient(OpenAIClient):
    PROVIDER_NAME = "fireworks"
    PROVIDER_LABEL = "Fireworks"
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    SUPPORTED_SCHEMA_FIELDS = [
        "$defs",
        "$ref",
        "anyOf",
        "definitions",
        "description",
        "enum",
        "items",
        "properties",
        "required",
        "type",
    ]

    def __init__(
        self,
        *,
        config: FireworksClientConfig,
        logger: Logger | None = None,
    ):
        super().__init__(
            config=OpenAIClientConfig(
                api_key=config.api_key,
                base_url=config.base_url or self.DEFAULT_BASE_URL,
                api_type=OpenAIApiType.COMPLETIONS,
            ),
            logger=logger,
        )

    def _openai_schema(
        self,
        schema: dict,
        *,
        strict: bool,
    ) -> dict:
        del strict
        return process_schema(
            schema,
            flatten_refs=True,
            flatten_allof=True,
            remove_additional_properties=True,
            supported_schema_fields=self.SUPPORTED_SCHEMA_FIELDS,
        )

    def _get_openai_response_format_or_omit(
        self,
        response_format: ResponseFormat | None,
    ):
        if not isinstance(response_format, JSONSchemaResponse):
            return super()._get_openai_response_format_or_omit(response_format)

        strict = get_response_format_strict(response_format, default=False)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": get_response_format_name(
                    response_format,
                    default="response",
                ),
                "schema": self._openai_schema(
                    get_response_schema(response_format, strict=bool(strict)) or {},
                    strict=bool(strict),
                ),
            },
        }

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
        fireworks_message = (
            super()._assistant_message_to_chat_completion_assistant_message_param(
                message
            )
        )
        reasoning_content = "\n".join(flatten_thinking_content(message.thinking))
        if reasoning_content:
            fireworks_message["reasoning_content"] = reasoning_content

        return fireworks_message
