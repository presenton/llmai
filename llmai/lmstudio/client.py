from __future__ import annotations

from logging import Logger
from urllib.parse import quote

import httpx

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import LMStudioClientConfig, OpenAIClientConfig
from llmai.shared.logs import LogLevel
from llmai.shared.model_metadata import metadata_items, metadata_value
from llmai.shared.models import (
    ModelInfo,
    ModelTokenLimits,
    model_token_limits,
)
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
        api_base_url = self._base_url(config.base_url)
        self._rest_base_url = api_base_url.removesuffix("/v1")
        self._lmstudio_api_key = config.api_key or self.DEFAULT_API_KEY
        super().__init__(
            config=OpenAIClientConfig(
                api_key=self._lmstudio_api_key,
                base_url=api_base_url,
                api_type=OpenAIApiType.COMPLETIONS,
            ),
            logger=logger,
        )

    def _rest_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._lmstudio_api_key}"}

    def _enhanced_model(self, model: str) -> object:
        response = httpx.get(
            (
                f"{self._rest_base_url}/api/v0/models/"
                f"{quote(model, safe='')}"
            ),
            headers=self._rest_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def _model_token_limits(self, model: object) -> ModelTokenLimits:
        enhanced_context = metadata_value(model, "max_context_length")
        if enhanced_context is not None:
            return model_token_limits(context_window=enhanced_context)
        return super()._model_token_limits(model)

    def get_model_context_window(self, *, model: str) -> ModelTokenLimits:
        try:
            return self._model_token_limits(self._enhanced_model(model))
        except Exception as exc:
            self.log(
                LogLevel.WARNING,
                (
                    f"LM Studio model metadata lookup failed for {model!r}; "
                    f"using the 4000-token default: {exc}"
                ),
            )
            return ModelTokenLimits()

    def list_models(self) -> list[ModelInfo]:
        try:
            response = httpx.get(
                f"{self._rest_base_url}/api/v0/models",
                headers=self._rest_headers(),
                timeout=30.0,
            )
            response.raise_for_status()
            results = []
            for model in metadata_items(response.json()):
                info = self._model_info(model)
                if info is not None:
                    results.append(info)
            return results
        except Exception as exc:
            self.log(
                LogLevel.WARNING,
                (
                    "LM Studio enhanced model listing failed; trying the "
                    f"OpenAI-compatible endpoint: {exc}"
                ),
            )
            return super().list_models()

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
