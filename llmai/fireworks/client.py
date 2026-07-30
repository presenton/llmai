from __future__ import annotations

from logging import Logger

import httpx

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import FireworksClientConfig, OpenAIClientConfig
from llmai.shared.logs import LogLevel
from llmai.shared.model_metadata import metadata_value
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
        self._fireworks_api_key = config.api_key
        self._uses_default_base_url = config.base_url is None
        super().__init__(
            config=OpenAIClientConfig(
                api_key=config.api_key,
                base_url=config.base_url or self.DEFAULT_BASE_URL,
                api_type=OpenAIApiType.COMPLETIONS,
            ),
            logger=logger,
        )

    def _management_resource(self, model: str) -> str | None:
        resource = model.rsplit("#", 1)[-1]
        parts = resource.split("/")
        if (
            len(parts) == 4
            and parts[0] == "accounts"
            and parts[2] in {"models", "deployments"}
            and all(parts)
        ):
            return resource
        return None

    def _get_management_model(self, model: str) -> object | None:
        resource = self._management_resource(model)
        if resource is None or not self._uses_default_base_url:
            return None
        response = httpx.get(
            f"https://api.fireworks.ai/v1/{resource}",
            headers={"Authorization": f"Bearer {self._fireworks_api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def _model_token_limits(self, model: object) -> ModelTokenLimits:
        direct_context = metadata_value(
            model,
            "contextLength",
            "maxContextLength",
        )
        if direct_context is not None:
            return model_token_limits(context_window=direct_context)
        return super()._model_token_limits(model)

    def get_model_context_window(self, *, model: str) -> ModelTokenLimits:
        try:
            model_data = self._get_management_model(model)
            if model_data is not None:
                return self._model_token_limits(model_data)
        except Exception as exc:
            self.log(
                LogLevel.WARNING,
                (
                    f"Fireworks model metadata lookup failed for {model!r}; "
                    f"using the 4000-token default: {exc}"
                ),
            )
            return ModelTokenLimits()
        return super().get_model_context_window(model=model)

    def list_models(self) -> list[ModelInfo]:
        models = super().list_models()
        if not self._uses_default_base_url:
            return models
        results = []
        for model in models:
            try:
                model_data = self._get_management_model(model.id)
                if model_data is not None:
                    model = model.model_copy(
                        update={
                            "token_limits": self._model_token_limits(model_data)
                        }
                    )
            except Exception as exc:
                self.log(
                    LogLevel.WARNING,
                    (
                        f"Fireworks metadata enrichment failed for {model.id!r}; "
                        f"using the 4000-token default: {exc}"
                    ),
                )
            results.append(model)
        return results

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
