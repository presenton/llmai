from __future__ import annotations

from logging import Logger

import httpx
from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import CerebrasClientConfig, OpenAIClientConfig
from llmai.shared.logs import LogLevel
from llmai.shared.model_metadata import metadata_items, metadata_value
from llmai.shared.models import ModelInfo, ModelTokenLimits
from llmai.shared.schema import get_schema_as_dict, process_schema
from llmai.shared.tools import Tool


class CerebrasClient(OpenAIClient):
    PROVIDER_NAME = "cerebras"
    PROVIDER_LABEL = "Cerebras"
    DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
    STRICT_SUPPORTED_SCHEMA_FIELDS = [
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "items",
        "maximum",
        "minimum",
        "multipleOf",
        "prefixItems",
        "properties",
        "required",
        "type",
    ]

    def __init__(
        self,
        *,
        config: CerebrasClientConfig,
        logger: Logger | None = None,
    ):
        self._uses_default_base_url = config.base_url is None
        super().__init__(
            config=OpenAIClientConfig(
                api_key=config.api_key,
                base_url=config.base_url or self.DEFAULT_BASE_URL,
                api_type=OpenAIApiType.COMPLETIONS,
            ),
            logger=logger,
        )

    def _public_models(self) -> list[object]:
        response = httpx.get(
            "https://api.cerebras.ai/public/v1/models",
            timeout=30.0,
        )
        response.raise_for_status()
        return metadata_items(response.json())

    def get_model_context_window(self, *, model: str) -> ModelTokenLimits:
        if not self._uses_default_base_url:
            return super().get_model_context_window(model=model)
        try:
            for model_data in self._public_models():
                if metadata_value(model_data, "id") == model:
                    return self._model_token_limits(model_data)
        except Exception as exc:
            self.log(
                LogLevel.WARNING,
                (
                    f"Cerebras model metadata lookup failed for {model!r}; "
                    f"using the 4000-token default: {exc}"
                ),
            )
        return ModelTokenLimits()

    def list_models(self) -> list[ModelInfo]:
        models = super().list_models()
        if not self._uses_default_base_url:
            return models
        try:
            public = {
                metadata_value(model, "id"): model
                for model in self._public_models()
            }
            return [
                model.model_copy(
                    update={
                        "token_limits": self._model_token_limits(public[model.id])
                    }
                )
                if model.id in public
                else model
                for model in models
            ]
        except Exception as exc:
            self.log(
                LogLevel.WARNING,
                (
                    "Cerebras public model enrichment failed; using default "
                    f"context windows: {exc}"
                ),
            )
            return models

    def _openai_schema(
        self,
        schema: dict,
        *,
        strict: bool,
    ) -> dict:
        if not strict:
            return schema

        return process_schema(
            schema,
            flatten_refs=False,
            flatten_allof=True,
            ensure_additional_properties=True,
            supported_schema_fields=self.STRICT_SUPPORTED_SCHEMA_FIELDS,
        )

    def _llm_tools_to_openai_tools(
        self,
        tools: list[Tool],
    ) -> list[ChatCompletionFunctionToolParam]:
        strict = any(tool.strict for tool in tools)
        return [
            ChatCompletionFunctionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=self._openai_schema(
                        get_schema_as_dict(tool.input_schema, strict=strict),
                        strict=strict,
                    ),
                    strict=strict,
                ),
            )
            for tool in tools
        ]
