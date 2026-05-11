from __future__ import annotations

from logging import Logger

from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import CerebrasClientConfig, OpenAIClientConfig
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
