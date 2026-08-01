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
                generation=config.generation,
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

    def _discover_model_capabilities(self, model: str) -> dict[str, object] | None:
        response = self._client.models.list()
        for item in getattr(response, "data", response) or []:
            raw = self._dump_model(item)
            if str(raw.get("id", "")) != model:
                continue
            parameters = set(raw.get("supported_parameters") or [])
            result: dict[str, object] = {}
            maximum = raw.get("max_completion_tokens") or raw.get("max_output_tokens")
            if isinstance(maximum, int):
                result["max_output_tokens"] = maximum
            if parameters:
                result["tool_call"] = bool(
                    parameters.intersection({"tools", "tool_choice"})
                )
                result["reasoning"] = bool(
                    parameters.intersection({"reasoning", "reasoning_effort"})
                )
            return result or None
        return None
