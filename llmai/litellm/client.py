from __future__ import annotations

from logging import Logger

import httpx
from openai import OpenAI

from llmai.openai.client import OpenAIClient
from llmai.shared.base import BaseClient
from llmai.shared.configs import LiteLLMClientConfig
from llmai.shared.errors import configuration_error, raise_llm_error
from llmai.shared.logs import LogLevel
from llmai.shared.model_metadata import metadata_items, metadata_value
from llmai.shared.messages import Message
from llmai.shared.models import (
    ModelInfo,
    ModelTokenLimits,
    model_token_limits,
)
from llmai.shared.reasoning import ReasoningEffort
from llmai.shared.response_formats import ResponseFormat
from llmai.shared.responses import ResponseResult
from llmai.shared.tools import LLMTool, ToolChoice


class LiteLLMClient(OpenAIClient):
    PROVIDER_NAME = "litellm"
    PROVIDER_LABEL = "LiteLLM"

    def __init__(
        self,
        *,
        config: LiteLLMClientConfig,
        logger: Logger | None = None,
    ):
        BaseClient.__init__(self, logger=logger)
        self._litellm_base_url = config.base_url
        self._litellm_api_key = config.api_key or "EMPTY"
        self._api_type = self._coerce_api_type(config.api_type)
        if self._api_type is None:
            raise configuration_error(
                f"Unsupported LiteLLM api_type: {config.api_type}",
                provider=self.PROVIDER_NAME,
            )

        self._provide_system_message_as_instructions = False
        self._extra_body = dict(config.extra_kwargs)

        try:
            self._client = OpenAI(
                base_url=config.base_url,
                api_key=self._litellm_api_key,
            )
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

        if self._logger:
            self._logger.info("%s client created", self.PROVIDER_LABEL)
            self._logger.info("Base URL: %s", config.base_url)

    def _model_info_data(self) -> list[object]:
        if self._litellm_base_url is None:
            return []
        response = httpx.get(
            f"{self._litellm_base_url.rstrip('/')}/model/info",
            headers={"Authorization": f"Bearer {self._litellm_api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        return metadata_items(response.json())

    def _model_token_limits(self, model: object) -> ModelTokenLimits:
        info = metadata_value(model, "model_info")
        if info is not None:
            return model_token_limits(
                context_window=metadata_value(info, "max_tokens"),
                max_input_tokens=metadata_value(info, "max_input_tokens"),
                max_output_tokens=metadata_value(info, "max_output_tokens"),
            )
        return super()._model_token_limits(model)

    def get_model_context_window(self, *, model: str) -> ModelTokenLimits:
        try:
            for model_data in self._model_info_data():
                candidates = {
                    metadata_value(model_data, "model_name"),
                    metadata_value(
                        metadata_value(model_data, "model_info") or {},
                        "key",
                        "id",
                    ),
                }
                if model in candidates:
                    return self._model_token_limits(model_data)
        except Exception as exc:
            self.log(
                LogLevel.WARNING,
                (
                    f"LiteLLM model metadata lookup failed for {model!r}; "
                    f"using the 4000-token default: {exc}"
                ),
            )
        return ModelTokenLimits()

    def list_models(self) -> list[ModelInfo]:
        try:
            model_data = self._model_info_data()
            if model_data:
                results = []
                for model in model_data:
                    info = self._model_info(model)
                    if info is not None:
                        results.append(info)
                return results
        except Exception as exc:
            self.log(
                LogLevel.WARNING,
                (
                    "LiteLLM model-info listing failed; trying the "
                    f"OpenAI-compatible endpoint: {exc}"
                ),
            )
        return super().list_models()

    def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[LLMTool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
        stream: bool = False,
    ) -> ResponseResult:
        request_extra_body = {
            **self._extra_body,
            **(extra_body or {}),
        }

        return super().generate(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra_body=request_extra_body or None,
            stream=stream,
        )

    def _openai_schema(
        self,
        schema: dict,
        *,
        strict: bool,
    ) -> dict:
        del strict
        return schema
