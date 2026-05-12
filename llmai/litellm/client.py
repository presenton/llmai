from __future__ import annotations

from logging import Logger

from openai import OpenAI

from llmai.openai.client import OpenAIClient
from llmai.shared.base import BaseClient
from llmai.shared.configs import LiteLLMClientConfig
from llmai.shared.errors import configuration_error, raise_llm_error
from llmai.shared.messages import Message
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
                api_key=config.api_key or "EMPTY",
            )
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

        if self._logger:
            self._logger.info("%s client created", self.PROVIDER_LABEL)
            self._logger.info("Base URL: %s", config.base_url)

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

    # def _openai_schema(
    #     self,
    #     schema: dict,
    #     *,
    #     strict: bool,
    # ) -> dict:
    #     del strict
    #     return schema
