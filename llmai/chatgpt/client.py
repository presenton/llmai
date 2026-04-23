from __future__ import annotations

from logging import Logger

from openai import Omit, OpenAI

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import ChatGPTClientConfig
from llmai.shared.errors import LLMError, configuration_error, raise_llm_error
from llmai.shared.messages import Message
from llmai.shared.reasoning import ReasoningEffort
from llmai.shared.response_formats import ResponseFormat
from llmai.shared.responses import ResponseContent, ResponseResult
from llmai.shared.tools import LLMTool, ToolChoice

CHATGPT_DEFAULT_INSTRUCTIONS = "Follow the prompt"


class ChatGPTClient(OpenAIClient):
    DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"

    def __init__(
        self,
        *,
        config: ChatGPTClientConfig,
        logger: Logger | None = None,
    ):
        self._logger = logger
        self._api_type = OpenAIApiType.RESPONSES
        self._provide_system_message_as_instructions = True
        self._base_url = config.base_url or self.DEFAULT_BASE_URL
        resolved_access_token = self._resolve_access_token(config.access_token)
        resolved_account_id = _strip_or_none(config.account_id)

        default_headers = {
            "OpenAI-Beta": "responses=experimental",
            "originator": "pi",
        }
        if resolved_account_id is not None:
            default_headers["chatgpt-account-id"] = resolved_account_id

        try:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=resolved_access_token,
                default_headers=default_headers,
                timeout=120.0,
            )
        except Exception as exc:
            raise_llm_error(exc, provider="chatgpt")

        if self._logger:
            self._logger.info("ChatGPT client created")
            self._logger.info("Base URL: %s", self._base_url)

    def _messages_to_openai_responses_instructions(
        self,
        messages: list[Message],
    ) -> str | None:
        return super()._messages_to_openai_responses_instructions(
            messages
        ) or CHATGPT_DEFAULT_INSTRUCTIONS

    def _get_openai_responses_max_output_tokens_or_omit(
        self,
        max_tokens: int | None,
    ) -> Omit:
        del max_tokens
        return Omit()

    def _get_openai_responses_temperature_or_omit(
        self,
        temperature: float | None,
    ) -> Omit:
        del temperature
        return Omit()

    def _generate_responses_once(
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
    ) -> ResponseContent:
        completion_chunk = None

        for chunk in super()._generate_responses_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        ):
            if getattr(chunk, "type", None) == "completion":
                completion_chunk = chunk

        if completion_chunk is None:
            raise LLMError(
                500,
                "No completion returned from streamed ChatGPT response",
                provider="chatgpt",
            )

        return ResponseContent(
            content=completion_chunk.content,
            thinking=completion_chunk.thinking,
            messages=completion_chunk.messages,
            tool_calls=completion_chunk.tool_calls,
            usage=completion_chunk.usage,
            duration_seconds=completion_chunk.duration_seconds,
        )

    def _resolve_access_token(self, access_token: str | None) -> str:
        resolved_access_token = _strip_or_none(access_token)
        if resolved_access_token is not None:
            return resolved_access_token

        raise configuration_error(
            "Missing ChatGPT access token. Provide config.access_token",
            provider="chatgpt",
        )

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
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": True,
            **(extra_body or {}),
        }
        if response_format is None and "text" not in request_extra_body:
            request_extra_body["text"] = {"verbosity": "medium"}

        return super().generate(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra_body=request_extra_body,
            stream=stream,
        )


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None
