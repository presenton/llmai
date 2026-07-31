from __future__ import annotations

from collections.abc import AsyncIterator
from logging import Logger
from typing import Any

from openai import AsyncOpenAI

from llmai.chatgpt.client import ChatGPTClient
from llmai.openai.async_client import (
    aiter_openai_responses_stream,
    create_async_provider_client,
)
from llmai.shared.async_utils import close_sync_provider_client
from llmai.shared.base import AsyncBaseClient
from llmai.shared.configs import ChatGPTClientConfig
from llmai.shared.errors import LLMError
from llmai.shared.messages import Message
from llmai.shared.responses import ResponseContent, ResponseStreamEvent


class AsyncChatGPTClient(AsyncBaseClient):
    def __init__(
        self,
        *,
        config: ChatGPTClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = ChatGPTClient(config=config, logger=logger)
        default_headers = {
            "OpenAI-Beta": "responses=experimental",
            "originator": "pi",
        }
        if config.account_id is not None:
            default_headers["chatgpt-account-id"] = config.account_id.strip()
        provider_client = create_async_provider_client(
            AsyncOpenAI,
            provider="chatgpt",
            base_url=config.base_url or ChatGPTClient.DEFAULT_BASE_URL,
            api_key=config.access_token,
            default_headers=default_headers,
            timeout=120.0,
        )
        self._provider_client = provider_client
        close_sync_provider_client(sync_client, provider_client)
        super().__init__(
            sync_client=sync_client,
            async_close=provider_client.close,
        )

    @property
    def _parser(self) -> ChatGPTClient:
        return self._sync_client

    async def _agenerate_once(self, **kwargs: Any) -> ResponseContent:
        completion_chunk = None
        async for chunk in self._agenerate_stream(**kwargs):
            if getattr(chunk, "type", None) == "completion":
                completion_chunk = chunk

        if completion_chunk is None:
            raise LLMError(
                500,
                "No completion returned from streamed ChatGPT response",
                provider=self._parser.PROVIDER_NAME,
            )

        return ResponseContent(
            content=completion_chunk.content,
            thinking=completion_chunk.thinking,
            messages=completion_chunk.messages,
            tool_calls=completion_chunk.tool_calls,
            usage=completion_chunk.usage,
            duration_seconds=completion_chunk.duration_seconds,
        )

    async def _agenerate_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any | None = None,
        response_format: Any | None = None,
        max_tokens: int | None = None,
        reasoning_effort: Any | None = None,
        extra_body: dict | None = None,
    ) -> AsyncIterator[ResponseStreamEvent]:
        del temperature
        del max_tokens

        request_extra_body = {
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": True,
            **(extra_body or {}),
        }
        if response_format is None and "text" not in request_extra_body:
            request_extra_body["text"] = {"verbosity": "medium"}

        request_kwargs = self._parser._responses_request_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            extra_body=request_extra_body,
            stream=True,
        )
        async for event in aiter_openai_responses_stream(
            parser=self._parser,
            provider_client=self._provider_client,
            request_kwargs=request_kwargs,
            messages=messages,
            response_format=response_format,
        ):
            yield event
