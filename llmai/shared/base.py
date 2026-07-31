from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections.abc import AsyncIterator, Awaitable
import contextvars
from functools import partial
import os
from logging import Logger
from typing import Any, Callable, Literal, overload
from uuid import uuid4

from llmai.shared.logs import LogLevel
from llmai.shared.messages import Message
from llmai.shared.reasoning import ReasoningEffort
from llmai.shared.response_formats import ResponseFormat
from llmai.shared.responses import (
    ResponseResult,
    ResponseContent,
    ResponseStreamEvent,
    ResponseStreamChunk,
    ResponseStreamChunkType,
)
from llmai.shared.errors import configuration_error
from llmai.shared.tools import LLMTool, ToolChoice


class BaseClient(ABC):
    def __init__(self, *, logger: Logger | None = None):
        self._logger = logger

    def log(self, level: LogLevel, message: Any) -> None:
        if not self._logger:
            return

        match level:
            case LogLevel.INFO:
                self._logger.info(message)
            case LogLevel.WARNING:
                self._logger.warning(message)
            case LogLevel.ERROR:
                self._logger.error(message)

    def _dump_value(self, value: Any) -> Any:
        if value is None:
            return None

        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)

        if isinstance(value, dict):
            return {
                key: self._dump_value(item)
                for key, item in value.items()
                if item is not None
            }

        if isinstance(value, (list, tuple)):
            return [self._dump_value(item) for item in value]

        if hasattr(value, "__dict__"):
            return {
                key: self._dump_value(item)
                for key, item in vars(value).items()
                if not key.startswith("_") and item is not None
            }

        return value

    def _dump_model(self, value: Any) -> dict[str, Any]:
        dumped = self._dump_value(value)
        return dumped if isinstance(dumped, dict) else {}

    def list_available_models(self) -> list[str]:
        """List models currently available to this configured provider account."""

        provider = getattr(self, "PROVIDER_NAME", self.__class__.__name__)
        raise configuration_error(
            f"Live model listing is not supported for provider {provider!r}.",
            provider=str(provider),
        )

    def _tool_call_id(self, tool_id: str | None = None) -> str:
        if tool_id:
            return tool_id
        return f"call_{uuid4().hex}"

    def _transition_stream_chunk(
        self,
        *,
        current_chunk_type: ResponseStreamChunkType | None,
        next_chunk_type: ResponseStreamChunkType,
        current_tool: str | None = None,
        next_tool: str | None = None,
    ) -> tuple[ResponseStreamChunkType, str | None, list[ResponseStreamChunk]]:
        if current_chunk_type == next_chunk_type and current_tool == next_tool:
            return current_chunk_type, current_tool, []

        chunks: list[ResponseStreamChunk] = []
        if current_chunk_type is not None:
            chunks.append(
                ResponseStreamChunk(
                    chunk_type=current_chunk_type,
                    event="end",
                    tool=current_tool if current_chunk_type == "tool" else None,
                )
            )

        chunks.append(
            ResponseStreamChunk(
                chunk_type=next_chunk_type,
                event="start",
                tool=next_tool if next_chunk_type == "tool" else None,
            )
        )
        return next_chunk_type, next_tool if next_chunk_type == "tool" else None, chunks

    def _close_stream_chunk(
        self,
        *,
        current_chunk_type: ResponseStreamChunkType | None,
        current_tool: str | None = None,
    ) -> ResponseStreamChunk | None:
        if current_chunk_type is None:
            return None

        return ResponseStreamChunk(
            chunk_type=current_chunk_type,
            event="end",
            tool=current_tool if current_chunk_type == "tool" else None,
        )

    @abstractmethod
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
        raise NotImplementedError


_STREAM_EXHAUSTED = object()
_ASYNC_PARSER_EXECUTOR_MAX_WORKERS = max(16, min(64, (os.cpu_count() or 1) * 8))
_ASYNC_PARSER_EXECUTOR = ThreadPoolExecutor(
    max_workers=_ASYNC_PARSER_EXECUTOR_MAX_WORKERS,
    thread_name_prefix="llmai-parser",
)


def _next_stream_event(stream: Any) -> Any:
    try:
        return next(stream)
    except StopIteration:
        return _STREAM_EXHAUSTED


class AsyncBaseClient:
    """Async fallback facade for an llmai sync provider client."""

    def __init__(
        self,
        *,
        sync_client: BaseClient,
        async_close: Callable[[], Awaitable[None]] | None = None,
    ):
        self._sync_client = sync_client
        self._async_close = async_close
        self._closed = False

    @overload
    def agenerate(
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
        stream: Literal[False] = False,
    ) -> Awaitable[ResponseContent]: ...

    @overload
    def agenerate(
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
        stream: Literal[True],
    ) -> AsyncIterator[ResponseStreamEvent]: ...

    def agenerate(
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
    ) -> Awaitable[ResponseContent] | AsyncIterator[ResponseStreamEvent]:
        self._ensure_open()
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "extra_body": extra_body,
        }
        if stream:
            return self._agenerate_stream(**kwargs)
        return self._agenerate_once(**kwargs)

    async def _agenerate_once(self, **kwargs: Any) -> ResponseContent:
        result = await self._run_in_parser_thread(
            self._sync_client.generate,
            **kwargs,
            stream=False,
        )
        if not isinstance(result, ResponseContent):
            raise TypeError("Non-streaming generation returned a stream")
        return result

    async def alist_available_models(self) -> list[str]:
        """Asynchronously list models available to this provider account."""

        self._ensure_open()
        result = await self._run_in_parser_thread(
            self._sync_client.list_available_models
        )
        if not isinstance(result, list) or not all(
            isinstance(model, str) for model in result
        ):
            raise TypeError("Provider model listing returned an invalid result")
        return result

    async def _agenerate_stream(
        self,
        **kwargs: Any,
    ) -> AsyncIterator[ResponseStreamEvent]:
        stream = None
        try:
            stream = await self._run_in_parser_thread(
                self._sync_client.generate,
                **kwargs,
                stream=True,
            )
            while True:
                event = await self._run_in_parser_thread(
                    _next_stream_event,
                    stream,
                )
                if event is _STREAM_EXHAUSTED:
                    break
                yield event
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                await self._run_in_parser_thread(close)

    async def _run_in_parser_thread(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        context = contextvars.copy_context()
        loop = asyncio.get_running_loop()
        call = partial(callback, *args, **kwargs)
        return await loop.run_in_executor(
            _ASYNC_PARSER_EXECUTOR,
            context.run,
            call,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Async client is closed")

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True
        if self._async_close is not None:
            await self._async_close()
            return

        provider_client = getattr(self._sync_client, "_client", None)
        close = getattr(provider_client, "close", None)
        if callable(close):
            await self._run_in_parser_thread(close)

    async def __aenter__(self) -> AsyncBaseClient:
        self._ensure_open()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()
