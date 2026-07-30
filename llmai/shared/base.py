from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import Future
from collections.abc import AsyncIterator, Awaitable
from contextvars import ContextVar
from logging import Logger
from typing import Any, Callable, Literal, overload
from uuid import uuid4

from llmai.shared.logs import LogLevel
from llmai.shared.messages import Message
from llmai.shared.models import ModelInfo, ModelTokenLimits
from llmai.shared.reasoning import ReasoningEffort
from llmai.shared.response_formats import ResponseFormat
from llmai.shared.responses import (
    ResponseResult,
    ResponseContent,
    ResponseStreamEvent,
    ResponseStreamChunk,
    ResponseStreamChunkType,
)
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

    def get_model_context_window(self, *, model: str) -> ModelTokenLimits:
        del model
        return ModelTokenLimits()

    def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement model listing"
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
_ASYNC_EVENT_LOOP: ContextVar[asyncio.AbstractEventLoop] = ContextVar(
    "llmai_async_event_loop"
)
_ACTIVE_PROVIDER_FUTURES: ContextVar[list[Future[Any]] | None] = ContextVar(
    "llmai_active_provider_futures",
    default=None,
)


def _next_stream_event(stream: Any) -> Any:
    try:
        return next(stream)
    except StopIteration:
        return _STREAM_EXHAUSTED


def run_awaitable_from_worker(awaitable: Awaitable[Any]) -> Any:
    """Run provider async I/O on the caller's event loop from a parser thread."""

    loop = _ASYNC_EVENT_LOOP.get()
    future = asyncio.run_coroutine_threadsafe(awaitable, loop)
    active_futures = _ACTIVE_PROVIDER_FUTURES.get()
    if active_futures is not None:
        active_futures.append(future)
    try:
        return future.result()
    finally:
        if active_futures is not None and future in active_futures:
            active_futures.remove(future)


class AsyncBaseClient:
    """Async facade for an llmai provider client.

    Response parsing runs outside the event-loop thread while provider-native
    async transports perform network I/O on the event loop. This keeps sync and
    async provider behavior identical while permitting concurrent requests.
    """

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

    async def aget_model_context_window(
        self,
        *,
        model: str,
    ) -> ModelTokenLimits:
        self._ensure_open()
        return await self._run_in_parser_thread(
            self._sync_client.get_model_context_window,
            model=model,
        )

    async def alist_models(self) -> list[ModelInfo]:
        self._ensure_open()
        return await self._run_in_parser_thread(
            self._sync_client.list_models,
        )

    async def _agenerate_once(self, **kwargs: Any) -> ResponseContent:
        result = await self._run_in_parser_thread(
            self._sync_client.generate,
            **kwargs,
            stream=False,
        )
        if not isinstance(result, ResponseContent):
            raise TypeError("Non-streaming generation returned a stream")
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
        loop_token = _ASYNC_EVENT_LOOP.set(asyncio.get_running_loop())
        active_futures: list[Future[Any]] = []
        futures_token = _ACTIVE_PROVIDER_FUTURES.set(active_futures)
        try:
            return await asyncio.to_thread(callback, *args, **kwargs)
        except asyncio.CancelledError:
            for future in tuple(active_futures):
                future.cancel()
            raise
        finally:
            _ACTIVE_PROVIDER_FUTURES.reset(futures_token)
            _ASYNC_EVENT_LOOP.reset(loop_token)

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
            await asyncio.to_thread(close)

    async def __aenter__(self) -> AsyncBaseClient:
        self._ensure_open()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()
