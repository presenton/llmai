from __future__ import annotations

import asyncio
import contextvars
import inspect
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from logging import Logger
from time import monotonic
from typing import Any, Literal, overload
from uuid import uuid4

from llmai.shared.errors import configuration_error
from llmai.shared.generation import (
    GenerationDefaults,
    GenerationProfile,
    PreparedGeneration,
    prepare_generation,
)
from llmai.shared.logs import LogLevel
from llmai.shared.messages import AssistantMessage, Message
from llmai.shared.reasoning import (
    ReasoningConfig,
    ReasoningEffort,
    ReasoningHistoryMode,
)
from llmai.shared.response_formats import ResponseFormat
from llmai.shared.responses import (
    ResponseContent,
    ResponseResult,
    ResponseStreamChunk,
    ResponseStreamChunkType,
    ResponseStreamEvent,
)
from llmai.shared.tools import LLMTool, ToolChoice


class BaseClient(ABC):
    def __init__(
        self,
        *,
        logger: Logger | None = None,
        generation_defaults: GenerationDefaults | None = None,
    ):
        self._logger = logger
        self._generation_defaults = generation_defaults or GenerationDefaults()
        self._compatibility_cache: dict[tuple[str, str], str] = {}
        self._capability_discovery_cache: dict[
            str, tuple[float, dict[str, Any] | None]
        ] = {}

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

    def get_model_capabilities(self, model: str):
        from llmai.capabilities import get_model_capabilities
        from llmai.shared.generation import CapabilitySource

        capabilities = get_model_capabilities(
            model,
            provider=getattr(self, "PROVIDER_NAME", None),
            overrides=self._generation_defaults.capability_overrides,
        )
        if not self._generation_defaults.discover_capabilities or getattr(
            self, "_skip_live_discovery", False
        ):
            return capabilities

        unknown = any(
            value.supported is None
            for value in (
                capabilities.reasoning,
                capabilities.tool_call.support,
                capabilities.max_output_tokens,
            )
        )
        if not (unknown or self._bundled_capabilities_are_stale(capabilities.raw)):
            return capabilities

        live = self._cached_live_capabilities(model)
        if not live:
            return capabilities
        application = self._generation_defaults.capability_overrides
        qualified = f"{getattr(self, 'PROVIDER_NAME', '')}:{model}"
        app_override = application.get(qualified, application.get(model, {}))
        mapping = {
            "max_output_tokens": capabilities.max_output_tokens,
            "reasoning": capabilities.reasoning,
            "reasoning_levels": capabilities.reasoning_levels,
            "reasoning_budget": capabilities.reasoning_budget,
            "reasoning_interleaved": capabilities.reasoning_interleaved,
            "tool_call": capabilities.tool_call.support,
            "parallel_tool_calls": capabilities.tool_call.parallel,
            "streaming_tool_calls": capabilities.tool_call.streaming,
        }
        for key, value in live.items():
            if key in app_override or key not in mapping:
                continue
            target = mapping[key]
            target.value = value
            target.source = CapabilitySource.LIVE
            target.fresh = True
            if isinstance(value, bool):
                target.status = "supported" if value else "unsupported"
            elif value is not None:
                target.status = "supported"
        return capabilities

    def _bundled_capabilities_are_stale(self, metadata: dict[str, Any]) -> bool:
        updated = metadata.get("last_updated")
        if not isinstance(updated, str):
            return True
        try:
            age = (
                datetime.now(timezone.utc).date()
                - datetime.fromisoformat(updated).date()
            )
        except ValueError:
            return True
        return age.days > self._generation_defaults.bundled_metadata_max_age_days

    def _cached_live_capabilities(self, model: str) -> dict[str, Any] | None:
        cached = self._capability_discovery_cache.get(model)
        if cached and cached[0] > monotonic():
            return cached[1]
        try:
            result = self._discover_model_capabilities(model)
        except Exception as exc:
            result = None
            self.log(
                LogLevel.WARNING,
                {
                    "code": "capability_discovery_failed",
                    "provider": getattr(self, "PROVIDER_NAME", None),
                    "model": model,
                    "message": str(exc),
                },
            )
        ttl = (
            self._generation_defaults.discovery_success_ttl_seconds
            if result is not None
            else self._generation_defaults.discovery_failure_ttl_seconds
        )
        self._capability_discovery_cache[model] = (monotonic() + ttl, result)
        return result

    def _discover_model_capabilities(self, model: str) -> dict[str, Any] | None:
        del model
        return None

    def supports_tool_call(self, model: str) -> bool | None:
        return self.get_model_capabilities(model).tool_call.support.supported

    def supports_thinking(self, model: str) -> bool | None:
        return self.get_model_capabilities(model).reasoning.supported

    def get_reasoning_levels(self, model: str) -> list[str]:
        value = self.get_model_capabilities(model).reasoning_levels.value
        return list(value) if isinstance(value, list) else []

    def require_tool_call_support(self, model: str) -> None:
        supported = self.supports_tool_call(model)
        if supported is False:
            raise configuration_error(
                f"Model {model!r} does not support tool calls",
                provider=getattr(self, "PROVIDER_NAME", None),
            )
        if supported is None:
            raise configuration_error(
                f"Tool-call support for model {model!r} is unknown",
                provider=getattr(self, "PROVIDER_NAME", None),
            )

    def prepare_generation(
        self,
        *,
        model: str,
        profile: GenerationProfile | str | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        tools_requested: bool = False,
    ) -> PreparedGeneration:
        prepared = prepare_generation(
            model=model,
            provider=getattr(self, "PROVIDER_NAME", None),
            defaults=self._generation_defaults,
            profile=profile,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            tools_requested=tools_requested,
            capabilities=self.get_model_capabilities(model),
        )
        for warning in prepared.warnings:
            self.log(LogLevel.WARNING, warning.model_dump())
        return prepared

    def _prepare_reasoning_history(
        self,
        messages: list[Message],
        mode: ReasoningHistoryMode,
    ) -> list[Message]:
        if mode != ReasoningHistoryMode.DISABLED:
            return messages

        prepared: list[Message] = []
        for message in messages:
            if not isinstance(message, AssistantMessage):
                prepared.append(message)
                continue
            assistant = message.model_copy(deep=True)
            assistant.thinking = None
            for tool_call in assistant.tool_calls:
                tool_call.thought_signature = None
            prepared.append(assistant)
        return prepared

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
        max_output_tokens: int | None = None,
        profile: GenerationProfile | str | None = None,
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
        stream: bool = False,
    ) -> ResponseResult:
        raise NotImplementedError


class AsyncBaseClient(ABC):
    """Base class for provider clients with native async request paths."""

    def __init__(
        self,
        *,
        sync_client: BaseClient,
        async_close: Callable[[], Awaitable[None]] | None = None,
    ):
        self._sync_client = sync_client
        self._sync_client._skip_live_discovery = True
        self._async_close = async_close
        self._closed = False

    def get_model_capabilities(self, model: str):
        return self._sync_client.get_model_capabilities(model)

    def supports_tool_call(self, model: str) -> bool | None:
        return self._sync_client.supports_tool_call(model)

    def supports_thinking(self, model: str) -> bool | None:
        return self._sync_client.supports_thinking(model)

    def get_reasoning_levels(self, model: str) -> list[str]:
        return self._sync_client.get_reasoning_levels(model)

    def require_tool_call_support(self, model: str) -> None:
        self._sync_client.require_tool_call_support(model)

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
        max_output_tokens: int | None = None,
        profile: GenerationProfile | str | None = None,
        reasoning: ReasoningConfig | None = None,
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
        max_output_tokens: int | None = None,
        profile: GenerationProfile | str | None = None,
        reasoning: ReasoningConfig | None = None,
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
        max_output_tokens: int | None = None,
        profile: GenerationProfile | str | None = None,
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
        stream: bool = False,
    ) -> Awaitable[ResponseContent] | AsyncIterator[ResponseStreamEvent]:
        self._ensure_open()
        prepare = getattr(self._sync_client, "prepare_generation", None)
        if callable(prepare):
            prepared = prepare(
                model=model,
                profile=profile,
                max_tokens=max_tokens,
                max_output_tokens=max_output_tokens,
                reasoning=reasoning,
                reasoning_effort=reasoning_effort,
                tools_requested=bool(tools),
            )
            max_tokens = prepared.max_output_tokens
            reasoning_effort = prepared.reasoning_effort
            messages = self._sync_client._prepare_reasoning_history(
                messages, prepared.reasoning.history
            )
        elif max_output_tokens is not None:
            if max_tokens is not None:
                raise ValueError("Pass only one of max_tokens or max_output_tokens")
            max_tokens = max_output_tokens
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

    @abstractmethod
    async def _agenerate_once(self, **kwargs: Any) -> ResponseContent:
        raise NotImplementedError

    async def alist_available_models(self) -> list[str]:
        """Asynchronously list models available to this provider account."""

        self._ensure_open()
        provider = getattr(
            self._sync_client,
            "PROVIDER_NAME",
            self.__class__.__name__,
        )
        raise configuration_error(
            f"Live model listing is not supported for provider {provider!r}.",
            provider=str(provider),
        )

    @abstractmethod
    async def _agenerate_stream(
        self,
        **kwargs: Any,
    ) -> AsyncIterator[ResponseStreamEvent]:
        raise NotImplementedError
        yield

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
            result = close()
            if inspect.isawaitable(result):
                await result

    async def __aenter__(self) -> AsyncBaseClient:
        self._ensure_open()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()
