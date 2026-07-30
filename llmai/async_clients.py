from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from logging import Logger
from types import SimpleNamespace
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncAzureOpenAI, AsyncOpenAI

from llmai.anthropic.client import AnthropicClient
from llmai.azure.client import AzureOpenAIClient
from llmai.bedrock.client import BedrockClient
from llmai.cerebras.client import CerebrasClient
from llmai.chatgpt.client import ChatGPTClient
from llmai.deepseek.client import DeepSeekClient
from llmai.fireworks.client import FireworksClient
from llmai.google.client import GoogleClient
from llmai.litellm.client import LiteLLMClient
from llmai.lmstudio.client import LMStudioClient
from llmai.openai.client import OpenAIClient
from llmai.openrouter.client import OpenRouterClient
from llmai.shared.base import (
    AsyncBaseClient,
    BaseClient,
    run_awaitable_from_worker,
)
from llmai.shared.configs import (
    AnthropicClientConfig,
    AzureOpenAIClientConfig,
    BedrockClientConfig,
    CerebrasClientConfig,
    ChatGPTClientConfig,
    DeepSeekClientConfig,
    FireworksClientConfig,
    GoogleClientConfig,
    LMStudioClientConfig,
    LiteLLMClientConfig,
    OpenAIClientConfig,
    OpenRouterClientConfig,
    TogetherAIClientConfig,
    VertexAIClientConfig,
)
from llmai.shared.errors import raise_llm_error
from llmai.togetherai.client import TogetherAIClient
from llmai.vertex.client import VertexAIClient


_ACTIVE_STREAM_BRIDGES: ContextVar[list[_AsyncIteratorBridge] | None] = (
    ContextVar("llmai_active_stream_bridges", default=None)
)


class _AsyncIteratorBridge:
    def __init__(self, iterator: Any):
        self._source = iterator
        self._iterator = iterator.__aiter__()
        active_bridges = _ACTIVE_STREAM_BRIDGES.get()
        if active_bridges is not None:
            active_bridges.append(self)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return run_awaitable_from_worker(anext(self._iterator))
        except StopAsyncIteration as exc:
            raise StopIteration from exc

    def close(self) -> None:
        close = getattr(self._source, "aclose", None)
        if not callable(close):
            close = getattr(self._iterator, "aclose", None)
        if callable(close):
            run_awaitable_from_worker(close())

    async def aclose(self) -> None:
        close = getattr(self._source, "aclose", None)
        if not callable(close):
            close = getattr(self._iterator, "aclose", None)
        if callable(close):
            await close()


def _resolve_async_value(value: Any) -> Any:
    if inspect.isawaitable(value):
        value = run_awaitable_from_worker(value)
    if hasattr(value, "__aiter__"):
        return _AsyncIteratorBridge(value)
    return value


class _AsyncCallableBridge:
    def __init__(self, callback: Callable[..., Any]):
        self._callback = callback

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_async_value(self._callback(*args, **kwargs))


class _AsyncStreamBridge(_AsyncIteratorBridge):
    def __getattr__(self, name: str) -> Any:
        value = getattr(self._source, name)
        if callable(value):
            return _AsyncCallableBridge(value)
        return value


class _AsyncContextManagerBridge:
    def __init__(self, context_manager: Any):
        self._context_manager = context_manager

    def __enter__(self):
        stream = run_awaitable_from_worker(self._context_manager.__aenter__())
        return _AsyncStreamBridge(stream)

    def __exit__(self, exc_type, exc, traceback):
        return run_awaitable_from_worker(
            self._context_manager.__aexit__(exc_type, exc, traceback)
        )


def _missing_async_resource(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise AttributeError("Provider client does not expose this resource")


def _resource_callback(resource: Any, name: str) -> Callable[..., Any]:
    return getattr(resource, name, _missing_async_resource)


class _AsyncAnthropicMessagesBridge:
    def __init__(self, messages: Any):
        self.create = _AsyncCallableBridge(messages.create)
        self._stream = messages.stream

    def stream(self, *args: Any, **kwargs: Any) -> _AsyncContextManagerBridge:
        return _AsyncContextManagerBridge(self._stream(*args, **kwargs))


def _openai_bridge(client: AsyncOpenAI | AsyncAzureOpenAI) -> Any:
    models = getattr(client, "models", None)
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=_AsyncCallableBridge(client.chat.completions.create)
            )
        ),
        responses=SimpleNamespace(
            create=_AsyncCallableBridge(client.responses.create)
        ),
        models=SimpleNamespace(
            list=_AsyncCallableBridge(_resource_callback(models, "list")),
            retrieve=_AsyncCallableBridge(
                _resource_callback(models, "retrieve")
            ),
        ),
    )


def _anthropic_bridge(client: AsyncAnthropic) -> Any:
    models = getattr(client, "models", None)
    return SimpleNamespace(
        messages=_AsyncAnthropicMessagesBridge(client.messages),
        models=SimpleNamespace(
            list=_AsyncCallableBridge(_resource_callback(models, "list")),
            retrieve=_AsyncCallableBridge(
                _resource_callback(models, "retrieve")
            ),
        ),
    )


def _google_bridge(client: Any) -> Any:
    models = client.aio.models
    return SimpleNamespace(
        models=SimpleNamespace(
            generate_content=_AsyncCallableBridge(
                models.generate_content
            ),
            generate_content_stream=_AsyncCallableBridge(
                models.generate_content_stream
            ),
            get=_AsyncCallableBridge(_resource_callback(models, "get")),
            list=_AsyncCallableBridge(_resource_callback(models, "list")),
        )
    )


async def _close_google_client(client: Any) -> None:
    await client.aio.aclose()
    client.close()


def _create_async_provider_client(
    factory: Callable[..., Any],
    *,
    provider: str,
    **kwargs: Any,
) -> Any:
    try:
        return factory(**kwargs)
    except Exception as exc:
        raise_llm_error(exc, provider=provider)


class _AsyncProviderClient(AsyncBaseClient):
    def __init__(
        self,
        *,
        sync_client: BaseClient,
        provider_client: Any | None = None,
        bridge: Any | None = None,
        async_close: Callable[[], Awaitable[None]] | None = None,
    ):
        if bridge is not None:
            sync_provider_client = getattr(sync_client, "_client", None)
            if sync_provider_client is not provider_client:
                close = getattr(sync_provider_client, "close", None)
                if callable(close):
                    close()
            sync_client._client = bridge
        if async_close is None and provider_client is not None:
            async_close = provider_client.close
        super().__init__(
            sync_client=sync_client,
            async_close=async_close,
        )

    async def _agenerate_stream(self, **kwargs: Any):
        active_bridges: list[_AsyncIteratorBridge] = []
        stream = super()._agenerate_stream(**kwargs)
        try:
            while True:
                token = _ACTIVE_STREAM_BRIDGES.set(active_bridges)
                try:
                    event = await anext(stream)
                except StopAsyncIteration:
                    break
                finally:
                    _ACTIVE_STREAM_BRIDGES.reset(token)
                yield event
        finally:
            await stream.aclose()
            for bridge in active_bridges:
                await bridge.aclose()


class AsyncOpenAIClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: OpenAIClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = OpenAIClient(config=config, logger=logger)
        provider_client = _create_async_provider_client(
            AsyncOpenAI,
            provider="openai",
            base_url=config.base_url,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_openai_bridge(provider_client),
        )


class AsyncAzureOpenAIClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: AzureOpenAIClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = AzureOpenAIClient(config=config, logger=logger)

        async_token_provider = None
        if config.azure_ad_token_provider is not None:
            token_provider = config.azure_ad_token_provider

            async def async_token_provider() -> str:
                value = await asyncio.to_thread(token_provider)
                if inspect.isawaitable(value):
                    value = await value
                return value

        client_kwargs: dict[str, Any] = {
            "api_version": config.api_version,
            "api_key": config.api_key,
            "azure_ad_token": config.azure_ad_token,
            "azure_ad_token_provider": async_token_provider,
            "base_url": config.base_url,
        }
        if config.endpoint is not None:
            client_kwargs["azure_endpoint"] = config.endpoint
        if config.deployment is not None and config.base_url is None:
            client_kwargs["azure_deployment"] = config.deployment

        provider_client = _create_async_provider_client(
            AsyncAzureOpenAI,
            provider="azure",
            **client_kwargs,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_openai_bridge(provider_client),
        )


class AsyncVertexAIClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: VertexAIClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = VertexAIClient(config=config, logger=logger)
        provider_client = sync_client._client
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_google_bridge(provider_client),
            async_close=lambda: _close_google_client(provider_client),
        )


class AsyncChatGPTClient(_AsyncProviderClient):
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
        provider_client = _create_async_provider_client(
            AsyncOpenAI,
            provider="chatgpt",
            base_url=config.base_url or ChatGPTClient.DEFAULT_BASE_URL,
            api_key=config.access_token,
            default_headers=default_headers,
            timeout=120.0,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_openai_bridge(provider_client),
        )


class AsyncDeepSeekClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: DeepSeekClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = DeepSeekClient(config=config, logger=logger)
        provider_client = _create_async_provider_client(
            AsyncOpenAI,
            provider="deepseek",
            base_url=config.base_url or DeepSeekClient.DEFAULT_BASE_URL,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_openai_bridge(provider_client),
        )


class AsyncOpenRouterClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: OpenRouterClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = OpenRouterClient(config=config, logger=logger)
        provider_client = _create_async_provider_client(
            AsyncOpenAI,
            provider="openrouter",
            base_url=config.base_url or OpenRouterClient.DEFAULT_BASE_URL,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_openai_bridge(provider_client),
        )


class AsyncCerebrasClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: CerebrasClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = CerebrasClient(config=config, logger=logger)
        provider_client = _create_async_provider_client(
            AsyncOpenAI,
            provider="cerebras",
            base_url=config.base_url or CerebrasClient.DEFAULT_BASE_URL,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_openai_bridge(provider_client),
        )


class AsyncFireworksClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: FireworksClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = FireworksClient(config=config, logger=logger)
        provider_client = _create_async_provider_client(
            AsyncOpenAI,
            provider="fireworks",
            base_url=config.base_url or FireworksClient.DEFAULT_BASE_URL,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_openai_bridge(provider_client),
        )


class AsyncTogetherAIClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: TogetherAIClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = TogetherAIClient(config=config, logger=logger)
        provider_client = _create_async_provider_client(
            AsyncOpenAI,
            provider="togetherai",
            base_url=config.base_url or TogetherAIClient.DEFAULT_BASE_URL,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_openai_bridge(provider_client),
        )


class AsyncLMStudioClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: LMStudioClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = LMStudioClient(config=config, logger=logger)
        provider_client = _create_async_provider_client(
            AsyncOpenAI,
            provider="lmstudio",
            base_url=sync_client._base_url(config.base_url),
            api_key=config.api_key or LMStudioClient.DEFAULT_API_KEY,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_openai_bridge(provider_client),
        )


class AsyncGoogleClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: GoogleClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = GoogleClient(config=config, logger=logger)
        provider_client = sync_client._client
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_google_bridge(provider_client),
            async_close=lambda: _close_google_client(provider_client),
        )


class AsyncAnthropicClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: AnthropicClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = AnthropicClient(config=config, logger=logger)
        provider_client = _create_async_provider_client(
            AsyncAnthropic,
            provider="anthropic",
            api_key=config.api_key,
            base_url=config.base_url,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_anthropic_bridge(provider_client),
        )


class AsyncBedrockClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: BedrockClientConfig,
        logger: Logger | None = None,
    ):
        super().__init__(sync_client=BedrockClient(config=config, logger=logger))


class AsyncLiteLLMClient(_AsyncProviderClient):
    def __init__(
        self,
        *,
        config: LiteLLMClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = LiteLLMClient(config=config, logger=logger)
        provider_client = _create_async_provider_client(
            AsyncOpenAI,
            provider="litellm",
            base_url=config.base_url,
            api_key=config.api_key or "EMPTY",
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
            bridge=_openai_bridge(provider_client),
        )


__all__ = [
    "AsyncAnthropicClient",
    "AsyncAzureOpenAIClient",
    "AsyncBedrockClient",
    "AsyncCerebrasClient",
    "AsyncChatGPTClient",
    "AsyncDeepSeekClient",
    "AsyncFireworksClient",
    "AsyncGoogleClient",
    "AsyncLiteLLMClient",
    "AsyncLMStudioClient",
    "AsyncOpenAIClient",
    "AsyncOpenRouterClient",
    "AsyncTogetherAIClient",
    "AsyncVertexAIClient",
]
