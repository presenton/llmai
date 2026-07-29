from __future__ import annotations

from logging import Logger
from typing import TypeVar

from llmai.async_clients import (
    AsyncAnthropicClient,
    AsyncAzureOpenAIClient,
    AsyncBedrockClient,
    AsyncCerebrasClient,
    AsyncChatGPTClient,
    AsyncDeepSeekClient,
    AsyncFireworksClient,
    AsyncGoogleClient,
    AsyncLiteLLMClient,
    AsyncLMStudioClient,
    AsyncOpenAIClient,
    AsyncOpenRouterClient,
    AsyncTogetherAIClient,
    AsyncVertexAIClient,
)
from llmai.anthropic.client import AnthropicClient
from llmai.azure.client import AzureOpenAIClient
from llmai.bedrock.client import BedrockClient
from llmai.cerebras.client import CerebrasClient
from llmai.chatgpt.client import ChatGPTClient
from llmai.deepseek.client import DeepSeekClient
from llmai.fireworks.client import FireworksClient
from llmai.google.client import GoogleClient
from llmai.lmstudio.client import LMStudioClient
from llmai.litellm.client import LiteLLMClient
from llmai.openai.client import OpenAIClient
from llmai.openrouter.client import OpenRouterClient
from llmai.shared.base import AsyncBaseClient, BaseClient
from llmai.shared.configs import (
    AnthropicClientConfig,
    AzureOpenAIClientConfig,
    BedrockClientConfig,
    CerebrasClientConfig,
    ChatGPTClientConfig,
    ClientConfig,
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
from llmai.shared.errors import configuration_error
from llmai.shared.providers import LLMProvider
from llmai.togetherai.client import TogetherAIClient
from llmai.vertex.client import VertexAIClient

__all__ = ["LLMProvider", "get_async_client", "get_client"]

TConfig = TypeVar("TConfig")


def get_client(
    *,
    config: ClientConfig,
    logger: Logger | None = None,
) -> BaseClient:
    provider = getattr(config, "provider", None)

    if provider == "openai":
        return OpenAIClient(
            config=_require_config(
                provider,
                config,
                OpenAIClientConfig,
            ),
            logger=logger,
        )

    if provider == "azure":
        return AzureOpenAIClient(
            config=_require_config(
                provider,
                config,
                AzureOpenAIClientConfig,
            ),
            logger=logger,
        )

    if provider == "vertex":
        return VertexAIClient(
            config=_require_config(
                provider,
                config,
                VertexAIClientConfig,
            ),
            logger=logger,
        )

    if provider == "chatgpt":
        return ChatGPTClient(
            config=_require_config(
                provider,
                config,
                ChatGPTClientConfig,
            ),
            logger=logger,
        )

    if provider == "deepseek":
        return DeepSeekClient(
            config=_require_config(
                provider,
                config,
                DeepSeekClientConfig,
            ),
            logger=logger,
        )

    if provider == "openrouter":
        return OpenRouterClient(
            config=_require_config(
                provider,
                config,
                OpenRouterClientConfig,
            ),
            logger=logger,
        )

    if provider == "cerebras":
        return CerebrasClient(
            config=_require_config(
                provider,
                config,
                CerebrasClientConfig,
            ),
            logger=logger,
        )

    if provider == "fireworks":
        return FireworksClient(
            config=_require_config(
                provider,
                config,
                FireworksClientConfig,
            ),
            logger=logger,
        )

    if provider == "togetherai":
        return TogetherAIClient(
            config=_require_config(
                provider,
                config,
                TogetherAIClientConfig,
            ),
            logger=logger,
        )

    if provider == "lmstudio":
        return LMStudioClient(
            config=_require_config(
                provider,
                config,
                LMStudioClientConfig,
            ),
            logger=logger,
        )

    if provider == "google":
        return GoogleClient(
            config=_require_config(
                provider,
                config,
                GoogleClientConfig,
            ),
            logger=logger,
        )

    if provider == "anthropic":
        return AnthropicClient(
            config=_require_config(
                provider,
                config,
                AnthropicClientConfig,
            ),
            logger=logger,
        )

    if provider == "bedrock":
        return BedrockClient(
            config=_require_config(
                provider,
                config,
                BedrockClientConfig,
            ),
            logger=logger,
        )

    if provider == "litellm":
        return LiteLLMClient(
            config=_require_config(
                provider,
                config,
                LiteLLMClientConfig,
            ),
            logger=logger,
        )

    supported = ", ".join(each.value for each in LLMProvider)
    raise configuration_error(
        f"Unsupported client config provider: {provider!r}. Expected one of: {supported}",
        provider=None,
    )


def get_async_client(
    *,
    config: ClientConfig,
    logger: Logger | None = None,
) -> AsyncBaseClient:
    provider = getattr(config, "provider", None)
    clients = {
        "openai": (OpenAIClientConfig, AsyncOpenAIClient),
        "azure": (AzureOpenAIClientConfig, AsyncAzureOpenAIClient),
        "vertex": (VertexAIClientConfig, AsyncVertexAIClient),
        "chatgpt": (ChatGPTClientConfig, AsyncChatGPTClient),
        "deepseek": (DeepSeekClientConfig, AsyncDeepSeekClient),
        "openrouter": (OpenRouterClientConfig, AsyncOpenRouterClient),
        "cerebras": (CerebrasClientConfig, AsyncCerebrasClient),
        "fireworks": (FireworksClientConfig, AsyncFireworksClient),
        "togetherai": (TogetherAIClientConfig, AsyncTogetherAIClient),
        "lmstudio": (LMStudioClientConfig, AsyncLMStudioClient),
        "google": (GoogleClientConfig, AsyncGoogleClient),
        "anthropic": (AnthropicClientConfig, AsyncAnthropicClient),
        "bedrock": (BedrockClientConfig, AsyncBedrockClient),
        "litellm": (LiteLLMClientConfig, AsyncLiteLLMClient),
    }
    client_entry = clients.get(provider)
    if client_entry is None:
        supported = ", ".join(each.value for each in LLMProvider)
        raise configuration_error(
            (
                f"Unsupported client config provider: {provider!r}. "
                f"Expected one of: {supported}"
            ),
            provider=None,
        )

    config_type, client_type = client_entry
    return client_type(
        config=_require_config(provider, config, config_type),
        logger=logger,
    )


def _require_config(
    provider: str,
    config: ClientConfig,
    config_type: type[TConfig],
) -> TConfig:
    if isinstance(config, config_type):
        return config

    raise configuration_error(
        (
            f"Invalid config for provider {provider!r}. "
            f"Expected {config_type.__name__}, got {type(config).__name__}"
        ),
        provider=provider,
    )
