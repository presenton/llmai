from __future__ import annotations

from logging import Logger
from typing import TypeVar

from llmai.anthropic.client import AnthropicClient
from llmai.azure.client import AzureOpenAIClient
from llmai.bedrock.client import BedrockClient
from llmai.chatgpt.client import ChatGPTClient
from llmai.deepseek.client import DeepSeekClient
from llmai.google.client import GoogleClient
from llmai.openai.client import OpenAIClient
from llmai.shared.base import BaseClient
from llmai.shared.configs import (
    AnthropicClientConfig,
    AzureOpenAIClientConfig,
    BedrockClientConfig,
    ChatGPTClientConfig,
    ClientConfig,
    DeepSeekClientConfig,
    GoogleClientConfig,
    OpenAIClientConfig,
    VertexAIClientConfig,
)
from llmai.shared.errors import configuration_error
from llmai.shared.providers import LLMProvider
from llmai.vertex.client import VertexAIClient

__all__ = ["LLMProvider", "get_client"]

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

    supported = ", ".join(each.value for each in LLMProvider)
    raise configuration_error(
        f"Unsupported client config provider: {provider!r}. Expected one of: {supported}",
        provider=None,
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
