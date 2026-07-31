from __future__ import annotations

import inspect
from logging import Logger
from typing import Any

from openai import AsyncAzureOpenAI

from llmai.azure.client import AzureOpenAIClient
from llmai.openai.async_client import (
    AsyncOpenAICompatibleClient,
    create_async_provider_client,
)
from llmai.shared.configs import AzureOpenAIClientConfig


class AsyncAzureOpenAIClient(AsyncOpenAICompatibleClient):
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
                value = token_provider()
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

        provider_client = create_async_provider_client(
            AsyncAzureOpenAI,
            provider="azure",
            **client_kwargs,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
        )
