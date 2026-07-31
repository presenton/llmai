from __future__ import annotations

from logging import Logger

from openai import AsyncOpenAI

from llmai.openai.async_client import (
    AsyncOpenAICompatibleClient,
    create_async_provider_client,
)
from llmai.openrouter.client import OpenRouterClient
from llmai.shared.configs import OpenRouterClientConfig


class AsyncOpenRouterClient(AsyncOpenAICompatibleClient):
    def __init__(
        self,
        *,
        config: OpenRouterClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = OpenRouterClient(config=config, logger=logger)
        provider_client = create_async_provider_client(
            AsyncOpenAI,
            provider="openrouter",
            base_url=config.base_url or OpenRouterClient.DEFAULT_BASE_URL,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
        )
