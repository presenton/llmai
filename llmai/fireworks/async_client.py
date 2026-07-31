from __future__ import annotations

from logging import Logger

from openai import AsyncOpenAI

from llmai.fireworks.client import FireworksClient
from llmai.openai.async_client import (
    AsyncOpenAICompatibleClient,
    create_async_provider_client,
)
from llmai.shared.configs import FireworksClientConfig


class AsyncFireworksClient(AsyncOpenAICompatibleClient):
    def __init__(
        self,
        *,
        config: FireworksClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = FireworksClient(config=config, logger=logger)
        provider_client = create_async_provider_client(
            AsyncOpenAI,
            provider="fireworks",
            base_url=config.base_url or FireworksClient.DEFAULT_BASE_URL,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
        )
