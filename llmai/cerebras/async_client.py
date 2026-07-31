from __future__ import annotations

from logging import Logger

from openai import AsyncOpenAI

from llmai.cerebras.client import CerebrasClient
from llmai.openai.async_client import (
    AsyncOpenAICompatibleClient,
    create_async_provider_client,
)
from llmai.shared.configs import CerebrasClientConfig


class AsyncCerebrasClient(AsyncOpenAICompatibleClient):
    def __init__(
        self,
        *,
        config: CerebrasClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = CerebrasClient(config=config, logger=logger)
        provider_client = create_async_provider_client(
            AsyncOpenAI,
            provider="cerebras",
            base_url=config.base_url or CerebrasClient.DEFAULT_BASE_URL,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
        )
