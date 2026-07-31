from __future__ import annotations

from logging import Logger

from openai import AsyncOpenAI

from llmai.lmstudio.client import LMStudioClient
from llmai.openai.async_client import (
    AsyncOpenAICompatibleClient,
    create_async_provider_client,
)
from llmai.shared.configs import LMStudioClientConfig


class AsyncLMStudioClient(AsyncOpenAICompatibleClient):
    def __init__(
        self,
        *,
        config: LMStudioClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = LMStudioClient(config=config, logger=logger)
        provider_client = create_async_provider_client(
            AsyncOpenAI,
            provider="lmstudio",
            base_url=sync_client._base_url(config.base_url),
            api_key=config.api_key or LMStudioClient.DEFAULT_API_KEY,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
        )
