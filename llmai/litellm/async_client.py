from __future__ import annotations

from logging import Logger

from openai import AsyncOpenAI

from llmai.litellm.client import LiteLLMClient
from llmai.openai.async_client import (
    AsyncOpenAICompatibleClient,
    create_async_provider_client,
)
from llmai.shared.configs import LiteLLMClientConfig


class AsyncLiteLLMClient(AsyncOpenAICompatibleClient):
    def __init__(
        self,
        *,
        config: LiteLLMClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = LiteLLMClient(config=config, logger=logger)
        provider_client = create_async_provider_client(
            AsyncOpenAI,
            provider="litellm",
            base_url=config.base_url,
            api_key=config.api_key or "EMPTY",
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
        )

    def _prepare_extra_body(self, extra_body: dict | None) -> dict | None:
        request_extra_body = {
            **self._parser._extra_body,
            **(extra_body or {}),
        }
        return request_extra_body or None
