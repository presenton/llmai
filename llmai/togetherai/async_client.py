from __future__ import annotations

from logging import Logger

import httpx
from openai import AsyncOpenAI

from llmai.openai.async_client import (
    AsyncOpenAICompatibleClient,
    create_async_provider_client,
)
from llmai.shared.configs import TogetherAIClientConfig
from llmai.shared.errors import LLMError, raise_llm_error
from llmai.shared.model_listing import openai_compatible_model_ids
from llmai.togetherai.client import TogetherAIClient


class AsyncTogetherAIClient(AsyncOpenAICompatibleClient):
    def __init__(
        self,
        *,
        config: TogetherAIClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = TogetherAIClient(config=config, logger=logger)
        provider_client = create_async_provider_client(
            AsyncOpenAI,
            provider="togetherai",
            base_url=config.base_url or TogetherAIClient.DEFAULT_BASE_URL,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
        )

    async def alist_available_models(self) -> list[str]:
        self._ensure_open()
        parser = self._parser
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{parser._models_base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {parser._models_api_key}"},
                )
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise LLMError(
                        502,
                        "Together AI returned an invalid model-list response.",
                        provider=parser.PROVIDER_NAME,
                        cause=exc,
                    ) from exc

            models = openai_compatible_model_ids(payload)
            if models is None:
                raise LLMError(
                    502,
                    "Together AI returned an invalid model-list response.",
                    provider=parser.PROVIDER_NAME,
                )
            return models
        except Exception as exc:
            raise_llm_error(exc, provider=parser.PROVIDER_NAME)
