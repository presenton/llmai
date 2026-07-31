from __future__ import annotations

from logging import Logger

from llmai.google.async_client import AsyncGoogleCompatibleClient
from llmai.shared.configs import VertexAIClientConfig
from llmai.vertex.client import VertexAIClient


class AsyncVertexAIClient(AsyncGoogleCompatibleClient):
    def __init__(
        self,
        *,
        config: VertexAIClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = VertexAIClient(config=config, logger=logger)
        super().__init__(
            sync_client=sync_client,
            provider_client=sync_client._client,
        )
