from __future__ import annotations

from logging import Logger

from llmai.google.client import GoogleClient
from llmai.shared.base import BaseClient
from llmai.shared.configs import VertexAIClientConfig


class VertexAIClient(GoogleClient):
    PROVIDER_NAME = "vertex"
    PROVIDER_LABEL = "Vertex AI"

    def __init__(
        self,
        *,
        config: VertexAIClientConfig,
        logger: Logger | None = None,
    ):
        BaseClient.__init__(self, logger=logger)
        self._client = self._create_genai_client(
            vertexai=True,
            api_key=config.api_key,
            credentials=config.credentials,
            project=config.project,
            location=config.location,
            http_options=self._http_options(config.base_url),
        )
