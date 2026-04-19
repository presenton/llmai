from __future__ import annotations

import os
from logging import Logger

from google.auth.credentials import Credentials

from llmai.google.client import GoogleClient
from llmai.shared.base import BaseClient


class VertexAIClient(GoogleClient):
    PROVIDER_NAME = "vertex"
    PROVIDER_LABEL = "Vertex AI"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: str | None = None,
        logger: Logger | None = None,
    ):
        BaseClient.__init__(self, logger=logger)
        self._client = self._create_genai_client(
            vertexai=True,
            api_key=_strip_or_none(api_key) or _first_env("VERTEX_API_KEY"),
            credentials=credentials,
            project=_strip_or_none(project) or _first_env("VERTEX_PROJECT"),
            location=_strip_or_none(location) or _first_env("VERTEX_LOCATION"),
        )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = _strip_or_none(os.getenv(name))
        if value is not None:
            return value
    return None


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None
