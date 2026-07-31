from __future__ import annotations

from logging import Logger
from typing import Any

from llmai.client import get_async_client, get_client
from llmai.shared.configs import ClientConfig


def _close_sync_client(client: Any) -> None:
    provider_client = getattr(client, "_client", None)
    close = getattr(provider_client, "close", None)
    if callable(close):
        close()


def list_available_models(
    *,
    config: ClientConfig,
    logger: Logger | None = None,
) -> list[str]:
    """List models available from the live provider configured by ``config``."""

    client = get_client(config=config, logger=logger)
    try:
        return client.list_available_models()
    finally:
        _close_sync_client(client)


async def alist_available_models(
    *,
    config: ClientConfig,
    logger: Logger | None = None,
) -> list[str]:
    """Asynchronously list models available from the configured live provider."""

    async with get_async_client(config=config, logger=logger) as client:
        return await client.alist_available_models()
