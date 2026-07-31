from __future__ import annotations

import inspect
from typing import Any

from llmai.shared.base import BaseClient


def close_sync_provider_client(
    sync_client: BaseClient,
    provider_client: Any | None = None,
) -> None:
    sync_provider_client = getattr(sync_client, "_client", None)
    if sync_provider_client is None or sync_provider_client is provider_client:
        return

    close = getattr(sync_provider_client, "close", None)
    if callable(close):
        close()


async def close_async_resource(resource: Any) -> None:
    if resource is None:
        return

    close = getattr(resource, "aclose", None)
    if not callable(close):
        close = getattr(resource, "close", None)
    if not callable(close):
        return

    result = close()
    if inspect.isawaitable(result):
        await result


async def resolve_async_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
