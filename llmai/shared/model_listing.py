from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def model_ids(models: Iterable[Any]) -> list[str]:
    """Return unique provider model IDs while preserving provider order."""

    result: list[str] = []
    seen: set[str] = set()
    for model in models:
        model_id: Any = None
        if isinstance(model, str):
            model_id = model
        elif isinstance(model, dict):
            model_id = model.get("id") or model.get("name")
        else:
            model_id = getattr(model, "id", None) or getattr(model, "name", None)

        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        result.append(model_id)

    return result


def openai_compatible_model_ids(payload: Any) -> list[str] | None:
    """Parse both standard OpenAI and top-level-list model payloads."""

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("data")
        if not isinstance(items, list):
            return None
    else:
        return None

    return model_ids(items)
