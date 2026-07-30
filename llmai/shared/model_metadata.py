from collections.abc import Iterable
from typing import Any


def metadata_value(value: object, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]

        resolved = getattr(value, name, None)
        if resolved is not None:
            return resolved

        model_extra = getattr(value, "model_extra", None)
        if isinstance(model_extra, dict) and name in model_extra:
            return model_extra[name]

    return None


def metadata_items(
    response: object,
    *,
    fields: tuple[str, ...] = ("data", "models", "model_summaries"),
) -> list[object]:
    for field in fields:
        items = metadata_value(response, field)
        if items is not None:
            return list(items)

    if isinstance(response, Iterable) and not isinstance(
        response,
        (str, bytes, dict),
    ):
        return list(response)

    return []


def all_metadata_items(
    response: object,
    *,
    fields: tuple[str, ...] = ("data", "models", "model_summaries"),
) -> list[object]:
    has_next_page = getattr(response, "has_next_page", None)
    get_next_page = getattr(response, "get_next_page", None)
    if callable(has_next_page) and callable(get_next_page):
        results: list[object] = []
        page = response
        while True:
            results.extend(metadata_items(page, fields=fields))
            if not page.has_next_page():
                return results
            page = page.get_next_page()

    return metadata_items(response, fields=fields)
