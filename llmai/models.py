"""Local model metadata sourced from https://models.dev/api.json."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

MODELS_DEV_API_URL = "https://models.dev/api.json"
DEFAULT_MODELS_PATH = Path(__file__).with_name("data") / "models.json"
DEFAULT_CONTEXT_WINDOW = 4_000
MODELS_DEV_PROVIDER_ALIASES = {
    "bedrock": "amazon-bedrock",
    "chatgpt": "openai",
    "codex": "openai",
    "fireworks": "fireworks-ai",
    "together": "togetherai",
    "vertex": "google-vertex",
}


class ModelLookupError(LookupError):
    """Base class for model metadata lookup errors."""


class ModelNotFoundError(ModelLookupError):
    """Raised when no model matches a reference."""


class AmbiguousModelError(ModelLookupError):
    """Raised when an unqualified model name or ID matches multiple providers."""

    def __init__(self, query: str, references: list[str]) -> None:
        self.query = query
        self.references = references
        choices = ", ".join(references)
        super().__init__(
            f"Model {query!r} is ambiguous; use a provider-qualified reference: "
            f"{choices}"
        )


def _data_path(data_path: str | os.PathLike[str] | None) -> Path:
    return Path(data_path) if data_path is not None else DEFAULT_MODELS_PATH


def _file_stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=8)
def _read_model_data(
    path_string: str, _stamp: tuple[int, int]
) -> dict[str, dict[str, Any]]:
    with Path(path_string).open(encoding="utf-8") as file:
        data = json.load(file)
    _validate_model_data(data)
    return data


def _model_data(
    data_path: str | os.PathLike[str] | None = None,
) -> dict[str, dict[str, Any]]:
    path = _data_path(data_path).resolve()
    return _read_model_data(str(path), _file_stamp(path))


def load_model_data(
    data_path: str | os.PathLike[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a copy of the complete, unmodified models.dev provider data."""

    return deepcopy(_model_data(data_path))


def _normalize(value: object) -> str:
    return str(value).strip().casefold()


def _provider_matches(
    provider_key: str, provider_data: dict[str, Any], provider: str | None
) -> bool:
    if provider is None:
        return True
    wanted = _normalize(provider)
    wanted = MODELS_DEV_PROVIDER_ALIASES.get(wanted, wanted)
    return wanted in {
        _normalize(provider_key),
        _normalize(provider_data.get("id", provider_key)),
        _normalize(provider_data.get("name", provider_key)),
    }


def _flatten_model(
    provider_key: str,
    provider_data: dict[str, Any],
    model_key: str,
    model_data: dict[str, Any],
) -> dict[str, Any]:
    provider_id = str(provider_data.get("id", provider_key))
    provider_name = str(provider_data.get("name", provider_id))
    model_id = str(model_data.get("id", model_key))
    model_name = str(model_data.get("name", model_id))
    reference = f"{provider_id}:{model_id}"
    limit = model_data.get("limit", {})
    if not isinstance(limit, dict):
        limit = {}
    context_window = limit.get("context")
    if (
        not isinstance(context_window, int)
        or isinstance(context_window, bool)
        or context_window <= 0
    ):
        context_window = DEFAULT_CONTEXT_WINDOW

    result = deepcopy(model_data)
    result.update(
        {
            "provider_id": provider_id,
            "provider_name": provider_name,
            "provider_metadata": deepcopy(
                {key: value for key, value in provider_data.items() if key != "models"}
            ),
            "model_key": model_key,
            "reference": reference,
            "references": [
                reference,
                f"{provider_id}/{model_id}",
                model_id,
                model_name,
            ],
            "context_window": context_window,
            "max_output_tokens": limit.get("output"),
        }
    )
    return result


def list_models(
    provider: str | None = None,
    *,
    data_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """List every model, optionally restricted by provider ID or name.

    Each result contains all upstream model fields plus ``provider_id``,
    ``provider_name``, a canonical ``reference``, accepted ``references``,
    ``context_window``, and ``max_output_tokens``.
    """

    results: list[dict[str, Any]] = []
    for provider_key, provider_data in _model_data(data_path).items():
        if not _provider_matches(provider_key, provider_data, provider):
            continue
        for model_key, model_data in provider_data["models"].items():
            results.append(
                _flatten_model(provider_key, provider_data, model_key, model_data)
            )
    return results


def _search_values(model: dict[str, Any]) -> list[str]:
    values = [
        *model["references"],
        f"{model['provider_name']}:{model['id']}",
        f"{model['provider_name']}/{model['id']}",
        f"{model['provider_id']}:{model['name']}",
        f"{model['provider_id']}/{model['name']}",
        model.get("family", ""),
        model.get("description", ""),
    ]
    return [_normalize(value) for value in values if value]


def _exact_search_values(
    provider_key: str,
    provider_data: dict[str, Any],
    model_key: str,
    model_data: dict[str, Any],
) -> set[str]:
    provider_id = str(provider_data.get("id", provider_key))
    provider_name = str(provider_data.get("name", provider_id))
    model_id = str(model_data.get("id", model_key))
    model_name = str(model_data.get("name", model_id))
    return {
        _normalize(value)
        for value in (
            model_id,
            model_name,
            f"{provider_id}:{model_id}",
            f"{provider_id}/{model_id}",
            f"{provider_name}:{model_id}",
            f"{provider_name}/{model_id}",
            f"{provider_id}:{model_name}",
            f"{provider_id}/{model_name}",
        )
    }


def query_models(
    query: str | None = None,
    *,
    provider: str | None = None,
    exact: bool = False,
    data_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Search model IDs, names, references, families, and descriptions.

    With no query, this is equivalent to :func:`list_models`. Searches are
    case-insensitive. Set ``exact=True`` to match only an ID, name, or
    provider-qualified reference.
    """

    if query is None or not query.strip():
        return list_models(provider, data_path=data_path)

    wanted = _normalize(query)
    if exact:
        results: list[dict[str, Any]] = []
        for provider_key, provider_data in _model_data(data_path).items():
            if not _provider_matches(provider_key, provider_data, provider):
                continue
            for model_key, model_data in provider_data["models"].items():
                if wanted in _exact_search_values(
                    provider_key, provider_data, model_key, model_data
                ):
                    results.append(
                        _flatten_model(
                            provider_key, provider_data, model_key, model_data
                        )
                    )
        return results

    models = list_models(provider, data_path=data_path)
    return [
        model
        for model in models
        if any(wanted in candidate for candidate in _search_values(model))
    ]


def get_model_metadata(
    reference: str,
    *,
    provider: str | None = None,
    data_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Get one model by ID, name, or provider-qualified reference.

    Bare IDs and names work when unique. If several providers expose the same
    model, pass ``provider=`` or use ``provider:model-id``.
    """

    matches = query_models(
        reference, provider=provider, exact=True, data_path=data_path
    )
    if not matches:
        provider_hint = f" for provider {provider!r}" if provider else ""
        raise ModelNotFoundError(f"No model found for {reference!r}{provider_hint}")
    if len(matches) > 1:
        raise AmbiguousModelError(
            reference, sorted(model["reference"] for model in matches)
        )
    return matches[0]


def get_context_window(
    reference: str,
    *,
    provider: str | None = None,
    default: int = DEFAULT_CONTEXT_WINDOW,
    data_path: str | os.PathLike[str] | None = None,
) -> int:
    """Return a model's maximum input context window in tokens.

    Unknown, ambiguous, and incomplete records return ``default`` (4000 tokens
    unless overridden). Provider aliases used by llmai clients are translated
    to their models.dev IDs. A leading Google SDK ``models/`` prefix is also
    accepted.
    """

    candidates = [reference]
    if reference.casefold().startswith("models/"):
        candidates.append(reference.split("/", 1)[1])

    for candidate in candidates:
        try:
            return int(
                get_model_metadata(candidate, provider=provider, data_path=data_path)[
                    "context_window"
                ]
            )
        except ModelLookupError:
            pass

        # A provider may have removed an older alias while other providers
        # still list it. Use the value only when all exact matches agree.
        if provider is not None:
            matches = query_models(candidate, exact=True, data_path=data_path)
            context_windows = {int(model["context_window"]) for model in matches}
            if len(context_windows) == 1:
                return context_windows.pop()

    return int(default)


def _validate_model_data(data: object) -> None:
    if not isinstance(data, dict) or not data:
        raise ValueError("models.dev data must be a non-empty JSON object")

    for provider_key, provider_data in data.items():
        if not isinstance(provider_data, dict):
            raise TypeError(f"Provider {provider_key!r} must be an object")
        models = provider_data.get("models")
        if not isinstance(models, dict):
            raise TypeError(f"Provider {provider_key!r} has no models object")
        for model_key, model_data in models.items():
            if not isinstance(model_data, dict):
                raise TypeError(f"Model {provider_key}:{model_key} must be an object")


def refresh_model_data(
    *,
    destination: str | os.PathLike[str] | None = None,
    url: str = MODELS_DEV_API_URL,
    timeout: float = 60.0,
) -> Path:
    """Download, validate, and atomically replace the local model JSON.

    ``destination`` defaults to the package's bundled ``models.json``. A custom
    destination can be queried by passing it as ``data_path`` to the lookup
    functions.
    """

    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "llmai-model-metadata/1",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()

    try:
        data = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("models.dev returned invalid JSON") from error
    _validate_model_data(data)

    path = _data_path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    _read_model_data.cache_clear()
    return path
