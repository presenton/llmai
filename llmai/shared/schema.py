from typing import Annotated, TypeAlias
from copy import deepcopy

from pydantic import BaseModel, Field

from llmai.shared.errors import configuration_error

SchemaLike: TypeAlias = Annotated[
    dict | type[BaseModel] | BaseModel | None,
    Field(union_mode="left_to_right"),
]


def get_schema_as_dict(
    schema: SchemaLike,
    default: dict | None = None,
    *,
    supported_keys: set[str] | None = None,
    supported_string_formats: set[str] | None = None,
    strict: bool = False,
) -> dict:
    if isinstance(schema, dict):
        normalized_schema = _normalize_object_schema(deepcopy(schema))
        if strict:
            return cleanup_schema_dict(
                normalized_schema,
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
            )
        return normalized_schema

    if isinstance(schema, BaseModel):
        normalized_schema = _normalize_object_schema(
            _model_json_schema(schema.__class__)
        )
        if strict:
            return cleanup_schema_dict(
                normalized_schema,
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
            )
        return normalized_schema

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        normalized_schema = _normalize_object_schema(_model_json_schema(schema))
        if strict:
            return cleanup_schema_dict(
                normalized_schema,
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
            )
        return normalized_schema

    if default is None:
        return {}

    default_schema = deepcopy(default)
    if strict:
        return cleanup_schema_dict(
            default_schema,
            supported_keys=supported_keys,
            supported_string_formats=supported_string_formats,
        )
    return default_schema


def cleanup_schema_dict(
    schema: dict,
    *,
    supported_keys: set[str] | None = None,
    supported_string_formats: set[str] | None = None,
) -> dict:
    normalized_schema = _normalize_object_schema(deepcopy(schema))
    return _cleanup_schema_dict(
        normalized_schema,
        supported_keys=supported_keys,
        supported_string_formats=supported_string_formats,
        root=normalized_schema,
    )


def filter_schema_dict(
    schema: dict,
    *,
    supported_keys: set[str] | None = None,
    supported_string_formats: set[str] | None = None,
) -> dict:
    return cleanup_schema_dict(
        schema,
        supported_keys=supported_keys,
        supported_string_formats=supported_string_formats,
    )


def _normalize_object_schema(schema: dict) -> dict:
    if not isinstance(schema, dict):
        return schema

    schema_type = schema.get("type")
    if schema_type == "object":
        schema.setdefault("additionalProperties", False)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for each in properties.values():
            _normalize_object_schema(each)

    items = schema.get("items")
    if isinstance(items, dict):
        _normalize_object_schema(items)
    elif isinstance(items, list):
        for each in items:
            _normalize_object_schema(each)

    for key in ("allOf", "anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list):
            for each in options:
                _normalize_object_schema(each)

    defs = schema.get("$defs")
    if isinstance(defs, dict):
        for each in defs.values():
            _normalize_object_schema(each)

    definitions = schema.get("definitions")
    if isinstance(definitions, dict):
        for each in definitions.values():
            _normalize_object_schema(each)

    return schema


def _model_json_schema(model_type: type[BaseModel]) -> dict:
    if model_type is BaseModel:
        raise configuration_error(
            "Schema must be a dict, a BaseModel subclass, or an instance of a BaseModel subclass",
            status_code=400,
        )

    return model_type.model_json_schema()


def _cleanup_schema_dict(
    schema: dict,
    *,
    supported_keys: set[str] | None = None,
    supported_string_formats: set[str] | None = None,
    root: dict,
) -> dict:
    if not isinstance(schema, dict):
        return schema

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for each in properties.values():
            _cleanup_schema_dict(
                each,
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
                root=root,
            )

    items = schema.get("items")
    if isinstance(items, dict):
        _cleanup_schema_dict(
            items,
            supported_keys=supported_keys,
            supported_string_formats=supported_string_formats,
            root=root,
        )
    elif isinstance(items, list):
        for each in items:
            _cleanup_schema_dict(
                each,
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
                root=root,
            )

    additional_properties = schema.get("additionalProperties")
    if isinstance(additional_properties, dict):
        _cleanup_schema_dict(
            additional_properties,
            supported_keys=supported_keys,
            supported_string_formats=supported_string_formats,
            root=root,
        )

    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list):
            for each in options:
                _cleanup_schema_dict(
                    each,
                    supported_keys=supported_keys,
                    supported_string_formats=supported_string_formats,
                    root=root,
                )

    defs = schema.get("$defs")
    if isinstance(defs, dict):
        for each in defs.values():
            _cleanup_schema_dict(
                each,
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
                root=root,
            )

    definitions = schema.get("definitions")
    if isinstance(definitions, dict):
        for each in definitions.values():
            _cleanup_schema_dict(
                each,
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
                root=root,
            )

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        schema["allOf"] = [
            _cleanup_schema_dict(
                each,
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
                root=root,
            )
            for each in all_of
        ]

        if len(schema["allOf"]) == 1 and isinstance(schema["allOf"][0], dict):
            flattened_schema = dict(schema["allOf"][0])
            sibling_schema = {
                key: value
                for key, value in schema.items()
                if key != "allOf"
            }
            schema.clear()
            schema.update({**flattened_schema, **sibling_schema})
            return _cleanup_schema_dict(
                schema,
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
                root=root,
            )

    schema_ref = schema.get("$ref")
    if isinstance(schema_ref, str) and _has_more_than_n_keys(schema, 1):
        resolved_schema = _resolve_ref(root=root, ref=schema_ref)
        if isinstance(resolved_schema, dict):
            sibling_schema = {
                key: value
                for key, value in schema.items()
                if key != "$ref"
            }
            schema.clear()
            schema.update({**deepcopy(resolved_schema), **sibling_schema})
            return _cleanup_schema_dict(
                schema,
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
                root=root,
            )

    if supported_keys is not None:
        for key in tuple(schema.keys()):
            if key not in supported_keys:
                schema.pop(key)

    schema_format = schema.get("format")
    if (
        supported_string_formats is not None
        and isinstance(schema_format, str)
        and schema_format not in supported_string_formats
    ):
        schema.pop("format", None)

    return schema


def _resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        return {}

    resolved: object = root
    for key in ref[2:].split("/"):
        if not isinstance(resolved, dict):
            return {}
        resolved = resolved.get(key)

    return resolved


def _has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    count = 0
    for _ in obj:
        count += 1
        if count > n:
            return True
    return False
