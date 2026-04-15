from typing import TypeAlias
from copy import deepcopy

from pydantic import BaseModel

SchemaLike: TypeAlias = type[BaseModel] | BaseModel | dict | None


def get_schema_as_dict(schema: SchemaLike, default: dict | None = None) -> dict:
    if isinstance(schema, dict):
        return _normalize_object_schema(deepcopy(schema))

    if isinstance(schema, BaseModel):
        return _normalize_object_schema(schema.__class__.model_json_schema())

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return _normalize_object_schema(schema.model_json_schema())

    return {} if default is None else default


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

    return schema
