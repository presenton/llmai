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
    strict: bool = False,
) -> dict:
    del strict

    if isinstance(schema, dict):
        return deepcopy(schema)

    if isinstance(schema, BaseModel):
        return _model_json_schema(schema.__class__)

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return _model_json_schema(schema)

    if default is None:
        return {}

    return deepcopy(default)


def strip_schema_keys(
    schema: dict,
    *,
    keys: set[str],
) -> dict:
    return _strip_schema_keys(deepcopy(schema), keys=keys)


def _strip_schema_keys(
    schema: object,
    *,
    keys: set[str],
) -> object:
    if isinstance(schema, dict):
        return {
            key: _strip_schema_keys(value, keys=keys)
            for key, value in schema.items()
            if key not in keys
        }

    if isinstance(schema, list):
        return [_strip_schema_keys(each, keys=keys) for each in schema]

    return schema


def _model_json_schema(model_type: type[BaseModel]) -> dict:
    if model_type is BaseModel:
        raise configuration_error(
            "Schema must be a dict, a BaseModel subclass, or an instance of a BaseModel subclass",
            status_code=400,
        )

    return model_type.model_json_schema()
