from typing import TypeAlias

from pydantic import BaseModel

SchemaLike: TypeAlias = type[BaseModel] | BaseModel | dict | None


def get_schema_as_dict(schema: SchemaLike, default: dict | None = None) -> dict:
    if isinstance(schema, dict):
        return schema

    if isinstance(schema, BaseModel):
        return schema.__class__.model_json_schema()

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()

    return {} if default is None else default
