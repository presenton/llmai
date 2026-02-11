from pydantic import BaseModel


def get_schema_as_dict(schema: BaseModel | dict | None, default: dict = {}) -> dict:
    if isinstance(schema, BaseModel):
        return schema.model_dump(mode="json")
    elif isinstance(schema, dict):
        return schema
    return default