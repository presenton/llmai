from typing import Literal

from pydantic import BaseModel

from llmai.shared.schema import SchemaLike, get_schema_as_dict


class ResponseFormat(BaseModel):
    pass


class JSONSchemaResponse(ResponseFormat):
    type: Literal["json_schema"] = "json_schema"
    json_schema: SchemaLike


class JSONObjectResponse(ResponseFormat):
    type: Literal["json_object"] = "json_object"


class TextResponse(ResponseFormat):
    type: Literal["text"] = "text"


def get_response_schema(response_format: ResponseFormat | None) -> dict | None:
    if isinstance(response_format, JSONSchemaResponse):
        return get_schema_as_dict(response_format.json_schema)

    if isinstance(response_format, JSONObjectResponse):
        return {"type": "object"}

    return None
