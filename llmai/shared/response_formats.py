from typing import Literal

from pydantic import BaseModel

from llmai.shared.schema import SchemaLike, cleanup_schema_dict, get_schema_as_dict


class ResponseFormat(BaseModel):
    pass


class JSONSchemaResponse(ResponseFormat):
    type: Literal["json_schema"] = "json_schema"
    name: str | None = None
    strict: bool = True
    json_schema: SchemaLike


class JSONObjectResponse(ResponseFormat):
    type: Literal["json_object"] = "json_object"


class TextResponse(ResponseFormat):
    type: Literal["text"] = "text"


def get_response_schema(
    response_format: ResponseFormat | None,
    *,
    supported_keys: set[str] | None = None,
    supported_string_formats: set[str] | None = None,
    strict: bool = False,
) -> dict | None:
    if isinstance(response_format, JSONSchemaResponse):
        return get_schema_as_dict(
            response_format.json_schema,
            supported_keys=supported_keys,
            supported_string_formats=supported_string_formats,
            strict=strict,
        )

    if isinstance(response_format, JSONObjectResponse):
        if strict:
            return cleanup_schema_dict(
                {"type": "object"},
                supported_keys=supported_keys,
                supported_string_formats=supported_string_formats,
            )
        return {"type": "object"}

    return None


def get_response_format_name(
    response_format: ResponseFormat | None,
    *,
    default: str | None = None,
) -> str | None:
    if isinstance(response_format, JSONSchemaResponse):
        return response_format.name or default

    return default


def get_response_format_strict(
    response_format: ResponseFormat | None,
    *,
    default: bool | None = None,
) -> bool | None:
    if isinstance(response_format, JSONSchemaResponse):
        return response_format.strict

    return default
