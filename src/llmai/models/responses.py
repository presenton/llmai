from typing import Literal
from pydantic import BaseModel

class ResponseFormat(BaseModel):
    pass

class JSONSchemaResponse(ResponseFormat):
    type: Literal["json_schema"] = "json_schema"
    json_schema: BaseModel | dict

class JSONObjectResponse(ResponseFormat):
    type: Literal["json_object"] = "json_object"

class TextResponse(ResponseFormat):
    type: Literal["text"] = "text"