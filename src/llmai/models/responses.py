from typing import Literal
from pydantic import BaseModel


class ResponseContent(BaseModel):
    type: Literal["content"] = "content"
    content: str | dict


class ResponseStreamChunk(BaseModel):
    type: Literal["content_content_chunk"]
    id: str


class ResponseStreamContentChunk(ResponseStreamChunk):
    type: Literal["content_content_chunk"] = "content_content_chunk"
    source: Literal["direct", "tool"]
    chunk: str