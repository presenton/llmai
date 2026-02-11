from pydantic import BaseModel


class LLMTool(BaseModel):
    name: str
    description: str
    strict: bool = False
    schema: BaseModel | dict | None = None