from abc import ABC, abstractmethod

from pydantic import BaseModel

from llmai.models.tools import LLMTool


class SyncToolHandler(ABC):
    def __init__(
        self, name: str, description: str, schema: BaseModel | dict | None = None
    ):
        self._name = name
        self._description = description
        self._schema = schema

    @property
    def tool(self) -> LLMTool:
        return LLMTool(
            name=self._name, description=self._description, schema=self._schema
        )

    @abstractmethod
    def execute(self, arguments: dict | None) -> str | None:
        raise NotImplementedError


class AsyncToolHandler(ABC):
    def __init__(self, name: str, description: str, schema: BaseModel | dict | None = None):
        self._name = name
        self._description = description
        self._schema = schema

    @property
    def tool(self) -> LLMTool:
        return LLMTool(name=self._name, description=self._description, schema=self._schema)

    @abstractmethod
    async def execute(self, arguments: dict | None) -> str | None:
        raise NotImplementedError
