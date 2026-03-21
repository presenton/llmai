from abc import ABC, abstractmethod

from llmai.shared.errors import ToolError
from llmai.shared.schema import SchemaLike
from llmai.shared.tools import Tool


class ToolHandler(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        schema: SchemaLike = None,
    ):
        self._name = name
        self._description = description
        self._schema = schema

    @property
    def tool(self) -> Tool:
        return Tool(
            name=self._name,
            description=self._description,
            input_schema=self._schema,
        )

    @abstractmethod
    def execute(self, arguments: dict | None) -> str | None:
        raise NotImplementedError


class ToolsManager:
    def __init__(self):
        self._registry: dict[str, ToolHandler] = {}

    @property
    def tools(self) -> list[str]:
        return list(self._registry.keys())

    def add(self, name: str, handler: ToolHandler) -> None:
        if name in self._registry:
            raise ToolError(400, f"Tool {name} already registered")
        self._registry[name] = handler

    def remove(self, name: str) -> None:
        self._registry.pop(name, None)

    def execute(self, name: str, arguments: dict | None) -> str | None:
        handler = self._registry.get(name)
        if not handler:
            raise ToolError(404, f"Tool {name} not registered")
        return handler.execute(arguments)
