from typing import Dict, List

from llmai.models.errors import ToolError
from tools.handlers import ToolHandler

class ToolsManager:
    def __init__(self):
        self._registry: Dict[str, ToolHandler] = {}
    
    @property
    def tools(self) -> List[str]:
        return list(self._registry.keys())

    def add(self, name: str, handler: ToolHandler):
        if name in self._registry:
            raise ToolError(400, f"Tool {name} already registered")
        self._registry[name] = handler
    
    def remove(self, name: str):
        self._registry.pop(name)

    def execute(
        self,
        name: str,
        arguments: dict | None,
    ) -> str | None:
        handler = self._registry.get(name)
        if not handler:
            raise ToolError(404, f"Tool {name} not registered")
        return handler.execute(arguments)