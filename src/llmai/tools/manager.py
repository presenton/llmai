from typing import Dict, List, Union

from llmai.models.errors import ToolError
from tools.handlers import SyncToolHandler, AsyncToolHandler

ToolHandler = Union[SyncToolHandler, AsyncToolHandler]

class ToolsManager:
    def __init__(self):
        self._registry: Dict[str, ToolHandler] = {}
    
    @property
    def tools(self) -> List[str]:
        return list(self._registry.keys())

    def add(self, name: str, handler: ToolHandler):
        self._registry[name] = handler
    
    def remove(self, name: str):
        self._registry.pop(name)

    async def execute(
        self,
        name: str,
        arguments: dict | None,
    ) -> str | None:
        handler = self._registry.get(name)
        if not handler:
            raise ToolError(404, f"Tool {name} not registered")
        
        if isinstance(handler, SyncToolHandler):
            return handler.execute(arguments)
        elif isinstance(handler, AsyncToolHandler):
            return await handler.execute(arguments)
        else:
            raise ToolError(500, "Invalid tool handler")