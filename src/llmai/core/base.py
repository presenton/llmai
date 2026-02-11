from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, AsyncGenerator, List, Optional

from llmai.models.logs import LogLevel
from llmai.models.messages import Message
from llmai.models.tools import LLMTool

class BaseClient(ABC):
    def __init__(self, *, logger: Optional[Logger] = None):
        self._logger = logger

    def log(self, level: LogLevel, message: any):
        if self._logger:
            match level:
                case LogLevel.INFO:
                    self._logger.info(message)
                case LogLevel.WARNING:
                    self._logger.warning(message)
                case LogLevel.ERROR:
                    self._logger.error(message)

    @abstractmethod
    async def generate(
        self,
        model: str,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        tools: Optional[List[LLMTool]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        raise NotImplementedError

    # @abstractmethod
    # async def generate_structured(
    #     self,
    #     model: str,
    #     messages: List[LLMMessage],
    #     response_format: dict,
    #     strict: bool = False,
    #     prune_length_constraints: bool = False,
    #     tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None,
    #     max_tokens: Optional[int] = None,
    #     extra_body: Optional[dict] = None,
    #     temperature: Optional[float] = None,
    #     tool_call_for_structured_output: Optional[bool] = None,
    # ) -> dict:
    #     raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        model: str,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        tools: Optional[List[LLMTool]] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[Any, None]:
        raise NotImplementedError
