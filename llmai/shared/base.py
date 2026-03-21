from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Generator

from llmai.shared.logs import LogLevel
from llmai.shared.messages import Message
from llmai.shared.response_formats import ResponseFormat
from llmai.shared.responses import ResponseContent, ResponseStreamChunk
from llmai.shared.tools import ToolChoices


class BaseClient(ABC):
    def __init__(self, *, logger: Logger | None = None):
        self._logger = logger

    def log(self, level: LogLevel, message: Any) -> None:
        if not self._logger:
            return

        match level:
            case LogLevel.INFO:
                self._logger.info(message)
            case LogLevel.WARNING:
                self._logger.warning(message)
            case LogLevel.ERROR:
                self._logger.error(message)

    @abstractmethod
    def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: ToolChoices | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        use_tools_for_structured_output: bool | None = None,
        depth: int = 0,
    ) -> ResponseContent:
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: ToolChoices | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        use_tools_for_structured_output: bool | None = None,
        depth: int = 0,
    ) -> Generator[ResponseStreamChunk, None, None]:
        raise NotImplementedError
