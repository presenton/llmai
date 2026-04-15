from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Generator

from llmai.shared.logs import LogLevel
from llmai.shared.messages import Message
from llmai.shared.response_formats import ResponseFormat
from llmai.shared.responses import ResponseContent, ResponseStreamChunk
from llmai.shared.tools import Tool, ToolChoice


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

    def _dump_value(self, value: Any) -> Any:
        if value is None:
            return None

        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)

        if isinstance(value, dict):
            return {
                key: self._dump_value(item)
                for key, item in value.items()
                if item is not None
            }

        if isinstance(value, (list, tuple)):
            return [self._dump_value(item) for item in value]

        if hasattr(value, "__dict__"):
            return {
                key: self._dump_value(item)
                for key, item in vars(value).items()
                if not key.startswith("_") and item is not None
            }

        return value

    def _dump_model(self, value: Any) -> dict[str, Any]:
        dumped = self._dump_value(value)
        return dumped if isinstance(dumped, dict) else {}

    @abstractmethod
    def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        use_tools_for_structured_output: bool | None = None,
    ) -> ResponseContent:
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        use_tools_for_structured_output: bool | None = None,
    ) -> Generator[ResponseStreamChunk, None, None]:
        raise NotImplementedError
