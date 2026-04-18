from abc import ABC, abstractmethod
from logging import Logger
from typing import Any
from uuid import uuid4

from llmai.shared.logs import LogLevel
from llmai.shared.messages import Message
from llmai.shared.reasoning import ReasoningEffort
from llmai.shared.response_formats import ResponseFormat
from llmai.shared.responses import (
    ResponseResult,
    ResponseStreamChunk,
    ResponseStreamChunkType,
)
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

    def _tool_call_id(self, tool_id: str | None = None) -> str:
        if tool_id:
            return tool_id
        return f"call_{uuid4().hex}"

    def _transition_stream_chunk(
        self,
        *,
        current_chunk_type: ResponseStreamChunkType | None,
        next_chunk_type: ResponseStreamChunkType,
        current_tool: str | None = None,
        next_tool: str | None = None,
    ) -> tuple[ResponseStreamChunkType, str | None, list[ResponseStreamChunk]]:
        if current_chunk_type == next_chunk_type and current_tool == next_tool:
            return current_chunk_type, current_tool, []

        chunks: list[ResponseStreamChunk] = []
        if current_chunk_type is not None:
            chunks.append(
                ResponseStreamChunk(
                    chunk_type=current_chunk_type,
                    event="end",
                    tool=current_tool if current_chunk_type == "tool" else None,
                )
            )

        chunks.append(
            ResponseStreamChunk(
                chunk_type=next_chunk_type,
                event="start",
                tool=next_tool if next_chunk_type == "tool" else None,
            )
        )
        return next_chunk_type, next_tool if next_chunk_type == "tool" else None, chunks

    def _close_stream_chunk(
        self,
        *,
        current_chunk_type: ResponseStreamChunkType | None,
        current_tool: str | None = None,
    ) -> ResponseStreamChunk | None:
        if current_chunk_type is None:
            return None

        return ResponseStreamChunk(
            chunk_type=current_chunk_type,
            event="end",
            tool=current_tool if current_chunk_type == "tool" else None,
        )

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
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
        use_tools_for_structured_output: bool | None = None,
        stream: bool = False,
    ) -> ResponseResult:
        raise NotImplementedError
