from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from llmai.shared.errors import ToolError
from llmai.shared.schema import SchemaLike


class Tool(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    strict: bool = False
    input_schema: SchemaLike = Field(
        default=None,
        alias="schema",
        serialization_alias="schema",
    )


class ToolChoiceMode(str, Enum):
    AUTO = "auto"
    REQUIRED = "required"


class ToolChoice(TypedDict, total=False):
    mode: ToolChoiceMode
    tools: list[str]


@dataclass(frozen=True)
class ResolvedToolChoice:
    tools: list[Tool]
    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    is_explicit: bool = False

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    @property
    def requires_tool(self) -> bool:
        return self.mode == ToolChoiceMode.REQUIRED


def resolve_tools(
    tools: list[Tool] | None,
    tool_choice: ToolChoice | None,
) -> ResolvedToolChoice:
    available_tools = tools or []
    tool_by_name = _tool_map(available_tools)

    if not tool_choice:
        return ResolvedToolChoice(
            tools=available_tools,
        )

    unknown_keys = sorted(set(tool_choice) - {"mode", "tools"})
    if unknown_keys:
        names = ", ".join(unknown_keys)
        raise ToolError(
            400,
            f"Unsupported keys in tool_choice: {names}. Use 'mode' and 'tools'.",
        )

    mode = _coerce_tool_choice_mode(tool_choice.get("mode", ToolChoiceMode.AUTO))
    if mode is None:
        raise ToolError(
            400,
            f"Unsupported tool_choice mode: {tool_choice.get('mode')}",
        )

    tool_names = _unique_names(tool_choice.get("tools"))
    unknown_names = [name for name in tool_names if name not in tool_by_name]
    if unknown_names:
        names = ", ".join(unknown_names)
        raise ToolError(400, f"Unknown tool names in tool_choice: {names}")

    visible_tools = (
        [tool_by_name[name] for name in tool_names]
        if tool_names
        else available_tools
    )
    if not visible_tools:
        if mode == ToolChoiceMode.REQUIRED:
            raise ToolError(
                400,
                "tool_choice mode='required' requires at least one visible tool",
            )
        return ResolvedToolChoice(
            tools=[],
            mode=mode,
            is_explicit=True,
        )

    return ResolvedToolChoice(
        tools=visible_tools,
        mode=mode,
        is_explicit=True,
    )


def _tool_map(tools: list[Tool]) -> dict[str, Tool]:
    tool_by_name: dict[str, Tool] = {}

    for tool in tools:
        if tool.name in tool_by_name:
            raise ToolError(400, f"Tool {tool.name} is defined multiple times")
        tool_by_name[tool.name] = tool

    return tool_by_name


def _unique_names(names: list[str] | None) -> list[str]:
    if not names:
        return []

    seen: set[str] = set()
    unique_names: list[str] = []

    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique_names.append(name)

    return unique_names


def _coerce_tool_choice_mode(
    value: ToolChoiceMode | str | None,
) -> ToolChoiceMode | None:
    if value is None:
        return None

    if isinstance(value, ToolChoiceMode):
        return value

    try:
        return ToolChoiceMode(value)
    except ValueError:
        return None
