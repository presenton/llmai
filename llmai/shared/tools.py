from __future__ import annotations

from dataclasses import dataclass

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


class ToolChoice(BaseModel):
    required: list[str] | None = None
    optional: list[str] | None = None


@dataclass(frozen=True)
class ResolvedToolChoice:
    tools: list[Tool]
    required_tools: list[Tool]
    optional_tools: list[Tool]

    @property
    def required_names(self) -> list[str]:
        return [tool.name for tool in self.required_tools]

    @property
    def optional_names(self) -> list[str]:
        return [tool.name for tool in self.optional_tools]


def resolve_tools(
    tools: list[Tool] | None,
    tool_choice: ToolChoice | None,
) -> ResolvedToolChoice:
    available_tools = tools or []
    tool_by_name = _tool_map(available_tools)

    if not tool_choice:
        return ResolvedToolChoice(
            tools=available_tools,
            required_tools=[],
            optional_tools=[],
        )

    required_names = _unique_names(tool_choice.required)
    optional_names = _unique_names(tool_choice.optional)
    overlapping_names = set(required_names) & set(optional_names)
    if overlapping_names:
        names = ", ".join(sorted(overlapping_names))
        raise ToolError(
            400,
            f"Tool names cannot be both required and optional: {names}",
        )

    unknown_names = [
        name
        for name in [*required_names, *optional_names]
        if name not in tool_by_name
    ]
    if unknown_names:
        names = ", ".join(unknown_names)
        raise ToolError(400, f"Unknown tool names in tool_choice: {names}")

    if not required_names and not optional_names:
        return ResolvedToolChoice(
            tools=available_tools,
            required_tools=[],
            optional_tools=[],
        )

    visible_names = [*required_names, *optional_names]
    return ResolvedToolChoice(
        tools=[tool_by_name[name] for name in visible_names],
        required_tools=[tool_by_name[name] for name in required_names],
        optional_tools=[tool_by_name[name] for name in optional_names],
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
