from pydantic import BaseModel, ConfigDict, Field

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


class ToolsChoice(BaseModel):
    required: list[Tool] = Field(default_factory=list)
    optional: list[Tool] = Field(default_factory=list)
    depth: int | None = None
    start: bool | None = None
    end: bool | None = None


ToolChoice = ToolsChoice


class ToolChoices(BaseModel):
    choices: list[ToolsChoice] = Field(default_factory=list)
    stop_on: list[str] | None = None

    def for_depth(self, depth: int) -> list[ToolsChoice]:
        return [
            choice
            for choice in self.choices
            if choice.depth is None or choice.depth == depth
        ]

    def required_for_depth(self, depth: int) -> list[Tool]:
        return _unique_tools(
            [
                tool
                for choice in self.for_depth(depth)
                for tool in choice.required
            ]
        )

    def optional_for_depth(self, depth: int) -> list[Tool]:
        return _unique_tools(
            [
                tool
                for choice in self.for_depth(depth)
                for tool in choice.optional
            ]
        )

    def all_for_depth(self, depth: int) -> list[Tool]:
        return _unique_tools(
            [
                *self.required_for_depth(depth),
                *self.optional_for_depth(depth),
            ]
        )


def _unique_tools(tools: list[Tool]) -> list[Tool]:
    seen: set[str] = set()
    unique_tools: list[Tool] = []

    for tool in tools:
        if tool.name in seen:
            continue
        seen.add(tool.name)
        unique_tools.append(tool)

    return unique_tools
