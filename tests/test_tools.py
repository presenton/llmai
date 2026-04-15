import unittest

from llmai.shared import Tool, ToolChoice
from llmai.shared.errors import ToolError
from llmai.shared.tools import resolve_tools


def make_tool(name: str) -> Tool:
    return Tool(name=name, description=f"{name} description")


class ResolveToolsTests(unittest.TestCase):
    def test_returns_all_tools_when_no_choice_is_provided(self):
        tools = [make_tool("weather"), make_tool("time")]

        resolved = resolve_tools(tools, None)

        self.assertEqual([tool.name for tool in resolved.tools], ["weather", "time"])
        self.assertEqual(resolved.required_names, [])
        self.assertEqual(resolved.optional_names, [])

    def test_filters_tools_for_required_and_optional_names(self):
        tools = [make_tool("weather"), make_tool("time"), make_tool("news")]

        resolved = resolve_tools(
            tools,
            ToolChoice(required=["weather"], optional=["time"]),
        )

        self.assertEqual([tool.name for tool in resolved.tools], ["weather", "time"])
        self.assertEqual(resolved.required_names, ["weather"])
        self.assertEqual(resolved.optional_names, ["time"])

    def test_rejects_unknown_tool_names(self):
        tools = [make_tool("weather")]

        with self.assertRaises(ToolError) as context:
            resolve_tools(tools, ToolChoice(required=["time"]))

        self.assertIn("Unknown tool names", str(context.exception))

    def test_rejects_overlapping_required_and_optional_names(self):
        tools = [make_tool("weather")]

        with self.assertRaises(ToolError) as context:
            resolve_tools(
                tools,
                ToolChoice(required=["weather"], optional=["weather"]),
            )

        self.assertIn("both required and optional", str(context.exception))

    def test_rejects_duplicate_tool_definitions(self):
        tools = [make_tool("weather"), make_tool("weather")]

        with self.assertRaises(ToolError) as context:
            resolve_tools(tools, None)

        self.assertIn("defined multiple times", str(context.exception))


if __name__ == "__main__":
    unittest.main()
