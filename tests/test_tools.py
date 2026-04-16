import unittest

from llmai.shared import Tool
from llmai.shared.errors import ToolError
from llmai.shared.schema import get_schema_as_dict
from llmai.shared.tools import resolve_tools


def make_tool(name: str) -> Tool:
    return Tool(name=name, description=f"{name} description")


class ResolveToolsTests(unittest.TestCase):
    def test_tool_preserves_dict_input_schema(self):
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

        tool = Tool(name="weather", description="weather description", schema=schema)

        self.assertIsInstance(tool.input_schema, dict)
        self.assertEqual(tool.input_schema, schema)

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
            {"required": ["weather"], "optional": ["time"]},
        )

        self.assertEqual([tool.name for tool in resolved.tools], ["weather", "time"])
        self.assertEqual(resolved.required_names, ["weather"])
        self.assertEqual(resolved.optional_names, ["time"])

    def test_rejects_unknown_tool_names(self):
        tools = [make_tool("weather")]

        with self.assertRaises(ToolError) as context:
            resolve_tools(tools, {"required": ["time"]})

        self.assertIn("Unknown tool names", str(context.exception))

    def test_rejects_overlapping_required_and_optional_names(self):
        tools = [make_tool("weather")]

        with self.assertRaises(ToolError) as context:
            resolve_tools(
                tools,
                {"required": ["weather"], "optional": ["weather"]},
            )

        self.assertIn("both required and optional", str(context.exception))

    def test_rejects_duplicate_tool_definitions(self):
        tools = [make_tool("weather"), make_tool("weather")]

        with self.assertRaises(ToolError) as context:
            resolve_tools(tools, None)

        self.assertIn("defined multiple times", str(context.exception))


class SchemaTests(unittest.TestCase):
    def test_get_schema_as_dict_filters_to_supported_keys_and_formats(self):
        schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 20,
                },
                "email": {
                    "type": "string",
                    "format": "email",
                },
                "url": {
                    "type": "string",
                    "format": "uri",
                    "examples": ["https://example.com"],
                },
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "bullet_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                },
            },
            "required": ["title", "email", "url", "score", "bullet_points"],
        }

        filtered = get_schema_as_dict(
            schema,
            supported_keys={
                "$defs",
                "$ref",
                "additionalProperties",
                "allOf",
                "anyOf",
                "const",
                "definitions",
                "description",
                "enum",
                "exclusiveMaximum",
                "exclusiveMinimum",
                "format",
                "items",
                "maximum",
                "maxItems",
                "minimum",
                "minItems",
                "multipleOf",
                "pattern",
                "properties",
                "required",
                "type",
            },
            supported_string_formats={"email"},
            strict=True,
        )

        self.assertFalse(filtered["additionalProperties"])
        self.assertEqual(filtered["properties"]["email"]["format"], "email")
        self.assertNotIn("format", filtered["properties"]["url"])
        self.assertNotIn("examples", filtered["properties"]["url"])
        self.assertNotIn("minLength", filtered["properties"]["title"])
        self.assertEqual(filtered["properties"]["score"]["minimum"], 0)
        self.assertEqual(filtered["properties"]["score"]["maximum"], 1)
        self.assertEqual(filtered["properties"]["bullet_points"]["minItems"], 1)
        self.assertEqual(filtered["properties"]["bullet_points"]["maxItems"], 3)
        self.assertEqual(schema["properties"]["url"]["format"], "uri")
        self.assertEqual(schema["properties"]["title"]["minLength"], 20)

    def test_get_schema_as_dict_keeps_non_strict_schema_intact(self):
        schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 20,
                },
                "url": {
                    "type": "string",
                    "format": "uri",
                    "examples": ["https://example.com"],
                },
            },
            "required": ["title", "url"],
        }

        preserved = get_schema_as_dict(
            schema,
            supported_keys={
                "format",
                "properties",
                "required",
                "type",
            },
            supported_string_formats={"email"},
            strict=False,
        )

        self.assertFalse(preserved["additionalProperties"])
        self.assertEqual(preserved["properties"]["url"]["format"], "uri")
        self.assertEqual(
            preserved["properties"]["url"]["examples"],
            ["https://example.com"],
        )
        self.assertEqual(preserved["properties"]["title"]["minLength"], 20)

    def test_get_schema_as_dict_flattens_all_of_and_ref_in_strict_cleanup(self):
        schema = {
            "type": "object",
            "properties": {
                "result": {
                    "allOf": [
                        {
                            "$ref": "#/$defs/Result",
                        }
                    ],
                    "description": "Resolved result",
                }
            },
            "required": ["result"],
            "$defs": {
                "Result": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "format": "email",
                        },
                        "url": {
                            "type": "string",
                            "format": "uri",
                        },
                    },
                    "required": ["email", "url"],
                }
            },
        }

        cleaned = get_schema_as_dict(
            schema,
            supported_keys={
                "$defs",
                "$ref",
                "additionalProperties",
                "allOf",
                "description",
                "format",
                "properties",
                "required",
                "type",
            },
            supported_string_formats={"email"},
            strict=True,
        )

        result = cleaned["properties"]["result"]
        self.assertNotIn("allOf", result)
        self.assertNotIn("$ref", result)
        self.assertEqual(result["type"], "object")
        self.assertEqual(result["description"], "Resolved result")
        self.assertEqual(result["properties"]["email"]["format"], "email")
        self.assertNotIn("format", result["properties"]["url"])


if __name__ == "__main__":
    unittest.main()
