import unittest

from llmai.shared import Tool, WebSearchTool
from llmai.shared.errors import ToolError
from llmai.shared.schema import get_schema_as_dict, process_schema
from llmai.shared.tools import filter_resolved_tools_for_provider, resolve_tools


def make_tool(name: str) -> Tool:
    return Tool(name=name, description=f"{name} description")


def make_web_search_tool() -> WebSearchTool:
    return WebSearchTool()


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
        self.assertEqual(resolved.tool_names, ["weather", "time"])
        self.assertFalse(resolved.requires_tool)
        self.assertFalse(resolved.is_explicit)

    def test_filters_tools_for_selected_names(self):
        tools = [make_tool("weather"), make_tool("time"), make_tool("news")]

        resolved = resolve_tools(
            tools,
            {"tools": ["weather", "time"]},
        )

        self.assertEqual([tool.name for tool in resolved.tools], ["weather", "time"])
        self.assertEqual(resolved.tool_names, ["weather", "time"])
        self.assertFalse(resolved.requires_tool)
        self.assertTrue(resolved.is_explicit)

    def test_required_mode_uses_all_tools_when_no_subset_is_provided(self):
        tools = [make_tool("weather"), make_tool("time")]

        resolved = resolve_tools(
            tools,
            {"mode": "required"},
        )

        self.assertEqual(resolved.tool_names, ["weather", "time"])
        self.assertTrue(resolved.requires_tool)
        self.assertTrue(resolved.is_explicit)

    def test_rejects_unknown_tool_names(self):
        tools = [make_tool("weather")]

        with self.assertRaises(ToolError) as context:
            resolve_tools(tools, {"tools": ["time"]})

        self.assertIn("Unknown tool names", str(context.exception))

    def test_rejects_legacy_required_and_optional_keys(self):
        tools = [make_tool("weather")]

        with self.assertRaises(ToolError) as context:
            resolve_tools(
                tools,
                {"required": ["weather"], "optional": ["weather"]},
            )

        self.assertIn("Use 'mode' and 'tools'", str(context.exception))

    def test_rejects_invalid_tool_choice_mode(self):
        tools = [make_tool("weather")]

        with self.assertRaises(ToolError) as context:
            resolve_tools(
                tools,
                {"mode": "always"},
            )

        self.assertIn("Unsupported tool_choice mode", str(context.exception))

    def test_rejects_required_mode_without_visible_tools(self):
        with self.assertRaises(ToolError) as context:
            resolve_tools(
                [],
                {"mode": "required"},
            )

        self.assertIn("requires at least one visible tool", str(context.exception))

    def test_rejects_duplicate_tool_definitions(self):
        tools = [make_tool("weather"), make_tool("weather")]

        with self.assertRaises(ToolError) as context:
            resolve_tools(tools, None)

        self.assertIn("defined multiple times", str(context.exception))

    def test_rejects_function_tool_named_web_search(self):
        with self.assertRaises(ToolError) as context:
            resolve_tools([make_tool("web_search")], None)

        self.assertIn("reserved for the hosted web search tool", str(context.exception))

    def test_rejects_duplicate_web_search_tools(self):
        with self.assertRaises(ToolError) as context:
            resolve_tools([make_web_search_tool(), make_web_search_tool()], None)

        self.assertIn("defined multiple times", str(context.exception))

    def test_resolves_hosted_web_search_tool(self):
        resolved = resolve_tools([make_tool("weather"), make_web_search_tool()], None)

        self.assertEqual(resolved.tool_names, ["weather", "web_search"])
        self.assertEqual([tool.name for tool in resolved.function_tools], ["weather"])
        self.assertTrue(resolved.has_web_search)

    def test_filters_tools_for_selected_web_search_and_function(self):
        resolved = resolve_tools(
            [make_tool("weather"), make_tool("time"), make_web_search_tool()],
            {"tools": ["web_search", "weather"]},
        )

        self.assertEqual(resolved.tool_names, ["web_search", "weather"])
        self.assertEqual([tool.name for tool in resolved.function_tools], ["weather"])
        self.assertTrue(resolved.has_web_search)
        self.assertTrue(resolved.is_explicit)

    def test_filters_hosted_web_search_for_unsupported_provider(self):
        resolved = resolve_tools([make_web_search_tool()], {"mode": "required"})
        filtered = filter_resolved_tools_for_provider(
            resolved,
            supports_web_search=False,
        )

        self.assertEqual(filtered.tools, [])
        self.assertFalse(filtered.requires_tool)
        self.assertFalse(filtered.is_explicit)


class SchemaTests(unittest.TestCase):
    def test_get_schema_as_dict_keeps_strict_schema_intact(self):
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

        cleaned = get_schema_as_dict(
            schema,
            strict=True,
        )

        self.assertNotIn("additionalProperties", cleaned)
        self.assertEqual(cleaned["properties"]["email"]["format"], "email")
        self.assertEqual(cleaned["properties"]["url"]["format"], "uri")
        self.assertEqual(
            cleaned["properties"]["url"]["examples"],
            ["https://example.com"],
        )
        self.assertEqual(cleaned["properties"]["title"]["minLength"], 20)
        self.assertEqual(cleaned["properties"]["score"]["minimum"], 0)
        self.assertEqual(cleaned["properties"]["score"]["maximum"], 1)
        self.assertEqual(cleaned["properties"]["bullet_points"]["minItems"], 1)
        self.assertEqual(cleaned["properties"]["bullet_points"]["maxItems"], 3)
        self.assertIsNot(cleaned, schema)
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
            strict=False,
        )

        self.assertNotIn("additionalProperties", preserved)
        self.assertEqual(preserved["properties"]["url"]["format"], "uri")
        self.assertEqual(
            preserved["properties"]["url"]["examples"],
            ["https://example.com"],
        )
        self.assertEqual(preserved["properties"]["title"]["minLength"], 20)

    def test_get_schema_as_dict_keeps_all_of_and_ref_in_strict_schema(self):
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
            strict=True,
        )

        result = cleaned["properties"]["result"]
        self.assertEqual(result["allOf"], [{"$ref": "#/$defs/Result"}])
        self.assertEqual(result["description"], "Resolved result")
        self.assertEqual(
            cleaned["$defs"]["Result"]["properties"]["email"]["format"],
            "email",
        )
        self.assertEqual(
            cleaned["$defs"]["Result"]["properties"]["url"]["format"],
            "uri",
        )

    def test_process_schema_filters_fields_without_removing_property_names(self):
        schema = {
            "type": "object",
            "title": "Search result",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email",
                    "minLength": 3,
                },
                "url": {
                    "type": "string",
                    "format": "uri",
                },
                "score": {
                    "type": "number",
                    "minimum": 0,
                },
            },
            "required": ["email", "url", "score"],
            "additionalProperties": False,
        }

        processed = process_schema(
            schema,
            flatten_refs=False,
            flatten_allof=False,
            supported_string_types=["email"],
            supported_schema_fields=[
                "type",
                "properties",
                "format",
                "required",
                "additionalProperties",
            ],
        )

        self.assertNotIn("title", processed)
        self.assertEqual(
            sorted(processed["properties"]),
            ["email", "score", "url"],
        )
        self.assertEqual(processed["properties"]["email"]["format"], "email")
        self.assertNotIn("format", processed["properties"]["url"])
        self.assertNotIn("minLength", processed["properties"]["email"])
        self.assertNotIn("minimum", processed["properties"]["score"])
        self.assertEqual(schema["properties"]["url"]["format"], "uri")

    def test_process_schema_flattens_local_refs_and_all_of(self):
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

        processed = process_schema(
            schema,
            flatten_refs=True,
            flatten_allof=True,
            supported_string_types=["email", "uri"],
            supported_schema_fields=[
                "type",
                "properties",
                "required",
                "format",
                "description",
            ],
        )

        result = processed["properties"]["result"]
        self.assertNotIn("$defs", processed)
        self.assertNotIn("allOf", result)
        self.assertNotIn("$ref", result)
        self.assertEqual(result["type"], "object")
        self.assertEqual(result["description"], "Resolved result")
        self.assertEqual(result["required"], ["email", "url"])
        self.assertEqual(
            result["properties"]["email"],
            {"type": "string", "format": "email"},
        )
        self.assertEqual(
            result["properties"]["url"],
            {"type": "string", "format": "uri"},
        )

    def test_process_schema_flattens_all_of_inside_preserved_defs(self):
        schema = {
            "type": "object",
            "properties": {
                "result": {"$ref": "#/$defs/Result"},
            },
            "$defs": {
                "Result": {
                    "allOf": [
                        {
                            "$ref": "#/$defs/TextValue",
                        }
                    ],
                    "description": "Resolved result",
                },
                "TextValue": {
                    "type": "string",
                    "format": "email",
                },
            },
        }

        processed = process_schema(
            schema,
            flatten_refs=False,
            flatten_allof=True,
            supported_string_types=["email"],
            supported_schema_fields=[
                "$defs",
                "$ref",
                "description",
                "format",
                "properties",
                "type",
            ],
        )

        self.assertEqual(processed["properties"]["result"], {"$ref": "#/$defs/Result"})
        self.assertEqual(
            processed["$defs"]["Result"],
            {
                "type": "string",
                "format": "email",
                "description": "Resolved result",
            },
        )

    def test_process_schema_can_ensure_additional_properties_false(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                        },
                    },
                },
            },
            "$defs": {
                "Existing": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "value": {
                            "type": "string",
                        }
                    },
                }
            },
        }

        processed = process_schema(
            schema,
            flatten_refs=False,
            flatten_allof=False,
            ensure_additional_properties=True,
            supported_string_types=[],
            supported_schema_fields=[
                "$defs",
                "additionalProperties",
                "properties",
                "type",
            ],
        )

        self.assertFalse(processed["additionalProperties"])
        self.assertFalse(
            processed["properties"]["metadata"]["additionalProperties"]
        )
        self.assertFalse(processed["$defs"]["Existing"]["additionalProperties"])
        self.assertNotIn("additionalProperties", processed["properties"]["name"])

    def test_process_schema_can_remove_additional_properties(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "source": {
                            "type": "string",
                        },
                    },
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additional_properties": False,
                        "properties": {
                            "name": {
                                "type": "string",
                            }
                        },
                    },
                },
            },
            "$defs": {
                "Existing": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "value": {
                            "type": "string",
                        }
                    },
                }
            },
        }

        processed = process_schema(
            schema,
            flatten_refs=False,
            flatten_allof=False,
            remove_additional_properties=True,
        )

        processed_text = repr(processed)
        self.assertNotIn("additionalProperties", processed_text)
        self.assertNotIn("additional_properties", processed_text)
        self.assertEqual(processed["properties"]["metadata"]["type"], "object")
        self.assertEqual(processed["$defs"]["Existing"]["type"], "object")

    def test_process_schema_remove_additional_properties_wins_over_ensure(self):
        processed = process_schema(
            {
                "type": "object",
                "properties": {
                    "metadata": {
                        "type": "object",
                    },
                },
            },
            flatten_refs=False,
            flatten_allof=False,
            ensure_additional_properties=True,
            remove_additional_properties=True,
        )

        self.assertNotIn("additionalProperties", repr(processed))

    def test_process_schema_preserves_multi_branch_any_of(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {
                            "type": "string",
                            "format": "email",
                        },
                        {
                            "type": "null",
                        },
                    ]
                }
            },
        }

        processed = process_schema(
            schema,
            flatten_refs=True,
            flatten_allof=True,
            supported_string_types=["email"],
            supported_schema_fields=[
                "anyOf",
                "format",
                "properties",
                "type",
            ],
        )

        self.assertEqual(
            processed["properties"]["value"]["anyOf"],
            [
                {
                    "type": "string",
                    "format": "email",
                },
                {
                    "type": "null",
                },
            ],
        )

    def test_process_schema_preserves_single_branch_any_of(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {
                            "type": "string",
                            "format": "email",
                        },
                    ],
                    "description": "Wrapped value",
                }
            },
        }

        processed = process_schema(
            schema,
            flatten_refs=True,
            flatten_allof=True,
            supported_string_types=["email"],
            supported_schema_fields=[
                "anyOf",
                "description",
                "format",
                "properties",
                "type",
            ],
        )

        self.assertEqual(
            processed["properties"]["value"],
            {
                "anyOf": [
                    {
                        "type": "string",
                        "format": "email",
                    },
                ],
                "description": "Wrapped value",
            },
        )

    def test_process_schema_keeps_defs_when_not_flattening(self):
        schema = {
            "type": "object",
            "properties": {
                "result": {"$ref": "#/$defs/Result"},
            },
            "$defs": {
                "Result": {
                    "type": "string",
                    "format": "uuid",
                }
            },
        }

        processed = process_schema(
            schema,
            flatten_refs=False,
            flatten_allof=False,
            supported_string_types=["email"],
            supported_schema_fields=["type", "properties", "$ref", "$defs", "format"],
        )

        self.assertEqual(processed["properties"]["result"]["$ref"], "#/$defs/Result")
        self.assertIn("$defs", processed)
        self.assertEqual(processed["$defs"]["Result"], {"type": "string"})

    def test_process_schema_flatten_keeps_recursive_defs_ref_resolvable(self):
        schema = {
            "type": "object",
            "properties": {
                "linked_list": {
                    "$ref": "#/$defs/linked_list_node",
                }
            },
            "$defs": {
                "linked_list_node": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                        },
                        "next": {
                            "anyOf": [
                                {
                                    "$ref": "#/$defs/linked_list_node",
                                },
                                {
                                    "type": "null",
                                },
                            ]
                        },
                    },
                    "additionalProperties": False,
                    "required": ["next", "value"],
                }
            },
            "additionalProperties": False,
            "required": ["linked_list"],
        }

        processed = process_schema(
            schema,
            flatten_refs=True,
            flatten_allof=True,
            supported_string_types=[],
            supported_schema_fields=[
                "$defs",
                "$ref",
                "additionalProperties",
                "anyOf",
                "properties",
                "required",
                "type",
            ],
        )

        linked_list = processed["properties"]["linked_list"]
        self.assertEqual(linked_list["type"], "object")
        self.assertIn("$defs", processed)
        self.assertEqual(
            linked_list["properties"]["next"]["anyOf"][0],
            {"$ref": "#/$defs/linked_list_node"},
        )

    def test_process_schema_flatten_handles_root_recursive_ref(self):
        schema = {
            "type": "object",
            "properties": {
                "children": {
                    "type": "array",
                    "items": {
                        "$ref": "#",
                    },
                }
            },
            "required": ["children"],
            "additionalProperties": False,
        }

        processed = process_schema(
            schema,
            flatten_refs=True,
            flatten_allof=True,
            supported_string_types=[],
            supported_schema_fields=[
                "$ref",
                "additionalProperties",
                "items",
                "properties",
                "required",
                "type",
            ],
        )

        self.assertEqual(
            processed["properties"]["children"]["items"]["properties"]["children"][
                "items"
            ],
            {"$ref": "#"},
        )


if __name__ == "__main__":
    unittest.main()
