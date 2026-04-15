import unittest
from types import SimpleNamespace

from pydantic import BaseModel

from llmai import AnthropicClient, GoogleClient, OpenAIClient
from llmai.shared import (
    AssistantMessage,
    ImageContentPart,
    JSONSchemaResponse,
    LLMError,
    TextContentPart,
    Tool,
    UserMessage,
)


def make_tool(name: str) -> Tool:
    return Tool(name=name, description=f"{name} description")


class FakeOpenAICompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeAnthropicMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeGoogleModels:
    def __init__(self, response=None, stream_response=None):
        self.response = response
        self.stream_response = stream_response
        self.calls = []
        self.stream_calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def generate_content_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return self.stream_response


class AnswerSchema(BaseModel):
    answer: str


class ClientBehaviorTests(unittest.TestCase):
    def test_openai_required_tool_choice_uses_standard_function_selector(self):
        client = OpenAIClient(api_key="test")

        openai_tools, tool_choice = client._get_openai_tools_and_tool_choice_or_omit(
            [make_tool("weather"), make_tool("time")],
            {"required": ["weather"]},
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in openai_tools],
            ["weather"],
        )
        self.assertEqual(
            tool_choice,
            {
                "type": "function",
                "function": {"name": "weather"},
            },
        )

    def test_openai_generate_filters_tools_without_custom_allowed_tools_wrapper(self):
        fake_message = SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(
                        name="get_weather",
                        arguments='{"city":"Kathmandu"}',
                    ),
                )
            ],
        )
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message)]
        )
        fake_completions = FakeOpenAICompletions(fake_response)

        client = OpenAIClient(api_key="test")
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        result = client.generate(
            model="gpt-test",
            messages=[UserMessage(content="Weather?")],
            tools=[make_tool("get_weather"), make_tool("time")],
            tool_choice={"optional": ["get_weather"]},
        )

        self.assertEqual(result.tool_calls[0].name, "get_weather")
        self.assertEqual(result.messages[-1].tool_calls[0].id, "call_1")
        self.assertEqual(
            [tool["function"]["name"] for tool in fake_completions.calls[0]["tools"]],
            ["get_weather"],
        )
        self.assertFalse(
            isinstance(fake_completions.calls[0]["tool_choice"], dict)
            and fake_completions.calls[0]["tool_choice"].get("type") == "allowed_tools"
        )

    def test_openai_serializes_user_image_parts(self):
        client = OpenAIClient(api_key="test")

        messages = client._messages_to_openai_messages(
            [
                UserMessage(
                    content=[
                        TextContentPart(text="Describe this"),
                        ImageContentPart(data=b"png-bytes", mime_type="image/png"),
                    ]
                )
            ]
        )

        self.assertEqual(messages[0]["content"][0]["type"], "text")
        self.assertEqual(messages[0]["content"][0]["text"], "Describe this")
        self.assertEqual(messages[0]["content"][1]["type"], "image_url")
        self.assertTrue(
            messages[0]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )

    def test_openai_rejects_assistant_image_history(self):
        client = OpenAIClient(api_key="test")

        with self.assertRaises(LLMError):
            client._messages_to_openai_messages(
                [
                    AssistantMessage(
                        content=[ImageContentPart(url="https://example.com/cat.png")]
                    )
                ]
            )


    def test_openai_stream_emits_tool_chunks_and_completion_tool_calls(self):
        events = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="Hello", tool_calls=None)
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call_1",
                                        function=SimpleNamespace(
                                            name="get_weather",
                                            arguments='{"city":"Kath',
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id=None,
                                        function=SimpleNamespace(
                                            name=None,
                                            arguments='mandu"}',
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                ),
            ]
        )
        fake_completions = FakeOpenAICompletions(events)

        client = OpenAIClient(api_key="test")
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        chunks = list(
            client.stream(
                model="gpt-test",
                messages=[UserMessage(content="Weather?")],
                tools=[make_tool("get_weather")],
            )
        )

        self.assertEqual(chunks[0].type, "stream_content")
        self.assertEqual(chunks[0].source, "direct")
        self.assertEqual(chunks[1].source, "tool")
        self.assertEqual(chunks[2].source, "tool")
        self.assertEqual(chunks[-1].type, "stream_completion")
        self.assertEqual(chunks[-1].tool_calls[0].name, "get_weather")
        self.assertEqual(
            chunks[-1].tool_calls[0].arguments,
            '{"city":"Kathmandu"}',
        )

    def test_anthropic_required_tool_choice_keeps_optional_tools_visible(self):
        client = AnthropicClient(api_key="test")

        anthropic_tools, tool_choice = (
            client._get_anthropic_tools_and_tool_choice_or_omit(
                [make_tool("weather"), make_tool("time")],
                {"required": ["weather"], "optional": ["time"]},
                None,
                None,
            )
        )

        self.assertEqual([tool["name"] for tool in anthropic_tools], ["weather", "time"])
        self.assertEqual(tool_choice, {"type": "any"})

    def test_anthropic_serializes_user_image_parts(self):
        client = AnthropicClient(api_key="test")

        messages = client._messages_to_anthropic_messages(
            [
                UserMessage(
                    content=[
                        TextContentPart(text="Describe this"),
                        ImageContentPart(data=b"png-bytes", mime_type="image/png"),
                    ]
                )
            ]
        )

        self.assertEqual(messages[0]["content"][0]["type"], "text")
        self.assertEqual(messages[0]["content"][1]["type"], "image")
        self.assertEqual(messages[0]["content"][1]["source"]["type"], "base64")
        self.assertEqual(messages[0]["content"][1]["source"]["media_type"], "image/png")

    def test_anthropic_generate_captures_thinking(self):
        fake_response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="internal"),
                SimpleNamespace(type="text", text="final answer"),
            ]
        )
        fake_messages = FakeAnthropicMessages(fake_response)

        client = AnthropicClient(api_key="test")
        client._client = SimpleNamespace(messages=fake_messages)

        result = client.generate(
            model="claude-test",
            messages=[UserMessage(content="Answer me")],
        )

        self.assertEqual(result.content, "final answer")
        self.assertEqual(result.messages[-1].thinking, "internal")

    def test_anthropic_generate_hides_internal_response_schema_tool(self):
        fake_response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="schema_1",
                    name="ResponseSchema",
                    input={"answer": "done"},
                )
            ]
        )
        fake_messages = FakeAnthropicMessages(fake_response)

        client = AnthropicClient(api_key="test")
        client._client = SimpleNamespace(messages=fake_messages)

        result = client.generate(
            model="claude-test",
            messages=[UserMessage(content="Answer in JSON")],
            response_format=JSONSchemaResponse(json_schema=AnswerSchema),
        )

        self.assertEqual(result.content, {"answer": "done"})
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.messages[-1].tool_calls, [])

    def test_google_required_tool_choice_uses_any_mode_with_allowed_names(self):
        client = GoogleClient(api_key="test")

        google_tools, tool_config = client._get_google_tools_and_tool_config(
            [make_tool("weather"), make_tool("time")],
            {"required": ["weather"]},
            None,
            None,
        )

        self.assertEqual(len(google_tools), 1)
        self.assertEqual(tool_config.function_calling_config.mode, "ANY")
        self.assertEqual(
            tool_config.function_calling_config.allowed_function_names,
            ["weather"],
        )

    def test_google_serializes_user_images(self):
        client = GoogleClient(api_key="test")

        messages = client._messages_to_google_messages(
            [
                UserMessage(
                    content=[
                        TextContentPart(text="Describe this"),
                        ImageContentPart(url="https://example.com/cat.png", mime_type="image/png"),
                        ImageContentPart(data=b"png-bytes", mime_type="image/png"),
                    ]
                )
            ]
        )

        self.assertEqual(messages[0].parts[0].text, "Describe this")
        self.assertEqual(messages[0].parts[1].file_data.file_uri, "https://example.com/cat.png")
        self.assertEqual(messages[0].parts[1].file_data.mime_type, "image/png")
        self.assertEqual(messages[0].parts[2].inline_data.data, b"png-bytes")
        self.assertEqual(messages[0].parts[2].inline_data.mime_type, "image/png")

    def test_google_generate_returns_multimodal_content_and_thinking(self):
        fake_response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text="I found an image.",
                                thought=False,
                                inline_data=None,
                                file_data=None,
                                function_call=None,
                            ),
                            SimpleNamespace(
                                text="hidden reasoning",
                                thought=True,
                                inline_data=None,
                                file_data=None,
                                function_call=None,
                            ),
                            SimpleNamespace(
                                text=None,
                                thought=None,
                                inline_data=SimpleNamespace(
                                    data=b"img-bytes",
                                    mime_type="image/png",
                                ),
                                file_data=None,
                                function_call=None,
                            ),
                        ]
                    )
                )
            ]
        )
        fake_models = FakeGoogleModels(response=fake_response)

        client = GoogleClient(api_key="test")
        client._client = SimpleNamespace(models=fake_models)

        result = client.generate(
            model="gemini-test",
            messages=[UserMessage(content="Show me something")],
        )

        self.assertEqual(result.messages[-1].thinking, "hidden reasoning")
        self.assertIsInstance(result.content, list)
        self.assertEqual(result.content[0].text, "I found an image.")
        self.assertEqual(result.content[1].mime_type, "image/png")
        self.assertEqual(result.content[1].data, b"img-bytes")

    def test_google_stream_includes_images_only_in_completion(self):
        fake_models = FakeGoogleModels(
            stream_response=iter(
                [
                    SimpleNamespace(
                        candidates=[
                            SimpleNamespace(
                                content=SimpleNamespace(
                                    parts=[
                                        SimpleNamespace(
                                            text="Hello",
                                            thought=False,
                                            inline_data=None,
                                            file_data=None,
                                            function_call=None,
                                        )
                                    ]
                                )
                            )
                        ]
                    ),
                    SimpleNamespace(
                        candidates=[
                            SimpleNamespace(
                                content=SimpleNamespace(
                                    parts=[
                                        SimpleNamespace(
                                            text=None,
                                            thought=None,
                                            inline_data=SimpleNamespace(
                                                data=b"img-bytes",
                                                mime_type="image/png",
                                            ),
                                            file_data=None,
                                            function_call=None,
                                        )
                                    ]
                                )
                            )
                        ]
                    ),
                ]
            )
        )

        client = GoogleClient(api_key="test")
        client._client = SimpleNamespace(models=fake_models)

        chunks = list(
            client.stream(
                model="gemini-test",
                messages=[UserMessage(content="Show me something")],
            )
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].type, "stream_content")
        self.assertEqual(chunks[0].chunk, "Hello")
        self.assertEqual(chunks[-1].type, "stream_completion")
        self.assertIsInstance(chunks[-1].content, list)
        self.assertEqual(chunks[-1].content[0].text, "Hello")
        self.assertEqual(chunks[-1].content[1].mime_type, "image/png")

    def test_google_generate_hides_internal_response_schema_tool(self):
        fake_response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text=None,
                                thought=None,
                                inline_data=None,
                                file_data=None,
                                function_call=SimpleNamespace(
                                    id="schema_1",
                                    name="ResponseSchema",
                                    args={"answer": "done"},
                                ),
                            )
                        ]
                    )
                )
            ]
        )
        fake_models = FakeGoogleModels(response=fake_response)

        client = GoogleClient(api_key="test")
        client._client = SimpleNamespace(models=fake_models)

        result = client.generate(
            model="gemini-test",
            messages=[UserMessage(content="Answer in JSON")],
            response_format=JSONSchemaResponse(json_schema=AnswerSchema),
            use_tools_for_structured_output=True,
        )

        self.assertEqual(result.content, {"answer": "done"})
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.messages[-1].tool_calls, [])


if __name__ == "__main__":
    unittest.main()
