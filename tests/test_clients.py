import unittest
from types import SimpleNamespace

from pydantic import BaseModel

from llmai import AnthropicClient, GoogleClient, OpenAIClient
from llmai.shared import JSONSchemaResponse, Tool, ToolChoice, UserMessage


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
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class AnswerSchema(BaseModel):
    answer: str


class ClientBehaviorTests(unittest.TestCase):
    def test_openai_tool_choice_mapping_uses_required_and_optional_lists(self):
        client = OpenAIClient(api_key="test")

        openai_tools, tool_choice = client._get_openai_tools_and_tool_choice_or_omit(
            [make_tool("weather"), make_tool("time")],
            ToolChoice(required=["weather"], optional=["time"]),
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in openai_tools],
            ["weather", "time"],
        )
        self.assertEqual(tool_choice["type"], "allowed_tools")
        self.assertEqual(
            [group["mode"] for group in tool_choice["allowed_tools"]],
            ["required", "auto"],
        )

    def test_openai_generate_returns_tool_calls_without_execution(self):
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
            tools=[make_tool("get_weather")],
            tool_choice=ToolChoice(optional=["get_weather"]),
        )

        self.assertEqual(result.tool_calls[0].name, "get_weather")
        self.assertEqual(result.messages[-1].tool_calls[0].id, "call_1")
        self.assertEqual(
            fake_completions.calls[0]["tool_choice"]["allowed_tools"][0]["mode"],
            "auto",
        )

    def test_openai_stream_emits_tool_chunks_and_completion_tool_calls(self):
        events = iter(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello", tool_calls=None))]
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

    def test_anthropic_required_tool_choice_forces_single_tool(self):
        client = AnthropicClient(api_key="test")

        anthropic_tools, tool_choice = (
            client._get_anthropic_tools_and_tool_choice_or_omit(
                [make_tool("weather"), make_tool("time")],
                ToolChoice(required=["weather"], optional=["time"]),
                None,
                None,
            )
        )

        self.assertEqual([tool["name"] for tool in anthropic_tools], ["weather"])
        self.assertEqual(tool_choice, {"type": "tool", "name": "weather"})

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
            ToolChoice(required=["weather"], optional=["time"]),
            None,
            None,
        )

        self.assertEqual(len(google_tools), 2)
        self.assertEqual(tool_config.function_calling_config.mode, "ANY")
        self.assertEqual(
            tool_config.function_calling_config.allowed_function_names,
            ["weather"],
        )

    def test_google_generate_hides_internal_response_schema_tool(self):
        fake_response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text=None,
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
        fake_models = FakeGoogleModels(fake_response)

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
