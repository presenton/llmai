import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import anthropic
import httpx
import openai
from botocore import exceptions as botocore_exceptions
from google.genai import errors as google_errors
from pydantic import BaseModel

from llmai import AnthropicClient, BedrockClient, GoogleClient, OpenAIClient
from llmai.shared import (
    AssistantMessage,
    ImageContentPart,
    LLMAuthenticationError,
    LLMConfigurationError,
    JSONSchemaResponse,
    LLMError,
    LLMRateLimitError,
    TextContentPart,
    Tool,
    UserMessage,
)


def make_tool(name: str) -> Tool:
    return Tool(name=name, description=f"{name} description")


def text_parts(text: str) -> list[TextContentPart]:
    return [TextContentPart(text=text)]


class FakeOpenAICompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeAnthropicMessages:
    def __init__(self, response=None, stream_response=None):
        self.response = response
        self.stream_response = stream_response
        self.calls = []
        self.stream_calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if isinstance(self.stream_response, Exception):
            raise self.stream_response
        return self.stream_response


class FakeAnthropicStream:
    def __init__(self, events, final_message=None):
        self.events = events
        self.final_message = final_message

    def __iter__(self):
        return iter(self.events)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def get_final_message(self):
        return self.final_message


class FakeGoogleModels:
    def __init__(self, response=None, stream_response=None):
        self.response = response
        self.stream_response = stream_response
        self.calls = []
        self.stream_calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def generate_content_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if isinstance(self.stream_response, Exception):
            raise self.stream_response
        return self.stream_response


class FakeBedrockRuntimeClient:
    def __init__(self, response=None, stream_response=None):
        self.response = response
        self.stream_response = stream_response
        self.calls = []
        self.stream_calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def converse_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if isinstance(self.stream_response, Exception):
            raise self.stream_response
        return self.stream_response


class FakeBoto3Session:
    def __init__(self, runtime_client, *, kwargs):
        self.runtime_client = runtime_client
        self.kwargs = kwargs
        self.client_calls = []

    def client(self, service_name, **kwargs):
        self.client_calls.append((service_name, kwargs))
        return self.runtime_client


class AnswerSchema(BaseModel):
    answer: str


class ClientBehaviorTests(unittest.TestCase):
    def make_bedrock_client(self, runtime_client, **client_kwargs):
        captured_sessions = []

        def fake_session_factory(**kwargs):
            session = FakeBoto3Session(runtime_client, kwargs=kwargs)
            captured_sessions.append(session)
            return session

        patcher = patch("llmai.bedrock.client.boto3.Session", side_effect=fake_session_factory)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        client = BedrockClient(**client_kwargs)
        return client, captured_sessions, mocked

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
            messages=[UserMessage(content=text_parts("Weather?"))],
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

    def test_openai_generate_returns_usage_and_duration(self):
        fake_message = SimpleNamespace(
            content="final answer",
            tool_calls=None,
        )
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message)],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
                prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
            ),
        )
        fake_completions = FakeOpenAICompletions(fake_response)

        client = OpenAIClient(api_key="test")
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        result = client.generate(
            model="gpt-test",
            messages=[UserMessage(content=text_parts("Hello"))],
        )

        self.assertEqual(result.usage.input_tokens, 11)
        self.assertEqual(result.usage.output_tokens, 7)
        self.assertEqual(result.usage.total_tokens, 18)
        self.assertEqual(result.usage.details["prompt_tokens_details"]["cached_tokens"], 2)
        self.assertEqual(
            result.usage.details["completion_tokens_details"]["reasoning_tokens"],
            3,
        )
        self.assertIsNotNone(result.duration_seconds)
        self.assertGreaterEqual(result.duration_seconds, 0)

    def test_openai_generate_wraps_provider_auth_errors(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(401, request=request)
        fake_completions = FakeOpenAICompletions(
            openai.AuthenticationError(
                "bad api key",
                response=response,
                body={},
            )
        )

        client = OpenAIClient(api_key="test")
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        with self.assertRaises(LLMAuthenticationError) as context:
            client.generate(
                model="gpt-test",
                messages=[UserMessage(content=text_parts("Hello"))],
            )

        self.assertEqual(context.exception.status_code, 401)
        self.assertEqual(context.exception.provider, "openai")

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
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=4,
                        total_tokens=14,
                        completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
                    ),
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
                messages=[UserMessage(content=text_parts("Weather?"))],
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
        self.assertEqual(chunks[-1].usage.input_tokens, 10)
        self.assertEqual(chunks[-1].usage.output_tokens, 4)
        self.assertEqual(chunks[-1].usage.total_tokens, 14)
        self.assertEqual(
            chunks[-1].usage.details["completion_tokens_details"]["reasoning_tokens"],
            1,
        )
        self.assertIsNotNone(chunks[-1].duration_seconds)
        self.assertGreaterEqual(chunks[-1].duration_seconds, 0)
        self.assertEqual(
            fake_completions.calls[0]["stream_options"],
            {"include_usage": True},
        )

    def test_openai_generate_parses_structured_output(self):
        fake_message = SimpleNamespace(
            content='{"answer":"pong"}',
            tool_calls=None,
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
            messages=[UserMessage(content=text_parts("Answer in JSON"))],
            response_format=JSONSchemaResponse(json_schema=AnswerSchema),
        )

        self.assertEqual(result.content, {"answer": "pong"})

    def test_openai_generate_accepts_dict_json_schema(self):
        fake_message = SimpleNamespace(
            content='{"answer":"pong"}',
            tool_calls=None,
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
            messages=[UserMessage(content=text_parts("Answer in JSON"))],
            response_format=JSONSchemaResponse(
                json_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                }
            ),
        )

        self.assertEqual(result.content, {"answer": "pong"})
        self.assertEqual(
            fake_completions.calls[0]["response_format"]["json_schema"]["schema"]["type"],
            "object",
        )

    def test_openai_generate_sanitizes_unsupported_json_schema_keywords(self):
        fake_message = SimpleNamespace(
            content='{"answer":"pong"}',
            tool_calls=None,
        )
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message)]
        )
        fake_completions = FakeOpenAICompletions(fake_response)
        schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 20,
                    "maxLength": 100,
                },
                "email": {
                    "type": "string",
                    "format": "email",
                },
                "username": {
                    "type": "string",
                    "pattern": "^@[a-zA-Z0-9_]+$",
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
                "bulletPoints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                },
                "image": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "minWords": 2,
                            "maxWords": 5,
                        }
                    },
                    "required": ["prompt"],
                },
            },
            "required": [
                "title",
                "email",
                "username",
                "url",
                "score",
                "bulletPoints",
                "image",
            ],
        }

        client = OpenAIClient(api_key="test")
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        client.generate(
            model="gpt-test",
            messages=[UserMessage(content=text_parts("Answer in JSON"))],
            response_format=JSONSchemaResponse(json_schema=schema),
        )

        sanitized_schema = fake_completions.calls[0]["response_format"]["json_schema"][
            "schema"
        ]
        self.assertEqual(
            sanitized_schema["properties"]["email"]["format"],
            "email",
        )
        self.assertEqual(
            sanitized_schema["properties"]["username"]["pattern"],
            "^@[a-zA-Z0-9_]+$",
        )
        self.assertNotIn("format", sanitized_schema["properties"]["url"])
        self.assertNotIn("examples", sanitized_schema["properties"]["url"])
        self.assertEqual(sanitized_schema["properties"]["score"]["minimum"], 0)
        self.assertEqual(sanitized_schema["properties"]["score"]["maximum"], 1)
        self.assertEqual(sanitized_schema["properties"]["bulletPoints"]["minItems"], 1)
        self.assertEqual(sanitized_schema["properties"]["bulletPoints"]["maxItems"], 3)
        self.assertNotIn("minLength", sanitized_schema["properties"]["title"])
        self.assertNotIn("maxLength", sanitized_schema["properties"]["title"])
        self.assertNotIn(
            "minWords",
            sanitized_schema["properties"]["image"]["properties"]["prompt"],
        )
        self.assertNotIn(
            "maxWords",
            sanitized_schema["properties"]["image"]["properties"]["prompt"],
        )
        self.assertEqual(schema["properties"]["url"]["format"], "uri")
        self.assertEqual(schema["properties"]["title"]["minLength"], 20)

    def test_openai_tools_sanitize_unsupported_json_schema_keywords(self):
        client = OpenAIClient(api_key="test")
        schema = {
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
                "title": {
                    "type": "string",
                    "minLength": 20,
                },
            },
            "required": ["email", "url", "title"],
        }

        tools = client._llm_tools_to_openai_tools(
            [
                Tool(
                    name="save_link",
                    description="Save a link",
                    schema=schema,
                    strict=True,
                )
            ]
        )

        parameters = tools[0]["function"]["parameters"]
        self.assertEqual(parameters["properties"]["email"]["format"], "email")
        self.assertNotIn("format", parameters["properties"]["url"])
        self.assertNotIn("minLength", parameters["properties"]["title"])
        self.assertEqual(schema["properties"]["url"]["format"], "uri")
        self.assertEqual(schema["properties"]["title"]["minLength"], 20)

    def test_openai_generate_uses_custom_json_schema_name_and_strict(self):
        fake_message = SimpleNamespace(
            content='{"answer":"pong"}',
            tool_calls=None,
        )
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message)]
        )
        fake_completions = FakeOpenAICompletions(fake_response)

        client = OpenAIClient(api_key="test")
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
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

        result = client.generate(
            model="gpt-test",
            messages=[UserMessage(content=text_parts("Answer in JSON"))],
            response_format=JSONSchemaResponse(
                name="final_answer",
                strict=False,
                json_schema=schema,
            ),
        )

        self.assertEqual(result.content, {"answer": "pong"})
        self.assertEqual(
            fake_completions.calls[0]["response_format"]["json_schema"]["name"],
            "final_answer",
        )
        self.assertFalse(
            fake_completions.calls[0]["response_format"]["json_schema"]["strict"]
        )
        sent_schema = fake_completions.calls[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(sent_schema["properties"]["title"]["minLength"], 20)
        self.assertEqual(sent_schema["properties"]["url"]["format"], "uri")
        self.assertEqual(
            sent_schema["properties"]["url"]["examples"],
            ["https://example.com"],
        )

    def test_openai_tools_keep_non_strict_json_schema_keywords(self):
        client = OpenAIClient(api_key="test")
        schema = {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                },
                "title": {
                    "type": "string",
                    "minLength": 20,
                },
            },
            "required": ["url", "title"],
        }

        tools = client._llm_tools_to_openai_tools(
            [
                Tool(
                    name="save_link",
                    description="Save a link",
                    schema=schema,
                    strict=False,
                )
            ]
        )

        parameters = tools[0]["function"]["parameters"]
        self.assertEqual(parameters["properties"]["url"]["format"], "uri")
        self.assertEqual(parameters["properties"]["title"]["minLength"], 20)

    def test_openai_generate_raises_configuration_error_for_bare_basemodel_schema(self):
        fake_completions = FakeOpenAICompletions(response=None)

        client = OpenAIClient(api_key="test")
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        with self.assertRaises(LLMConfigurationError) as context:
            client.generate(
                model="gpt-test",
                messages=[UserMessage(content=text_parts("Answer in JSON"))],
                response_format=JSONSchemaResponse(json_schema=BaseModel),
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.provider, "openai")
        self.assertIn("BaseModel subclass", context.exception.message)
        self.assertEqual(fake_completions.calls, [])

    def test_openai_stream_parses_structured_output_in_final_completion(self):
        events = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content='{"answer":"', tool_calls=None)
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content='pong"}', tool_calls=None)
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
                messages=[UserMessage(content=text_parts("Answer in JSON"))],
                response_format=JSONSchemaResponse(json_schema=AnswerSchema),
            )
        )

        self.assertEqual(chunks[0].chunk, '{"answer":"')
        self.assertEqual(chunks[1].chunk, 'pong"}')
        self.assertEqual(chunks[-1].content, {"answer": "pong"})

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
            messages=[UserMessage(content=text_parts("Answer me"))],
        )

        self.assertEqual(result.content[0].text, "final answer")
        self.assertEqual(result.messages[-1].thinking, "internal")

    def test_anthropic_generate_returns_usage_and_duration(self):
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="final answer")],
            usage=SimpleNamespace(
                input_tokens=10,
                cache_creation_input_tokens=2,
                cache_read_input_tokens=3,
                output_tokens=5,
            ),
        )
        fake_messages = FakeAnthropicMessages(fake_response)

        client = AnthropicClient(api_key="test")
        client._client = SimpleNamespace(messages=fake_messages)

        result = client.generate(
            model="claude-test",
            messages=[UserMessage(content=text_parts("Answer me"))],
        )

        self.assertEqual(result.usage.input_tokens, 15)
        self.assertEqual(result.usage.output_tokens, 5)
        self.assertEqual(result.usage.total_tokens, 20)
        self.assertEqual(result.usage.details["cache_creation_input_tokens"], 2)
        self.assertEqual(result.usage.details["cache_read_input_tokens"], 3)
        self.assertIsNotNone(result.duration_seconds)
        self.assertGreaterEqual(result.duration_seconds, 0)

    def test_anthropic_generate_wraps_provider_rate_limit_errors(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(429, request=request)
        fake_messages = FakeAnthropicMessages(
            anthropic.RateLimitError(
                "slow down",
                response=response,
                body={},
            )
        )

        client = AnthropicClient(api_key="test")
        client._client = SimpleNamespace(messages=fake_messages)

        with self.assertRaises(LLMRateLimitError) as context:
            client.generate(
                model="claude-test",
                messages=[UserMessage(content=text_parts("Answer me"))],
            )

        self.assertEqual(context.exception.status_code, 429)
        self.assertEqual(context.exception.provider, "anthropic")

    def test_anthropic_stream_returns_usage_and_duration(self):
        fake_stream = FakeAnthropicStream(
            events=[
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="Hello"),
                )
            ],
            final_message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=8,
                    cache_creation_input_tokens=1,
                    output_tokens=2,
                )
            ),
        )
        fake_messages = FakeAnthropicMessages(stream_response=fake_stream)

        client = AnthropicClient(api_key="test")
        client._client = SimpleNamespace(messages=fake_messages)

        chunks = list(
            client.stream(
                model="claude-test",
                messages=[UserMessage(content=text_parts("Stream please"))],
            )
        )

        self.assertEqual(chunks[0].chunk, "Hello")
        self.assertEqual(chunks[-1].usage.input_tokens, 9)
        self.assertEqual(chunks[-1].usage.output_tokens, 2)
        self.assertEqual(chunks[-1].usage.total_tokens, 11)
        self.assertEqual(chunks[-1].usage.details["cache_creation_input_tokens"], 1)
        self.assertIsNotNone(chunks[-1].duration_seconds)
        self.assertGreaterEqual(chunks[-1].duration_seconds, 0)

    def test_anthropic_generate_hides_internal_response_schema_tool(self):
        fake_response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="schema_1",
                    name="final_answer",
                    input={"answer": "done"},
                )
            ]
        )
        fake_messages = FakeAnthropicMessages(fake_response)

        client = AnthropicClient(api_key="test")
        client._client = SimpleNamespace(messages=fake_messages)

        result = client.generate(
            model="claude-test",
            messages=[UserMessage(content=text_parts("Answer in JSON"))],
            response_format=JSONSchemaResponse(
                name="final_answer",
                strict=False,
                json_schema=AnswerSchema,
            ),
        )

        self.assertEqual(result.content, {"answer": "done"})
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.messages[-1].tool_calls, [])
        self.assertEqual(fake_messages.calls[0]["tools"][0]["name"], "final_answer")
        self.assertFalse(fake_messages.calls[0]["tools"][0]["strict"])

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
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                tool_use_prompt_token_count=4,
                candidates_token_count=7,
                thoughts_token_count=3,
                total_token_count=24,
            ),
        )
        fake_models = FakeGoogleModels(response=fake_response)

        client = GoogleClient(api_key="test")
        client._client = SimpleNamespace(models=fake_models)

        result = client.generate(
            model="gemini-test",
            messages=[UserMessage(content=text_parts("Show me something"))],
        )

        self.assertEqual(result.messages[-1].thinking, "hidden reasoning")
        self.assertIsInstance(result.content, list)
        self.assertEqual(result.content[0].text, "I found an image.")
        self.assertEqual(result.content[1].mime_type, "image/png")
        self.assertEqual(result.content[1].data, b"img-bytes")
        self.assertEqual(result.usage.input_tokens, 14)
        self.assertEqual(result.usage.output_tokens, 10)
        self.assertEqual(result.usage.total_tokens, 24)
        self.assertEqual(result.usage.details["tool_use_prompt_token_count"], 4)
        self.assertEqual(result.usage.details["thoughts_token_count"], 3)
        self.assertIsNotNone(result.duration_seconds)
        self.assertGreaterEqual(result.duration_seconds, 0)

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
                    SimpleNamespace(
                        candidates=[],
                        usage_metadata=SimpleNamespace(
                            prompt_token_count=6,
                            candidates_token_count=2,
                            total_token_count=8,
                        ),
                    ),
                ]
            )
        )

        client = GoogleClient(api_key="test")
        client._client = SimpleNamespace(models=fake_models)

        chunks = list(
            client.stream(
                model="gemini-test",
                messages=[UserMessage(content=text_parts("Show me something"))],
            )
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].type, "stream_content")
        self.assertEqual(chunks[0].chunk, "Hello")
        self.assertEqual(chunks[-1].type, "stream_completion")
        self.assertIsInstance(chunks[-1].content, list)
        self.assertEqual(chunks[-1].content[0].text, "Hello")
        self.assertEqual(chunks[-1].content[1].mime_type, "image/png")
        self.assertEqual(chunks[-1].usage.input_tokens, 6)
        self.assertEqual(chunks[-1].usage.output_tokens, 2)
        self.assertEqual(chunks[-1].usage.total_tokens, 8)
        self.assertIsNotNone(chunks[-1].duration_seconds)
        self.assertGreaterEqual(chunks[-1].duration_seconds, 0)

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
                                    name="final_answer",
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
            messages=[UserMessage(content=text_parts("Answer in JSON"))],
            response_format=JSONSchemaResponse(
                name="final_answer",
                strict=False,
                json_schema=AnswerSchema,
            ),
            use_tools_for_structured_output=True,
        )

        self.assertEqual(result.content, {"answer": "done"})
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.messages[-1].tool_calls, [])
        self.assertEqual(
            fake_models.calls[0]["config"].tools[0].function_declarations[0].name,
            "final_answer",
        )

    def test_google_generate_wraps_provider_auth_errors(self):
        fake_models = FakeGoogleModels(
            response=google_errors.ClientError(
                401,
                {"message": "bad api key", "status": "UNAUTHENTICATED"},
            )
        )

        client = GoogleClient(api_key="test")
        client._client = SimpleNamespace(models=fake_models)

        with self.assertRaises(LLMAuthenticationError) as context:
            client.generate(
                model="gemini-test",
                messages=[UserMessage(content=text_parts("Hello"))],
            )

        self.assertEqual(context.exception.status_code, 401)
        self.assertEqual(context.exception.provider, "google")

    def test_bedrock_init_supports_api_key_auth(self):
        fake_runtime = FakeBedrockRuntimeClient(response={})

        with patch.dict(os.environ, {}, clear=False):
            client, sessions, _ = self.make_bedrock_client(
                fake_runtime,
                api_key="bedrock-api-key",
                region_name="us-east-1",
            )
            self.assertEqual(os.environ["AWS_BEARER_TOKEN_BEDROCK"], "bedrock-api-key")

        self.assertIsNotNone(client)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].kwargs["region_name"], "us-east-1")
        self.assertEqual(
            sessions[0].client_calls,
            [("bedrock-runtime", {"region_name": "us-east-1"})],
        )

    def test_bedrock_init_uses_unified_configuration_errors(self):
        fake_runtime = FakeBedrockRuntimeClient(response={})

        with self.assertRaises(LLMConfigurationError) as context:
            self.make_bedrock_client(
                fake_runtime,
                api_key="bedrock-api-key",
                region_name="us-east-1",
                aws_access_key_id="aws-id",
                aws_secret_access_key="aws-secret",
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.provider, "bedrock")

    def test_bedrock_init_supports_aws_credentials(self):
        fake_runtime = FakeBedrockRuntimeClient(response={})
        client, sessions, _ = self.make_bedrock_client(
            fake_runtime,
            region_name="us-west-2",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
            aws_session_token="aws-session",
        )

        self.assertIsNotNone(client)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].kwargs["aws_access_key_id"], "aws-id")
        self.assertEqual(sessions[0].kwargs["aws_secret_access_key"], "aws-secret")
        self.assertEqual(sessions[0].kwargs["aws_session_token"], "aws-session")
        self.assertEqual(sessions[0].kwargs["region_name"], "us-west-2")

    def test_bedrock_generate_returns_usage_and_duration(self):
        fake_runtime = FakeBedrockRuntimeClient(
            response={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "hello from bedrock"}],
                    }
                },
                "usage": {
                    "inputTokens": 12,
                    "outputTokens": 5,
                    "totalTokens": 17,
                    "cacheReadInputTokens": 2,
                },
            }
        )
        client, _, _ = self.make_bedrock_client(
            fake_runtime,
            region_name="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        result = client.generate(
            model="anthropic.claude-3-5-haiku",
            messages=[UserMessage(content=text_parts("Hello"))],
            temperature=0.2,
            max_tokens=123,
        )

        self.assertEqual(result.content[0].text, "hello from bedrock")
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertEqual(result.usage.output_tokens, 5)
        self.assertEqual(result.usage.total_tokens, 17)
        self.assertEqual(result.usage.details["cacheReadInputTokens"], 2)
        self.assertIsNotNone(result.duration_seconds)
        self.assertGreaterEqual(result.duration_seconds, 0)
        self.assertEqual(fake_runtime.calls[0]["modelId"], "anthropic.claude-3-5-haiku")
        self.assertEqual(fake_runtime.calls[0]["inferenceConfig"]["temperature"], 0.2)
        self.assertEqual(fake_runtime.calls[0]["inferenceConfig"]["maxTokens"], 123)

    def test_bedrock_generate_uses_native_structured_output(self):
        fake_runtime = FakeBedrockRuntimeClient(
            response={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": '{"answer":"pong"}'}],
                    }
                }
            }
        )
        client, _, _ = self.make_bedrock_client(
            fake_runtime,
            region_name="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        result = client.generate(
            model="anthropic.claude-3-5-haiku",
            messages=[UserMessage(content=text_parts("Answer in JSON"))],
            response_format=JSONSchemaResponse(json_schema=AnswerSchema),
        )

        self.assertEqual(result.content, {"answer": "pong"})
        self.assertEqual(
            fake_runtime.calls[0]["outputConfig"]["textFormat"]["type"],
            "json_schema",
        )
        self.assertIn(
            '"type": "object"',
            fake_runtime.calls[0]["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"],
        )

    def test_bedrock_tool_structured_output_uses_custom_name_and_strict(self):
        fake_runtime = FakeBedrockRuntimeClient(response={})
        client, _, _ = self.make_bedrock_client(
            fake_runtime,
            region_name="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        tool_config = client._get_bedrock_tool_config(
            None,
            None,
            JSONSchemaResponse(
                name="final_answer",
                strict=False,
                json_schema=AnswerSchema,
            ),
            True,
        )

        self.assertEqual(tool_config["tools"][0]["toolSpec"]["name"], "final_answer")
        self.assertFalse(tool_config["tools"][0]["toolSpec"]["strict"])

    def test_bedrock_generate_returns_tool_calls(self):
        fake_runtime = FakeBedrockRuntimeClient(
            response={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call_1",
                                    "name": "get_weather",
                                    "input": {"city": "Kathmandu"},
                                }
                            }
                        ],
                    }
                }
            }
        )
        client, _, _ = self.make_bedrock_client(
            fake_runtime,
            region_name="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        result = client.generate(
            model="anthropic.claude-3-5-haiku",
            messages=[UserMessage(content=text_parts("Weather?"))],
            tools=[make_tool("get_weather"), make_tool("time")],
            tool_choice={"required": ["get_weather"]},
        )

        self.assertEqual(result.tool_calls[0].name, "get_weather")
        self.assertEqual(result.tool_calls[0].arguments, '{"city": "Kathmandu"}')
        self.assertEqual(
            fake_runtime.calls[0]["toolConfig"]["toolChoice"],
            {"tool": {"name": "get_weather"}},
        )

    def test_bedrock_serializes_user_images_and_rejects_assistant_image_history(self):
        fake_runtime = FakeBedrockRuntimeClient(
            response={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "done"}],
                    }
                }
            }
        )
        client, _, _ = self.make_bedrock_client(
            fake_runtime,
            region_name="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        client.generate(
            model="anthropic.claude-3-5-haiku",
            messages=[
                UserMessage(
                    content=[
                        TextContentPart(text="Describe this"),
                        ImageContentPart(data=b"png-bytes", mime_type="image/png"),
                    ]
                )
            ],
        )

        image_block = fake_runtime.calls[0]["messages"][0]["content"][1]["image"]
        self.assertEqual(image_block["format"], "png")
        self.assertEqual(image_block["source"]["bytes"], b"png-bytes")

        with self.assertRaises(LLMError):
            client.generate(
                model="anthropic.claude-3-5-haiku",
                messages=[
                    AssistantMessage(
                        content=[ImageContentPart(data=b"png-bytes", mime_type="image/png")]
                    )
                ],
            )

    def test_bedrock_stream_emits_tool_chunks_and_completion_usage(self):
        fake_runtime = FakeBedrockRuntimeClient(
            stream_response={
                "stream": iter(
                    [
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 0,
                                "delta": {"text": "Hello"},
                            }
                        },
                        {
                            "contentBlockStart": {
                                "contentBlockIndex": 1,
                                "start": {
                                    "toolUse": {
                                        "toolUseId": "call_1",
                                        "name": "get_weather",
                                        "type": "server_tool_use",
                                    }
                                },
                            }
                        },
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 1,
                                "delta": {
                                    "toolUse": {"input": '{"city":"Kath'}
                                },
                            }
                        },
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 1,
                                "delta": {
                                    "toolUse": {"input": 'mandu"}'}
                                },
                            }
                        },
                        {
                            "metadata": {
                                "usage": {
                                    "inputTokens": 10,
                                    "outputTokens": 4,
                                    "totalTokens": 14,
                                }
                            }
                        },
                    ]
                )
            }
        )
        client, _, _ = self.make_bedrock_client(
            fake_runtime,
            region_name="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        chunks = list(
            client.stream(
                model="anthropic.claude-3-5-haiku",
                messages=[UserMessage(content=text_parts("Weather?"))],
                tools=[make_tool("get_weather")],
            )
        )

        self.assertEqual(chunks[0].type, "stream_content")
        self.assertEqual(chunks[0].chunk, "Hello")
        self.assertEqual(chunks[1].source, "tool")
        self.assertEqual(chunks[2].source, "tool")
        self.assertEqual(chunks[-1].tool_calls[0].name, "get_weather")
        self.assertEqual(
            chunks[-1].tool_calls[0].arguments,
            '{"city":"Kathmandu"}',
        )
        self.assertEqual(chunks[-1].usage.input_tokens, 10)
        self.assertEqual(chunks[-1].usage.output_tokens, 4)
        self.assertEqual(chunks[-1].usage.total_tokens, 14)
        self.assertIsNotNone(chunks[-1].duration_seconds)
        self.assertGreaterEqual(chunks[-1].duration_seconds, 0)

    def test_bedrock_generate_wraps_provider_rate_limit_errors(self):
        fake_runtime = FakeBedrockRuntimeClient(
            response=botocore_exceptions.ClientError(
                {
                    "Error": {
                        "Code": "ThrottlingException",
                        "Message": "slow down",
                    },
                    "ResponseMetadata": {
                        "HTTPStatusCode": 429,
                    },
                },
                "Converse",
            )
        )
        client, _, _ = self.make_bedrock_client(
            fake_runtime,
            region_name="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        with self.assertRaises(LLMRateLimitError) as context:
            client.generate(
                model="anthropic.claude-3-5-haiku",
                messages=[UserMessage(content=text_parts("Hello"))],
            )

        self.assertEqual(context.exception.status_code, 429)
        self.assertEqual(context.exception.provider, "bedrock")


if __name__ == "__main__":
    unittest.main()
