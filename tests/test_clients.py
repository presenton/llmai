import inspect
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import anthropic
import httpx
import openai
from botocore import exceptions as botocore_exceptions
from google.genai.types import HttpOptions
from google.genai import errors as google_errors
from pydantic import BaseModel, ValidationError

from llmai import (
    AnthropicClient,
    AnthropicClientConfig,
    AzureOpenAIClient,
    AzureOpenAIClientConfig,
    BedrockClient,
    BedrockClientConfig,
    ChatGPTClient,
    ChatGPTClientConfig,
    DeepSeekClient,
    DeepSeekClientConfig,
    GoogleClient,
    GoogleClientConfig,
    OpenAIApiType,
    OpenAIClient,
    OpenAIClientConfig,
    VertexAIClient,
    VertexAIClientConfig,
    get_client,
)
from llmai.shared import (
    AssistantMessage,
    BaseClient,
    ImageContentPart,
    LLMAuthenticationError,
    LLMConfigurationError,
    JSONSchemaResponse,
    LLMError,
    LLMRateLimitError,
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
    SystemMessage,
    TextContentPart,
    Tool,
    UserMessage,
    WebSearchTool,
)


def make_tool(name: str) -> Tool:
    return Tool(name=name, description=f"{name} description")


def make_web_search_tool() -> WebSearchTool:
    return WebSearchTool()


def text_parts(text: str) -> list[TextContentPart]:
    return [TextContentPart(text=text)]


def stream_marker_chunks(chunks):
    return [chunk for chunk in chunks if chunk.type == "event"]


def stream_payload_chunks(chunks):
    return [chunk for chunk in chunks if chunk.type != "event"]


def stream_marker_events(chunks):
    return [
        (chunk.event, chunk.chunk_type, chunk.tool)
        for chunk in stream_marker_chunks(chunks)
    ]


class FakeOpenAICompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeOpenAIResponses:
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
    def test_get_client_openai_uses_explicit_config(self):
        config = OpenAIClientConfig(
            api_key="openai-key",
            base_url="https://openai.example/v1",
        )

        with patch("llmai.client.OpenAIClient") as openai_client_cls:
            client = get_client(config=config)

        self.assertIs(client, openai_client_cls.return_value)
        openai_client_cls.assert_called_once_with(config=config, logger=None)

    def test_get_client_azure_uses_explicit_config(self):
        config = AzureOpenAIClientConfig(
            api_key="azure-key",
            endpoint="https://azure.example.openai.azure.com",
            api_version="2024-10-21",
            deployment="gpt-4.1",
        )

        with patch("llmai.client.AzureOpenAIClient") as azure_client_cls:
            client = get_client(config=config)

        self.assertIs(client, azure_client_cls.return_value)
        azure_client_cls.assert_called_once_with(config=config, logger=None)

    def test_get_client_vertex_uses_explicit_config(self):
        config = VertexAIClientConfig(
            project="vertex-project",
            location="us-central1",
        )

        with patch("llmai.client.VertexAIClient") as vertex_client_cls:
            client = get_client(config=config)

        self.assertIs(client, vertex_client_cls.return_value)
        vertex_client_cls.assert_called_once_with(config=config, logger=None)

    def test_get_client_chatgpt_uses_explicit_config(self):
        config = ChatGPTClientConfig(
            access_token="chatgpt-token",
            account_id="account-123",
            base_url="https://chatgpt.example/codex",
        )

        with patch("llmai.client.ChatGPTClient") as chatgpt_client_cls:
            client = get_client(config=config)

        self.assertIs(client, chatgpt_client_cls.return_value)
        chatgpt_client_cls.assert_called_once_with(config=config, logger=None)

    def test_get_client_deepseek_uses_explicit_config(self):
        config = DeepSeekClientConfig(
            api_key="deepseek-key",
            base_url="https://api.deepseek.com/beta",
        )

        with patch("llmai.client.DeepSeekClient") as deepseek_client_cls:
            client = get_client(config=config)

        self.assertIs(client, deepseek_client_cls.return_value)
        deepseek_client_cls.assert_called_once_with(config=config, logger=None)

    def test_get_client_google_uses_explicit_config(self):
        config = GoogleClientConfig(api_key="google-key")

        with patch("llmai.client.GoogleClient") as google_client_cls:
            client = get_client(config=config)

        self.assertIs(client, google_client_cls.return_value)
        google_client_cls.assert_called_once_with(config=config, logger=None)

    def test_get_client_anthropic_uses_explicit_config(self):
        config = AnthropicClientConfig(api_key="anthropic-key")

        with patch("llmai.client.AnthropicClient") as anthropic_client_cls:
            client = get_client(config=config)

        self.assertIs(client, anthropic_client_cls.return_value)
        anthropic_client_cls.assert_called_once_with(config=config, logger=None)

    def test_get_client_bedrock_uses_explicit_config(self):
        config = BedrockClientConfig(
            api_key="bedrock-key",
            region="us-east-1",
        )

        with patch("llmai.client.BedrockClient") as bedrock_client_cls:
            client = get_client(config=config)

        self.assertIs(client, bedrock_client_cls.return_value)
        bedrock_client_cls.assert_called_once_with(config=config, logger=None)

    def test_get_client_rejects_mismatched_config_type(self):
        config = OpenAIClientConfig(api_key="openai-key")
        config.provider = "google"  # type: ignore[assignment]

        with self.assertRaises(LLMConfigurationError) as context:
            get_client(config=config)

        self.assertEqual(context.exception.provider, "google")
        self.assertIn("GoogleClientConfig", context.exception.message)

    def test_get_client_uses_provider_literal_from_config(self):
        self.assertEqual(OpenAIClientConfig(api_key="openai-key").provider, "openai")
        self.assertEqual(
            ChatGPTClientConfig(access_token="chatgpt-token").provider,
            "chatgpt",
        )

    def test_openai_config_coerces_api_type_to_enum(self):
        config = OpenAIClientConfig(
            api_key="openai-key",
            api_type="responses",
        )

        self.assertEqual(config.api_type, OpenAIApiType.RESPONSES)

    def test_openai_config_rejects_invalid_api_type(self):
        with self.assertRaises(ValidationError):
            OpenAIClientConfig(
                api_key="openai-key",
                api_type="invalid",
            )

    def test_openai_config_defaults_system_message_instructions_to_false(self):
        config = OpenAIClientConfig(api_key="openai-key")

        self.assertFalse(config.provide_system_message_as_instructions)

    def test_all_clients_share_base_generate_signature(self):
        expected = tuple(inspect.signature(BaseClient.generate).parameters)

        for client_type in (
            OpenAIClient,
            AzureOpenAIClient,
            ChatGPTClient,
            DeepSeekClient,
            AnthropicClient,
            GoogleClient,
            VertexAIClient,
            BedrockClient,
        ):
            with self.subTest(client=client_type.__name__):
                self.assertEqual(
                    tuple(inspect.signature(client_type.generate).parameters),
                    expected,
                )

    def make_bedrock_client(self, runtime_client, **client_kwargs):
        captured_sessions = []

        def fake_session_factory(**kwargs):
            session = FakeBoto3Session(runtime_client, kwargs=kwargs)
            captured_sessions.append(session)
            return session

        patcher = patch(
            "llmai.bedrock.client.boto3.Session",
            side_effect=fake_session_factory,
        )
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        client = BedrockClient(config=BedrockClientConfig(**client_kwargs))
        return client, captured_sessions, mocked

    def test_openai_required_tool_choice_uses_standard_function_selector(self):
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))

        openai_tools, tool_choice = client._get_openai_tools_and_tool_choice_or_omit(
            [make_tool("weather"), make_tool("time")],
            {"mode": "required", "tools": ["weather"]},
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

    def test_openai_responses_web_search_uses_allowed_tools_wrapper(self):
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))

        openai_tools, tool_choice = (
            client._get_openai_responses_tools_and_tool_choice_or_omit(
                [make_tool("weather"), make_web_search_tool()],
                {"mode": "required", "tools": ["web_search", "weather"]},
            )
        )

        self.assertEqual(len(openai_tools), 2)
        self.assertEqual(
            {tool["type"] for tool in openai_tools},
            {"function", "web_search"},
        )
        self.assertEqual(tool_choice["type"], "allowed_tools")
        self.assertEqual(tool_choice["mode"], "required")
        self.assertEqual(tool_choice["tools"], openai_tools)

    def test_openai_responses_web_search_defaults_to_auto_without_tool_choice_wrapper(self):
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))

        openai_tools, tool_choice = (
            client._get_openai_responses_tools_and_tool_choice_or_omit(
                [make_web_search_tool()],
                None,
            )
        )

        self.assertEqual(openai_tools, [{"type": "web_search"}])
        self.assertIsInstance(tool_choice, openai.Omit)

    def test_openai_completions_ignore_web_search_when_unsupported(self):
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))

        openai_tools, tool_choice = client._get_openai_tools_and_tool_choice_or_omit(
            [make_web_search_tool()],
            {"mode": "required", "tools": ["web_search"]},
        )

        self.assertIsInstance(openai_tools, openai.Omit)
        self.assertIsInstance(tool_choice, openai.Omit)

    def test_openai_strict_tool_schemas_strip_unsupported_keywords(self):
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))

        openai_tools = client._llm_tools_to_openai_tools(
            [
                Tool(
                    name="weather",
                    description="weather description",
                    strict=True,
                    schema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "allOf": [
                                    {"type": "string"},
                                ],
                                "description": "Resolved location",
                            }
                        },
                        "required": ["location"],
                    },
                )
            ]
        )

        parameters = openai_tools[0]["function"]["parameters"]
        self.assertNotIn("allOf", parameters["properties"]["location"])
        self.assertEqual(parameters["properties"]["location"]["type"], "string")
        self.assertEqual(
            parameters["properties"]["location"]["description"],
            "Resolved location",
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

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        result = client.generate(
            model="gpt-test",
            messages=[UserMessage(content=text_parts("Weather?"))],
            tools=[make_tool("get_weather"), make_tool("time")],
            tool_choice={"tools": ["get_weather"]},
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

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
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

    def test_openai_generate_passes_reasoning_effort_model_to_chat_completions(self):
        fake_message = SimpleNamespace(
            content="final answer",
            tool_calls=None,
        )
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message)]
        )
        fake_completions = FakeOpenAICompletions(fake_response)

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        client.generate(
            model="gpt-test",
            messages=[UserMessage(content=text_parts("Hello"))],
            reasoning_effort=ReasoningEffort(
                effort=ReasoningEffortValue.HIGH,
                tokens=2048,
                summary=ReasoningSummary.DETAILED,
            ),
        )

        self.assertEqual(fake_completions.calls[0]["reasoning_effort"], "high")

    def test_chatgpt_init_uses_codex_base_url_and_headers(self):
        with patch("llmai.chatgpt.client.OpenAI") as openai_cls:
            ChatGPTClient(
                config=ChatGPTClientConfig(
                    access_token="token-123",
                    account_id="account-123",
                )
            )

        openai_cls.assert_called_once()
        kwargs = openai_cls.call_args.kwargs
        self.assertEqual(
            kwargs["base_url"],
            "https://chatgpt.com/backend-api/codex",
        )
        self.assertEqual(kwargs["api_key"], "token-123")
        self.assertEqual(kwargs["timeout"], 120.0)
        self.assertEqual(
            kwargs["default_headers"]["OpenAI-Beta"],
            "responses=experimental",
        )
        self.assertEqual(kwargs["default_headers"]["originator"], "pi")
        self.assertEqual(
            kwargs["default_headers"]["chatgpt-account-id"],
            "account-123",
        )

    def test_chatgpt_generate_uses_openai_responses_api(self):
        completed_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="final answer",
                        )
                    ],
                )
            ]
        )
        fake_responses = FakeOpenAIResponses(
            iter(
                [
                    SimpleNamespace(
                        type="response.completed",
                        response=completed_response,
                    )
                ]
            )
        )
        fake_completions = FakeOpenAICompletions(
            Exception("chat completions should not be used")
        )

        client = ChatGPTClient(
            config=ChatGPTClientConfig(access_token="test")
        )
        client._client = SimpleNamespace(
            responses=fake_responses,
            chat=SimpleNamespace(completions=fake_completions),
        )

        result = client.generate(
            model="chatgpt-4o-latest",
            messages=[
                SystemMessage(content="First instruction"),
                SystemMessage(content="Second instruction"),
                UserMessage(content=text_parts("Hello")),
            ],
        )

        self.assertEqual(result.content[0].text, "final answer")
        self.assertEqual(fake_responses.calls[0]["model"], "chatgpt-4o-latest")
        self.assertEqual(
            fake_responses.calls[0]["instructions"],
            "First instruction\n\nSecond instruction",
        )
        self.assertEqual(len(fake_responses.calls[0]["input"]), 1)
        self.assertNotIn(
            "system",
            [item.get("role") for item in fake_responses.calls[0]["input"]],
        )
        self.assertEqual(fake_responses.calls[0]["input"][0]["role"], "user")
        self.assertEqual(fake_responses.calls[0]["extra_body"]["store"], False)
        self.assertEqual(
            fake_responses.calls[0]["extra_body"]["include"],
            ["reasoning.encrypted_content"],
        )
        self.assertEqual(
            fake_responses.calls[0]["extra_body"]["parallel_tool_calls"],
            True,
        )
        self.assertEqual(
            fake_responses.calls[0]["extra_body"]["text"],
            {"verbosity": "medium"},
        )
        self.assertTrue(fake_responses.calls[0]["stream"])
        self.assertFalse(fake_completions.calls)

    def test_chatgpt_generate_attaches_web_search_tool(self):
        completed_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="final answer",
                        )
                    ],
                )
            ]
        )
        fake_responses = FakeOpenAIResponses(
            iter(
                [
                    SimpleNamespace(
                        type="response.completed",
                        response=completed_response,
                    )
                ]
            )
        )

        client = ChatGPTClient(
            config=ChatGPTClientConfig(access_token="test")
        )
        client._client = SimpleNamespace(
            responses=fake_responses,
            chat=SimpleNamespace(completions=FakeOpenAICompletions(None)),
        )

        client.generate(
            model="chatgpt-4o-latest",
            messages=[UserMessage(content=text_parts("Hello"))],
            tools=[make_web_search_tool()],
        )

        self.assertEqual(
            fake_responses.calls[0]["tools"],
            [{"type": "web_search"}],
        )
        self.assertIsInstance(fake_responses.calls[0]["tool_choice"], openai.Omit)

    def test_chatgpt_generate_falls_back_to_streamed_content_when_completed_output_is_empty(self):
        completed_response = SimpleNamespace(
            output=[],
            usage=SimpleNamespace(
                input_tokens=17,
                output_tokens=141,
                total_tokens=158,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )
        fake_responses = FakeOpenAIResponses(
            iter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta",
                        delta="final answer",
                        item_id="msg_1",
                        content_index=0,
                        output_index=0,
                        logprobs=[],
                        sequence_number=1,
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=completed_response,
                        sequence_number=2,
                    ),
                ]
            )
        )

        client = ChatGPTClient(
            config=ChatGPTClientConfig(access_token="test")
        )
        client._client = SimpleNamespace(
            responses=fake_responses,
            chat=SimpleNamespace(completions=FakeOpenAICompletions(None)),
        )

        result = client.generate(
            model="chatgpt-4o-latest",
            messages=[UserMessage(content=text_parts("Hello"))],
        )

        self.assertEqual(result.content[0].text, "final answer")
        self.assertEqual(result.messages[-1].content[0].text, "final answer")
        self.assertEqual(result.usage.input_tokens, 17)
        self.assertEqual(result.usage.output_tokens, 141)
        self.assertEqual(result.usage.total_tokens, 158)

    def test_chatgpt_generate_uses_fallback_instructions_without_system_message(self):
        completed_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="final answer",
                        )
                    ],
                )
            ]
        )
        fake_responses = FakeOpenAIResponses(
            iter(
                [
                    SimpleNamespace(
                        type="response.completed",
                        response=completed_response,
                    )
                ]
            )
        )

        client = ChatGPTClient(
            config=ChatGPTClientConfig(access_token="test")
        )
        client._client = SimpleNamespace(
            responses=fake_responses,
            chat=SimpleNamespace(completions=FakeOpenAICompletions(None)),
        )

        client.generate(
            model="chatgpt-4o-latest",
            messages=[UserMessage(content=text_parts("Hello"))],
            temperature=0.2,
            max_tokens=123,
        )

        self.assertEqual(
            fake_responses.calls[0]["instructions"],
            "Follow the prompt",
        )
        self.assertIsInstance(
            fake_responses.calls[0]["temperature"],
            openai.Omit,
        )
        self.assertIsInstance(
            fake_responses.calls[0]["max_output_tokens"],
            openai.Omit,
        )
        self.assertTrue(fake_responses.calls[0]["stream"])
        self.assertEqual(len(fake_responses.calls[0]["input"]), 1)
        self.assertEqual(fake_responses.calls[0]["input"][0]["role"], "user")

    def test_chatgpt_generate_does_not_expose_api_type(self):
        self.assertNotIn(
            "api_type",
            inspect.signature(ChatGPTClient.generate).parameters,
        )

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

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
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
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))

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
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))

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

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        chunks = list(
            client.generate(
                model="gpt-test",
                messages=[UserMessage(content=text_parts("Weather?"))],
                tools=[make_tool("get_weather")],
                stream=True,
            )
        )

        marker_chunks = stream_marker_chunks(chunks)
        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            [chunk.type for chunk in chunks],
            [
                "event",
                "content",
                "event",
                "event",
                "tool",
                "tool",
                "tool_complete",
                "event",
                "completion",
            ],
        )
        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "content", None),
                ("end", "content", None),
                ("start", "tool", "get_weather"),
                ("end", "tool", "get_weather"),
            ],
        )
        self.assertEqual(payload_chunks[0].type, "content")
        self.assertEqual(payload_chunks[1].type, "tool")
        self.assertEqual(payload_chunks[2].type, "tool")
        self.assertEqual(payload_chunks[1].tool, "get_weather")
        self.assertEqual(payload_chunks[2].tool, "get_weather")
        self.assertEqual(payload_chunks[3].type, "tool_complete")
        self.assertEqual(payload_chunks[3].tool, "get_weather")
        self.assertEqual(payload_chunks[3].arguments, '{"city":"Kathmandu"}')
        self.assertEqual(payload_chunks[-1].type, "completion")
        self.assertEqual(payload_chunks[-1].tool_calls[0].name, "get_weather")
        self.assertEqual(
            payload_chunks[-1].tool_calls[0].arguments,
            '{"city":"Kathmandu"}',
        )
        self.assertEqual(payload_chunks[-1].usage.input_tokens, 10)
        self.assertEqual(payload_chunks[-1].usage.output_tokens, 4)
        self.assertEqual(payload_chunks[-1].usage.total_tokens, 14)
        self.assertEqual(
            payload_chunks[-1].usage.details["completion_tokens_details"]["reasoning_tokens"],
            1,
        )
        self.assertIsNotNone(payload_chunks[-1].duration_seconds)
        self.assertGreaterEqual(payload_chunks[-1].duration_seconds, 0)
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

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
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

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
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

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
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
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
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

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
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
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
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

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
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

        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        chunks = list(
            client.generate(
                model="gpt-test",
                messages=[UserMessage(content=text_parts("Answer in JSON"))],
                response_format=JSONSchemaResponse(json_schema=AnswerSchema),
                stream=True,
            )
        )

        marker_chunks = stream_marker_chunks(chunks)
        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "content", None),
                ("end", "content", None),
            ],
        )
        self.assertEqual(payload_chunks[0].type, "content")
        self.assertEqual(payload_chunks[0].chunk, '{"answer":"')
        self.assertEqual(payload_chunks[1].type, "content")
        self.assertEqual(payload_chunks[1].chunk, 'pong"}')
        self.assertEqual(payload_chunks[-1].content, {"answer": "pong"})

    def test_openai_responses_generate_parses_structured_output_and_thinking(self):
        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="Compare speed and creativity.")],
                ),
                SimpleNamespace(
                    id="msg_1",
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text='{"answer":"pong"}',
                        )
                    ],
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=7,
                total_tokens=18,
                input_tokens_details=SimpleNamespace(cached_tokens=2),
                output_tokens_details=SimpleNamespace(reasoning_tokens=3),
            ),
        )
        fake_responses = FakeOpenAIResponses(fake_response)

        client = OpenAIClient(
            config=OpenAIClientConfig(
                api_key="test",
                api_type=OpenAIApiType.RESPONSES,
            )
        )
        client._client = SimpleNamespace(responses=fake_responses)

        result = client.generate(
            model="gpt-test",
            messages=[UserMessage(content=text_parts("Answer in JSON"))],
            response_format=JSONSchemaResponse(json_schema=AnswerSchema),
        )

        self.assertEqual(result.content, {"answer": "pong"})
        self.assertEqual(
            result.thinking,
            ["Compare speed and creativity."],
        )
        self.assertEqual(
            result.messages[-1].thinking,
            ["Compare speed and creativity."],
        )
        self.assertEqual(result.messages[-1].id, "msg_1")
        self.assertEqual(result.usage.input_tokens, 11)
        self.assertEqual(result.usage.output_tokens, 7)
        self.assertEqual(result.usage.total_tokens, 18)
        self.assertEqual(result.usage.details["input_tokens_details"]["cached_tokens"], 2)
        self.assertEqual(
            result.usage.details["output_tokens_details"]["reasoning_tokens"],
            3,
        )
        self.assertEqual(
            fake_responses.calls[0]["text"]["format"]["type"],
            "json_schema",
        )
        self.assertEqual(
            fake_responses.calls[0]["reasoning"],
            {"summary": "auto"},
        )
        self.assertNotIn("instructions", fake_responses.calls[0])
        self.assertEqual(fake_responses.calls[0]["input"][0]["role"], "user")

    def test_openai_responses_omits_assistant_message_id_when_unset(self):
        client = OpenAIClient(
            config=OpenAIClientConfig(
                api_key="test",
                api_type=OpenAIApiType.RESPONSES,
            )
        )

        openai_input = client._messages_to_openai_responses_input(
            [AssistantMessage(content=text_parts("Hello"))]
        )

        self.assertEqual(len(openai_input), 1)
        self.assertEqual(openai_input[0]["role"], "assistant")
        self.assertNotIn("id", openai_input[0])

    def test_openai_responses_preserves_assistant_message_id_in_history(self):
        client = OpenAIClient(
            config=OpenAIClientConfig(
                api_key="test",
                api_type=OpenAIApiType.RESPONSES,
            )
        )

        openai_input = client._messages_to_openai_responses_input(
            [AssistantMessage(id="msg_1", content=text_parts("Hello"))]
        )

        self.assertEqual(len(openai_input), 1)
        self.assertEqual(openai_input[0]["id"], "msg_1")

    def test_openai_responses_generate_can_lift_system_messages_to_instructions(self):
        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="done",
                        )
                    ],
                )
            ]
        )
        fake_responses = FakeOpenAIResponses(fake_response)

        client = OpenAIClient(
            config=OpenAIClientConfig(
                api_key="test",
                api_type=OpenAIApiType.RESPONSES,
                provide_system_message_as_instructions=True,
            )
        )
        client._client = SimpleNamespace(responses=fake_responses)

        client.generate(
            model="gpt-test",
            messages=[
                SystemMessage(content="First instruction"),
                SystemMessage(content="Second instruction"),
                UserMessage(content=text_parts("Hello")),
            ],
        )

        self.assertEqual(
            fake_responses.calls[0]["instructions"],
            "First instruction\n\nSecond instruction",
        )
        self.assertEqual(len(fake_responses.calls[0]["input"]), 1)
        self.assertNotIn(
            "system",
            [item.get("role") for item in fake_responses.calls[0]["input"]],
        )
        self.assertEqual(fake_responses.calls[0]["input"][0]["role"], "user")

    def test_openai_responses_generate_preserves_multiple_thinking_blocks(self):
        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="Plan")],
                ),
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="Reflect")],
                ),
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="done",
                        )
                    ],
                ),
            ],
        )
        fake_responses = FakeOpenAIResponses(fake_response)

        client = OpenAIClient(
            config=OpenAIClientConfig(
                api_key="test",
                api_type=OpenAIApiType.RESPONSES,
            )
        )
        client._client = SimpleNamespace(responses=fake_responses)

        result = client.generate(
            model="gpt-test",
            messages=[UserMessage(content=text_parts("Hello"))],
        )

        self.assertEqual(result.thinking, ["Plan", "Reflect"])
        self.assertEqual(result.messages[-1].thinking, ["Plan", "Reflect"])

    def test_openai_responses_generate_passes_reasoning_effort_model(self):
        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="done",
                        )
                    ],
                )
            ]
        )
        fake_responses = FakeOpenAIResponses(fake_response)

        client = OpenAIClient(
            config=OpenAIClientConfig(
                api_key="test",
                api_type=OpenAIApiType.RESPONSES,
            )
        )
        client._client = SimpleNamespace(responses=fake_responses)

        client.generate(
            model="gpt-test",
            messages=[UserMessage(content=text_parts("Hello"))],
            reasoning_effort=ReasoningEffort(
                effort=ReasoningEffortValue.HIGH,
                tokens=2048,
                summary=ReasoningSummary.DETAILED,
            ),
        )

        self.assertEqual(
            fake_responses.calls[0]["reasoning"],
            {"effort": "high", "summary": "detailed"},
        )

    def test_openai_responses_stream_emits_thinking_tool_chunks_and_completion(self):
        completed_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="Plan")],
                ),
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="Hello",
                        )
                    ],
                ),
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="get_weather",
                    arguments='{"city":"Kathmandu"}',
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
                input_tokens_details=SimpleNamespace(cached_tokens=1),
                output_tokens_details=SimpleNamespace(reasoning_tokens=2),
            ),
        )
        events = iter(
            [
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        type="function_call",
                        id="fc_1",
                        call_id="call_1",
                        name="get_weather",
                        arguments="",
                    ),
                    output_index=2,
                    sequence_number=1,
                ),
                SimpleNamespace(
                    type="response.reasoning_summary_text.delta",
                    delta="Plan",
                    item_id="rs_1",
                    output_index=0,
                    sequence_number=2,
                    summary_index=0,
                ),
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta="Hello",
                    item_id="msg_1",
                    content_index=0,
                    output_index=1,
                    logprobs=[],
                    sequence_number=3,
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    delta='{"city":"Kath',
                    item_id="fc_1",
                    output_index=2,
                    sequence_number=4,
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    delta='mandu"}',
                    item_id="fc_1",
                    output_index=2,
                    sequence_number=5,
                ),
                SimpleNamespace(
                    type="response.completed",
                    response=completed_response,
                    sequence_number=6,
                ),
            ]
        )
        fake_responses = FakeOpenAIResponses(events)

        client = OpenAIClient(
            config=OpenAIClientConfig(
                api_key="test",
                api_type=OpenAIApiType.RESPONSES,
            )
        )
        client._client = SimpleNamespace(responses=fake_responses)

        chunks = list(
            client.generate(
                model="gpt-test",
                messages=[UserMessage(content=text_parts("Weather?"))],
                tools=[make_tool("get_weather")],
                stream=True,
            )
        )

        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            [chunk.type for chunk in chunks],
            [
                "event",
                "thinking",
                "event",
                "event",
                "content",
                "event",
                "event",
                "tool",
                "tool",
                "tool_complete",
                "event",
                "completion",
            ],
        )
        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "content", None),
                ("end", "content", None),
                ("start", "tool", "get_weather"),
                ("end", "tool", "get_weather"),
            ],
        )
        self.assertEqual(payload_chunks[0].type, "thinking")
        self.assertEqual(payload_chunks[0].chunk, "Plan")
        self.assertEqual(payload_chunks[1].type, "content")
        self.assertEqual(payload_chunks[1].chunk, "Hello")
        self.assertEqual(payload_chunks[2].type, "tool")
        self.assertEqual(payload_chunks[2].tool, "get_weather")
        self.assertEqual(payload_chunks[3].type, "tool")
        self.assertEqual(payload_chunks[3].tool, "get_weather")
        self.assertEqual(payload_chunks[4].type, "tool_complete")
        self.assertEqual(payload_chunks[4].tool, "get_weather")
        self.assertEqual(payload_chunks[4].arguments, '{"city":"Kathmandu"}')
        self.assertEqual(payload_chunks[-1].thinking, ["Plan"])
        self.assertEqual(payload_chunks[-1].messages[-1].thinking, ["Plan"])
        self.assertEqual(
            payload_chunks[-1].tool_calls[0].arguments,
            '{"city":"Kathmandu"}',
        )
        self.assertEqual(payload_chunks[-1].usage.input_tokens, 10)
        self.assertEqual(payload_chunks[-1].usage.output_tokens, 4)
        self.assertEqual(payload_chunks[-1].usage.total_tokens, 14)
        self.assertEqual(
            payload_chunks[-1].usage.details["output_tokens_details"]["reasoning_tokens"],
            2,
        )
        self.assertTrue(fake_responses.calls[0]["stream"])
        self.assertNotIn("instructions", fake_responses.calls[0])

    def test_openai_responses_stream_falls_back_to_streamed_message_id_when_completed_output_is_empty(self):
        completed_response = SimpleNamespace(
            output=[],
            usage=SimpleNamespace(
                input_tokens=17,
                output_tokens=141,
                total_tokens=158,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )
        events = iter(
            [
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta="final answer",
                    item_id="msg_1",
                    content_index=0,
                    output_index=0,
                    logprobs=[],
                    sequence_number=1,
                ),
                SimpleNamespace(
                    type="response.completed",
                    response=completed_response,
                    sequence_number=2,
                ),
            ]
        )
        fake_responses = FakeOpenAIResponses(events)

        client = OpenAIClient(
            config=OpenAIClientConfig(
                api_key="test",
                api_type=OpenAIApiType.RESPONSES,
            )
        )
        client._client = SimpleNamespace(responses=fake_responses)

        chunks = list(
            client.generate(
                model="gpt-test",
                messages=[UserMessage(content=text_parts("Hello"))],
                stream=True,
            )
        )

        completion = stream_payload_chunks(chunks)[-1]

        self.assertEqual(completion.content[0].text, "final answer")
        self.assertEqual(completion.messages[-1].content[0].text, "final answer")
        self.assertEqual(completion.messages[-1].id, "msg_1")
        self.assertEqual(completion.usage.input_tokens, 17)
        self.assertEqual(completion.usage.output_tokens, 141)
        self.assertEqual(completion.usage.total_tokens, 158)

    def test_openai_responses_stream_wraps_multiple_thinking_blocks(self):
        completed_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="Plan")],
                ),
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="Reflect")],
                ),
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="Hello",
                        )
                    ],
                ),
            ],
        )
        events = iter(
            [
                SimpleNamespace(
                    type="response.reasoning_summary_text.delta",
                    delta="Plan",
                    item_id="rs_1",
                    output_index=0,
                    sequence_number=1,
                    summary_index=0,
                ),
                SimpleNamespace(
                    type="response.reasoning_summary_text.done",
                    text="Plan",
                    item_id="rs_1",
                    output_index=0,
                    sequence_number=2,
                    summary_index=0,
                ),
                SimpleNamespace(
                    type="response.reasoning_summary_text.delta",
                    delta="Reflect",
                    item_id="rs_1",
                    output_index=0,
                    sequence_number=3,
                    summary_index=1,
                ),
                SimpleNamespace(
                    type="response.reasoning_summary_text.done",
                    text="Reflect",
                    item_id="rs_1",
                    output_index=0,
                    sequence_number=4,
                    summary_index=1,
                ),
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta="Hello",
                    item_id="msg_1",
                    content_index=0,
                    output_index=1,
                    logprobs=[],
                    sequence_number=5,
                ),
                SimpleNamespace(
                    type="response.completed",
                    response=completed_response,
                    sequence_number=6,
                ),
            ]
        )
        fake_responses = FakeOpenAIResponses(events)

        client = OpenAIClient(
            config=OpenAIClientConfig(
                api_key="test",
                api_type=OpenAIApiType.RESPONSES,
            )
        )
        client._client = SimpleNamespace(responses=fake_responses)

        chunks = list(
            client.generate(
                model="gpt-test",
                messages=[UserMessage(content=text_parts("Hi"))],
                stream=True,
            )
        )

        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "content", None),
                ("end", "content", None),
            ],
        )
        self.assertEqual(payload_chunks[0].chunk, "Plan")
        self.assertEqual(payload_chunks[1].chunk, "Reflect")
        self.assertEqual(payload_chunks[2].chunk, "Hello")
        self.assertEqual(payload_chunks[-1].thinking, ["Plan", "Reflect"])
        self.assertEqual(payload_chunks[-1].messages[-1].thinking, ["Plan", "Reflect"])
        self.assertEqual(
            fake_responses.calls[0]["reasoning"],
            {"summary": "auto"},
        )

    def test_deepseek_init_uses_official_default_base_url(self):
        with patch("llmai.deepseek.client.OpenAI") as openai_cls:
            DeepSeekClient(config=DeepSeekClientConfig(api_key="test"))

        openai_cls.assert_called_once_with(
            base_url="https://api.deepseek.com",
            api_key="test",
        )

    def test_azure_init_uses_endpoint_api_key_and_api_version(self):
        with patch("llmai.azure.client.AzureOpenAI") as azure_openai_cls:
            AzureOpenAIClient(
                config=AzureOpenAIClientConfig(
                    api_key="test",
                    endpoint="https://azure.example.openai.azure.com",
                    api_version="2024-10-21",
                    deployment="gpt-4.1",
                )
            )

        azure_openai_cls.assert_called_once_with(
            api_version="2024-10-21",
            api_key="test",
            azure_ad_token=None,
            azure_ad_token_provider=None,
            base_url=None,
            azure_endpoint="https://azure.example.openai.azure.com",
            azure_deployment="gpt-4.1",
        )

    def test_azure_init_uses_base_url_ad_token_and_api_version(self):
        with patch("llmai.azure.client.AzureOpenAI") as azure_openai_cls:
            AzureOpenAIClient(
                config=AzureOpenAIClientConfig(
                    azure_ad_token="azure-ad-token",
                    base_url="https://azure.example.openai.azure.com/openai/v1",
                    api_version="2024-10-21",
                )
            )

        azure_openai_cls.assert_called_once_with(
            api_version="2024-10-21",
            api_key=None,
            azure_ad_token="azure-ad-token",
            azure_ad_token_provider=None,
            base_url="https://azure.example.openai.azure.com/openai/v1",
        )

    def test_azure_init_requires_credentials(self):
        with self.assertRaises(LLMConfigurationError) as context:
            AzureOpenAIClientConfig(
                endpoint="https://azure.example.openai.azure.com",
                api_version="2024-10-21",
            )

        self.assertEqual(context.exception.provider, "azure")
        self.assertIn("credentials", context.exception.message)

    def test_azure_init_rejects_ambiguous_endpoint_values(self):
        with self.assertRaises(LLMConfigurationError) as context:
            AzureOpenAIClientConfig(
                api_key="test",
                endpoint="https://first.azure.com",
                base_url="https://second.azure.com/openai/v1",
                api_version="2024-10-21",
            )

        self.assertEqual(context.exception.provider, "azure")
        self.assertIn("base_url or endpoint", context.exception.message)

    def test_azure_config_does_not_accept_api_type(self):
        with self.assertRaises(ValidationError):
            AzureOpenAIClientConfig(
                api_key="test",
                endpoint="https://azure.example.openai.azure.com",
                api_version="2024-10-21",
                api_type=OpenAIApiType.RESPONSES,
            )

    def test_azure_generate_wraps_provider_auth_errors(self):
        request = httpx.Request(
            "POST",
            "https://azure.example.openai.azure.com/openai/deployments/gpt/chat/completions",
        )
        response = httpx.Response(401, request=request)
        fake_completions = FakeOpenAICompletions(
            openai.AuthenticationError(
                "bad api key",
                response=response,
                body={},
            )
        )

        client = AzureOpenAIClient(
            config=AzureOpenAIClientConfig(
                api_key="test",
                endpoint="https://azure.example.openai.azure.com",
                api_version="2024-10-21",
            )
        )
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        with self.assertRaises(LLMAuthenticationError) as context:
            client.generate(
                model="gpt-test",
                messages=[UserMessage(content=text_parts("Hello"))],
            )

        self.assertEqual(context.exception.provider, "azure")

    def test_azure_always_uses_chat_completions(self):
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="hello",
                        tool_calls=None,
                    )
                )
            ]
        )
        fake_completions = FakeOpenAICompletions(fake_response)
        fake_responses = FakeOpenAIResponses(
            Exception("responses API should not be used")
        )

        client = AzureOpenAIClient(
            config=AzureOpenAIClientConfig(
                api_key="test",
                endpoint="https://azure.example.openai.azure.com",
                api_version="2024-10-21",
            )
        )
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions),
            responses=fake_responses,
        )

        result = client.generate(
            model="gpt-test",
            messages=[UserMessage(content=text_parts("Hello"))],
        )

        self.assertEqual(result.content[0].text, "hello")
        self.assertTrue(fake_completions.calls)
        self.assertFalse(fake_responses.calls)

    def test_vertex_init_uses_vertex_genai_client_configuration(self):
        with patch("llmai.google.client.genai.Client") as genai_client_cls:
            VertexAIClient(
                config=VertexAIClientConfig(
                    project="vertex-project",
                    location="us-central1",
                )
            )

        genai_client_cls.assert_called_once_with(
            vertexai=True,
            api_key=None,
            credentials=None,
            project="vertex-project",
            location="us-central1",
            http_options=None,
        )

    def test_vertex_init_uses_base_url_when_provided(self):
        with patch("llmai.google.client.genai.Client") as genai_client_cls:
            VertexAIClient(
                config=VertexAIClientConfig(
                    project="vertex-project",
                    location="us-central1",
                    base_url="https://vertex.example",
                )
            )

        genai_client_cls.assert_called_once()
        self.assertEqual(genai_client_cls.call_args.kwargs["vertexai"], True)
        self.assertIsNone(genai_client_cls.call_args.kwargs["api_key"])
        self.assertEqual(genai_client_cls.call_args.kwargs["project"], "vertex-project")
        self.assertEqual(genai_client_cls.call_args.kwargs["location"], "us-central1")
        self.assertIsInstance(genai_client_cls.call_args.kwargs["http_options"], HttpOptions)
        self.assertEqual(
            genai_client_cls.call_args.kwargs["http_options"].base_url,
            "https://vertex.example",
        )

    def test_vertex_config_rejects_api_key_with_project_or_location(self):
        with self.assertRaises(LLMConfigurationError) as context:
            VertexAIClientConfig(
                api_key="vertex-key",
                project="vertex-project",
                location="us-central1",
            )

        self.assertEqual(context.exception.provider, "vertex")
        self.assertIn("either api_key or project/location/credentials", context.exception.message)

    def test_vertex_init_wraps_configuration_errors(self):
        with patch(
            "llmai.google.client.genai.Client",
            side_effect=ValueError("Project or API key must be set when using the Vertex AI API."),
        ):
            with self.assertRaises(LLMConfigurationError) as context:
                VertexAIClient(config=VertexAIClientConfig(api_key="test"))

        self.assertEqual(context.exception.provider, "vertex")
        self.assertIn("Project or API key", context.exception.message)

    def test_vertex_generate_wraps_provider_auth_errors(self):
        fake_models = FakeGoogleModels(
            response=google_errors.ClientError(
                401,
                {"message": "bad credentials", "status": "UNAUTHENTICATED"},
            )
        )

        client = VertexAIClient(config=VertexAIClientConfig(api_key="test"))
        client._client = SimpleNamespace(models=fake_models)

        with self.assertRaises(LLMAuthenticationError) as context:
            client.generate(
                model="gemini-test",
                messages=[UserMessage(content=text_parts("Hello"))],
            )

        self.assertEqual(context.exception.status_code, 401)
        self.assertEqual(context.exception.provider, "vertex")

    def test_deepseek_init_uses_explicit_api_key(self):
        with (
            patch("llmai.deepseek.client.OpenAI") as openai_cls,
        ):
            DeepSeekClient(config=DeepSeekClientConfig(api_key="explicit-key"))

        openai_cls.assert_called_once_with(
            base_url="https://api.deepseek.com",
            api_key="explicit-key",
        )

    def test_deepseek_init_requires_explicit_api_key(self):
        with self.assertRaises(ValidationError):
            DeepSeekClientConfig(api_key="")

    def test_deepseek_stream_uses_internal_response_schema_tool(self):
        events = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="schema_1",
                                        function=SimpleNamespace(
                                            name="final_answer",
                                            arguments='{"answer":"',
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
                                            arguments='done"}',
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
                        prompt_tokens=8,
                        completion_tokens=3,
                        total_tokens=11,
                    ),
                ),
            ]
        )
        fake_completions = FakeOpenAICompletions(events)

        client = DeepSeekClient(config=DeepSeekClientConfig(api_key="test"))
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        chunks = list(
            client.generate(
                model="deepseek-chat",
                messages=[UserMessage(content=text_parts("Answer in JSON"))],
                response_format=JSONSchemaResponse(
                    name="final_answer",
                    json_schema=AnswerSchema,
                ),
                stream=True,
            )
        )

        marker_chunks = stream_marker_chunks(chunks)
        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "content", None),
                ("end", "content", None),
            ],
        )
        self.assertEqual(payload_chunks[0].type, "content")
        self.assertEqual(payload_chunks[0].chunk, '{"answer":"')
        self.assertEqual(payload_chunks[1].type, "content")
        self.assertEqual(payload_chunks[1].chunk, 'done"}')
        self.assertEqual(payload_chunks[-1].type, "completion")
        self.assertEqual(payload_chunks[-1].content, {"answer": "done"})
        self.assertEqual(payload_chunks[-1].tool_calls, [])
        self.assertEqual(payload_chunks[-1].messages[-1].tool_calls, [])
        self.assertEqual(payload_chunks[-1].messages[-1].content[0].text, '{"answer":"done"}')
        self.assertEqual(payload_chunks[-1].usage.input_tokens, 8)
        self.assertEqual(payload_chunks[-1].usage.output_tokens, 3)
        self.assertEqual(payload_chunks[-1].usage.total_tokens, 11)
        self.assertEqual(
            [tool["function"]["name"] for tool in fake_completions.calls[0]["tools"]],
            ["final_answer"],
        )
        self.assertFalse(fake_completions.calls[0]["tools"][0]["function"]["strict"])
        self.assertEqual(
            fake_completions.calls[0]["tool_choice"],
            {
                "type": "function",
                "function": {"name": "final_answer"},
            },
        )
        self.assertEqual(
            fake_completions.calls[0]["stream_options"],
            {"include_usage": True},
        )

    def test_deepseek_generate_uses_internal_response_schema_tool(self):
        fake_message = SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="schema_1",
                    function=SimpleNamespace(
                        name="final_answer",
                        arguments='{"answer":"done"}',
                    ),
                )
            ],
        )
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message)]
        )
        fake_completions = FakeOpenAICompletions(fake_response)

        client = DeepSeekClient(config=DeepSeekClientConfig(api_key="test"))
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        result = client.generate(
            model="deepseek-chat",
            messages=[UserMessage(content=text_parts("Answer in JSON"))],
            response_format=JSONSchemaResponse(
                name="final_answer",
                json_schema=AnswerSchema,
            ),
        )

        self.assertEqual(result.content, {"answer": "done"})
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.messages[-1].tool_calls, [])
        self.assertEqual(
            [tool["function"]["name"] for tool in fake_completions.calls[0]["tools"]],
            ["final_answer"],
        )
        self.assertEqual(
            fake_completions.calls[0]["tool_choice"],
            {
                "type": "function",
                "function": {"name": "final_answer"},
            },
        )
        self.assertIsInstance(fake_completions.calls[0]["response_format"], openai.Omit)

    def test_deepseek_stream_emits_tool_complete_chunk(self):
        events = iter(
            [
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

        client = DeepSeekClient(config=DeepSeekClientConfig(api_key="test"))
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        chunks = list(
            client.generate(
                model="deepseek-chat",
                messages=[UserMessage(content=text_parts("Weather?"))],
                tools=[make_tool("get_weather")],
                stream=True,
            )
        )

        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            [chunk.type for chunk in chunks],
            [
                "event",
                "tool",
                "tool",
                "tool_complete",
                "event",
                "completion",
            ],
        )
        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "tool", "get_weather"),
                ("end", "tool", "get_weather"),
            ],
        )
        self.assertEqual(payload_chunks[0].type, "tool")
        self.assertEqual(payload_chunks[1].type, "tool")
        self.assertEqual(payload_chunks[2].type, "tool_complete")
        self.assertEqual(payload_chunks[2].tool, "get_weather")
        self.assertEqual(payload_chunks[2].arguments, '{"city":"Kathmandu"}')
        self.assertEqual(payload_chunks[-1].tool_calls[0].arguments, '{"city":"Kathmandu"}')

    def test_deepseek_generate_does_not_expose_structured_output_toggle(self):
        self.assertNotIn(
            "use_tools_for_structured_output",
            inspect.signature(DeepSeekClient.generate).parameters,
        )

    def test_deepseek_tool_structured_output_keeps_selected_user_tools_visible(self):
        client = DeepSeekClient(config=DeepSeekClientConfig(api_key="test"))

        deepseek_tools, deepseek_tool_choice, _ = (
            client._get_deepseek_tools_and_tool_choice_or_omit(
                [make_tool("get_weather")],
                {"tools": ["get_weather"]},
                JSONSchemaResponse(
                    name="final_answer",
                    json_schema=AnswerSchema,
                ),
            )
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in deepseek_tools],
            ["get_weather", "final_answer"],
        )
        self.assertEqual(deepseek_tool_choice, "required")

    def test_deepseek_ignores_web_search_tool(self):
        client = DeepSeekClient(config=DeepSeekClientConfig(api_key="test"))

        deepseek_tools, deepseek_tool_choice, _ = (
            client._get_deepseek_tools_and_tool_choice_or_omit(
                [make_web_search_tool()],
                {"mode": "required", "tools": ["web_search"]},
                None,
            )
        )

        self.assertIsInstance(deepseek_tools, openai.Omit)
        self.assertIsInstance(deepseek_tool_choice, openai.Omit)

    def test_anthropic_required_tool_choice_uses_any_for_multiple_visible_tools(self):
        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))

        anthropic_tools, tool_choice = (
            client._get_anthropic_tools_and_tool_choice_or_omit(
                [make_tool("weather"), make_tool("time")],
                {"mode": "required", "tools": ["weather", "time"]},
                None,
            )
        )

        self.assertEqual([tool["name"] for tool in anthropic_tools], ["weather", "time"])
        self.assertEqual(tool_choice, {"type": "any"})

    def test_anthropic_web_search_maps_to_server_tool(self):
        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))

        anthropic_tools, tool_choice = (
            client._get_anthropic_tools_and_tool_choice_or_omit(
                [make_web_search_tool()],
                {"mode": "required", "tools": ["web_search"]},
                None,
            )
        )

        self.assertEqual(
            anthropic_tools,
            [{"type": "web_search_20250305", "name": "web_search"}],
        )
        self.assertEqual(tool_choice, {"type": "tool", "name": "web_search"})

    def test_anthropic_tool_schemas_strip_max_items(self):
        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))

        anthropic_tools, _ = client._get_anthropic_tools_and_tool_choice_or_omit(
            [
                Tool(
                    name="weather",
                    description="weather description",
                    strict=True,
                    schema={
                        "type": "object",
                        "properties": {
                            "cities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 3,
                            }
                        },
                        "required": ["cities"],
                    },
                )
            ],
            None,
            None,
        )

        input_schema = anthropic_tools[0]["input_schema"]
        self.assertNotIn("maxItems", input_schema["properties"]["cities"])

    def test_anthropic_tool_schemas_keep_non_strict_keywords(self):
        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))

        anthropic_tools, _ = client._get_anthropic_tools_and_tool_choice_or_omit(
            [
                Tool(
                    name="weather",
                    description="weather description",
                    strict=False,
                    schema={
                        "type": "object",
                        "properties": {
                            "cities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 3,
                            }
                        },
                        "required": ["cities"],
                    },
                )
            ],
            None,
            None,
        )

        input_schema = anthropic_tools[0]["input_schema"]
        self.assertEqual(input_schema["properties"]["cities"]["maxItems"], 3)

    def test_anthropic_serializes_user_image_parts(self):
        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))

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

        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))
        client._client = SimpleNamespace(messages=fake_messages)

        result = client.generate(
            model="claude-test",
            messages=[UserMessage(content=text_parts("Answer me"))],
        )

        self.assertEqual(result.content[0].text, "final answer")
        self.assertEqual(result.thinking, ["internal"])
        self.assertEqual(result.messages[-1].thinking, ["internal"])

    def test_anthropic_generate_preserves_multiple_thinking_blocks(self):
        fake_response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="internal-1"),
                SimpleNamespace(type="thinking", thinking="internal-2"),
                SimpleNamespace(type="text", text="final answer"),
            ]
        )
        fake_messages = FakeAnthropicMessages(fake_response)

        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))
        client._client = SimpleNamespace(messages=fake_messages)

        result = client.generate(
            model="claude-test",
            messages=[UserMessage(content=text_parts("Answer me"))],
        )

        self.assertEqual(result.thinking, ["internal-1", "internal-2"])
        self.assertEqual(result.messages[-1].thinking, ["internal-1", "internal-2"])

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

        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))
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

    def test_anthropic_generate_passes_reasoning_effort_tokens_as_thinking_budget(self):
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="final answer")],
        )
        fake_messages = FakeAnthropicMessages(fake_response)

        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))
        client._client = SimpleNamespace(messages=fake_messages)

        client.generate(
            model="claude-test",
            messages=[UserMessage(content=text_parts("Answer me"))],
            reasoning_effort=ReasoningEffort(tokens=2048),
        )

        self.assertEqual(
            fake_messages.calls[0]["thinking"],
            {"type": "enabled", "budget_tokens": 2048},
        )

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

        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))
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
                    delta=SimpleNamespace(type="thinking_delta", thinking="Plan"),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="Hello"),
                ),
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

        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))
        client._client = SimpleNamespace(messages=fake_messages)

        chunks = list(
            client.generate(
                model="claude-test",
                messages=[UserMessage(content=text_parts("Stream please"))],
                stream=True,
            )
        )

        marker_chunks = stream_marker_chunks(chunks)
        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            [chunk.type for chunk in chunks],
            [
                "event",
                "thinking",
                "event",
                "event",
                "content",
                "event",
                "completion",
            ],
        )
        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "content", None),
                ("end", "content", None),
            ],
        )
        self.assertEqual(payload_chunks[0].type, "thinking")
        self.assertEqual(payload_chunks[0].chunk, "Plan")
        self.assertEqual(payload_chunks[1].type, "content")
        self.assertEqual(payload_chunks[1].chunk, "Hello")
        self.assertEqual(payload_chunks[-1].thinking, ["Plan"])
        self.assertEqual(payload_chunks[-1].messages[-1].thinking, ["Plan"])
        self.assertEqual(payload_chunks[-1].usage.input_tokens, 9)
        self.assertEqual(payload_chunks[-1].usage.output_tokens, 2)
        self.assertEqual(payload_chunks[-1].usage.total_tokens, 11)
        self.assertEqual(payload_chunks[-1].usage.details["cache_creation_input_tokens"], 1)
        self.assertIsNotNone(payload_chunks[-1].duration_seconds)
        self.assertGreaterEqual(payload_chunks[-1].duration_seconds, 0)

    def test_anthropic_stream_wraps_multiple_thinking_blocks(self):
        fake_stream = FakeAnthropicStream(
            events=[
                SimpleNamespace(
                    type="content_block_start",
                    content_block=SimpleNamespace(type="thinking"),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="thinking_delta", thinking="Plan"),
                ),
                SimpleNamespace(
                    type="content_block_stop",
                    content_block=SimpleNamespace(type="thinking"),
                ),
                SimpleNamespace(
                    type="content_block_start",
                    content_block=SimpleNamespace(type="thinking"),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="thinking_delta", thinking="Reflect"),
                ),
                SimpleNamespace(
                    type="content_block_stop",
                    content_block=SimpleNamespace(type="thinking"),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="Hello"),
                ),
            ]
        )
        fake_messages = FakeAnthropicMessages(stream_response=fake_stream)

        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))
        client._client = SimpleNamespace(messages=fake_messages)

        chunks = list(
            client.generate(
                model="claude-test",
                messages=[UserMessage(content=text_parts("Stream please"))],
                stream=True,
            )
        )

        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "content", None),
                ("end", "content", None),
            ],
        )
        self.assertEqual(payload_chunks[0].chunk, "Plan")
        self.assertEqual(payload_chunks[1].chunk, "Reflect")
        self.assertEqual(payload_chunks[2].chunk, "Hello")
        self.assertEqual(payload_chunks[-1].thinking, ["Plan", "Reflect"])
        self.assertEqual(payload_chunks[-1].messages[-1].thinking, ["Plan", "Reflect"])

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

        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))
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
        self.assertEqual(
            fake_messages.calls[0]["tool_choice"],
            {"type": "tool", "name": "final_answer"},
        )

    def test_anthropic_stream_hides_internal_response_schema_tool(self):
        fake_stream = FakeAnthropicStream(
            events=[
                SimpleNamespace(
                    type="content_block_start",
                    content_block=SimpleNamespace(
                        type="tool_use",
                        id="schema_1",
                        name="final_answer",
                    ),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(
                        type="input_json_delta",
                        partial_json='{"answer": ',
                    ),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(
                        type="input_json_delta",
                        partial_json='"done"}',
                    ),
                ),
                SimpleNamespace(
                    type="content_block_stop",
                    content_block=SimpleNamespace(
                        type="tool_use",
                        id="schema_1",
                        name="final_answer",
                        input={"answer": "done"},
                    ),
                ),
            ],
            final_message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=8,
                    output_tokens=2,
                )
            ),
        )
        fake_messages = FakeAnthropicMessages(stream_response=fake_stream)

        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))
        client._client = SimpleNamespace(messages=fake_messages)

        chunks = list(
            client.generate(
                model="claude-test",
                messages=[UserMessage(content=text_parts("Answer in JSON"))],
                response_format=JSONSchemaResponse(
                    name="final_answer",
                    strict=False,
                    json_schema=AnswerSchema,
                ),
                stream=True,
            )
        )

        marker_chunks = stream_marker_chunks(chunks)
        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(marker_chunks, [])
        self.assertEqual(len(payload_chunks), 1)
        self.assertEqual(payload_chunks[0].type, "completion")
        self.assertEqual(payload_chunks[0].content, {"answer": "done"})
        self.assertEqual(payload_chunks[0].tool_calls, [])
        self.assertEqual(payload_chunks[0].messages[-1].tool_calls, [])
        self.assertEqual(
            fake_messages.stream_calls[0]["tool_choice"],
            {"type": "tool", "name": "final_answer"},
        )

    def test_anthropic_stream_emits_tool_complete_chunk(self):
        fake_stream = FakeAnthropicStream(
            events=[
                SimpleNamespace(
                    type="content_block_start",
                    content_block=SimpleNamespace(
                        type="tool_use",
                        id="call_1",
                        name="get_weather",
                    ),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(
                        type="input_json_delta",
                        partial_json='{"city":"Kath',
                    ),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(
                        type="input_json_delta",
                        partial_json='mandu"}',
                    ),
                ),
                SimpleNamespace(
                    type="content_block_stop",
                    content_block=SimpleNamespace(
                        type="tool_use",
                        id="call_1",
                        name="get_weather",
                        input={"city": "Kathmandu"},
                    ),
                ),
            ],
            final_message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=8,
                    output_tokens=2,
                )
            ),
        )
        fake_messages = FakeAnthropicMessages(stream_response=fake_stream)

        client = AnthropicClient(config=AnthropicClientConfig(api_key="test"))
        client._client = SimpleNamespace(messages=fake_messages)

        chunks = list(
            client.generate(
                model="claude-test",
                messages=[UserMessage(content=text_parts("Weather?"))],
                tools=[make_tool("get_weather")],
                stream=True,
            )
        )

        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            [chunk.type for chunk in chunks],
            [
                "event",
                "tool",
                "tool",
                "tool_complete",
                "event",
                "completion",
            ],
        )
        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "tool", "get_weather"),
                ("end", "tool", "get_weather"),
            ],
        )
        self.assertEqual(payload_chunks[0].type, "tool")
        self.assertEqual(payload_chunks[1].type, "tool")
        self.assertEqual(payload_chunks[2].type, "tool_complete")
        self.assertEqual(payload_chunks[2].tool, "get_weather")
        self.assertEqual(payload_chunks[2].arguments, '{"city": "Kathmandu"}')
        self.assertEqual(payload_chunks[-1].tool_calls[0].arguments, '{"city": "Kathmandu"}')

    def test_google_required_tool_choice_uses_any_mode_with_allowed_names(self):
        client = GoogleClient(config=GoogleClientConfig(api_key="test"))

        google_tools, tool_config = client._get_google_tools_and_tool_config(
            [make_tool("weather"), make_tool("time")],
            {"mode": "required", "tools": ["weather"]},
        )

        self.assertEqual(len(google_tools), 1)
        self.assertEqual(tool_config.function_calling_config.mode, "ANY")
        self.assertEqual(
            tool_config.function_calling_config.allowed_function_names,
            ["weather"],
        )

    def test_google_web_search_attaches_without_function_tool_config(self):
        client = GoogleClient(config=GoogleClientConfig(api_key="test"))

        google_tools, tool_config = client._get_google_tools_and_tool_config(
            [make_web_search_tool()],
            {"mode": "required", "tools": ["web_search"]},
        )

        self.assertEqual(len(google_tools), 1)
        self.assertIsNotNone(google_tools[0].google_search)
        self.assertIsNone(tool_config)

    def test_google_required_web_search_keeps_function_targeting_best_effort(self):
        client = GoogleClient(config=GoogleClientConfig(api_key="test"))

        google_tools, tool_config = client._get_google_tools_and_tool_config(
            [make_tool("weather"), make_web_search_tool()],
            {"mode": "required", "tools": ["web_search", "weather"]},
        )

        self.assertEqual(len(google_tools), 2)
        self.assertIsNotNone(google_tools[1].google_search)
        self.assertEqual(tool_config.function_calling_config.mode, "ANY")
        self.assertEqual(
            tool_config.function_calling_config.allowed_function_names,
            ["weather"],
        )

    def test_google_tool_schemas_strip_unsupported_keywords_when_strict(self):
        client = GoogleClient(config=GoogleClientConfig(api_key="test"))

        google_tools, _ = client._get_google_tools_and_tool_config(
            [
                Tool(
                    name="weather",
                    description="weather description",
                    strict=True,
                    schema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "example": "Kathmandu",
                                    },
                                },
                                "required": ["city"],
                            }
                        },
                        "required": ["location"],
                    },
                )
            ],
            None,
        )

        parameters = google_tools[0].function_declarations[0].parameters
        payload = parameters.model_dump(exclude_none=True, by_alias=True)
        self.assertNotIn("example", payload["properties"]["location"]["properties"]["city"])
        self.assertNotIn("additionalProperties", payload)
        self.assertNotIn("additional_properties", payload)

    def test_google_tool_schemas_keep_non_strict_keywords(self):
        client = GoogleClient(config=GoogleClientConfig(api_key="test"))

        google_tools, _ = client._get_google_tools_and_tool_config(
            [
                Tool(
                    name="weather",
                    description="weather description",
                    strict=False,
                    schema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "example": "Kathmandu",
                                    },
                                },
                                "required": ["city"],
                            }
                        },
                        "required": ["location"],
                    },
                )
            ],
            None,
        )

        parameters = google_tools[0].function_declarations[0].parameters
        payload = parameters.model_dump(exclude_none=True, by_alias=True)
        self.assertEqual(
            payload["properties"]["location"]["properties"]["city"]["example"],
            "Kathmandu",
        )

    def test_google_stream_generates_tool_id_when_missing(self):
        fake_models = FakeGoogleModels(
            stream_response=iter(
                [
                    SimpleNamespace(
                        candidates=[
                            SimpleNamespace(
                                content=SimpleNamespace(
                                    parts=[
                                        SimpleNamespace(
                                            text=None,
                                            thought=False,
                                            inline_data=None,
                                            file_data=None,
                                            function_call=SimpleNamespace(
                                                id=None,
                                                name="get_weather",
                                                args={"city": "Kathmandu"},
                                            ),
                                        )
                                    ]
                                )
                            )
                        ]
                    )
                ]
            )
        )

        client = GoogleClient(config=GoogleClientConfig(api_key="test"))
        client._client = SimpleNamespace(models=fake_models)

        chunks = list(
            client.generate(
                model="gemini-test",
                messages=[UserMessage(content=text_parts("Weather?"))],
                tools=[make_tool("get_weather")],
                stream=True,
            )
        )

        payload_chunks = stream_payload_chunks(chunks)
        tool_chunk = payload_chunks[0]
        tool_complete_chunk = payload_chunks[1]
        completion_chunk = payload_chunks[-1]

        self.assertEqual(tool_chunk.type, "tool")
        self.assertEqual(tool_chunk.tool, "get_weather")
        self.assertTrue(tool_chunk.id.startswith("call_"))
        self.assertNotEqual(tool_chunk.id, "get_weather")
        self.assertEqual(tool_complete_chunk.type, "tool_complete")
        self.assertEqual(tool_complete_chunk.id, tool_chunk.id)
        self.assertEqual(tool_complete_chunk.tool, "get_weather")
        self.assertEqual(tool_complete_chunk.arguments, '{"city": "Kathmandu"}')
        self.assertEqual(completion_chunk.tool_calls[0].id, tool_chunk.id)
        self.assertEqual(completion_chunk.messages[-1].tool_calls[0].id, tool_chunk.id)

    def test_google_serializes_user_images(self):
        client = GoogleClient(config=GoogleClientConfig(api_key="test"))

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

        client = GoogleClient(config=GoogleClientConfig(api_key="test"))
        client._client = SimpleNamespace(models=fake_models)

        result = client.generate(
            model="gemini-test",
            messages=[UserMessage(content=text_parts("Show me something"))],
        )

        self.assertEqual(result.thinking, ["hidden reasoning"])
        self.assertEqual(result.messages[-1].thinking, ["hidden reasoning"])
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

    def test_google_generate_preserves_multiple_thinking_blocks(self):
        fake_response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text="Plan",
                                thought=True,
                                inline_data=None,
                                file_data=None,
                                function_call=None,
                            ),
                            SimpleNamespace(
                                text="Hello",
                                thought=False,
                                inline_data=None,
                                file_data=None,
                                function_call=None,
                            ),
                            SimpleNamespace(
                                text="Reflect",
                                thought=True,
                                inline_data=None,
                                file_data=None,
                                function_call=None,
                            ),
                        ]
                    )
                )
            ]
        )
        fake_models = FakeGoogleModels(response=fake_response)

        client = GoogleClient(config=GoogleClientConfig(api_key="test"))
        client._client = SimpleNamespace(models=fake_models)

        result = client.generate(
            model="gemini-test",
            messages=[UserMessage(content=text_parts("Show me something"))],
        )

        self.assertEqual(result.thinking, ["Plan", "Reflect"])
        self.assertEqual(result.messages[-1].thinking, ["Plan", "Reflect"])

    def test_google_generate_passes_reasoning_effort_to_thinking_config(self):
        fake_response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text="hello",
                                thought=False,
                                inline_data=None,
                                file_data=None,
                                function_call=None,
                            ),
                        ]
                    )
                )
            ]
        )
        fake_models = FakeGoogleModels(response=fake_response)

        client = GoogleClient(config=GoogleClientConfig(api_key="test"))
        client._client = SimpleNamespace(models=fake_models)

        client.generate(
            model="gemini-test",
            messages=[UserMessage(content=text_parts("Show me something"))],
            reasoning_effort=ReasoningEffort(
                effort=ReasoningEffortValue.HIGH,
                tokens=1024,
                summary=ReasoningSummary.DETAILED,
            ),
        )

        thinking_config = fake_models.calls[0]["config"].thinking_config
        self.assertEqual(thinking_config.include_thoughts, True)
        self.assertEqual(thinking_config.thinking_budget, 1024)
        self.assertEqual(thinking_config.thinking_level, "HIGH")

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
                                            text="Plan",
                                            thought=True,
                                            inline_data=None,
                                            file_data=None,
                                            function_call=None,
                                        ),
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

        client = GoogleClient(config=GoogleClientConfig(api_key="test"))
        client._client = SimpleNamespace(models=fake_models)

        chunks = list(
            client.generate(
                model="gemini-test",
                messages=[UserMessage(content=text_parts("Show me something"))],
                stream=True,
            )
        )

        marker_chunks = stream_marker_chunks(chunks)
        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            [chunk.type for chunk in chunks],
            [
                "event",
                "thinking",
                "event",
                "event",
                "content",
                "event",
                "completion",
            ],
        )
        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "content", None),
                ("end", "content", None),
            ],
        )
        self.assertEqual(payload_chunks[0].type, "thinking")
        self.assertEqual(payload_chunks[0].chunk, "Plan")
        self.assertEqual(payload_chunks[1].type, "content")
        self.assertEqual(payload_chunks[1].chunk, "Hello")
        self.assertEqual(payload_chunks[-1].type, "completion")
        self.assertEqual(payload_chunks[-1].thinking, ["Plan"])
        self.assertEqual(payload_chunks[-1].messages[-1].thinking, ["Plan"])
        self.assertIsInstance(payload_chunks[-1].content, list)
        self.assertEqual(payload_chunks[-1].content[0].text, "Hello")
        self.assertEqual(payload_chunks[-1].content[1].mime_type, "image/png")
        self.assertEqual(payload_chunks[-1].usage.input_tokens, 6)
        self.assertEqual(payload_chunks[-1].usage.output_tokens, 2)
        self.assertEqual(payload_chunks[-1].usage.total_tokens, 8)
        self.assertIsNotNone(payload_chunks[-1].duration_seconds)
        self.assertGreaterEqual(payload_chunks[-1].duration_seconds, 0)

    def test_google_stream_wraps_multiple_thinking_blocks(self):
        fake_models = FakeGoogleModels(
            stream_response=iter(
                [
                    SimpleNamespace(
                        candidates=[
                            SimpleNamespace(
                                content=SimpleNamespace(
                                    parts=[
                                        SimpleNamespace(
                                            text="Plan",
                                            thought=True,
                                            inline_data=None,
                                            file_data=None,
                                            function_call=None,
                                        ),
                                        SimpleNamespace(
                                            text="Reflect",
                                            thought=True,
                                            inline_data=None,
                                            file_data=None,
                                            function_call=None,
                                        ),
                                        SimpleNamespace(
                                            text="Hello",
                                            thought=False,
                                            inline_data=None,
                                            file_data=None,
                                            function_call=None,
                                        ),
                                    ]
                                )
                            )
                        ]
                    )
                ]
            )
        )

        client = GoogleClient(config=GoogleClientConfig(api_key="test"))
        client._client = SimpleNamespace(models=fake_models)

        chunks = list(
            client.generate(
                model="gemini-test",
                messages=[UserMessage(content=text_parts("Show me something"))],
                stream=True,
            )
        )

        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "content", None),
                ("end", "content", None),
            ],
        )
        self.assertEqual(payload_chunks[0].chunk, "Plan")
        self.assertEqual(payload_chunks[1].chunk, "Reflect")
        self.assertEqual(payload_chunks[2].chunk, "Hello")
        self.assertEqual(payload_chunks[-1].thinking, ["Plan", "Reflect"])
        self.assertEqual(payload_chunks[-1].messages[-1].thinking, ["Plan", "Reflect"])

    def test_google_generate_uses_native_structured_output(self):
        fake_response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text='{"answer":"done"}',
                                thought=False,
                                inline_data=None,
                                file_data=None,
                                function_call=None,
                            )
                        ]
                    )
                )
            ]
        )
        fake_models = FakeGoogleModels(response=fake_response)

        client = GoogleClient(config=GoogleClientConfig(api_key="test"))
        client._client = SimpleNamespace(models=fake_models)

        result = client.generate(
            model="gemini-test",
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
        self.assertEqual(fake_models.calls[0]["config"].response_mime_type, "application/json")
        self.assertIsNotNone(fake_models.calls[0]["config"].response_json_schema)

    def test_google_generate_wraps_provider_auth_errors(self):
        fake_models = FakeGoogleModels(
            response=google_errors.ClientError(
                401,
                {"message": "bad api key", "status": "UNAUTHENTICATED"},
            )
        )

        client = GoogleClient(config=GoogleClientConfig(api_key="test"))
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

        client, sessions, _ = self.make_bedrock_client(
            fake_runtime,
            api_key="bedrock-api-key",
            region="us-east-1",
        )

        self.assertIsNotNone(client)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].kwargs["region_name"], "us-east-1")
        self.assertEqual(
            sessions[0]
            .kwargs["botocore_session"]
            .get_auth_token(signing_name="bedrock")
            .token,
            "bedrock-api-key",
        )
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
                region="us-east-1",
                aws_access_key_id="aws-id",
                aws_secret_access_key="aws-secret",
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.provider, "bedrock")

    def test_bedrock_init_supports_aws_credentials(self):
        fake_runtime = FakeBedrockRuntimeClient(response={})
        client, sessions, _ = self.make_bedrock_client(
            fake_runtime,
            region="us-west-2",
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
            region="us-east-1",
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
            region="us-east-1",
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

    def test_bedrock_native_structured_output_sanitizes_response_schema_for_both_strict_values(self):
        schema = {
            "type": "object",
            "properties": {
                "bulletPoints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "image": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "minWords": 2,
                        }
                    },
                    "required": ["prompt"],
                },
            },
            "required": ["bulletPoints", "image"],
        }

        for strict in (False, True):
            with self.subTest(strict=strict):
                fake_runtime = FakeBedrockRuntimeClient(response={})
                client, _, _ = self.make_bedrock_client(
                    fake_runtime,
                    region="us-east-1",
                    aws_access_key_id="aws-id",
                    aws_secret_access_key="aws-secret",
                )

                client.generate(
                    model="anthropic.claude-3-5-haiku",
                    messages=[UserMessage(content=text_parts("Answer in JSON"))],
                    response_format=JSONSchemaResponse(
                        strict=strict,
                        json_schema=schema,
                    ),
                )

                sent_schema = json.loads(
                    fake_runtime.calls[0]["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"]
                )
                self.assertNotIn("maxItems", sent_schema["properties"]["bulletPoints"])
                self.assertNotIn(
                    "minWords",
                    sent_schema["properties"]["image"]["properties"]["prompt"],
                )

    def test_bedrock_ignores_web_search_tool(self):
        fake_runtime = FakeBedrockRuntimeClient(response={})
        client, _, _ = self.make_bedrock_client(
            fake_runtime,
            region="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        tool_config = client._get_bedrock_tool_config(
            [make_web_search_tool()],
            {"mode": "required", "tools": ["web_search"]},
        )

        self.assertIsNone(tool_config)

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
            region="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        result = client.generate(
            model="anthropic.claude-3-5-haiku",
            messages=[UserMessage(content=text_parts("Weather?"))],
            tools=[make_tool("get_weather"), make_tool("time")],
            tool_choice={"mode": "required", "tools": ["get_weather"]},
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
            region="us-east-1",
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
                                "contentBlockIndex": 0,
                                "delta": {
                                    "reasoningContent": {"text": "Plan"}
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
            region="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        chunks = list(
            client.generate(
                model="anthropic.claude-3-5-haiku",
                messages=[UserMessage(content=text_parts("Weather?"))],
                tools=[make_tool("get_weather")],
                stream=True,
            )
        )

        marker_chunks = stream_marker_chunks(chunks)
        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            [chunk.type for chunk in chunks],
            [
                "event",
                "content",
                "event",
                "event",
                "thinking",
                "event",
                "event",
                "tool",
                "tool",
                "tool_complete",
                "event",
                "completion",
            ],
        )
        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "content", None),
                ("end", "content", None),
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "tool", "get_weather"),
                ("end", "tool", "get_weather"),
            ],
        )
        self.assertEqual(payload_chunks[0].type, "content")
        self.assertEqual(payload_chunks[0].chunk, "Hello")
        self.assertEqual(payload_chunks[1].type, "thinking")
        self.assertEqual(payload_chunks[1].chunk, "Plan")
        self.assertEqual(payload_chunks[2].type, "tool")
        self.assertEqual(payload_chunks[3].type, "tool")
        self.assertEqual(payload_chunks[2].tool, "get_weather")
        self.assertEqual(payload_chunks[3].tool, "get_weather")
        self.assertEqual(payload_chunks[4].type, "tool_complete")
        self.assertEqual(payload_chunks[4].tool, "get_weather")
        self.assertEqual(payload_chunks[4].arguments, '{"city":"Kathmandu"}')
        self.assertEqual(payload_chunks[-1].thinking, ["Plan"])
        self.assertEqual(payload_chunks[-1].messages[-1].thinking, ["Plan"])
        self.assertEqual(payload_chunks[-1].tool_calls[0].name, "get_weather")
        self.assertEqual(
            payload_chunks[-1].tool_calls[0].arguments,
            '{"city":"Kathmandu"}',
        )
        self.assertEqual(payload_chunks[-1].usage.input_tokens, 10)
        self.assertEqual(payload_chunks[-1].usage.output_tokens, 4)
        self.assertEqual(payload_chunks[-1].usage.total_tokens, 14)
        self.assertIsNotNone(payload_chunks[-1].duration_seconds)
        self.assertGreaterEqual(payload_chunks[-1].duration_seconds, 0)

    def test_bedrock_stream_wraps_multiple_thinking_blocks(self):
        fake_runtime = FakeBedrockRuntimeClient(
            stream_response={
                "stream": iter(
                    [
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 0,
                                "delta": {
                                    "reasoningContent": {"text": "Plan"}
                                },
                            }
                        },
                        {
                            "contentBlockStop": {
                                "contentBlockIndex": 0,
                            }
                        },
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 1,
                                "delta": {
                                    "reasoningContent": {"text": "Reflect"}
                                },
                            }
                        },
                        {
                            "contentBlockStop": {
                                "contentBlockIndex": 1,
                            }
                        },
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 2,
                                "delta": {"text": "Hello"},
                            }
                        },
                    ]
                )
            }
        )
        client, _, _ = self.make_bedrock_client(
            fake_runtime,
            region="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        chunks = list(
            client.generate(
                model="anthropic.claude-3-5-haiku",
                messages=[UserMessage(content=text_parts("Weather?"))],
                stream=True,
            )
        )

        payload_chunks = stream_payload_chunks(chunks)

        self.assertEqual(
            stream_marker_events(chunks),
            [
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "thinking", None),
                ("end", "thinking", None),
                ("start", "content", None),
                ("end", "content", None),
            ],
        )
        self.assertEqual(payload_chunks[0].chunk, "Plan")
        self.assertEqual(payload_chunks[1].chunk, "Reflect")
        self.assertEqual(payload_chunks[2].chunk, "Hello")
        self.assertEqual(payload_chunks[-1].thinking, ["Plan", "Reflect"])
        self.assertEqual(payload_chunks[-1].messages[-1].thinking, ["Plan", "Reflect"])

    def test_bedrock_stream_generates_tool_id_when_missing(self):
        fake_runtime = FakeBedrockRuntimeClient(
            stream_response={
                "stream": iter(
                    [
                        {
                            "contentBlockStart": {
                                "contentBlockIndex": 0,
                                "start": {
                                    "toolUse": {
                                        "name": "get_weather",
                                        "type": "server_tool_use",
                                    }
                                },
                            }
                        },
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 0,
                                "delta": {
                                    "toolUse": {"input": '{"city":"Kath'}
                                },
                            }
                        },
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 0,
                                "delta": {
                                    "toolUse": {"input": 'mandu"}'}
                                },
                            }
                        },
                    ]
                )
            }
        )
        client, _, _ = self.make_bedrock_client(
            fake_runtime,
            region="us-east-1",
            aws_access_key_id="aws-id",
            aws_secret_access_key="aws-secret",
        )

        chunks = list(
            client.generate(
                model="anthropic.claude-3-5-haiku",
                messages=[UserMessage(content=text_parts("Weather?"))],
                tools=[make_tool("get_weather")],
                stream=True,
            )
        )

        payload_chunks = stream_payload_chunks(chunks)
        tool_chunk_1 = payload_chunks[0]
        tool_chunk_2 = payload_chunks[1]
        completion_chunk = payload_chunks[-1]

        self.assertEqual(tool_chunk_1.type, "tool")
        self.assertEqual(tool_chunk_2.type, "tool")
        self.assertTrue(tool_chunk_1.id.startswith("call_"))
        self.assertEqual(tool_chunk_1.id, tool_chunk_2.id)
        self.assertEqual(completion_chunk.tool_calls[0].id, tool_chunk_1.id)
        self.assertEqual(completion_chunk.messages[-1].tool_calls[0].id, tool_chunk_1.id)

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
            region="us-east-1",
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
