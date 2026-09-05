import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import llmai.client as client_module
import openai
from llmai import (
    AsyncBedrockClient,
    AsyncChatGPTClient,
    AsyncOpenAIClient,
    OpenAIClientConfig,
    get_async_client,
)
from llmai.shared import (
    AnthropicClientConfig,
    AsyncBaseClient,
    AzureOpenAIClientConfig,
    BedrockClientConfig,
    CerebrasClientConfig,
    ChatGPTClientConfig,
    DeepSeekClientConfig,
    FireworksClientConfig,
    GoogleClientConfig,
    LiteLLMClientConfig,
    LMStudioClientConfig,
    OpenRouterClientConfig,
    ResponseContent,
    ResponseStreamCompletionChunk,
    ResponseStreamContentChunk,
    TogetherAIClientConfig,
    UserMessage,
    VertexAIClientConfig,
)


class FakeSyncClient:
    def __init__(self):
        self.closed = 0
        self._client = SimpleNamespace(close=self.close)

    def close(self):
        self.closed += 1

    def generate(self, **kwargs):  # pragma: no cover - AsyncBaseClient must not call it.
        raise AssertionError("sync generate should not be used by AsyncBaseClient")


class FakeAsyncClient(AsyncBaseClient):
    def __init__(self, *, sync_client: FakeSyncClient | None = None):
        super().__init__(sync_client=sync_client or FakeSyncClient())
        self.calls = []
        self.stream_closed = asyncio.Event()

    async def _agenerate_once(self, **kwargs):
        self.calls.append(kwargs)
        return ResponseContent(content=kwargs["model"])

    async def _agenerate_stream(self, **kwargs):
        self.calls.append(kwargs)
        try:
            yield ResponseStreamContentChunk(chunk="hello")
            yield ResponseStreamCompletionChunk(content="hello")
        finally:
            self.stream_closed.set()


class AsyncClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_agenerate_returns_response_content(self):
        client = FakeAsyncClient()

        result = await client.agenerate(
            model="model",
            messages=[UserMessage(content="Hello")],
        )

        self.assertEqual(result.content, "model")
        self.assertEqual(client.calls[0]["model"], "model")

    async def test_agenerate_stream_is_directly_async_iterable(self):
        client = FakeAsyncClient()

        stream = client.agenerate(
            model="model",
            messages=[UserMessage(content="Hello")],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

        self.assertEqual([chunk.type for chunk in chunks], ["content", "completion"])
        self.assertTrue(client.stream_closed.is_set())

    async def test_agenerate_stream_closes_when_consumer_exits_early(self):
        client = FakeAsyncClient()
        stream = client.agenerate(
            model="model",
            messages=[UserMessage(content="Hello")],
            stream=True,
        )

        first = await anext(stream)
        self.assertEqual(first.type, "content")
        await stream.aclose()

        self.assertTrue(client.stream_closed.is_set())

    async def test_context_manager_closes_once_and_rejects_reuse(self):
        sync_client = FakeSyncClient()
        client = FakeAsyncClient(sync_client=sync_client)

        async with client as entered:
            self.assertIs(entered, client)

        await client.aclose()
        self.assertEqual(sync_client.closed, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            client.agenerate(model="model", messages=[])

    async def test_async_openai_client_preserves_sync_provider_behavior(self):
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="answer",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
        )
        provider_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=fake_response)),
            ),
            responses=SimpleNamespace(create=AsyncMock()),
            close=AsyncMock(),
        )
        with (
            patch("llmai.openai.client.OpenAI") as openai_cls,
            patch(
                "llmai.openai.async_client.AsyncOpenAI",
                return_value=provider_client,
            ),
        ):
            client = AsyncOpenAIClient(
                config=OpenAIClientConfig(api_key="key"),
            )

        result = await client.agenerate(
            model="gpt-test",
            messages=[UserMessage(content="Hello")],
        )

        self.assertEqual(result.content[0].text, "answer")
        self.assertEqual(result.usage.total_tokens, 3)
        self.assertEqual(result.finish_reason, "stop")
        openai_cls.assert_called_once()
        openai_cls.return_value.close.assert_called_once()
        await client.aclose()

    async def test_async_chatgpt_stream_exposes_completed_finish_reason(self):
        completed_response = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="final answer")
                    ],
                )
            ],
            usage=None,
        )

        async def response_stream():
            yield SimpleNamespace(
                type="response.completed",
                response=completed_response,
            )

        provider_client = SimpleNamespace(
            responses=SimpleNamespace(
                create=AsyncMock(return_value=response_stream()),
            ),
            close=AsyncMock(),
        )
        with (
            patch("llmai.chatgpt.client.OpenAI") as openai_cls,
            patch(
                "llmai.chatgpt.async_client.AsyncOpenAI",
                return_value=provider_client,
            ),
        ):
            client = AsyncChatGPTClient(
                config=ChatGPTClientConfig(access_token="token"),
            )

        chunks = [
            chunk
            async for chunk in client.agenerate(
                model="gpt-test",
                messages=[UserMessage(content="Hello")],
                stream=True,
            )
        ]

        self.assertEqual(chunks[-1].type, "completion")
        self.assertEqual(chunks[-1].content[0].text, "final answer")
        self.assertEqual(chunks[-1].finish_reason, "stop")
        openai_cls.return_value.close.assert_called_once()
        await client.aclose()

    async def test_async_openai_uses_native_async_transport(self):
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="native answer",
                        tool_calls=None,
                    )
                )
            ],
            usage=None,
        )
        create = AsyncMock(return_value=fake_response)
        close = AsyncMock()
        provider_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
            responses=SimpleNamespace(create=AsyncMock()),
            close=close,
        )

        with (
            patch("llmai.openai.client.OpenAI"),
            patch(
                "llmai.openai.async_client.AsyncOpenAI",
                return_value=provider_client,
            ),
        ):
            client = AsyncOpenAIClient(
                config=OpenAIClientConfig(api_key="key"),
            )

        result = await client.agenerate(
            model="gpt-test",
            messages=[UserMessage(content="Hello")],
        )
        await client.aclose()

        self.assertEqual(result.content[0].text, "native answer")
        create.assert_awaited_once()
        self.assertIsInstance(create.await_args.kwargs["temperature"], openai.Omit)
        close.assert_awaited_once()

    async def test_async_openai_streams_native_async_events_incrementally(self):
        released = asyncio.Event()
        stream_closed = asyncio.Event()

        async def events():
            try:
                yield SimpleNamespace(
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content="first",
                                tool_calls=None,
                            )
                        )
                    ],
                )
                await released.wait()
                yield SimpleNamespace(
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=" second",
                                tool_calls=None,
                            )
                        )
                    ],
                )
            finally:
                stream_closed.set()

        create = AsyncMock(side_effect=lambda **kwargs: events())
        provider_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
            responses=SimpleNamespace(create=AsyncMock()),
            close=AsyncMock(),
        )

        with (
            patch("llmai.openai.client.OpenAI"),
            patch(
                "llmai.openai.async_client.AsyncOpenAI",
                return_value=provider_client,
            ),
        ):
            client = AsyncOpenAIClient(
                config=OpenAIClientConfig(api_key="key"),
            )

        stream = client.agenerate(
            model="gpt-test",
            messages=[UserMessage(content="Hello")],
            stream=True,
        )
        marker = await asyncio.wait_for(anext(stream), timeout=1)
        first = await asyncio.wait_for(anext(stream), timeout=1)

        self.assertEqual(marker.type, "event")
        self.assertEqual(first.chunk, "first")
        released.set()
        remaining = [chunk async for chunk in stream]
        await client.aclose()

        self.assertIn("completion", [chunk.type for chunk in remaining])
        self.assertTrue(stream_closed.is_set())
        self.assertIsInstance(create.await_args.kwargs["temperature"], openai.Omit)

    async def test_cancellation_reaches_native_provider_request(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def create(**kwargs):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        provider_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
            responses=SimpleNamespace(create=AsyncMock()),
            close=AsyncMock(),
        )
        with (
            patch("llmai.openai.client.OpenAI"),
            patch(
                "llmai.openai.async_client.AsyncOpenAI",
                return_value=provider_client,
            ),
        ):
            client = AsyncOpenAIClient(
                config=OpenAIClientConfig(api_key="key"),
            )

        task = asyncio.create_task(
            client.agenerate(model="gpt-test", messages=[])
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        await client.aclose()

    async def test_async_bedrock_uses_async_http_for_converse(self):
        captured = {}
        client = AsyncBedrockClient(
            config=BedrockClientConfig(region="us-east-1", api_key="bedrock-key"),
        )
        await client._http_client.aclose()

        async def request(method, url, *, headers, content):
            captured.update(
                {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "content": content,
                }
            )
            return httpx.Response(
                200,
                json={
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [{"text": "answer"}],
                        }
                    },
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 2,
                        "totalTokens": 3,
                    },
                },
                request=httpx.Request(method, url),
            )

        client._http_client = SimpleNamespace(
            request=request,
            aclose=AsyncMock(),
        )

        result = await client.agenerate(
            model="anthropic.claude-test-v1:0",
            messages=[UserMessage(content="Hello")],
        )
        await client.aclose()

        self.assertEqual(result.content[0].text, "answer")
        self.assertEqual(result.usage.total_tokens, 3)
        self.assertEqual(captured["method"], "POST")
        self.assertIn("/converse", captured["url"])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer bedrock-key")

    async def test_async_bedrock_uses_async_http_for_converse_stream(self):
        captured = {}
        stream_closed = asyncio.Event()

        class FakeStreamResponse:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                if False:
                    yield b""

            async def aclose(self):
                stream_closed.set()

        class FakeStreamContext:
            async def __aenter__(self):
                return FakeStreamResponse()

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        def stream(method, url, *, headers, content):
            captured.update(
                {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "content": content,
                }
            )
            return FakeStreamContext()

        client = AsyncBedrockClient(
            config=BedrockClientConfig(region="us-east-1", api_key="bedrock-key"),
        )
        await client._http_client.aclose()
        client._http_client = SimpleNamespace(
            stream=stream,
            aclose=AsyncMock(),
        )

        chunks = [
            chunk
            async for chunk in client.agenerate(
                model="anthropic.claude-test-v1:0",
                messages=[UserMessage(content="Hello")],
                stream=True,
            )
        ]
        await client.aclose()

        self.assertEqual(chunks[-1].type, "completion")
        self.assertEqual(captured["method"], "POST")
        self.assertIn("/converse-stream", captured["url"])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer bedrock-key")
        self.assertTrue(stream_closed.is_set())

    async def test_async_bedrock_parses_unwrapped_stream_event_payloads(self):
        client = AsyncBedrockClient(
            config=BedrockClientConfig(region="us-east-1", api_key="bedrock-key"),
        )
        output_shape = (
            client._service_model.operation_model("ConverseStream")
            .output_shape.members["stream"]
        )

        parsed = client._parse_stream_event_response(
            {
                "status_code": 200,
                "headers": {
                    ":message-type": "event",
                    ":event-type": "contentBlockDelta",
                    ":content-type": "application/json",
                },
                "body": b'{"contentBlockIndex":0,"delta":{"text":"Hello"}}',
            },
            output_shape,
        )
        await client.aclose()

        self.assertEqual(
            parsed,
            {
                "contentBlockDelta": {
                    "delta": {"text": "Hello"},
                    "contentBlockIndex": 0,
                }
            },
        )

    async def test_get_async_client_uses_provider_config(self):
        config = OpenAIClientConfig(api_key="key")

        with patch("llmai.client.AsyncOpenAIClient") as client_type:
            client = get_async_client(config=config)

        self.assertIs(client, client_type.return_value)
        client_type.assert_called_once_with(config=config, logger=None)

    async def test_get_async_client_supports_every_provider(self):
        cases = [
            (
                OpenAIClientConfig(api_key="key"),
                "AsyncOpenAIClient",
            ),
            (
                AzureOpenAIClientConfig(
                    api_key="key",
                    endpoint="https://azure.example",
                    api_version="2025-01-01",
                ),
                "AsyncAzureOpenAIClient",
            ),
            (
                VertexAIClientConfig(project="project", location="location"),
                "AsyncVertexAIClient",
            ),
            (
                ChatGPTClientConfig(access_token="token"),
                "AsyncChatGPTClient",
            ),
            (
                DeepSeekClientConfig(api_key="key"),
                "AsyncDeepSeekClient",
            ),
            (
                OpenRouterClientConfig(api_key="key"),
                "AsyncOpenRouterClient",
            ),
            (
                CerebrasClientConfig(api_key="key"),
                "AsyncCerebrasClient",
            ),
            (
                FireworksClientConfig(api_key="key"),
                "AsyncFireworksClient",
            ),
            (
                TogetherAIClientConfig(api_key="key"),
                "AsyncTogetherAIClient",
            ),
            (
                LMStudioClientConfig(),
                "AsyncLMStudioClient",
            ),
            (
                GoogleClientConfig(api_key="key"),
                "AsyncGoogleClient",
            ),
            (
                AnthropicClientConfig(api_key="key"),
                "AsyncAnthropicClient",
            ),
            (
                BedrockClientConfig(region="us-east-1", api_key="key"),
                "AsyncBedrockClient",
            ),
            (
                LiteLLMClientConfig(),
                "AsyncLiteLLMClient",
            ),
        ]

        for config, class_name in cases:
            with self.subTest(provider=config.provider):
                with patch.object(client_module, class_name) as client_type:
                    client = get_async_client(config=config)

                self.assertIs(client, client_type.return_value)
                client_type.assert_called_once_with(config=config, logger=None)


if __name__ == "__main__":
    unittest.main()
