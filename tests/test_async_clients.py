import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import llmai.client as client_module
from llmai.async_clients import _anthropic_bridge
from llmai import (
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
    def __init__(self, *, barrier: threading.Barrier | None = None):
        self.barrier = barrier
        self.calls = []
        self.closed = 0
        self.stream_closed = threading.Event()
        self._client = SimpleNamespace(close=self.close)

    def close(self):
        self.closed += 1

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["stream"]:
            return self._stream()

        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        return ResponseContent(content=kwargs["model"])

    def _stream(self):
        try:
            yield ResponseStreamContentChunk(chunk="hello")
            yield ResponseStreamCompletionChunk(content="hello")
        finally:
            self.stream_closed.set()


class AsyncClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_agenerate_returns_response_content(self):
        sync_client = FakeSyncClient()
        client = AsyncBaseClient(sync_client=sync_client)

        result = await client.agenerate(
            model="model",
            messages=[UserMessage(content="Hello")],
        )

        self.assertEqual(result.content, "model")
        self.assertEqual(sync_client.calls[0]["stream"], False)

    async def test_agenerate_stream_is_directly_async_iterable(self):
        sync_client = FakeSyncClient()
        client = AsyncBaseClient(sync_client=sync_client)

        stream = client.agenerate(
            model="model",
            messages=[UserMessage(content="Hello")],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

        self.assertEqual([chunk.type for chunk in chunks], ["content", "completion"])
        self.assertTrue(sync_client.stream_closed.is_set())

    async def test_agenerate_stream_closes_when_consumer_exits_early(self):
        sync_client = FakeSyncClient()
        client = AsyncBaseClient(sync_client=sync_client)
        stream = client.agenerate(
            model="model",
            messages=[UserMessage(content="Hello")],
            stream=True,
        )

        first = await anext(stream)
        self.assertEqual(first.type, "content")
        await stream.aclose()

        self.assertTrue(sync_client.stream_closed.is_set())

    async def test_multiple_requests_run_concurrently(self):
        sync_client = FakeSyncClient(barrier=threading.Barrier(2))
        client = AsyncBaseClient(sync_client=sync_client)

        first, second = await asyncio.wait_for(
            asyncio.gather(
                client.agenerate(model="first", messages=[]),
                client.agenerate(model="second", messages=[]),
            ),
            timeout=3,
        )

        self.assertEqual({first.content, second.content}, {"first", "second"})

    async def test_context_manager_closes_once_and_rejects_reuse(self):
        sync_client = FakeSyncClient()
        client = AsyncBaseClient(sync_client=sync_client)

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
                    )
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
        completions = SimpleNamespace(
            create=lambda **kwargs: fake_response,
        )

        provider_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock()),
            ),
            responses=SimpleNamespace(create=AsyncMock()),
            close=AsyncMock(),
        )
        with (
            patch("llmai.openai.client.OpenAI") as openai_cls,
            patch(
                "llmai.async_clients.AsyncOpenAI",
                return_value=provider_client,
            ),
        ):
            client = AsyncOpenAIClient(
                config=OpenAIClientConfig(api_key="key"),
            )
        client._sync_client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        result = await client.agenerate(
            model="gpt-test",
            messages=[UserMessage(content="Hello")],
        )

        self.assertEqual(result.content[0].text, "answer")
        self.assertEqual(result.usage.total_tokens, 3)
        openai_cls.assert_called_once()
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
                "llmai.async_clients.AsyncOpenAI",
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
                "llmai.async_clients.AsyncOpenAI",
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
                "llmai.async_clients.AsyncOpenAI",
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

    async def test_anthropic_stream_bridge_preserves_stream_methods(self):
        exited = asyncio.Event()

        class NativeStream:
            async def __aiter__(self):
                yield "event"

            async def get_final_message(self):
                return "final"

        class NativeStreamContext:
            async def __aenter__(self):
                return NativeStream()

            async def __aexit__(self, exc_type, exc, traceback):
                exited.set()

        native_messages = SimpleNamespace(
            create=AsyncMock(),
            stream=lambda **kwargs: NativeStreamContext(),
        )

        class SyncParser:
            def __init__(self):
                self._client = SimpleNamespace(
                    messages=native_messages,
                )

            def generate(self, **kwargs):
                with self._client.messages.stream() as stream:
                    events = list(stream)
                    final = stream.get_final_message()
                return ResponseContent(content=[events, final])

        sync_parser = SyncParser()
        sync_parser._client = _anthropic_bridge(
            SimpleNamespace(messages=native_messages)
        )
        client = AsyncBaseClient(sync_client=sync_parser)

        result = await client.agenerate(model="claude-test", messages=[])

        self.assertEqual(result.content, [["event"], "final"])
        self.assertTrue(exited.is_set())

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
