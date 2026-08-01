import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import openai

from llmai.anthropic.client import AnthropicClient
from llmai.async_clients import AsyncAnthropicClient, AsyncOpenAIClient
from llmai.google.client import GoogleClient
from llmai.openai.client import OpenAIClient
from llmai.shared.configs import AnthropicClientConfig, OpenAIClientConfig
from llmai.shared.base import AsyncBaseClient, BaseClient
from llmai.shared.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    normalize_llm_error,
)
from llmai.shared.model_listing import (
    amodel_ids,
    model_ids,
    openai_compatible_model_ids,
)
from llmai.togetherai.client import TogetherAIClient


class FakeModels:
    def __init__(self, models):
        self._models = models

    def list(self, **kwargs):
        return self._models


class FakeAsyncModelsPage:
    def __init__(self, models):
        self._models = models

    def __iter__(self):
        # Pydantic async page models expose a synchronous mapping iterator.
        return iter([("data", self._models), ("has_more", False)])

    async def __aiter__(self):
        for model in self._models:
            yield model


class FakeHTTPClient:
    def __init__(self, response):
        self.response = response
        self.request = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def get(self, url, *, headers):
        self.request = (url, headers)
        return self.response


class FakeListingClient(BaseClient):
    PROVIDER_NAME = "fake"

    def list_available_models(self):
        return ["model-a", "model-b"]

    def generate(self, **kwargs):
        raise NotImplementedError


class FakeAsyncListingClient(AsyncBaseClient):
    async def _agenerate_once(self, **kwargs):
        raise NotImplementedError

    async def _agenerate_stream(self, **kwargs):
        raise NotImplementedError
        yield


class UnsupportedListingClient(BaseClient):
    PROVIDER_NAME = "unsupported"

    def generate(self, **kwargs):
        raise NotImplementedError


def response(status_code, payload):
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("GET", "https://provider.example/v1/models"),
    )


class ModelListingHelpersTests(unittest.TestCase):
    def test_model_ids_accept_provider_objects_dicts_and_strings(self):
        models = [
            SimpleNamespace(id="model-a"),
            {"name": "model-b"},
            "model-c",
            {"id": "model-a"},
            None,
        ]

        self.assertEqual(model_ids(models), ["model-a", "model-b", "model-c"])

    def test_openai_compatible_payload_supports_both_shapes(self):
        self.assertEqual(
            openai_compatible_model_ids({"data": [{"id": "model-a"}]}),
            ["model-a"],
        )
        self.assertEqual(
            openai_compatible_model_ids([{"name": "model-b"}]),
            ["model-b"],
        )
        self.assertIsNone(openai_compatible_model_ids({"models": []}))

    def test_async_model_ids_prefers_the_native_async_iterator(self):
        page = FakeAsyncModelsPage(
            [SimpleNamespace(id="model-a"), SimpleNamespace(id="model-b")]
        )

        self.assertEqual(
            asyncio.run(amodel_ids(page)),
            ["model-a", "model-b"],
        )


class ProviderClientModelListingTests(unittest.TestCase):
    def test_openai_client_lists_live_models(self):
        client = object.__new__(OpenAIClient)
        client._client = SimpleNamespace(
            models=FakeModels(
                [
                    SimpleNamespace(id="gpt-a"),
                    SimpleNamespace(id="gpt-b"),
                ]
            )
        )

        self.assertEqual(
            client.list_available_models(),
            ["gpt-a", "gpt-b"],
        )

    def test_anthropic_client_lists_live_models(self):
        client = object.__new__(AnthropicClient)
        client._client = SimpleNamespace(
            models=FakeModels([SimpleNamespace(id="claude-a")])
        )

        self.assertEqual(client.list_available_models(), ["claude-a"])

    def test_google_client_lists_live_models(self):
        client = object.__new__(GoogleClient)
        client._client = SimpleNamespace(
            models=FakeModels([SimpleNamespace(name="models/gemini-a")])
        )

        self.assertEqual(
            client.list_available_models(),
            ["models/gemini-a"],
        )

    def test_base_async_listing_does_not_fall_back_to_sync_listing(self):
        client = FakeAsyncListingClient(sync_client=FakeListingClient())

        with self.assertRaises(LLMConfigurationError) as raised:
            asyncio.run(client.alist_available_models())

        self.assertEqual(raised.exception.status_code, 400)

    def test_unsupported_provider_is_a_configuration_error(self):
        client = UnsupportedListingClient()

        with self.assertRaises(LLMConfigurationError) as raised:
            client.list_available_models()

        self.assertEqual(raised.exception.status_code, 400)


class NativeAsyncModelListingTests(unittest.IsolatedAsyncioTestCase):
    async def test_anthropic_async_client_iterates_the_async_models_page(self):
        list_models = AsyncMock(
            return_value=FakeAsyncModelsPage(
                [
                    SimpleNamespace(id="claude-a"),
                    SimpleNamespace(id="claude-b"),
                ]
            )
        )
        provider_client = SimpleNamespace(
            models=SimpleNamespace(list=list_models),
            close=AsyncMock(),
        )
        sync_provider_client = SimpleNamespace(close=Mock())

        with (
            patch(
                "llmai.anthropic.client.Anthropic",
                return_value=sync_provider_client,
            ),
            patch(
                "llmai.anthropic.async_client.AsyncAnthropic",
                return_value=provider_client,
            ),
        ):
            client = AsyncAnthropicClient(
                config=AnthropicClientConfig(api_key="key"),
            )

        models = await client.alist_available_models()
        await client.aclose()

        self.assertEqual(models, ["claude-a", "claude-b"])
        list_models.assert_awaited_once_with(limit=100)
        provider_client.close.assert_awaited_once()

    async def test_openai_async_client_iterates_the_async_models_page(self):
        list_models = Mock(
            return_value=FakeAsyncModelsPage(
                [
                    SimpleNamespace(id="gpt-a"),
                    SimpleNamespace(id="gpt-b"),
                ]
            )
        )
        provider_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock()),
            ),
            responses=SimpleNamespace(create=AsyncMock()),
            models=SimpleNamespace(list=list_models),
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

        models = await client.alist_available_models()
        await client.aclose()

        self.assertEqual(models, ["gpt-a", "gpt-b"])
        list_models.assert_called_once_with()

    async def test_openai_async_client_uses_native_model_listing(self):
        list_models = AsyncMock(
            return_value=[
                SimpleNamespace(id="gpt-a"),
                SimpleNamespace(id="gpt-b"),
            ]
        )
        provider_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock()),
            ),
            responses=SimpleNamespace(create=AsyncMock()),
            models=SimpleNamespace(list=list_models),
            close=AsyncMock(),
        )
        sync_provider_client = SimpleNamespace(close=Mock())

        with (
            patch(
                "llmai.openai.client.OpenAI",
                return_value=sync_provider_client,
            ),
            patch(
                "llmai.openai.async_client.AsyncOpenAI",
                return_value=provider_client,
            ),
        ):
            client = AsyncOpenAIClient(
                config=OpenAIClientConfig(api_key="key"),
            )

        models = await client.alist_available_models()
        await client.aclose()

        self.assertEqual(models, ["gpt-a", "gpt-b"])
        list_models.assert_awaited_once()
        provider_client.close.assert_awaited_once()

    async def test_openai_async_client_accepts_an_awaitable_pager(self):
        class AwaitablePager:
            def __await__(self):
                async def resolve():
                    return [SimpleNamespace(id="gpt-from-pager")]

                return resolve().__await__()

        list_models = Mock(return_value=AwaitablePager())
        provider_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock()),
            ),
            responses=SimpleNamespace(create=AsyncMock()),
            models=SimpleNamespace(list=list_models),
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

        models = await client.alist_available_models()
        await client.aclose()

        self.assertEqual(models, ["gpt-from-pager"])


class TogetherModelListingTests(unittest.TestCase):
    def _client(self):
        client = object.__new__(TogetherAIClient)
        client._models_base_url = "https://api.together.ai/v1"
        client._models_api_key = "secret"
        return client

    def test_together_accepts_its_top_level_list_payload(self):
        fake_http = FakeHTTPClient(
            response(
                200,
                [
                    {"id": "openai/gpt-oss-20b"},
                    {"id": "meta-llama/llama"},
                ],
            )
        )
        with patch(
            "llmai.togetherai.client.httpx.Client",
            return_value=fake_http,
        ):
            models = self._client().list_available_models()

        self.assertEqual(
            models,
            ["openai/gpt-oss-20b", "meta-llama/llama"],
        )
        self.assertEqual(
            fake_http.request,
            (
                "https://api.together.ai/v1/models",
                {"Authorization": "Bearer secret"},
            ),
        )

    def test_together_invalid_key_is_normalized(self):
        fake_http = FakeHTTPClient(
            response(
                401,
                {"error": {"message": "The supplied API key is invalid."}},
            )
        )
        with (
            patch(
                "llmai.togetherai.client.httpx.Client",
                return_value=fake_http,
            ),
            self.assertRaises(LLMAuthenticationError) as raised,
        ):
            self._client().list_available_models()

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(
            raised.exception.message,
            "The supplied API key is invalid.",
        )

    def test_together_rejects_malformed_success_payloads(self):
        fake_http = FakeHTTPClient(response(200, {"models": []}))
        with (
            patch(
                "llmai.togetherai.client.httpx.Client",
                return_value=fake_http,
            ),
            self.assertRaises(LLMError) as raised,
        ):
            self._client().list_available_models()

        self.assertEqual(raised.exception.status_code, 502)


class HTTPXErrorNormalizationTests(unittest.TestCase):
    def _status_error(self, status_code, payload):
        provider_response = response(status_code, payload)
        return httpx.HTTPStatusError(
            "provider error",
            request=provider_response.request,
            response=provider_response,
        )

    def test_authentication_error_preserves_status_and_message(self):
        normalized = normalize_llm_error(
            self._status_error(
                401,
                {"error": {"message": "Incorrect API key."}},
            ),
            provider="openai",
        )

        self.assertIsInstance(normalized, LLMAuthenticationError)
        self.assertEqual(normalized.status_code, 401)
        self.assertEqual(normalized.message, "Incorrect API key.")
        self.assertEqual(normalized.provider, "openai")

    def test_rate_limit_error_preserves_status_and_message(self):
        normalized = normalize_llm_error(
            self._status_error(
                429,
                {"detail": "Too many model-list requests."},
            ),
            provider="openrouter",
        )

        self.assertIsInstance(normalized, LLMRateLimitError)
        self.assertEqual(normalized.status_code, 429)
        self.assertEqual(normalized.message, "Too many model-list requests.")

    def test_timeout_and_connection_errors_have_gateway_statuses(self):
        request = httpx.Request("GET", "https://provider.example/v1/models")

        timeout = normalize_llm_error(
            httpx.ReadTimeout("timed out", request=request),
            provider="openai",
        )
        connection = normalize_llm_error(
            httpx.ConnectError("unreachable", request=request),
            provider="openai",
        )

        self.assertIsInstance(timeout, LLMConnectionError)
        self.assertEqual(timeout.status_code, 504)
        self.assertIsInstance(connection, LLMConnectionError)
        self.assertEqual(connection.status_code, 503)

    def test_provider_internal_error_remains_distinguishable(self):
        normalized = normalize_llm_error(
            self._status_error(
                500,
                {"error": {"message": "Provider is temporarily unavailable."}},
            ),
            provider="openai",
        )

        self.assertIsInstance(normalized, LLMError)
        self.assertEqual(normalized.status_code, 500)
        self.assertEqual(
            normalized.message,
            "Provider is temporarily unavailable.",
        )

    def test_openai_sdk_error_uses_the_provider_body_message(self):
        provider_response = response(
            401,
            {"error": {"message": "Incorrect API key."}},
        )
        error = openai.AuthenticationError(
            "Error code: 401 - raw payload",
            response=provider_response,
            body={
                "message": "Incorrect API key.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            },
        )

        normalized = normalize_llm_error(error, provider="openai")

        self.assertIsInstance(normalized, LLMAuthenticationError)
        self.assertEqual(normalized.status_code, 401)
        self.assertEqual(normalized.message, "Incorrect API key.")


if __name__ == "__main__":
    unittest.main()
