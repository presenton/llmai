import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from llmai import (
    AnthropicClient,
    AzureOpenAIClient,
    BedrockClient,
    CerebrasClient,
    ChatGPTClient,
    DeepSeekClient,
    FireworksClient,
    GoogleClient,
    LiteLLMClient,
    LMStudioClient,
    OpenAIClient,
    OpenRouterClient,
    TogetherAIClient,
    VertexAIClient,
)
from llmai.shared import (
    AsyncBaseClient,
    BaseClient,
    DEFAULT_MODEL_CONTEXT_WINDOW,
    ModelInfo,
    ModelTokenLimits,
    LLMAuthenticationError,
)
from llmai.shared.models import model_token_limits


class FakeModels:
    def __init__(self, *, listed=None, retrieved=None):
        self.listed = listed
        self.retrieved = retrieved
        self.list_calls = []
        self.retrieve_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        if isinstance(self.listed, Exception):
            raise self.listed
        return self.listed

    def retrieve(self, *args, **kwargs):
        self.retrieve_calls.append((args, kwargs))
        if isinstance(self.retrieved, Exception):
            raise self.retrieved
        return self.retrieved


class FakePage:
    def __init__(self, data, next_page=None):
        self.data = data
        self.next_page = next_page

    def has_next_page(self):
        return self.next_page is not None

    def get_next_page(self):
        return self.next_page


def bare_client(client_type, **attributes):
    client = object.__new__(client_type)
    BaseClient.__init__(client)
    for name, value in attributes.items():
        setattr(client, name, value)
    return client


def response(data):
    return SimpleNamespace(
        json=lambda: data,
        raise_for_status=lambda: None,
    )


class FakeMetadataClient(BaseClient):
    def generate(self, **kwargs):
        del kwargs
        raise NotImplementedError


class SharedModelMetadataTests(unittest.TestCase):
    def test_all_clients_share_model_discovery_signatures(self):
        for client_type in (
            OpenAIClient,
            AzureOpenAIClient,
            ChatGPTClient,
            DeepSeekClient,
            OpenRouterClient,
            CerebrasClient,
            FireworksClient,
            TogetherAIClient,
            LMStudioClient,
            AnthropicClient,
            GoogleClient,
            VertexAIClient,
            BedrockClient,
            LiteLLMClient,
        ):
            with self.subTest(client=client_type.__name__):
                self.assertEqual(
                    inspect.signature(
                        client_type.get_model_context_window
                    ),
                    inspect.signature(
                        BaseClient.get_model_context_window
                    ),
                )
                self.assertEqual(
                    inspect.signature(client_type.list_models),
                    inspect.signature(BaseClient.list_models),
                )

    def test_model_token_limits_use_a_labeled_default(self):
        limits = model_token_limits(context_window=0, max_input_tokens=None)

        self.assertEqual(
            limits.context_window,
            DEFAULT_MODEL_CONTEXT_WINDOW,
        )
        self.assertEqual(limits.source, "default")
        self.assertIsNone(limits.max_input_tokens)
        self.assertIsNone(limits.max_output_tokens)

    def test_model_token_limits_preserve_separate_input_and_output(self):
        limits = model_token_limits(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        )

        self.assertEqual(limits.context_window, 200_000)
        self.assertEqual(limits.max_input_tokens, 200_000)
        self.assertEqual(limits.max_output_tokens, 8_192)
        self.assertEqual(limits.source, "provider")

    def test_models_are_json_serializable(self):
        info = ModelInfo(
            id="model",
            provider="provider",
            token_limits=ModelTokenLimits(),
        )

        self.assertEqual(
            info.model_dump(mode="json")["token_limits"]["source"],
            "default",
        )

    def test_openai_compatible_listing_follows_pages(self):
        pages = FakePage(
            [{"id": "first", "name": "First"}],
            FakePage([{"id": "second", "context_length": 32_000}]),
        )
        client = bare_client(
            OpenAIClient,
            _client=SimpleNamespace(
                models=FakeModels(listed=pages),
            ),
        )

        models = client.list_models()

        self.assertEqual([model.id for model in models], ["first", "second"])
        self.assertEqual(models[0].token_limits.source, "default")
        self.assertEqual(models[1].token_limits.context_window, 32_000)

    def test_openai_compatible_provider_names_and_limits(self):
        cases = [
            (OpenAIClient, {}, "openai", 4_000),
            (AzureOpenAIClient, {}, "azure", 4_000),
            (DeepSeekClient, {}, "deepseek", 4_000),
            (OpenRouterClient, {}, "openrouter", 131_072),
            (TogetherAIClient, {}, "togetherai", 131_072),
            (
                CerebrasClient,
                {"_uses_default_base_url": False},
                "cerebras",
                131_072,
            ),
            (
                FireworksClient,
                {"_uses_default_base_url": False},
                "fireworks",
                131_072,
            ),
        ]
        for client_type, attributes, provider, expected_context in cases:
            with self.subTest(provider=provider):
                payload = {"id": "model"}
                if expected_context != 4_000:
                    payload["context_length"] = expected_context
                client = bare_client(
                    client_type,
                    _client=SimpleNamespace(
                        models=FakeModels(
                            listed={"data": [payload]},
                            retrieved=payload,
                        )
                    ),
                    **attributes,
                )

                listed = client.list_models()
                limits = client.get_model_context_window(model="model")

                self.assertEqual(listed[0].provider, provider)
                self.assertEqual(limits.context_window, expected_context)

    def test_openai_compatible_lookup_failure_returns_default(self):
        client = bare_client(
            OpenAIClient,
            _client=SimpleNamespace(
                models=FakeModels(retrieved=RuntimeError("offline"))
            ),
        )

        limits = client.get_model_context_window(model="model")

        self.assertEqual(limits.context_window, 4_000)
        self.assertEqual(limits.source, "default")

    def test_anthropic_lists_and_retrieves_token_limits(self):
        item = SimpleNamespace(
            id="claude",
            display_name="Claude",
            max_input_tokens=1_000_000,
            max_tokens=64_000,
        )
        client = bare_client(
            AnthropicClient,
            _client=SimpleNamespace(
                models=FakeModels(
                    listed=FakePage([item]),
                    retrieved=item,
                )
            ),
        )

        listed = client.list_models()
        limits = client.get_model_context_window(model="claude")

        self.assertEqual(listed[0].display_name, "Claude")
        self.assertEqual(limits.context_window, 1_000_000)
        self.assertEqual(limits.max_output_tokens, 64_000)

    def test_google_and_vertex_list_and_retrieve_token_limits(self):
        item = SimpleNamespace(
            name="models/gemini-test",
            base_model_id="gemini-test",
            display_name="Gemini Test",
            input_token_limit=1_048_576,
            output_token_limit=65_536,
        )
        for client_type, provider in (
            (GoogleClient, "google"),
            (VertexAIClient, "vertex"),
        ):
            with self.subTest(provider=provider):
                models = FakeModels(listed=[item], retrieved=item)
                models.get = lambda **kwargs: item
                client = bare_client(
                    client_type,
                    _client=SimpleNamespace(models=models),
                )

                listed = client.list_models()
                limits = client.get_model_context_window(
                    model="gemini-test"
                )

                self.assertEqual(listed[0].id, "gemini-test")
                self.assertEqual(listed[0].provider, provider)
                self.assertEqual(limits.context_window, 1_048_576)
                self.assertEqual(limits.max_output_tokens, 65_536)

    @patch("llmai.chatgpt.client.httpx.get")
    def test_chatgpt_uses_codex_model_catalog(self, get):
        get.return_value = response(
            {
                "models": [
                    {
                        "slug": "gpt-codex",
                        "display_name": "GPT Codex",
                        "context_window": 272_000,
                    }
                ]
            }
        )
        client = bare_client(
            ChatGPTClient,
            _base_url="https://chatgpt.test/backend-api/codex",
            _model_headers={"Authorization": "Bearer token"},
        )

        listed = client.list_models()
        limits = client.get_model_context_window(model="gpt-codex")

        self.assertEqual(listed[0].id, "gpt-codex")
        self.assertEqual(limits.context_window, 272_000)

    @patch("llmai.chatgpt.client.httpx.get")
    def test_http_metadata_listing_errors_are_normalized(self, get):
        request = httpx.Request("GET", "https://chatgpt.test/models")
        failed_response = httpx.Response(
            401,
            request=request,
            json={"error": {"message": "invalid token"}},
        )
        get.return_value = SimpleNamespace(
            json=failed_response.json,
            raise_for_status=lambda: failed_response.raise_for_status(),
        )
        client = bare_client(
            ChatGPTClient,
            _base_url="https://chatgpt.test",
            _model_headers={"Authorization": "Bearer token"},
        )

        with self.assertRaises(LLMAuthenticationError) as context:
            client.list_models()

        self.assertEqual(context.exception.status_code, 401)
        self.assertEqual(context.exception.provider, "chatgpt")
        self.assertEqual(context.exception.message, "invalid token")

    @patch("llmai.cerebras.client.httpx.get")
    def test_cerebras_enriches_accessible_models_from_public_catalog(self, get):
        get.return_value = response(
            {
                "data": [
                    {
                        "id": "gpt-oss",
                        "limits": {
                            "max_context_length": 131_072,
                            "max_completion_tokens": 40_960,
                        },
                    }
                ]
            }
        )
        client = bare_client(
            CerebrasClient,
            _uses_default_base_url=True,
            _client=SimpleNamespace(
                models=FakeModels(
                    listed={"data": [{"id": "gpt-oss"}]},
                )
            ),
        )

        model = client.list_models()[0]

        self.assertEqual(model.token_limits.context_window, 131_072)
        self.assertEqual(model.token_limits.max_output_tokens, 40_960)

    @patch("llmai.fireworks.client.httpx.get")
    def test_fireworks_enriches_full_model_resources(self, get):
        get.return_value = response({"contextLength": 131_072})
        model_id = "accounts/fireworks/models/test"
        client = bare_client(
            FireworksClient,
            _uses_default_base_url=True,
            _fireworks_api_key="key",
            _client=SimpleNamespace(
                models=FakeModels(
                    listed={"data": [{"id": model_id}]},
                )
            ),
        )

        model = client.list_models()[0]

        self.assertEqual(model.token_limits.context_window, 131_072)
        self.assertIn(model_id, get.call_args.args[0])

    @patch("llmai.lmstudio.client.httpx.get")
    def test_lmstudio_uses_enhanced_model_endpoint(self, get):
        get.return_value = response(
            {
                "data": [
                    {
                        "id": "local-model",
                        "max_context_length": 32_768,
                    }
                ]
            }
        )
        client = bare_client(
            LMStudioClient,
            _rest_base_url="http://localhost:1234",
            _lmstudio_api_key="key",
        )

        model = client.list_models()[0]

        self.assertEqual(model.token_limits.context_window, 32_768)

    @patch("llmai.litellm.client.httpx.get")
    def test_litellm_uses_model_info(self, get):
        get.return_value = response(
            {
                "data": [
                    {
                        "model_name": "deployment",
                        "model_info": {
                            "max_tokens": 128_000,
                            "max_input_tokens": 120_000,
                            "max_output_tokens": 8_000,
                        },
                    }
                ]
            }
        )
        client = bare_client(
            LiteLLMClient,
            _litellm_base_url="https://litellm.test",
            _litellm_api_key="key",
        )

        listed = client.list_models()
        limits = client.get_model_context_window(model="deployment")

        self.assertEqual(listed[0].id, "deployment")
        self.assertEqual(limits.context_window, 128_000)
        self.assertEqual(limits.max_input_tokens, 120_000)
        self.assertEqual(limits.max_output_tokens, 8_000)

    def test_bedrock_lists_foundation_models_with_default_limits(self):
        client = bare_client(
            BedrockClient,
            _model_client=SimpleNamespace(
                list_foundation_models=lambda: {
                    "modelSummaries": [
                        {
                            "modelId": "anthropic.claude",
                            "modelName": "Claude",
                        }
                    ]
                }
            ),
        )

        model = client.list_models()[0]

        self.assertEqual(model.id, "anthropic.claude")
        self.assertEqual(model.display_name, "Claude")
        self.assertEqual(model.token_limits.context_window, 4_000)


class AsyncModelMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_facade_exposes_model_methods(self):
        sync_client = bare_client(FakeMetadataClient)
        sync_client.get_model_context_window = lambda **kwargs: ModelTokenLimits(
            context_window=8_000,
            source="provider",
        )
        sync_client.list_models = lambda: [
            ModelInfo(
                id="model",
                provider="test",
                token_limits=ModelTokenLimits(),
            )
        ]
        client = AsyncBaseClient(sync_client=sync_client)

        limits = await client.aget_model_context_window(model="model")
        models = await client.alist_models()

        self.assertEqual(limits.context_window, 8_000)
        self.assertEqual(models[0].id, "model")
