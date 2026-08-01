import unittest
from types import SimpleNamespace

from llmai.capabilities import (
    get_model_capabilities,
    get_reasoning_levels,
    supports_thinking,
    supports_tool_call,
)
from llmai.deepseek import DeepSeekClient
from llmai.google import GoogleClient
from llmai.openai import OpenAIClient
from llmai.shared import (
    AssistantMessage,
    AssistantReasoningItem,
    AssistantToolCall,
    GenerationDefaults,
    OpenAIClientConfig,
    ReasoningConfig,
    ReasoningEffortValue,
    ReasoningHistoryMode,
    ResponseContent,
    ResponseUsage,
    ToolChoiceMode,
    ValidationMode,
    prepare_generation,
)


class GenerationPolicyTests(unittest.TestCase):
    def test_deepseek_thinking_adapts_forced_tool_choice(self):
        from llmai.shared.base import BaseClient

        client = DeepSeekClient.__new__(DeepSeekClient)
        BaseClient.__init__(client)
        adapted = client._adapt_tool_choice_for_thinking(
            "deepseek-v4-flash",
            {"mode": ToolChoiceMode.REQUIRED, "tools": ["lookup"]},
        )

        self.assertEqual(adapted["mode"], ToolChoiceMode.AUTO)
        self.assertEqual(adapted["tools"], ["lookup"])

    def test_google_budget_model_compiles_effort_to_budget(self):
        client = GoogleClient.__new__(GoogleClient)
        from llmai.shared.base import BaseClient

        BaseClient.__init__(client)
        prepared = client.prepare_generation(
            model="gemini-2.5-flash",
            reasoning=ReasoningConfig(
                enabled=True,
                effort=ReasoningEffortValue.LOW,
            ),
        )

        self.assertIsNone(prepared.reasoning.effort)
        self.assertEqual(prepared.reasoning.budget_tokens, 1_024)

    def test_profiles_use_metadata_and_safe_fallbacks(self):
        balanced = prepare_generation(model="gpt-5", provider="openai")
        fast = prepare_generation(model="gpt-5", provider="openai", profile="fast")
        unknown = prepare_generation(model="not-listed", provider="custom")

        self.assertEqual(balanced.max_output_tokens, 32_768)
        self.assertEqual(fast.max_output_tokens, 4_096)
        self.assertEqual(unknown.max_output_tokens, 8_192)

    def test_explicit_output_aliases_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "only one"):
            prepare_generation(
                model="gpt-5",
                provider="openai",
                max_tokens=100,
                max_output_tokens=200,
            )

    def test_adaptive_effort_uses_nearest_advertised_level(self):
        prepared = prepare_generation(
            model="gpt-5",
            provider="openai",
            reasoning=ReasoningConfig(
                enabled=True,
                effort=ReasoningEffortValue.MAX,
            ),
        )

        self.assertEqual(prepared.reasoning.effort, ReasoningEffortValue.HIGH)
        self.assertEqual(prepared.warnings[0].code, "reasoning_effort_adapted")

    def test_strict_metadata_conflict_raises(self):
        with self.assertRaisesRegex(ValueError, "not advertised"):
            prepare_generation(
                model="gpt-5",
                provider="openai",
                defaults=GenerationDefaults(validation=ValidationMode.STRICT),
                reasoning=ReasoningConfig(
                    enabled=True,
                    effort=ReasoningEffortValue.MAX,
                ),
            )

    def test_reasoning_config_rejects_contradictions(self):
        with self.assertRaises(ValueError):
            ReasoningConfig(
                enabled=False,
                effort=ReasoningEffortValue.HIGH,
            )

    def test_disabled_history_removes_trace_and_signatures_without_mutation(self):
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
        original = AssistantMessage(
            thinking=[AssistantReasoningItem(summary=["trace"])],
            tool_calls=[
                AssistantToolCall(
                    id="call_1",
                    name="lookup",
                    thought_signature=b"opaque",
                )
            ],
        )

        prepared = client._prepare_reasoning_history(
            [original], ReasoningHistoryMode.DISABLED
        )[0]

        self.assertIsNone(prepared.thinking)
        self.assertIsNone(prepared.tool_calls[0].thought_signature)
        self.assertIsNotNone(original.thinking)
        self.assertEqual(original.tool_calls[0].thought_signature, b"opaque")


class CapabilityTests(unittest.TestCase):
    def test_google_sdk_model_prefix_is_normalized(self):
        capabilities = get_model_capabilities(
            "models/gemini-2.5-flash", provider="google"
        )

        self.assertTrue(capabilities.reasoning.supported)
        self.assertTrue(capabilities.tool_call.support.supported)

    def test_known_capabilities_and_levels_are_typed(self):
        capabilities = get_model_capabilities("gpt-5", provider="openai")

        self.assertTrue(capabilities.reasoning.supported)
        self.assertTrue(capabilities.tool_call.support.supported)
        self.assertIn("high", get_reasoning_levels("gpt-5", provider="openai"))
        self.assertTrue(supports_thinking("gpt-5", provider="openai"))
        self.assertTrue(supports_tool_call("gpt-5", provider="openai"))

    def test_unknown_model_does_not_guess(self):
        capabilities = get_model_capabilities("not-listed", provider="custom")

        self.assertIsNone(capabilities.reasoning.supported)
        self.assertIsNone(capabilities.tool_call.support.supported)

    def test_budget_models_expose_llmai_inferred_effort_levels(self):
        capabilities = get_model_capabilities("claude-sonnet-4-5", provider="anthropic")

        self.assertIn("high", capabilities.reasoning_levels.value)
        self.assertEqual(capabilities.reasoning_levels.source, "inferred")


class ReasoningResponseTests(unittest.TestCase):
    def test_native_reasoning_usage_wins_over_visible_estimate(self):
        response = ResponseContent(
            thinking=[AssistantReasoningItem(summary=["a visible trace"])],
            usage=ResponseUsage(
                details={"completion_tokens_details": {"reasoning_tokens": 17}}
            ),
        )

        self.assertEqual(response.usage.thinking_tokens, 17)
        self.assertEqual(response.usage.reasoning.billed_tokens, 17)
        self.assertTrue(response.usage.reasoning.visible_estimated)

    def test_visible_trace_is_estimated_and_opaque_state_has_safe_dump(self):
        response = ResponseContent(
            thinking=[
                AssistantReasoningItem(
                    summary=["abcd"],
                    signature="opaque",
                    provider="anthropic",
                )
            ]
        )

        self.assertEqual(response.usage.thinking_tokens, 1)
        self.assertEqual(response.reasoning_trace.text, "abcd")
        self.assertNotIn("reasoning_state", response.safe_model_dump())
        self.assertIn("reasoning_state", response.lossless_model_dump())


class OpenAICompatibilityTests(unittest.TestCase):
    def test_client_generation_defaults_supply_omitted_output_limit(self):
        class Completions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="done",
                                tool_calls=[],
                            )
                        )
                    ],
                    usage=None,
                )

        completions = Completions()
        client = OpenAIClient(
            config=OpenAIClientConfig(
                api_key="test",
                generation=GenerationDefaults(max_output_tokens=1_234),
            )
        )
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        client.generate(model="custom-model", messages=[])

        self.assertEqual(completions.calls[0]["max_completion_tokens"], 1_234)

    def test_unsupported_token_field_retries_once_and_is_cached(self):
        class UnsupportedParameter(Exception):
            status_code = 400

        class Completions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise UnsupportedParameter(
                        "unsupported parameter max_completion_tokens"
                    )
                return SimpleNamespace(ok=True)

        completions = Completions()
        client = OpenAIClient(config=OpenAIClientConfig(api_key="test"))
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        client._create_chat_completion(
            model="custom-model", max_tokens=123, messages=[]
        )
        client._create_chat_completion(
            model="custom-model", max_tokens=456, messages=[]
        )

        self.assertEqual(completions.calls[0]["max_completion_tokens"], 123)
        self.assertEqual(completions.calls[1]["max_tokens"], 123)
        self.assertEqual(completions.calls[2]["max_tokens"], 456)


if __name__ == "__main__":
    unittest.main()
