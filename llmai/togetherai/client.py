from __future__ import annotations

from logging import Logger

import httpx

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import OpenAIClientConfig, TogetherAIClientConfig
from llmai.shared.errors import LLMError, raise_llm_error
from llmai.shared.messages import (
    AssistantMessage,
    AssistantReasoningItem,
    flatten_thinking_content,
)
from llmai.shared.model_listing import openai_compatible_model_ids


class TogetherAIClient(OpenAIClient):
    PROVIDER_NAME = "togetherai"
    PROVIDER_LABEL = "Together AI"
    DEFAULT_BASE_URL = "https://api.together.ai/v1"

    def __init__(
        self,
        *,
        config: TogetherAIClientConfig,
        logger: Logger | None = None,
    ):
        self._models_base_url = config.base_url or self.DEFAULT_BASE_URL
        self._models_api_key = config.api_key
        super().__init__(
            config=OpenAIClientConfig(
                api_key=config.api_key,
                base_url=self._models_base_url,
                api_type=OpenAIApiType.COMPLETIONS,
                generation=config.generation,
            ),
            logger=logger,
        )

    def list_available_models(self) -> list[str]:
        """Handle Together's nonstandard top-level-list response."""

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self._models_base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {self._models_api_key}"},
                )
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise LLMError(
                        502,
                        "Together AI returned an invalid model-list response.",
                        provider=self.PROVIDER_NAME,
                        cause=exc,
                    ) from exc

            models = openai_compatible_model_ids(payload)
            if models is None:
                raise LLMError(
                    502,
                    "Together AI returned an invalid model-list response.",
                    provider=self.PROVIDER_NAME,
                )
            return models
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

    def _chat_completion_message_to_thinking_items(
        self,
        message: object,
    ) -> list[AssistantReasoningItem]:
        reasoning = getattr(message, "reasoning", None)
        if not reasoning:
            reasoning = getattr(message, "reasoning_content", None)
        if not reasoning:
            return []

        return [AssistantReasoningItem(summary=[reasoning])]

    def _chat_completion_delta_to_thinking_text(self, delta: object) -> str | None:
        reasoning = getattr(delta, "reasoning", None)
        if not reasoning:
            reasoning = getattr(delta, "reasoning_content", None)
        return reasoning or None

    def _get_openai_chat_max_tokens_kwargs(
        self,
        max_tokens: int | None,
        *,
        model: str | None = None,
    ) -> dict[str, int | None]:
        if model is not None:
            self._compatibility_cache[(model, "output_token_field")] = "max_tokens"
        return {"max_tokens": max_tokens}

    def _assistant_message_to_chat_completion_assistant_message_param(
        self,
        message: AssistantMessage,
    ):
        together_message = (
            super()._assistant_message_to_chat_completion_assistant_message_param(
                message
            )
        )
        reasoning_content = "\n".join(flatten_thinking_content(message.thinking))
        if reasoning_content:
            together_message["reasoning_content"] = reasoning_content

        return together_message
