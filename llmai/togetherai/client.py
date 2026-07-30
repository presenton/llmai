from __future__ import annotations

from logging import Logger

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import OpenAIClientConfig, TogetherAIClientConfig
from llmai.shared.logs import LogLevel
from llmai.shared.messages import (
    AssistantMessage,
    AssistantReasoningItem,
    flatten_thinking_content,
)
from llmai.shared.models import ModelTokenLimits


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
        super().__init__(
            config=OpenAIClientConfig(
                api_key=config.api_key,
                base_url=config.base_url or self.DEFAULT_BASE_URL,
                api_type=OpenAIApiType.COMPLETIONS,
            ),
            logger=logger,
        )

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

    def get_model_context_window(self, *, model: str) -> ModelTokenLimits:
        try:
            for model_info in self.list_models():
                if model_info.id == model:
                    return model_info.token_limits
        except Exception as exc:
            self.log(
                LogLevel.WARNING,
                (
                    f"Together AI model metadata lookup failed for {model!r}; "
                    f"using the 4000-token default: {exc}"
                ),
            )
        return ModelTokenLimits()

    def _get_openai_chat_max_tokens_kwargs(
        self,
        max_tokens: int | None,
    ) -> dict[str, int | None]:
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
