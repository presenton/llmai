from __future__ import annotations

from logging import Logger

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import OpenAIClientConfig, OpenRouterClientConfig
from llmai.shared.messages import AssistantMessage


class OpenRouterClient(OpenAIClient):
    PROVIDER_NAME = "openrouter"
    PROVIDER_LABEL = "OpenRouter"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        *,
        config: OpenRouterClientConfig,
        logger: Logger | None = None,
    ):
        super().__init__(
            config=OpenAIClientConfig(
                api_key=config.api_key,
                base_url=config.base_url or self.DEFAULT_BASE_URL,
                api_type=OpenAIApiType.COMPLETIONS,
                generation=config.generation,
            ),
            logger=logger,
        )

    def _openai_schema(
        self,
        schema: dict,
        *,
        strict: bool,
    ) -> dict:
        if not strict:
            return schema
        return super()._openai_schema(schema, strict=True)

    def _assistant_message_to_chat_completion_assistant_message_param(
        self,
        message: AssistantMessage,
    ):
        result = super()._assistant_message_to_chat_completion_assistant_message_param(
            message
        )
        details = [item.raw for item in message.thinking or [] if item.raw]
        if details:
            result["reasoning_details"] = details
        return result

    def _discover_model_capabilities(self, model: str) -> dict[str, object] | None:
        response = self._client.models.list()
        for item in getattr(response, "data", response) or []:
            raw = self._dump_model(item)
            item_id = str(raw.get("id", ""))
            if item_id not in {model, model.removeprefix("models/")}:
                continue
            parameters = set(raw.get("supported_parameters") or [])
            top_provider = raw.get("top_provider") or {}
            maximum = raw.get("max_completion_tokens")
            if not isinstance(maximum, int) and isinstance(top_provider, dict):
                maximum = top_provider.get("max_completion_tokens")
            result: dict[str, object] = {}
            if isinstance(maximum, int):
                result["max_output_tokens"] = maximum
            if parameters:
                result["tool_call"] = bool(
                    parameters.intersection({"tools", "tool_choice"})
                )
                result["reasoning"] = bool(
                    parameters.intersection(
                        {"reasoning", "reasoning_effort", "include_reasoning"}
                    )
                )
            return result or None
        return None
