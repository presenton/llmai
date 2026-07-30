from __future__ import annotations

from logging import Logger

from llmai.openai.client import OpenAIApiType, OpenAIClient
from llmai.shared.configs import OpenAIClientConfig, OpenRouterClientConfig
from llmai.shared.logs import LogLevel
from llmai.shared.models import ModelTokenLimits


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
            ),
            logger=logger,
        )

    def _openai_schema(
        self,
        schema: dict,
        *,
        strict: bool,
    ) -> dict:
        del strict
        return schema

    def get_model_context_window(self, *, model: str) -> ModelTokenLimits:
        try:
            for model_info in self.list_models():
                if model_info.id == model:
                    return model_info.token_limits
        except Exception as exc:
            self.log(
                LogLevel.WARNING,
                (
                    f"OpenRouter model metadata lookup failed for {model!r}; "
                    f"using the 4000-token default: {exc}"
                ),
            )
        return ModelTokenLimits()
