from __future__ import annotations

from collections.abc import Callable
from logging import Logger

from openai import AzureOpenAI

from llmai.openai.client import OpenAIClient
from llmai.shared.configs import AzureOpenAIClientConfig, OpenAIApiType
from llmai.shared.errors import configuration_error, raise_llm_error


class AzureOpenAIClient(OpenAIClient):
    PROVIDER_NAME = "azure"
    PROVIDER_LABEL = "Azure OpenAI"

    def __init__(
        self,
        *,
        config: AzureOpenAIClientConfig,
        logger: Logger | None = None,
    ):
        self._logger = logger
        self._api_type = OpenAIApiType.COMPLETIONS

        resolved_base_url, resolved_endpoint = self._resolve_base_url_and_endpoint(
            base_url=config.base_url,
            endpoint=config.endpoint,
        )
        resolved_api_key, resolved_ad_token = self._resolve_auth(
            api_key=config.api_key,
            azure_ad_token=config.azure_ad_token,
            azure_ad_token_provider=config.azure_ad_token_provider,
        )
        resolved_api_version = self._resolve_api_version(config.api_version)
        resolved_deployment = self._resolve_deployment(config.deployment)

        client_kwargs: dict[str, object] = {
            "api_version": resolved_api_version,
            "api_key": resolved_api_key,
            "azure_ad_token": resolved_ad_token,
            "azure_ad_token_provider": config.azure_ad_token_provider,
            "base_url": resolved_base_url,
        }
        if resolved_endpoint is not None:
            client_kwargs["azure_endpoint"] = resolved_endpoint
        if resolved_deployment is not None and resolved_base_url is None:
            client_kwargs["azure_deployment"] = resolved_deployment

        try:
            self._client = AzureOpenAI(**client_kwargs)
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

        if self._logger:
            self._logger.info("%s client created", self.PROVIDER_LABEL)
            self._logger.info(
                "Base URL: %s",
                resolved_base_url or resolved_endpoint,
            )
            self._logger.info("API Version: %s", resolved_api_version)

    def _resolve_base_url_and_endpoint(
        self,
        *,
        base_url: str | None,
        endpoint: str | None,
    ) -> tuple[str | None, str | None]:
        explicit_base_url = _strip_or_none(base_url)
        explicit_endpoint = _strip_or_none(endpoint)
        if explicit_base_url is not None and explicit_endpoint is not None:
            raise configuration_error(
                "Azure OpenAI endpoint is ambiguous. Pass either base_url or endpoint, not both",
                provider=self.PROVIDER_NAME,
            )

        if explicit_base_url is not None:
            return explicit_base_url, None

        if explicit_endpoint is not None:
            return None, explicit_endpoint

        raise configuration_error(
            "Missing Azure OpenAI endpoint. Pass base_url or endpoint",
            provider=self.PROVIDER_NAME,
        )

    def _resolve_auth(
        self,
        *,
        api_key: str | None,
        azure_ad_token: str | None,
        azure_ad_token_provider: Callable[[], str] | None,
    ) -> tuple[str | None, str | None]:
        explicit_api_key = _strip_or_none(api_key)
        explicit_ad_token = _strip_or_none(azure_ad_token)
        if explicit_api_key is not None and explicit_ad_token is not None:
            raise configuration_error(
                "Azure OpenAI auth is ambiguous. Pass either api_key or azure_ad_token, not both",
                provider=self.PROVIDER_NAME,
            )
        if azure_ad_token_provider is not None and (
            explicit_api_key is not None or explicit_ad_token is not None
        ):
            raise configuration_error(
                "Azure OpenAI auth is ambiguous. azure_ad_token_provider cannot be combined with api_key or azure_ad_token",
                provider=self.PROVIDER_NAME,
            )

        if explicit_api_key is not None or explicit_ad_token is not None:
            return explicit_api_key, explicit_ad_token
        if azure_ad_token_provider is not None:
            return None, None
        raise configuration_error(
            "Missing Azure OpenAI credentials. Pass api_key, azure_ad_token, or azure_ad_token_provider",
            provider=self.PROVIDER_NAME,
        )

    def _resolve_api_version(self, api_version: str | None) -> str:
        resolved_api_version = _strip_or_none(api_version)
        if resolved_api_version is not None:
            return resolved_api_version

        raise configuration_error(
            "Missing Azure OpenAI API version. Pass api_version",
            provider=self.PROVIDER_NAME,
        )

    def _resolve_deployment(
        self,
        deployment: str | None,
    ) -> str | None:
        return _strip_or_none(deployment)


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None
