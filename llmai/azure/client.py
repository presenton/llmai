from __future__ import annotations

import os
from collections.abc import Callable
from logging import Logger

from openai import AzureOpenAI

from llmai.openai.client import OpenAIClient
from llmai.shared.errors import configuration_error, raise_llm_error


class AzureOpenAIClient(OpenAIClient):
    PROVIDER_NAME = "azure"
    PROVIDER_LABEL = "Azure OpenAI"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: Callable[[], str] | None = None,
        endpoint: str | None = None,
        azure_endpoint: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        deployment: str | None = None,
        azure_deployment: str | None = None,
        logger: Logger | None = None,
    ):
        self._logger = logger

        resolved_base_url, resolved_endpoint = self._resolve_base_url_and_endpoint(
            base_url=base_url,
            endpoint=endpoint,
            azure_endpoint=azure_endpoint,
        )
        resolved_api_key, resolved_ad_token = self._resolve_auth(
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
        )
        resolved_api_version = self._resolve_api_version(api_version)
        resolved_deployment = self._resolve_deployment(
            deployment=deployment,
            azure_deployment=azure_deployment,
        )

        client_kwargs: dict[str, object] = {
            "api_version": resolved_api_version,
            "api_key": resolved_api_key,
            "azure_ad_token": resolved_ad_token,
            "azure_ad_token_provider": azure_ad_token_provider,
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
        azure_endpoint: str | None,
    ) -> tuple[str | None, str | None]:
        explicit_base_url = _strip_or_none(base_url)
        explicit_endpoint = _resolve_alias_pair(
            "endpoint",
            endpoint,
            "azure_endpoint",
            azure_endpoint,
            provider=self.PROVIDER_NAME,
        )
        if explicit_base_url is not None and explicit_endpoint is not None:
            raise configuration_error(
                "Azure OpenAI endpoint is ambiguous. Pass either base_url or endpoint/azure_endpoint, not both",
                provider=self.PROVIDER_NAME,
            )

        env_base_url = _first_env("AZURE_OPENAI_BASE_URL")
        env_endpoint = _first_env("AZURE_OPENAI_ENDPOINT")

        if explicit_base_url is not None:
            return explicit_base_url, None

        if explicit_endpoint is not None:
            return None, explicit_endpoint

        if env_base_url is not None and env_endpoint is not None:
            raise configuration_error(
                "Azure OpenAI endpoint envs are ambiguous. Set either AZURE_OPENAI_BASE_URL or AZURE_OPENAI_ENDPOINT, not both",
                provider=self.PROVIDER_NAME,
            )

        resolved_base_url = env_base_url
        resolved_endpoint = env_endpoint
        if resolved_base_url is None and resolved_endpoint is None:
            raise configuration_error(
                "Missing Azure OpenAI endpoint. Pass base_url or endpoint/azure_endpoint, or set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_ENDPOINT",
                provider=self.PROVIDER_NAME,
            )

        return resolved_base_url, resolved_endpoint

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

        env_api_key = _first_env("AZURE_OPENAI_API_KEY")
        env_ad_token = _first_env("AZURE_OPENAI_AD_TOKEN")
        if env_api_key is not None and env_ad_token is not None:
            raise configuration_error(
                "Azure OpenAI auth envs are ambiguous. Set either AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN, not both",
                provider=self.PROVIDER_NAME,
            )
        if env_api_key is None and env_ad_token is None:
            raise configuration_error(
                "Missing Azure OpenAI credentials. Pass api_key, azure_ad_token, or azure_ad_token_provider, or set AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN",
                provider=self.PROVIDER_NAME,
            )

        return env_api_key, env_ad_token

    def _resolve_api_version(self, api_version: str | None) -> str:
        resolved_api_version = _strip_or_none(api_version) or _first_env(
            "AZURE_OPENAI_API_VERSION",
            "OPENAI_API_VERSION",
        )
        if resolved_api_version is not None:
            return resolved_api_version

        raise configuration_error(
            "Missing Azure OpenAI API version. Pass api_version or set AZURE_OPENAI_API_VERSION or OPENAI_API_VERSION",
            provider=self.PROVIDER_NAME,
        )

    def _resolve_deployment(
        self,
        *,
        deployment: str | None,
        azure_deployment: str | None,
    ) -> str | None:
        return _resolve_alias_pair(
            "deployment",
            deployment,
            "azure_deployment",
            azure_deployment,
            provider=self.PROVIDER_NAME,
        ) or _first_env("AZURE_OPENAI_DEPLOYMENT")


def _resolve_alias_pair(
    left_name: str,
    left_value: str | None,
    right_name: str,
    right_value: str | None,
    *,
    provider: str,
) -> str | None:
    resolved_left = _strip_or_none(left_value)
    resolved_right = _strip_or_none(right_value)
    if resolved_left is not None and resolved_right is not None:
        if resolved_left == resolved_right:
            return resolved_left
        raise configuration_error(
            f"Azure OpenAI configuration is ambiguous. Pass either {left_name} or {right_name}, not both",
            provider=provider,
        )

    return resolved_left or resolved_right


def _first_env(*names: str) -> str | None:
    for name in names:
        value = _strip_or_none(os.getenv(name))
        if value is not None:
            return value
    return None


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None
