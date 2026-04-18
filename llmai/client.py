from __future__ import annotations

import os
from logging import Logger

from llmai.anthropic.client import AnthropicClient
from llmai.bedrock.client import BedrockClient
from llmai.deepseek.client import DeepSeekClient
from llmai.google.client import GoogleClient
from llmai.openai.client import OpenAIClient
from llmai.shared.base import BaseClient
from llmai.shared.errors import configuration_error
from llmai.shared.providers import LLMProvider

__all__ = ["LLMProvider", "get_client"]


def get_client(
    provider: LLMProvider | str,
    *,
    logger: Logger | None = None,
) -> BaseClient:
    resolved_provider = _normalize_provider(provider)

    if resolved_provider == LLMProvider.OPENAI:
        api_key = _required_env(
            resolved_provider,
            "OPENAI_API_KEY",
        )
        return OpenAIClient(
            api_key=api_key,
            base_url=_env("OPENAI_BASE_URL"),
            logger=logger,
        )

    if resolved_provider == LLMProvider.DEEPSEEK:
        api_key = _required_env(
            resolved_provider,
            "DEEPSEEK_API_KEY",
        )
        return DeepSeekClient(
            api_key=api_key,
            base_url=_env("DEEPSEEK_BASE_URL"),
            logger=logger,
        )

    if resolved_provider == LLMProvider.GOOGLE:
        api_key = _required_env(
            resolved_provider,
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        )
        return GoogleClient(
            api_key=api_key,
            logger=logger,
        )

    if resolved_provider == LLMProvider.ANTHROPIC:
        api_key = _required_env(
            resolved_provider,
            "ANTHROPIC_API_KEY",
        )
        return AnthropicClient(
            api_key=api_key,
            logger=logger,
        )

    if resolved_provider == LLMProvider.BEDROCK:
        return _get_bedrock_client(logger=logger)

    supported = ", ".join(each.value for each in LLMProvider)
    raise configuration_error(
        f"Unsupported LLM provider: {resolved_provider!r}. Expected one of: {supported}",
        provider=None,
    )


def _get_bedrock_client(*, logger: Logger | None = None) -> BedrockClient:
    provider = LLMProvider.BEDROCK
    region = _first_env("BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")
    api_key = _first_env("BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK")
    aws_access_key_id = _env("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = _env("AWS_SECRET_ACCESS_KEY")
    aws_session_token = _env("AWS_SESSION_TOKEN")
    profile_name = _first_env("BEDROCK_PROFILE_NAME", "AWS_PROFILE")

    if not region:
        raise configuration_error(
            "Missing Bedrock region. Set one of: BEDROCK_REGION, AWS_REGION, AWS_DEFAULT_REGION",
            provider=provider.value,
        )

    has_api_key = api_key is not None
    has_access_key_pair = (
        aws_access_key_id is not None or aws_secret_access_key is not None
    )
    has_session_token = aws_session_token is not None
    has_profile = profile_name is not None
    has_aws_auth = has_access_key_pair or has_session_token or has_profile

    if has_api_key and has_aws_auth:
        raise configuration_error(
            "Bedrock auth is ambiguous. Configure either BEDROCK_API_KEY/AWS_BEARER_TOKEN_BEDROCK or AWS credentials/profile envs, not both",
            provider=provider.value,
        )

    if has_api_key:
        return BedrockClient(
            api_key=api_key,
            region=region,
            logger=logger,
        )

    if has_profile and has_access_key_pair:
        raise configuration_error(
            "Bedrock auth is ambiguous. Configure either AWS_PROFILE/BEDROCK_PROFILE_NAME or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY",
            provider=provider.value,
        )

    if has_profile:
        return BedrockClient(
            region=region,
            profile_name=profile_name,
            logger=logger,
        )

    if (aws_access_key_id is None) != (aws_secret_access_key is None):
        raise configuration_error(
            "Bedrock credential envs require both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
            provider=provider.value,
        )

    if aws_access_key_id and aws_secret_access_key:
        return BedrockClient(
            region=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            logger=logger,
        )

    raise configuration_error(
        "Missing Bedrock auth envs. Set BEDROCK_API_KEY, or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or AWS_PROFILE",
        provider=provider.value,
    )


def _normalize_provider(provider: LLMProvider | str) -> LLMProvider:
    if isinstance(provider, LLMProvider):
        return provider

    if isinstance(provider, str):
        normalized = provider.strip().lower()
        try:
            return LLMProvider(normalized)
        except ValueError:
            pass

    supported = ", ".join(each.value for each in LLMProvider)
    raise configuration_error(
        f"Unsupported LLM provider: {provider!r}. Expected one of: {supported}",
        provider=None,
    )


def _required_env(provider: LLMProvider, *names: str) -> str:
    value = _first_env(*names)
    if value is not None:
        return value

    missing = ", ".join(names)
    raise configuration_error(
        f"Missing required environment variable for {provider.value}: {missing}",
        provider=provider.value,
    )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = _env(name)
        if value is not None:
            return value
    return None


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None
