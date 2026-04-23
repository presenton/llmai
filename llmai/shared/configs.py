from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any, Literal

from google.auth.credentials import Credentials
from pydantic import BaseModel, BeforeValidator, ConfigDict, StringConstraints, model_validator

from llmai.shared.errors import configuration_error


def _strip_or_none(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    return value


RequiredStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OptionalStr = Annotated[str | None, BeforeValidator(_strip_or_none)]


class BaseClientConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class APIKeyClientConfig(BaseClientConfig):
    api_key: RequiredStr
    base_url: OptionalStr = None


class OpenAIApiType(str, Enum):
    COMPLETIONS = "completions"
    RESPONSES = "responses"


class OpenAIClientConfig(APIKeyClientConfig):
    provider: Literal["openai"] = "openai"
    api_type: OpenAIApiType = OpenAIApiType.COMPLETIONS


class AnthropicClientConfig(APIKeyClientConfig):
    provider: Literal["anthropic"] = "anthropic"


class ChatGPTClientConfig(BaseClientConfig):
    provider: Literal["chatgpt"] = "chatgpt"
    access_token: RequiredStr
    base_url: OptionalStr = None
    account_id: OptionalStr = None


class DeepSeekClientConfig(APIKeyClientConfig):
    provider: Literal["deepseek"] = "deepseek"


class GoogleClientConfig(APIKeyClientConfig):
    provider: Literal["google"] = "google"


class VertexAIClientConfig(BaseClientConfig):
    provider: Literal["vertex"] = "vertex"
    api_key: OptionalStr = None
    base_url: OptionalStr = None
    project: OptionalStr = None
    location: OptionalStr = None
    credentials: Credentials | None = None

    @model_validator(mode="after")
    def _validate_vertex_auth(self) -> VertexAIClientConfig:
        if self.api_key is not None and (
            self.project is not None
            or self.location is not None
            or self.credentials is not None
        ):
            raise configuration_error(
                "Vertex AI auth is ambiguous. Configure either api_key or project/location/credentials, not both",
                provider="vertex",
            )

        return self


class AzureOpenAIClientConfig(BaseClientConfig):
    provider: Literal["azure"] = "azure"
    api_key: OptionalStr = None
    azure_ad_token: OptionalStr = None
    azure_ad_token_provider: Callable[[], str] | None = None
    endpoint: OptionalStr = None
    base_url: OptionalStr = None
    api_version: RequiredStr
    deployment: OptionalStr = None

    @model_validator(mode="after")
    def _validate_auth_and_endpoint(self) -> AzureOpenAIClientConfig:
        if self.api_key is not None and self.azure_ad_token is not None:
            raise configuration_error(
                "Azure OpenAI auth is ambiguous. Configure either api_key or azure_ad_token, not both",
                provider="azure",
            )

        if self.azure_ad_token_provider is not None and (
            self.api_key is not None or self.azure_ad_token is not None
        ):
            raise configuration_error(
                "Azure OpenAI auth is ambiguous. azure_ad_token_provider cannot be combined with api_key or azure_ad_token",
                provider="azure",
            )

        if (
            self.api_key is None
            and self.azure_ad_token is None
            and self.azure_ad_token_provider is None
        ):
            raise configuration_error(
                "Missing Azure OpenAI credentials. Provide api_key, azure_ad_token, or azure_ad_token_provider",
                provider="azure",
            )

        if self.base_url is not None and self.endpoint is not None:
            raise configuration_error(
                "Azure OpenAI endpoint is ambiguous. Configure either base_url or endpoint, not both",
                provider="azure",
            )

        if self.base_url is None and self.endpoint is None:
            raise configuration_error(
                "Missing Azure OpenAI endpoint. Provide base_url or endpoint",
                provider="azure",
            )

        return self


class BedrockClientConfig(BaseClientConfig):
    provider: Literal["bedrock"] = "bedrock"
    region: RequiredStr
    api_key: OptionalStr = None
    aws_access_key_id: OptionalStr = None
    aws_secret_access_key: OptionalStr = None
    aws_session_token: OptionalStr = None
    profile_name: OptionalStr = None

    @model_validator(mode="after")
    def _validate_auth(self) -> BedrockClientConfig:
        has_api_key = self.api_key is not None
        has_access_key_pair = (
            self.aws_access_key_id is not None
            or self.aws_secret_access_key is not None
        )
        has_session_token = self.aws_session_token is not None
        has_profile = self.profile_name is not None
        has_aws_auth = has_access_key_pair or has_session_token or has_profile

        if has_api_key and has_aws_auth:
            raise configuration_error(
                "Bedrock auth is ambiguous. Configure either api_key or AWS credentials/profile, not both",
                provider="bedrock",
            )

        if (self.aws_access_key_id is None) != (self.aws_secret_access_key is None):
            raise configuration_error(
                "Bedrock credentials require both aws_access_key_id and aws_secret_access_key",
                provider="bedrock",
            )

        if self.profile_name is not None and has_access_key_pair:
            raise configuration_error(
                "Bedrock auth is ambiguous. Configure either profile_name or aws_access_key_id/aws_secret_access_key",
                provider="bedrock",
            )

        if not has_api_key and not has_aws_auth:
            raise configuration_error(
                "Missing Bedrock auth. Provide api_key, aws_access_key_id/aws_secret_access_key, or profile_name",
                provider="bedrock",
            )

        return self


ClientConfig = (
    OpenAIClientConfig
    | AzureOpenAIClientConfig
    | VertexAIClientConfig
    | ChatGPTClientConfig
    | DeepSeekClientConfig
    | GoogleClientConfig
    | AnthropicClientConfig
    | BedrockClientConfig
)


__all__ = [
    "APIKeyClientConfig",
    "AnthropicClientConfig",
    "AzureOpenAIClientConfig",
    "BaseClientConfig",
    "BedrockClientConfig",
    "ChatGPTClientConfig",
    "ClientConfig",
    "DeepSeekClientConfig",
    "GoogleClientConfig",
    "OpenAIApiType",
    "OpenAIClientConfig",
    "VertexAIClientConfig",
]
