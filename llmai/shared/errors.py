from __future__ import annotations

from typing import NoReturn

import anthropic
import openai
from botocore import exceptions as botocore_exceptions
from google.genai import errors as google_errors


class BaseError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        provider: str | None = None,
        cause: Exception | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.provider = provider
        self.cause = cause
        super().__init__(f"{status_code}: {message}")


class LLMError(BaseError):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMAuthenticationError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


class ToolError(BaseError):
    pass


_BEDROCK_AUTH_ERROR_CODES = {
    "AccessDenied",
    "AccessDeniedException",
    "ExpiredToken",
    "ExpiredTokenException",
    "IncompleteSignature",
    "InvalidClientTokenId",
    "MissingAuthenticationToken",
    "NotAuthorized",
    "NotAuthorizedException",
    "SignatureDoesNotMatch",
    "UnauthorizedOperation",
    "UnrecognizedClientException",
}
_BEDROCK_RATE_LIMIT_ERROR_CODES = {
    "ProvisionedThroughputExceededException",
    "RequestLimitExceeded",
    "ThrottledException",
    "Throttling",
    "ThrottlingException",
    "TooManyRequestsException",
}
_GOOGLE_AUTH_STATUSES = {
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
}
_GOOGLE_RATE_LIMIT_STATUSES = {
    "RESOURCE_EXHAUSTED",
}


def configuration_error(
    message: str,
    *,
    provider: str | None = None,
    status_code: int = 400,
    cause: Exception | None = None,
) -> LLMConfigurationError:
    return LLMConfigurationError(
        status_code,
        message,
        provider=provider,
        cause=cause,
    )


def normalize_llm_error(
    error: Exception,
    *,
    provider: str | None = None,
) -> BaseError:
    if isinstance(error, BaseError):
        if provider is not None and error.provider is None:
            error.provider = provider
        return error

    openai_status_error_types = (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.RateLimitError,
        openai.BadRequestError,
        openai.UnprocessableEntityError,
        openai.NotFoundError,
        openai.ConflictError,
        openai.InternalServerError,
        openai.APIStatusError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.APIError,
    )
    anthropic_status_error_types = (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.RateLimitError,
        anthropic.BadRequestError,
        anthropic.UnprocessableEntityError,
        anthropic.NotFoundError,
        anthropic.ConflictError,
        anthropic.InternalServerError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.APIError,
    )

    if isinstance(error, (openai.APITimeoutError, anthropic.APITimeoutError)):
        return LLMConnectionError(
            504,
            _error_message(error, default="Request timed out."),
            provider=provider,
            cause=error,
        )

    if isinstance(error, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return LLMConnectionError(
            503,
            _error_message(error, default="Connection error."),
            provider=provider,
            cause=error,
        )

    if isinstance(error, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return LLMAuthenticationError(
            _status_code(error, default=401),
            _error_message(error),
            provider=provider,
            cause=error,
        )

    if isinstance(error, (openai.PermissionDeniedError, anthropic.PermissionDeniedError)):
        return LLMAuthenticationError(
            _status_code(error, default=403),
            _error_message(error),
            provider=provider,
            cause=error,
        )

    if isinstance(error, (openai.RateLimitError, anthropic.RateLimitError)):
        return LLMRateLimitError(
            _status_code(error, default=429),
            _error_message(error),
            provider=provider,
            cause=error,
        )

    if isinstance(error, openai_status_error_types + anthropic_status_error_types):
        return LLMError(
            _status_code(error, default=500),
            _error_message(error),
            provider=provider,
            cause=error,
        )

    if isinstance(error, google_errors.ClientError):
        status_code = _status_code(error, default=400)
        status = getattr(error, "status", None)
        if status_code in (401, 403) or status in _GOOGLE_AUTH_STATUSES:
            return LLMAuthenticationError(
                status_code,
                _error_message(error),
                provider=provider,
                cause=error,
            )
        if status_code == 429 or status in _GOOGLE_RATE_LIMIT_STATUSES:
            return LLMRateLimitError(
                429,
                _error_message(error),
                provider=provider,
                cause=error,
            )
        return LLMError(
            status_code,
            _error_message(error),
            provider=provider,
            cause=error,
        )

    if isinstance(error, google_errors.APIError):
        return LLMError(
            _status_code(error, default=500),
            _error_message(error),
            provider=provider,
            cause=error,
        )

    if isinstance(
        error,
        (
            botocore_exceptions.NoAuthTokenError,
            botocore_exceptions.NoCredentialsError,
            botocore_exceptions.PartialCredentialsError,
            botocore_exceptions.CredentialRetrievalError,
        ),
    ):
        return LLMAuthenticationError(
            401,
            _error_message(error, default="AWS credentials are not configured."),
            provider=provider,
            cause=error,
        )

    if isinstance(
        error,
        (
            botocore_exceptions.NoRegionError,
            botocore_exceptions.InvalidRegionError,
            botocore_exceptions.InvalidEndpointConfigurationError,
            botocore_exceptions.InvalidEndpointDiscoveryConfigurationError,
        ),
    ):
        return configuration_error(
            _error_message(error, default="AWS region or endpoint configuration is invalid."),
            provider=provider,
            cause=error,
        )

    if isinstance(error, botocore_exceptions.ConnectTimeoutError):
        return LLMConnectionError(
            504,
            _error_message(error, default="Request timed out."),
            provider=provider,
            cause=error,
        )

    if isinstance(
        error,
        (
            botocore_exceptions.EndpointConnectionError,
            botocore_exceptions.ConnectionError,
            botocore_exceptions.HTTPClientError,
        ),
    ):
        return LLMConnectionError(
            503,
            _error_message(error, default="Connection error."),
            provider=provider,
            cause=error,
        )

    if isinstance(error, botocore_exceptions.ClientError):
        error_response = getattr(error, "response", {}) or {}
        details = error_response.get("Error", {}) or {}
        error_code = details.get("Code")
        status_code = (
            (error_response.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode")
            or 500
        )
        message = details.get("Message") or _error_message(error)

        if error_code in _BEDROCK_AUTH_ERROR_CODES:
            return LLMAuthenticationError(
                status_code or 401,
                message,
                provider=provider,
                cause=error,
            )

        if error_code in _BEDROCK_RATE_LIMIT_ERROR_CODES:
            return LLMRateLimitError(
                429,
                message,
                provider=provider,
                cause=error,
            )

        return LLMError(
            status_code,
            message,
            provider=provider,
            cause=error,
        )

    if isinstance(error, botocore_exceptions.BotoCoreError):
        return LLMError(
            500,
            _error_message(error),
            provider=provider,
            cause=error,
        )

    return LLMError(
        500,
        _error_message(error),
        provider=provider,
        cause=error,
    )


def raise_llm_error(error: Exception, *, provider: str | None = None) -> NoReturn:
    normalized = normalize_llm_error(error, provider=provider)
    if normalized is error:
        raise normalized
    raise normalized from error


def _status_code(error: Exception, *, default: int) -> int:
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    return default


def _error_message(error: Exception, *, default: str = "Request failed.") -> str:
    message = getattr(error, "message", None)
    if isinstance(message, str) and message:
        return message

    text = str(error)
    return text or default
