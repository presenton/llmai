import os

from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS, get_dev_logger
from llmai.litellm import LiteLLMClient, LiteLLMClientConfig
from llmai.openai import OpenAIApiType
from llmai.shared.messages import SystemMessage, UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse

MODEL = os.getenv("LITELLM_MODEL", "gpt-5.4-mini")
LOGGER = get_dev_logger("litellm")


def make_client(api_type: OpenAIApiType | None = None) -> LiteLLMClient:
    return LiteLLMClient(
        config=LiteLLMClientConfig(
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url=os.getenv("LITELLM_BASE_URL"),
            api_type=api_type
            or os.getenv("LITELLM_API_TYPE", OpenAIApiType.COMPLETIONS),
        ),
        logger=LOGGER,
    )


def make_completions_client() -> LiteLLMClient:
    return make_client(OpenAIApiType.COMPLETIONS)


def make_responses_client() -> LiteLLMClient:
    return make_client(OpenAIApiType.RESPONSES)


def make_reasoning_effort() -> ReasoningEffort:
    return ReasoningEffort(
        effort=ReasoningEffortValue.LOW,
        summary=ReasoningSummary.DETAILED,
    )


def make_response_format(*, strict: bool | None = None) -> JSONSchemaResponse:
    if strict is None:
        return JSONSchemaResponse(
            name="ResponseSchema",
            json_schema=SLIDE_SCHEMA,
        )

    return JSONSchemaResponse(
        name="ResponseSchema",
        strict=strict,
        json_schema=SLIDE_SCHEMA,
    )


def _generate(client: LiteLLMClient, label: str):
    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )
    print(label)
    print(response)
    print("-" * 50)


def test_generate_completions():
    _generate(make_completions_client(), "LiteLLM completions plain generation")


def test_generate_responses():
    _generate(make_responses_client(), "LiteLLM responses plain generation")


def _generate_structured(
    client: LiteLLMClient, label: str, *, strict: bool | None = None
):
    response = client.generate(
        model=MODEL,
        messages=[
            SystemMessage(content="create slide about global warming"),
            UserMessage(content="Create a presentation slide"),
        ],
        response_format=make_response_format(strict=strict),
    )
    print(label)
    print(response)
    print("-" * 50)


def test_generate_structured_completions():
    _generate_structured(
        make_completions_client(),
        "LiteLLM completions structured generation",
    )


def test_generate_structured_responses():
    _generate_structured(
        make_responses_client(),
        "LiteLLM responses structured generation",
    )


def test_generate_structured_strict_completions():
    _generate_structured(
        make_completions_client(),
        "LiteLLM completions strict structured generation",
        strict=True,
    )


def test_generate_structured_strict_responses():
    _generate_structured(
        make_responses_client(),
        "LiteLLM responses strict structured generation",
        strict=True,
    )


def _generate_tool_calls(client: LiteLLMClient, label: str):
    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
    )
    print(label)
    print(response)
    print("-" * 50)


def test_generate_tool_calls_completions():
    _generate_tool_calls(
        make_completions_client(),
        "LiteLLM completions tool-call generation",
    )


def test_generate_tool_calls_responses():
    _generate_tool_calls(
        make_responses_client(),
        "LiteLLM responses tool-call generation",
    )


def _stream(client: LiteLLMClient, label: str):
    print(label)
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_completions():
    _stream(make_completions_client(), "LiteLLM completions plain stream")


def test_stream_responses():
    _stream(make_responses_client(), "LiteLLM responses plain stream")


def _stream_structured(
    client: LiteLLMClient, label: str, *, strict: bool | None = None
):
    print(label)
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(strict=strict),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_structured_completions():
    _stream_structured(
        make_completions_client(),
        "LiteLLM completions structured stream",
    )


def test_stream_structured_responses():
    _stream_structured(
        make_responses_client(),
        "LiteLLM responses structured stream",
    )


def test_stream_structured_strict_completions():
    _stream_structured(
        make_completions_client(),
        "LiteLLM completions strict structured stream",
        strict=True,
    )


def test_stream_structured_strict_responses():
    _stream_structured(
        make_responses_client(),
        "LiteLLM responses strict structured stream",
        strict=True,
    )


def _generate_reasoning(client: LiteLLMClient, label: str):
    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="Think carefully about whether AI or humans are better at math."
            ),
        ],
        reasoning_effort=make_reasoning_effort(),
    )
    print(label)
    print(response)
    print("-" * 50)


def test_generate_reasoning_completions():
    _generate_reasoning(
        make_completions_client(),
        "LiteLLM completions reasoning generation",
    )


def test_generate_reasoning_responses():
    _generate_reasoning(
        make_responses_client(),
        "LiteLLM responses reasoning generation",
    )


def _stream_reasoning(client: LiteLLMClient, label: str):
    print(label)
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="Think carefully about whether AI or humans are better at math."
            ),
        ],
        reasoning_effort=make_reasoning_effort(),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_reasoning_completions():
    _stream_reasoning(
        make_completions_client(),
        "LiteLLM completions reasoning stream",
    )


def test_stream_reasoning_responses():
    _stream_reasoning(
        make_responses_client(),
        "LiteLLM responses reasoning stream",
    )


# test_generate_completions()
# test_generate_responses()
# test_generate_structured_completions()
# test_generate_structured_responses()
# test_generate_structured_strict_completions()
# test_generate_structured_strict_responses()
# test_generate_tool_calls_completions()
# test_generate_tool_calls_responses()
# test_stream_completions()
# test_stream_responses()
# test_stream_structured_completions()
# test_stream_structured_responses()
# test_stream_structured_strict_completions()
# test_stream_structured_strict_responses()
# test_generate_reasoning_completions()
# test_generate_reasoning_responses()
# test_stream_reasoning_completions()
# test_stream_reasoning_responses()
