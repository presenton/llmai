import json
import os

from dev.shared import (
    SLIDE_SCHEMA,
    TOOL_CHOICE,
    TOOL_DEFINITIONS,
    WEB_SEARCH_TOOL,
    get_dev_logger,
)
from llmai import TogetherAIClient, TogetherAIClientConfig
from llmai.shared.messages import SystemMessage, UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse

MODEL = os.getenv("TOGETHERAI_MODEL", "openai/gpt-oss-20b")
REASONING_MODEL = os.getenv("TOGETHERAI_REASONING_MODEL", MODEL)
MAX_TOKENS = int(os.getenv("TOGETHERAI_MAX_TOKENS", "4096"))
LOGGER = get_dev_logger("togetherai")


def make_client() -> TogetherAIClient:
    return TogetherAIClient(
        config=TogetherAIClientConfig(
            api_key=os.getenv("TOGETHERAI_API_KEY"),
            base_url=os.getenv("TOGETHERAI_BASE_URL"),
        ),
        logger=LOGGER,
    )


def make_reasoning_effort() -> ReasoningEffort:
    return ReasoningEffort(
        effort=ReasoningEffortValue.LOW,
        summary=ReasoningSummary.DETAILED,
    )


def make_response_format(*, strict: bool = False) -> JSONSchemaResponse:
    return JSONSchemaResponse(
        name="ResponseSchema",
        strict=strict,
        json_schema=SLIDE_SCHEMA,
    )


def make_structured_messages(prompt: str):
    return [
        SystemMessage(
            content=(
                "Only answer in JSON. Follow this JSON schema exactly: "
                f"{json.dumps(SLIDE_SCHEMA)}"
            ),
        ),
        UserMessage(content=prompt),
    ]


def test_generate():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )
    print("Together AI plain generation")
    print(response)
    print("-" * 50)


def test_generate_structured():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=make_structured_messages("What is presentation?"),
        response_format=make_response_format(),
        max_tokens=MAX_TOKENS,
    )
    print("Together AI structured generation")
    print(response)
    print("-" * 50)


def test_generate_structured_strict():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=make_structured_messages("What is presentation?"),
        response_format=make_response_format(strict=True),
        max_tokens=MAX_TOKENS,
    )
    print("Together AI strict structured generation")
    print(response)
    print("-" * 50)


def test_generate_tool_calls():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
    )
    print("Together AI tool-call generation")
    print(response)
    print("-" * 50)


def test_generate_web_search():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="What was a positive news story from today? Cite sources."
            ),
        ],
        tools=[WEB_SEARCH_TOOL],
    )
    print("Together AI web-search generation (ignored by provider adapter)")
    print(response)
    print("-" * 50)


def test_generate_reasoning():
    client = make_client()

    response = client.generate(
        model=REASONING_MODEL,
        messages=[
            UserMessage(content="Which number is bigger, 9.11 or 9.9?"),
        ],
        reasoning_effort=make_reasoning_effort(),
    )
    print("Together AI reasoning generation")
    print(response)
    print("-" * 50)


def test_stream():
    client = make_client()

    print("Together AI plain stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_reasoning():
    client = make_client()

    print("Together AI reasoning stream")
    for chunk in client.generate(
        model=REASONING_MODEL,
        messages=[
            UserMessage(content="Which number is bigger, 9.11 or 9.9?"),
        ],
        reasoning_effort=make_reasoning_effort(),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_tool_calls():
    client = make_client()

    print("Together AI tool-call stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


# test_generate()
# test_generate_structured()
# test_generate_structured_strict()
# test_generate_tool_calls()
# test_generate_web_search()
# test_generate_reasoning()
# test_stream()
# test_stream_reasoning()
# test_stream_tool_calls()
