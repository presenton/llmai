import os

from dev.shared import (
    SLIDE_SCHEMA,
    TOOL_CHOICE,
    TOOL_DEFINITIONS,
    WEB_SEARCH_TOOL,
    get_dev_logger,
)
from llmai.google import GoogleClient, GoogleClientConfig
from llmai.shared.messages import UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse

MODEL = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")
LOGGER = get_dev_logger("google")


def make_client() -> GoogleClient:
    return GoogleClient(
        config=GoogleClientConfig(
            api_key=os.getenv("GOOGLE_API_KEY"),
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


def test_generate():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )
    print("Google plain generation")
    print(response)
    print("-" * 50)


def test_generate_structured():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(),
    )
    print("Google structured generation")
    print(response)
    print("-" * 50)


def test_generate_structured_strict():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(strict=True),
    )
    print("Google strict structured generation")
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
    print("Google tool-call generation")
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
    print("Google web-search generation")
    print(response)
    print("-" * 50)


def test_stream():
    client = make_client()

    print("Google plain stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_structured():
    client = make_client()

    print("Google structured stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_structured_strict():
    client = make_client()

    print("Google strict structured stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(strict=True),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_tool_calls():
    client = make_client()

    print("Google tool-call stream")
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


def test_stream_web_search():
    client = make_client()

    print("Google web-search stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="What was a positive news story from today? Cite sources."
            ),
        ],
        tools=[WEB_SEARCH_TOOL],
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_generate_reasoning():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="Think carefully about whether AI or humans are better at math."
            ),
        ],
        reasoning_effort=make_reasoning_effort(),
    )
    print("Google reasoning generation")
    print(response)
    print("-" * 50)


def test_stream_reasoning():
    client = make_client()

    print("Google reasoning stream")
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


# test_generate()
# test_generate_structured()
# test_generate_structured_strict()
# test_generate_tool_calls()
# test_generate_web_search()
# test_stream()
# test_stream_structured()
# test_stream_structured_strict()
# test_stream_tool_calls()
# test_stream_web_search()
test_generate_reasoning()
# test_stream_reasoning()
