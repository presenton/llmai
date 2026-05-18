import os

from dev.shared import (
    SLIDE_SCHEMA,
    TOOL_CHOICE,
    TOOL_DEFINITIONS,
    WEB_SEARCH_TOOL,
    get_dev_logger,
)
from llmai import FireworksClient, FireworksClientConfig
from llmai.shared.messages import UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse

MODEL = os.getenv(
    "FIREWORKS_MODEL",
    "accounts/fireworks/models/glm-5p1",
    # "accounts/fireworks/models/deepseek-v4-pro",
)
REASONING_MODEL = os.getenv("FIREWORKS_REASONING_MODEL", MODEL)
LOGGER = get_dev_logger("fireworks")


def make_client() -> FireworksClient:
    return FireworksClient(
        config=FireworksClientConfig(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url=os.getenv("FIREWORKS_BASE_URL"),
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
    print("Fireworks plain generation")
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
    print("Fireworks structured generation")
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
    print("Fireworks strict structured generation")
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
    print("Fireworks tool-call generation")
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
    print("Fireworks web-search generation (ignored by provider adapter)")
    print(response)
    print("-" * 50)


def test_stream():
    client = make_client()

    print("Fireworks plain stream")
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

    print("Fireworks structured stream")
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

    print("Fireworks strict structured stream")
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

    print("Fireworks tool-call stream")
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

    print("Fireworks web-search stream (ignored by provider adapter)")
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
        model=REASONING_MODEL,
        messages=[
            UserMessage(
                content="Think carefully about whether AI or humans are better at math."
            ),
        ],
        reasoning_effort=make_reasoning_effort(),
    )
    print("Fireworks reasoning generation")
    print(response)
    print("-" * 50)


def test_stream_reasoning():
    client = make_client()

    print("Fireworks reasoning stream")
    for chunk in client.generate(
        model=REASONING_MODEL,
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
# test_generate_reasoning()
# test_stream_reasoning()
