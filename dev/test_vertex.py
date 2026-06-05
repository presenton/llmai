import os

from dev.shared import (
    SLIDE_SCHEMA,
    TOOL_CHOICE,
    TOOL_DEFINITIONS,
    WEB_SEARCH_TOOL,
    get_dev_logger,
)
from llmai.shared.messages import UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse
from llmai.vertex import VertexAIClient, VertexAIClientConfig


MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")
LOGGER = get_dev_logger("vertex")


def make_client() -> VertexAIClient:
    api_key = os.getenv("VERTEX_API_KEY")
    project = os.getenv("VERTEX_PROJECT")
    location = os.getenv("VERTEX_LOCATION")

    config_kwargs: dict[str, str] = {}
    if api_key:
        config_kwargs["api_key"] = api_key
    else:
        if project:
            config_kwargs["project"] = project
        if project or location:
            config_kwargs["location"] = location or "us-central1"

    return VertexAIClient(
        config=VertexAIClientConfig(**config_kwargs),
        logger=LOGGER,
    )


def make_reasoning_effort() -> ReasoningEffort:
    return ReasoningEffort(
        effort=ReasoningEffortValue.HIGH,
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


def test_generate():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )
    print("Vertex plain generation")
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
    print("Vertex structured generation")
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
    print("Vertex strict structured generation")
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
    print("Vertex tool-call generation")
    print(response)
    print("-" * 50)


def test_generate_web_search():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What was a positive news story from today? Cite sources."),
        ],
        tools=[WEB_SEARCH_TOOL],
    )
    print("Vertex web-search generation")
    print(response)
    print("-" * 50)


def test_stream():
    client = make_client()

    print("Vertex plain stream")
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

    print("Vertex structured stream")
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

    print("Vertex strict structured stream")
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

    print("Vertex tool-call stream")
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

    print("Vertex web-search stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What was a positive news story from today? Cite sources."),
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
    print("Vertex reasoning generation")
    print(response)
    print("-" * 50)


def test_stream_reasoning():
    client = make_client()

    print("Vertex reasoning stream")
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
test_generate_structured()
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
