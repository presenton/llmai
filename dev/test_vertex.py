import os

from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS, WEB_SEARCH_TOOL
from llmai.shared.messages import UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse
from llmai.vertex import VertexAIClient, VertexAIClientConfig


MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")


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
        config=VertexAIClientConfig(**config_kwargs)
    )


def test_generate():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        reasoning_effort=ReasoningEffort(
            # effort=ReasoningEffortValue.HIGH,
            summary=ReasoningSummary.DETAILED,
        ),
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
        response_format=JSONSchemaResponse(
            name="ResponseSchema",
            strict=True,
            json_schema=SLIDE_SCHEMA,
        ),
    )
    print("Vertex structured generation")
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
        reasoning_effort=ReasoningEffort(summary=ReasoningSummary.DETAILED),
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
        response_format=JSONSchemaResponse(
            name="ResponseSchema", strict=True, json_schema=SLIDE_SCHEMA
        ),
        reasoning_effort=ReasoningEffort(summary=ReasoningSummary.DETAILED),
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
        reasoning_effort=ReasoningEffort(summary=ReasoningSummary.DETAILED),
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
        reasoning_effort=ReasoningEffort(summary=ReasoningSummary.DETAILED),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


# test_generate()
test_generate_structured()
# test_generate_tool_calls()
# test_generate_web_search()
# test_stream()
# test_stream_structured()
# test_stream_tool_calls()
# test_stream_web_search()
