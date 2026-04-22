from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS, WEB_SEARCH_TOOL
from llmai.shared.messages import UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse
from llmai.vertex import VertexAIClient, VertexAIClientConfig


MODEL = "gemini-2.5-flash"
CLIENT_CONFIG = VertexAIClientConfig(
    api_key="<your-vertex-api-key>",
    project="your-gcp-project",
    location="us-central1",
)


def test_generate():
    client = VertexAIClient(config=CLIENT_CONFIG)

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
    client = VertexAIClient(config=CLIENT_CONFIG)

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
    client = VertexAIClient(config=CLIENT_CONFIG)

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
    client = VertexAIClient(config=CLIENT_CONFIG)

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
    client = VertexAIClient(config=CLIENT_CONFIG)

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
    client = VertexAIClient(config=CLIENT_CONFIG)

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
    client = VertexAIClient(config=CLIENT_CONFIG)

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
    client = VertexAIClient(config=CLIENT_CONFIG)

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
# test_generate_structured()
# test_generate_tool_calls()
# test_generate_web_search()
# test_stream()
# test_stream_structured()
# test_stream_tool_calls()
# test_stream_web_search()
