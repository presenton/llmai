import os

from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS, WEB_SEARCH_TOOL
from llmai.openai import OpenAIApiType, OpenAIClient, OpenAIClientConfig
from llmai.shared.messages import SystemMessage, UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse


MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
API_TYPE = OpenAIApiType.RESPONSES


def make_client() -> OpenAIClient:
    return OpenAIClient(
        config=OpenAIClientConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_type=API_TYPE,
        )
    )


def test_generate():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="Think as long as you want to define who is better at math AI or Human? You must think and answer"
            ),
        ],
        reasoning_effort=ReasoningEffort(
            effort=ReasoningEffortValue.HIGH,
            summary=ReasoningSummary.DETAILED,
        ),
    )
    print("OpenAI plain generation")
    print(response)
    print("-" * 50)


def test_generate_structured():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            SystemMessage(content="create slide about global warming"),
            UserMessage(content="Create a presentation slide"),
        ],
        response_format=JSONSchemaResponse(
            name="ResponseSchema",
            strict=True,
            json_schema=SLIDE_SCHEMA,
        ),
        reasoning_effort=ReasoningEffort(
            effort=ReasoningEffortValue.HIGH,
            summary=ReasoningSummary.DETAILED,
        ),
    )
    print("OpenAI structured generation")
    print(response)
    print("-" * 50)


def test_generate_tool_calls():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        reasoning_effort=ReasoningEffort(
            effort=ReasoningEffortValue.HIGH,
            summary=ReasoningSummary.DETAILED,
        ),
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
    )
    print("OpenAI tool-call generation")
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
        reasoning_effort=ReasoningEffort(
            effort=ReasoningEffortValue.LOW,
        ),
    )
    print("OpenAI web-search generation")
    print(response)
    print("-" * 50)


def test_stream():
    client = make_client()

    print("OpenAI plain stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        reasoning_effort=ReasoningEffort(
            effort=ReasoningEffortValue.HIGH,
            summary=ReasoningSummary.DETAILED,
        ),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_structured():
    client = make_client()

    print("OpenAI structured stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=JSONSchemaResponse(
            name="ResponseSchema", strict=True, json_schema=SLIDE_SCHEMA
        ),
        reasoning_effort=ReasoningEffort(
            effort=ReasoningEffortValue.HIGH,
            summary=ReasoningSummary.DETAILED,
        ),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_tool_calls():
    client = make_client()

    print("OpenAI tool-call stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
        reasoning_effort=ReasoningEffort(
            effort=ReasoningEffortValue.HIGH,
            summary=ReasoningSummary.DETAILED,
        ),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_web_search():
    client = make_client()

    print("OpenAI web-search stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="What was a positive news story from today? Cite sources."
            ),
        ],
        tools=[WEB_SEARCH_TOOL],
        reasoning_effort=ReasoningEffort(
            effort=ReasoningEffortValue.LOW,
        ),
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
