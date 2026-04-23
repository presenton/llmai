import os

from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS, WEB_SEARCH_TOOL
from llmai import ChatGPTClient, ChatGPTClientConfig
from llmai.shared.messages import UserMessage
from llmai.shared.reasoning import ReasoningEffort, ReasoningSummary
from llmai.shared.response_formats import JSONSchemaResponse


MODEL = os.getenv("CHATGPT_MODEL", "chatgpt-4o-latest")


def make_client() -> ChatGPTClient:
    return ChatGPTClient(
        config=ChatGPTClientConfig(
            access_token=os.getenv("CHATGPT_ACCESS_TOKEN"),
            account_id=os.getenv("CHATGPT_ACCOUNT_ID"),
        )
    )


def test_generate():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        reasoning_effort=ReasoningEffort(summary=ReasoningSummary.DETAILED),
    )
    print("ChatGPT plain generation")
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
        reasoning_effort=ReasoningEffort(summary=ReasoningSummary.DETAILED),
    )
    print("ChatGPT structured generation")
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
        reasoning_effort=ReasoningEffort(summary=ReasoningSummary.DETAILED),
    )
    print("ChatGPT tool-call generation")
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
    print("ChatGPT web-search generation")
    print(response)
    print("-" * 50)


def test_stream():
    client = make_client()

    print("ChatGPT plain stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        reasoning_effort=ReasoningEffort(summary=ReasoningSummary.DETAILED),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_structured():
    client = make_client()

    print("ChatGPT structured stream")
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

    print("ChatGPT tool-call stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
        reasoning_effort=ReasoningEffort(summary=ReasoningSummary.DETAILED),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_web_search():
    client = make_client()

    print("ChatGPT web-search stream")
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
