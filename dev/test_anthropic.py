import os

from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS, WEB_SEARCH_TOOL
from llmai.anthropic import AnthropicClient, AnthropicClientConfig
from llmai.shared.messages import UserMessage
from llmai.shared.response_formats import JSONSchemaResponse


MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")


def make_client() -> AnthropicClient:
    return AnthropicClient(
        config=AnthropicClientConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    )


def test_generate():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )
    print("Anthropic plain generation")
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
    print("Anthropic structured generation")
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
    print("Anthropic tool-call generation")
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
    print("Anthropic web-search generation")
    print(response)
    print("-" * 50)


def test_stream():
    client = make_client()

    print("Anthropic plain stream")
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

    print("Anthropic structured stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=JSONSchemaResponse(
            name="ResponseSchema", strict=True, json_schema=SLIDE_SCHEMA
        ),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_tool_calls():
    client = make_client()

    print("Anthropic tool-call stream")
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

    print("Anthropic web-search stream")
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


# test_generate()
test_generate_structured()
# test_generate_tool_calls()
# test_generate_web_search()
# test_stream()
# test_stream_structured()
# test_stream_tool_calls()
# test_stream_web_search()
