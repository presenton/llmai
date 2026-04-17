from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS
from llmai.anthropic import AnthropicClient
from llmai.shared.messages import UserMessage
from llmai.shared.response_formats import JSONSchemaResponse


MODEL = "claude-haiku-4-5"


def test_generate():
    client = AnthropicClient()

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
    client = AnthropicClient()

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
    client = AnthropicClient()

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


def test_stream():
    client = AnthropicClient()

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
    client = AnthropicClient()

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
    client = AnthropicClient()

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


# test_generate()
# test_generate_structured()
# test_generate_tool_calls()
# test_stream()
# test_stream_structured()
# test_stream_tool_calls()
