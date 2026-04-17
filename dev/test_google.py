from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS
from llmai.google import GoogleClient
from llmai.shared.messages import UserMessage
from llmai.shared.response_formats import JSONSchemaResponse


MODEL = "gemini-3.1-pro-preview"


def test_generate():
    client = GoogleClient()

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
    client = GoogleClient()

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
    print("Google structured generation")
    print(response)
    print("-" * 50)


def test_generate_tool_calls():
    client = GoogleClient()

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


def test_stream():
    client = GoogleClient()

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
    client = GoogleClient()

    print("Google structured stream")
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
    client = GoogleClient()

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


# test_generate()
# test_generate_structured()
test_generate_tool_calls()
# test_stream()
# test_stream_structured()
# test_stream_tool_calls()
