import os

from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS, WEB_SEARCH_TOOL
from llmai.bedrock import BedrockClient, BedrockClientConfig
from llmai.shared.messages import UserMessage
from llmai.shared.response_formats import JSONSchemaResponse


MODEL = os.getenv(
    "BEDROCK_MODEL",
    "arn:aws:bedrock:eu-central-1:471112542209:inference-profile/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
)


def make_client() -> BedrockClient:
    return BedrockClient(
        config=BedrockClientConfig(
            region=os.getenv("BEDROCK_REGION", "us-east-1"),
            api_key=os.getenv("BEDROCK_API_KEY"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            profile_name=os.getenv("AWS_PROFILE"),
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
    print("Bedrock plain generation")
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
    print("Bedrock structured generation")
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
    print("Bedrock tool-call generation")
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
    print("Bedrock web-search generation (ignored by provider adapter)")
    print(response)
    print("-" * 50)


def test_stream():
    client = make_client()

    print("Bedrock plain stream")
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

    print("Bedrock structured stream")
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

    print("Bedrock tool-call stream")
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

    print("Bedrock web-search stream (ignored by provider adapter)")
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
