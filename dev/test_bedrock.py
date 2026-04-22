from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS, WEB_SEARCH_TOOL
from llmai.bedrock import BedrockClient, BedrockClientConfig
from llmai.shared.messages import UserMessage
from llmai.shared.response_formats import JSONSchemaResponse


MODEL = "arn:aws:bedrock:eu-central-1:471112542209:inference-profile/eu.anthropic.claude-haiku-4-5-20251001-v1:0"
CLIENT_CONFIG = BedrockClientConfig(
    api_key="<your-bedrock-api-key>",
    region="us-east-1",
)


def test_generate():
    client = BedrockClient(config=CLIENT_CONFIG)

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
    client = BedrockClient(config=CLIENT_CONFIG)

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
    client = BedrockClient(config=CLIENT_CONFIG)

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
    client = BedrockClient(config=CLIENT_CONFIG)

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
    client = BedrockClient(config=CLIENT_CONFIG)

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
    client = BedrockClient(config=CLIENT_CONFIG)

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
    client = BedrockClient(config=CLIENT_CONFIG)

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
    client = BedrockClient(config=CLIENT_CONFIG)

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
# test_generate_structured()
# test_generate_tool_calls()
# test_generate_web_search()
# test_stream()
# test_stream_structured()
# test_stream_tool_calls()
# test_stream_web_search()
