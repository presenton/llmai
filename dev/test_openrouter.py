import os

from dev.shared import (
    SLIDE_SCHEMA,
    TOOL_CHOICE,
    TOOL_DEFINITIONS,
    WEB_SEARCH_TOOL,
    get_dev_logger,
)
from llmai import OpenRouterClient, OpenRouterClientConfig
from llmai.shared.messages import UserMessage
from llmai.shared.response_formats import JSONSchemaResponse

MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")
LOGGER = get_dev_logger("openrouter")


def make_client() -> OpenRouterClient:
    return OpenRouterClient(
        config=OpenRouterClientConfig(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
        ),
        logger=LOGGER,
    )


def make_response_format(*, strict: bool = False) -> JSONSchemaResponse:
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
    print("OpenRouter plain generation")
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
    print("OpenRouter structured generation")
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
    print("OpenRouter strict structured generation")
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
    print("OpenRouter tool-call generation")
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
    )
    print("OpenRouter web-search generation (ignored by provider adapter)")
    print(response)
    print("-" * 50)


def test_stream():
    client = make_client()

    print("OpenRouter plain stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_tool_calls():
    client = make_client()

    print("OpenRouter tool-call stream")
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
# test_generate_structured_strict()
# test_generate_tool_calls()
# test_generate_web_search()
# test_stream()
# test_stream_tool_calls()
