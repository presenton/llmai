import asyncio
import os

from dev.shared import (
    SLIDE_SCHEMA,
    TOOL_CHOICE,
    TOOL_DEFINITIONS,
    WEB_SEARCH_TOOL,
    arun_tool_loop,
    get_dev_logger,
    run_tool_loop,
)
from llmai import AsyncBedrockClient
from llmai.bedrock import BedrockClient, BedrockClientConfig
from llmai.shared.messages import UserMessage
from llmai.shared.reasoning import ReasoningEffort
from llmai.shared.response_formats import JSONSchemaResponse

MODEL = os.getenv(
    "BEDROCK_MODEL",
    "arn:aws:bedrock:eu-central-1:471112542209:inference-profile/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
)
LOGGER = get_dev_logger("bedrock")


def make_client() -> BedrockClient:
    return BedrockClient(
        config=BedrockClientConfig(
            region=os.getenv("BEDROCK_REGION", "us-east-1"),
            api_key=os.getenv("BEDROCK_API_KEY"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            profile_name=os.getenv("AWS_PROFILE"),
        ),
        logger=LOGGER,
    )


def make_async_client() -> AsyncBedrockClient:
    return AsyncBedrockClient(
        config=BedrockClientConfig(
            region=os.getenv("BEDROCK_REGION", "us-east-1"),
            api_key=os.getenv("BEDROCK_API_KEY"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            profile_name=os.getenv("AWS_PROFILE"),
        ),
        logger=LOGGER,
    )


def make_reasoning_effort() -> ReasoningEffort:
    return ReasoningEffort(tokens=2048)


def make_response_format(*, strict: bool | None = None) -> JSONSchemaResponse:
    if strict is None:
        return JSONSchemaResponse(
            name="ResponseSchema",
            json_schema=SLIDE_SCHEMA,
        )

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
        response_format=make_response_format(),
    )
    print("Bedrock structured generation")
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
    print("Bedrock strict structured generation")
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


def test_generate_tool_loop():
    run_tool_loop(make_client(), model=MODEL, label="Bedrock")


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
        response_format=make_response_format(),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_stream_structured_strict():
    client = make_client()

    print("Bedrock strict structured stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(strict=True),
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
            UserMessage(
                content="What was a positive news story from today? Cite sources."
            ),
        ],
        tools=[WEB_SEARCH_TOOL],
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


def test_generate_reasoning():
    client = make_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="Think carefully about whether AI or humans are better at math."
            ),
        ],
        reasoning_effort=make_reasoning_effort(),
    )
    print("Bedrock reasoning generation")
    print(response)
    print("-" * 50)


def test_stream_reasoning():
    client = make_client()

    print("Bedrock reasoning stream")
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="Think carefully about whether AI or humans are better at math."
            ),
        ],
        reasoning_effort=make_reasoning_effort(),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


async def _agenerate(label: str, *, model: str = MODEL, **kwargs):
    async with make_async_client() as client:
        response = await client.agenerate(model=model, **kwargs)
    print(label)
    print(response)
    print("-" * 50)


async def _astream(label: str, *, model: str = MODEL, **kwargs):
    print(label)
    async with make_async_client() as client:
        async for chunk in client.agenerate(model=model, stream=True, **kwargs):
            print(chunk)
    print("-" * 50)


async def test_agenerate():
    await _agenerate(
        "Bedrock async plain generation",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )


async def test_agenerate_structured():
    await _agenerate(
        "Bedrock async structured generation",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(),
    )


async def test_agenerate_structured_strict():
    await _agenerate(
        "Bedrock async strict structured generation",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(strict=True),
    )


async def test_agenerate_tool_calls():
    await _agenerate(
        "Bedrock async tool-call generation",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
    )


async def test_agenerate_tool_loop():
    async with make_async_client() as client:
        await arun_tool_loop(client, model=MODEL, label="Bedrock async")


async def test_agenerate_web_search():
    await _agenerate(
        "Bedrock async web-search generation (ignored by provider adapter)",
        messages=[
            UserMessage(
                content="What was a positive news story from today? Cite sources."
            ),
        ],
        tools=[WEB_SEARCH_TOOL],
    )


async def test_astream():
    await _astream(
        "Bedrock async plain stream",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )


async def test_astream_structured():
    await _astream(
        "Bedrock async structured stream",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(),
    )


async def test_astream_structured_strict():
    await _astream(
        "Bedrock async strict structured stream",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(strict=True),
    )


async def test_astream_tool_calls():
    await _astream(
        "Bedrock async tool-call stream",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
    )


async def test_astream_web_search():
    await _astream(
        "Bedrock async web-search stream (ignored by provider adapter)",
        messages=[
            UserMessage(
                content="What was a positive news story from today? Cite sources."
            ),
        ],
        tools=[WEB_SEARCH_TOOL],
    )


async def test_agenerate_reasoning():
    await _agenerate(
        "Bedrock async reasoning generation",
        messages=[
            UserMessage(
                content="Think carefully about whether AI or humans are better at math."
            ),
        ],
        reasoning_effort=make_reasoning_effort(),
    )


async def test_astream_reasoning():
    await _astream(
        "Bedrock async reasoning stream",
        messages=[
            UserMessage(
                content="Think carefully about whether AI or humans are better at math."
            ),
        ],
        reasoning_effort=make_reasoning_effort(),
    )


# test_generate()
# asyncio.run(test_agenerate())
# test_generate_structured()
# asyncio.run(test_agenerate_structured())
# test_generate_structured_strict()
# asyncio.run(test_agenerate_structured_strict())
# test_generate_tool_calls()
# asyncio.run(test_agenerate_tool_calls())
# test_generate_tool_loop()
# asyncio.run(test_agenerate_tool_loop())
# test_generate_web_search()
# asyncio.run(test_agenerate_web_search())
# test_stream()
# asyncio.run(test_astream())
# test_stream_structured()
# asyncio.run(test_astream_structured())
# test_stream_structured_strict()
# asyncio.run(test_astream_structured_strict())
# test_stream_tool_calls()
# asyncio.run(test_astream_tool_calls())
# test_stream_web_search()
# asyncio.run(test_astream_web_search())
# test_generate_reasoning()
# asyncio.run(test_agenerate_reasoning())
# test_stream_reasoning()
# asyncio.run(test_astream_reasoning())
