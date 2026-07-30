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
from llmai import AsyncCerebrasClient, CerebrasClient, CerebrasClientConfig
from llmai.shared.messages import UserMessage
from llmai.shared.response_formats import JSONSchemaResponse

MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
LOGGER = get_dev_logger("cerebras")


def make_client() -> CerebrasClient:
    return CerebrasClient(
        config=CerebrasClientConfig(
            api_key=os.getenv("CEREBRAS_API_KEY"),
            base_url=os.getenv("CEREBRAS_BASE_URL"),
        ),
        logger=LOGGER,
    )


def make_async_client() -> AsyncCerebrasClient:
    return AsyncCerebrasClient(
        config=CerebrasClientConfig(
            api_key=os.getenv("CEREBRAS_API_KEY"),
            base_url=os.getenv("CEREBRAS_BASE_URL"),
        ),
        logger=LOGGER,
    )


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
    print("Cerebras plain generation")
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
    print("Cerebras structured generation")
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
    print("Cerebras strict structured generation")
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
    print("Cerebras tool-call generation")
    print(response)
    print("-" * 50)


def test_generate_tool_loop():
    run_tool_loop(make_client(), model=MODEL, label="Cerebras")


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
    print("Cerebras web-search generation (ignored by provider adapter)")
    print(response)
    print("-" * 50)


def test_stream():
    client = make_client()

    print("Cerebras plain stream")
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

    print("Cerebras tool-call stream")
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


async def _agenerate(label: str, **kwargs):
    async with make_async_client() as client:
        response = await client.agenerate(model=MODEL, **kwargs)
    print(label)
    print(response)
    print("-" * 50)


async def _astream(label: str, **kwargs):
    print(label)
    async with make_async_client() as client:
        async for chunk in client.agenerate(model=MODEL, stream=True, **kwargs):
            print(chunk)
    print("-" * 50)


async def test_agenerate():
    await _agenerate(
        "Cerebras async plain generation",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )


async def test_agenerate_structured():
    await _agenerate(
        "Cerebras async structured generation",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(),
    )


async def test_agenerate_structured_strict():
    await _agenerate(
        "Cerebras async strict structured generation",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(strict=True),
    )


async def test_agenerate_tool_calls():
    await _agenerate(
        "Cerebras async tool-call generation",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
    )


async def test_agenerate_tool_loop():
    async with make_async_client() as client:
        await arun_tool_loop(client, model=MODEL, label="Cerebras async")


async def test_agenerate_web_search():
    await _agenerate(
        "Cerebras async web-search generation (ignored by provider adapter)",
        messages=[
            UserMessage(
                content="What was a positive news story from today? Cite sources."
            ),
        ],
        tools=[WEB_SEARCH_TOOL],
    )


async def test_astream():
    await _astream(
        "Cerebras async plain stream",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )


async def test_astream_tool_calls():
    await _astream(
        "Cerebras async tool-call stream",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
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
# test_stream_tool_calls()
# asyncio.run(test_astream_tool_calls())
