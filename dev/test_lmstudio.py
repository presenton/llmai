import asyncio
import os

from dev.shared import (
    SLIDE_SCHEMA,
    TOOL_CHOICE,
    TOOL_DEFINITIONS,
    get_dev_logger,
)
from llmai import AsyncLMStudioClient, LMStudioClient, LMStudioClientConfig
from llmai.shared.messages import UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse

MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-1.7b")
LOGGER = get_dev_logger("lmstudio")


def make_client() -> LMStudioClient:
    return LMStudioClient(
        config=LMStudioClientConfig(
            api_key=os.getenv("LMSTUDIO_API_KEY"),
            base_url=os.getenv("LMSTUDIO_BASE_URL"),
        ),
        logger=LOGGER,
    )


def make_async_client() -> AsyncLMStudioClient:
    return AsyncLMStudioClient(
        config=LMStudioClientConfig(
            api_key=os.getenv("LMSTUDIO_API_KEY"),
            base_url=os.getenv("LMSTUDIO_BASE_URL"),
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


def make_reasoning_effort() -> ReasoningEffort:
    return ReasoningEffort(
        effort=ReasoningEffortValue.LOW,
        summary=ReasoningSummary.DETAILED,
    )


def _generate(client: LMStudioClient, label: str):
    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )
    print(label)
    print(response)
    print("-" * 50)


async def _agenerate(client: AsyncLMStudioClient, label: str):
    async with client:
        response = await client.agenerate(
            model=MODEL,
            messages=[
                UserMessage(content="What is presentation?"),
            ],
        )
    print(label)
    print(response)
    print("-" * 50)


def test_generate_completions():
    _generate(make_client(), "LM Studio completions plain generation")


async def test_agenerate_completions():
    await _agenerate(
        make_async_client(),
        "LM Studio completions async plain generation",
    )


def _generate_structured(
    client: LMStudioClient, label: str, *, strict: bool | None = None
):
    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="Create a presentation slide about global warming"),
        ],
        response_format=make_response_format(strict=strict),
    )
    print(label)
    print(response)
    print("-" * 50)


async def _agenerate_structured(
    client: AsyncLMStudioClient, label: str, *, strict: bool | None = None
):
    async with client:
        response = await client.agenerate(
            model=MODEL,
            messages=[
                UserMessage(content="Create a presentation slide about global warming"),
            ],
            response_format=make_response_format(strict=strict),
        )
    print(label)
    print(response)
    print("-" * 50)


def test_generate_structured_completions():
    _generate_structured(
        make_client(),
        "LM Studio completions structured generation",
    )


def test_generate_structured_strict_completions():
    _generate_structured(
        make_client(),
        "LM Studio completions strict structured generation",
        strict=True,
    )


async def test_agenerate_structured_completions():
    await _agenerate_structured(
        make_async_client(),
        "LM Studio completions async structured generation",
    )


async def test_agenerate_structured_strict_completions():
    await _agenerate_structured(
        make_async_client(),
        "LM Studio completions async strict structured generation",
        strict=True,
    )


def test_generate_tool_calls_completions():
    response = make_client().generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is the weather and local time in Kathmandu?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
    )
    print("LM Studio completions tool-call generation")
    print(response)
    print("-" * 50)


async def test_agenerate_tool_calls_completions():
    async with make_async_client() as client:
        response = await client.agenerate(
            model=MODEL,
            messages=[
                UserMessage(content="What is the weather and local time in Kathmandu?"),
            ],
            tools=TOOL_DEFINITIONS,
            tool_choice=TOOL_CHOICE,
        )
    print("LM Studio completions async tool-call generation")
    print(response)
    print("-" * 50)


def test_generate_reasoning_completions():
    response = make_client().generate(
        model=MODEL,
        messages=[
            UserMessage(content="Which number is bigger, 9.11 or 9.9?"),
        ],
        reasoning_effort=make_reasoning_effort(),
    )
    print("LM Studio completions reasoning generation")
    print(response)
    print("-" * 50)


async def test_agenerate_reasoning_completions():
    async with make_async_client() as client:
        response = await client.agenerate(
            model=MODEL,
            messages=[
                UserMessage(content="Which number is bigger, 9.11 or 9.9?"),
            ],
            reasoning_effort=make_reasoning_effort(),
        )
    print("LM Studio completions async reasoning generation")
    print(response)
    print("-" * 50)


def _stream(client: LMStudioClient, label: str):
    print(label)
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


async def _astream(client: AsyncLMStudioClient, label: str):
    print(label)
    async with client:
        async for chunk in client.agenerate(
            model=MODEL,
            messages=[
                UserMessage(content="What is presentation?"),
            ],
            stream=True,
        ):
            print(chunk)
    print("-" * 50)


def test_stream_completions():
    _stream(make_client(), "LM Studio completions plain stream")


async def test_astream_completions():
    await _astream(make_async_client(), "LM Studio completions async plain stream")


# test_generate_completions()
# asyncio.run(test_agenerate_completions())
# test_generate_structured_completions()
# asyncio.run(test_agenerate_structured_completions())
# test_generate_structured_strict_completions()
# asyncio.run(test_agenerate_structured_strict_completions())
# test_generate_tool_calls_completions()
# asyncio.run(test_agenerate_tool_calls_completions())
test_generate_reasoning_completions()
# asyncio.run(test_agenerate_reasoning_completions())
# test_stream_completions()
# asyncio.run(test_astream_completions())
