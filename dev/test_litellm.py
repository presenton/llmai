import asyncio
import os

from dev.shared import SLIDE_SCHEMA, TOOL_CHOICE, TOOL_DEFINITIONS, get_dev_logger
from llmai import AsyncLiteLLMClient
from llmai.litellm import LiteLLMClient, LiteLLMClientConfig
from llmai.openai import OpenAIApiType
from llmai.shared.messages import SystemMessage, UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse

MODEL = os.getenv("LITELLM_MODEL", "gpt-5.4-mini")
LOGGER = get_dev_logger("litellm")


def make_client(api_type: OpenAIApiType | None = None) -> LiteLLMClient:
    return LiteLLMClient(
        config=LiteLLMClientConfig(
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url=os.getenv("LITELLM_BASE_URL"),
            api_type=api_type
            or os.getenv("LITELLM_API_TYPE", OpenAIApiType.COMPLETIONS),
        ),
        logger=LOGGER,
    )


def make_async_client(api_type: OpenAIApiType | None = None) -> AsyncLiteLLMClient:
    return AsyncLiteLLMClient(
        config=LiteLLMClientConfig(
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url=os.getenv("LITELLM_BASE_URL"),
            api_type=api_type
            or os.getenv("LITELLM_API_TYPE", OpenAIApiType.COMPLETIONS),
        ),
        logger=LOGGER,
    )


def make_completions_client() -> LiteLLMClient:
    return make_client(OpenAIApiType.COMPLETIONS)


def make_responses_client() -> LiteLLMClient:
    return make_client(OpenAIApiType.RESPONSES)


def make_async_completions_client() -> AsyncLiteLLMClient:
    return make_async_client(OpenAIApiType.COMPLETIONS)


def make_async_responses_client() -> AsyncLiteLLMClient:
    return make_async_client(OpenAIApiType.RESPONSES)


def make_reasoning_effort() -> ReasoningEffort:
    return ReasoningEffort(
        effort=ReasoningEffortValue.LOW,
        summary=ReasoningSummary.DETAILED,
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


def _generate(client: LiteLLMClient, label: str):
    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )
    print(label)
    print(response)
    print("-" * 50)


async def _agenerate(client: AsyncLiteLLMClient, label: str):
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
    _generate(make_completions_client(), "LiteLLM completions plain generation")


def test_generate_responses():
    _generate(make_responses_client(), "LiteLLM responses plain generation")


async def test_agenerate_completions():
    await _agenerate(
        make_async_completions_client(),
        "LiteLLM completions async plain generation",
    )


async def test_agenerate_responses():
    await _agenerate(
        make_async_responses_client(),
        "LiteLLM responses async plain generation",
    )


def _generate_structured(
    client: LiteLLMClient, label: str, *, strict: bool | None = None
):
    response = client.generate(
        model=MODEL,
        messages=[
            SystemMessage(content="create slide about global warming"),
            UserMessage(content="Create a presentation slide"),
        ],
        response_format=make_response_format(strict=strict),
    )
    print(label)
    print(response)
    print("-" * 50)


async def _agenerate_structured(
    client: AsyncLiteLLMClient, label: str, *, strict: bool | None = None
):
    async with client:
        response = await client.agenerate(
            model=MODEL,
            messages=[
                SystemMessage(content="create slide about global warming"),
                UserMessage(content="Create a presentation slide"),
            ],
            response_format=make_response_format(strict=strict),
        )
    print(label)
    print(response)
    print("-" * 50)


def test_generate_structured_completions():
    _generate_structured(
        make_completions_client(),
        "LiteLLM completions structured generation",
    )


def test_generate_structured_responses():
    _generate_structured(
        make_responses_client(),
        "LiteLLM responses structured generation",
    )


def test_generate_structured_strict_completions():
    _generate_structured(
        make_completions_client(),
        "LiteLLM completions strict structured generation",
        strict=True,
    )


def test_generate_structured_strict_responses():
    _generate_structured(
        make_responses_client(),
        "LiteLLM responses strict structured generation",
        strict=True,
    )


async def test_agenerate_structured_completions():
    await _agenerate_structured(
        make_async_completions_client(),
        "LiteLLM completions async structured generation",
    )


async def test_agenerate_structured_responses():
    await _agenerate_structured(
        make_async_responses_client(),
        "LiteLLM responses async structured generation",
    )


async def test_agenerate_structured_strict_completions():
    await _agenerate_structured(
        make_async_completions_client(),
        "LiteLLM completions async strict structured generation",
        strict=True,
    )


async def test_agenerate_structured_strict_responses():
    await _agenerate_structured(
        make_async_responses_client(),
        "LiteLLM responses async strict structured generation",
        strict=True,
    )


def _generate_tool_calls(client: LiteLLMClient, label: str):
    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice=TOOL_CHOICE,
    )
    print(label)
    print(response)
    print("-" * 50)


async def _agenerate_tool_calls(client: AsyncLiteLLMClient, label: str):
    async with client:
        response = await client.agenerate(
            model=MODEL,
            messages=[
                UserMessage(content="What is presentation?"),
            ],
            tools=TOOL_DEFINITIONS,
            tool_choice=TOOL_CHOICE,
        )
    print(label)
    print(response)
    print("-" * 50)


def test_generate_tool_calls_completions():
    _generate_tool_calls(
        make_completions_client(),
        "LiteLLM completions tool-call generation",
    )


def test_generate_tool_calls_responses():
    _generate_tool_calls(
        make_responses_client(),
        "LiteLLM responses tool-call generation",
    )


async def test_agenerate_tool_calls_completions():
    await _agenerate_tool_calls(
        make_async_completions_client(),
        "LiteLLM completions async tool-call generation",
    )


async def test_agenerate_tool_calls_responses():
    await _agenerate_tool_calls(
        make_async_responses_client(),
        "LiteLLM responses async tool-call generation",
    )


def _stream(client: LiteLLMClient, label: str):
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


async def _astream(client: AsyncLiteLLMClient, label: str):
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
    _stream(make_completions_client(), "LiteLLM completions plain stream")


def test_stream_responses():
    _stream(make_responses_client(), "LiteLLM responses plain stream")


async def test_astream_completions():
    await _astream(
        make_async_completions_client(),
        "LiteLLM completions async plain stream",
    )


async def test_astream_responses():
    await _astream(
        make_async_responses_client(),
        "LiteLLM responses async plain stream",
    )


def _stream_structured(
    client: LiteLLMClient, label: str, *, strict: bool | None = None
):
    print(label)
    for chunk in client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=make_response_format(strict=strict),
        stream=True,
    ):
        print(chunk)
    print("-" * 50)


async def _astream_structured(
    client: AsyncLiteLLMClient, label: str, *, strict: bool | None = None
):
    print(label)
    async with client:
        async for chunk in client.agenerate(
            model=MODEL,
            messages=[
                UserMessage(content="What is presentation?"),
            ],
            response_format=make_response_format(strict=strict),
            stream=True,
        ):
            print(chunk)
    print("-" * 50)


def test_stream_structured_completions():
    _stream_structured(
        make_completions_client(),
        "LiteLLM completions structured stream",
    )


def test_stream_structured_responses():
    _stream_structured(
        make_responses_client(),
        "LiteLLM responses structured stream",
    )


def test_stream_structured_strict_completions():
    _stream_structured(
        make_completions_client(),
        "LiteLLM completions strict structured stream",
        strict=True,
    )


def test_stream_structured_strict_responses():
    _stream_structured(
        make_responses_client(),
        "LiteLLM responses strict structured stream",
        strict=True,
    )


async def test_astream_structured_completions():
    await _astream_structured(
        make_async_completions_client(),
        "LiteLLM completions async structured stream",
    )


async def test_astream_structured_responses():
    await _astream_structured(
        make_async_responses_client(),
        "LiteLLM responses async structured stream",
    )


async def test_astream_structured_strict_completions():
    await _astream_structured(
        make_async_completions_client(),
        "LiteLLM completions async strict structured stream",
        strict=True,
    )


async def test_astream_structured_strict_responses():
    await _astream_structured(
        make_async_responses_client(),
        "LiteLLM responses async strict structured stream",
        strict=True,
    )


def _generate_reasoning(client: LiteLLMClient, label: str):
    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="Think carefully about whether AI or humans are better at math."
            ),
        ],
        reasoning_effort=make_reasoning_effort(),
    )
    print(label)
    print(response)
    print("-" * 50)


async def _agenerate_reasoning(client: AsyncLiteLLMClient, label: str):
    async with client:
        response = await client.agenerate(
            model=MODEL,
            messages=[
                UserMessage(
                    content=(
                        "Think carefully about whether AI or humans are better at math."
                    )
                ),
            ],
            reasoning_effort=make_reasoning_effort(),
        )
    print(label)
    print(response)
    print("-" * 50)


def test_generate_reasoning_completions():
    _generate_reasoning(
        make_completions_client(),
        "LiteLLM completions reasoning generation",
    )


def test_generate_reasoning_responses():
    _generate_reasoning(
        make_responses_client(),
        "LiteLLM responses reasoning generation",
    )


async def test_agenerate_reasoning_completions():
    await _agenerate_reasoning(
        make_async_completions_client(),
        "LiteLLM completions async reasoning generation",
    )


async def test_agenerate_reasoning_responses():
    await _agenerate_reasoning(
        make_async_responses_client(),
        "LiteLLM responses async reasoning generation",
    )


def _stream_reasoning(client: LiteLLMClient, label: str):
    print(label)
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


async def _astream_reasoning(client: AsyncLiteLLMClient, label: str):
    print(label)
    async with client:
        async for chunk in client.agenerate(
            model=MODEL,
            messages=[
                UserMessage(
                    content=(
                        "Think carefully about whether AI or humans are better at math."
                    )
                ),
            ],
            reasoning_effort=make_reasoning_effort(),
            stream=True,
        ):
            print(chunk)
    print("-" * 50)


def test_stream_reasoning_completions():
    _stream_reasoning(
        make_completions_client(),
        "LiteLLM completions reasoning stream",
    )


def test_stream_reasoning_responses():
    _stream_reasoning(
        make_responses_client(),
        "LiteLLM responses reasoning stream",
    )


async def test_astream_reasoning_completions():
    await _astream_reasoning(
        make_async_completions_client(),
        "LiteLLM completions async reasoning stream",
    )


async def test_astream_reasoning_responses():
    await _astream_reasoning(
        make_async_responses_client(),
        "LiteLLM responses async reasoning stream",
    )


# test_generate_completions()
# test_generate_responses()
# asyncio.run(test_agenerate_completions())
# asyncio.run(test_agenerate_responses())
# test_generate_structured_completions()
# test_generate_structured_responses()
# asyncio.run(test_agenerate_structured_completions())
# asyncio.run(test_agenerate_structured_responses())
# test_generate_structured_strict_completions()
# test_generate_structured_strict_responses()
# asyncio.run(test_agenerate_structured_strict_completions())
# asyncio.run(test_agenerate_structured_strict_responses())
# test_generate_tool_calls_completions()
# test_generate_tool_calls_responses()
# asyncio.run(test_agenerate_tool_calls_completions())
# asyncio.run(test_agenerate_tool_calls_responses())
# test_stream_completions()
# test_stream_responses()
# asyncio.run(test_astream_completions())
# asyncio.run(test_astream_responses())
# test_stream_structured_completions()
# test_stream_structured_responses()
# asyncio.run(test_astream_structured_completions())
# asyncio.run(test_astream_structured_responses())
# test_stream_structured_strict_completions()
# test_stream_structured_strict_responses()
# asyncio.run(test_astream_structured_strict_completions())
# asyncio.run(test_astream_structured_strict_responses())
# test_generate_reasoning_completions()
# test_generate_reasoning_responses()
# asyncio.run(test_agenerate_reasoning_completions())
# asyncio.run(test_agenerate_reasoning_responses())
# test_stream_reasoning_completions()
# test_stream_reasoning_responses()
# asyncio.run(test_astream_reasoning_completions())
# asyncio.run(test_astream_reasoning_responses())
