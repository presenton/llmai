import asyncio
import os

from llmai import AsyncOpenAIClient
from dev.shared import (
    SLIDE_SCHEMA,
    TOOL_CHOICE,
    TOOL_DEFINITIONS,
    WEB_SEARCH_TOOL,
    get_dev_logger,
)
from llmai.openai import OpenAIApiType, OpenAIClient, OpenAIClientConfig
from llmai.shared.messages import SystemMessage, UserMessage
from llmai.shared.reasoning import (
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
)
from llmai.shared.response_formats import JSONSchemaResponse

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
LOGGER = get_dev_logger("openai")


def make_client(api_type: OpenAIApiType) -> OpenAIClient:
    return OpenAIClient(
        config=OpenAIClientConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_type=api_type,
        ),
        logger=LOGGER,
    )


def make_async_client(api_type: OpenAIApiType) -> AsyncOpenAIClient:
    return AsyncOpenAIClient(
        config=OpenAIClientConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_type=api_type,
        ),
        logger=LOGGER,
    )


def make_completions_client() -> OpenAIClient:
    return make_client(OpenAIApiType.COMPLETIONS)


def make_responses_client() -> OpenAIClient:
    return make_client(OpenAIApiType.RESPONSES)


def make_async_completions_client() -> AsyncOpenAIClient:
    return make_async_client(OpenAIApiType.COMPLETIONS)


def make_async_responses_client() -> AsyncOpenAIClient:
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


def _generate(client: OpenAIClient, label: str):
    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )
    print(label)
    print(response)
    print("-" * 50)


async def _agenerate(client: AsyncOpenAIClient, label: str):
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
    _generate(make_completions_client(), "OpenAI completions plain generation")


def test_generate_responses():
    _generate(make_responses_client(), "OpenAI responses plain generation")


async def test_agenerate_completions():
    await _agenerate(
        make_async_completions_client(),
        "OpenAI completions async plain generation",
    )


async def test_agenerate_responses():
    await _agenerate(
        make_async_responses_client(),
        "OpenAI responses async plain generation",
    )


def _generate_structured(
    client: OpenAIClient, label: str, *, strict: bool | None = None
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
    client: AsyncOpenAIClient, label: str, *, strict: bool | None = None
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
        "OpenAI completions structured generation",
    )


def test_generate_structured_responses():
    _generate_structured(
        make_responses_client(),
        "OpenAI responses structured generation",
    )


def test_generate_structured_strict_completions():
    _generate_structured(
        make_completions_client(),
        "OpenAI completions strict structured generation",
        strict=True,
    )


def test_generate_structured_strict_responses():
    _generate_structured(
        make_responses_client(),
        "OpenAI responses strict structured generation",
        strict=True,
    )


async def test_agenerate_structured_completions():
    await _agenerate_structured(
        make_async_completions_client(),
        "OpenAI completions async structured generation",
    )


async def test_agenerate_structured_responses():
    await _agenerate_structured(
        make_async_responses_client(),
        "OpenAI responses async structured generation",
    )


async def test_agenerate_structured_strict_completions():
    await _agenerate_structured(
        make_async_completions_client(),
        "OpenAI completions async strict structured generation",
        strict=True,
    )


async def test_agenerate_structured_strict_responses():
    await _agenerate_structured(
        make_async_responses_client(),
        "OpenAI responses async strict structured generation",
        strict=True,
    )


def _generate_tool_calls(client: OpenAIClient, label: str):
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


async def _agenerate_tool_calls(client: AsyncOpenAIClient, label: str):
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
        "OpenAI completions tool-call generation",
    )


def test_generate_tool_calls_responses():
    _generate_tool_calls(
        make_responses_client(),
        "OpenAI responses tool-call generation",
    )


async def test_agenerate_tool_calls_completions():
    await _agenerate_tool_calls(
        make_async_completions_client(),
        "OpenAI completions async tool-call generation",
    )


async def test_agenerate_tool_calls_responses():
    await _agenerate_tool_calls(
        make_async_responses_client(),
        "OpenAI responses async tool-call generation",
    )


def test_generate_web_search_responses():
    client = make_responses_client()

    response = client.generate(
        model=MODEL,
        messages=[
            UserMessage(
                content="What was a positive news story from today? Cite sources."
            ),
        ],
        tools=[WEB_SEARCH_TOOL],
    )
    print("OpenAI responses web-search generation")
    print(response)
    print("-" * 50)


async def test_agenerate_web_search_responses():
    async with make_async_responses_client() as client:
        response = await client.agenerate(
            model=MODEL,
            messages=[
                UserMessage(
                    content="What was a positive news story from today? Cite sources."
                ),
            ],
            tools=[WEB_SEARCH_TOOL],
        )
    print("OpenAI responses async web-search generation")
    print(response)
    print("-" * 50)


def _stream(client: OpenAIClient, label: str):
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


async def _astream(client: AsyncOpenAIClient, label: str):
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
    _stream(make_completions_client(), "OpenAI completions plain stream")


def test_stream_responses():
    _stream(make_responses_client(), "OpenAI responses plain stream")


async def test_astream_completions():
    await _astream(
        make_async_completions_client(),
        "OpenAI completions async plain stream",
    )


async def test_astream_responses():
    await _astream(make_async_responses_client(), "OpenAI responses async plain stream")


def _stream_structured(client: OpenAIClient, label: str, *, strict: bool | None = None):
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
    client: AsyncOpenAIClient, label: str, *, strict: bool | None = None
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
        "OpenAI completions structured stream",
    )


def test_stream_structured_responses():
    _stream_structured(
        make_responses_client(),
        "OpenAI responses structured stream",
    )


def test_stream_structured_strict_completions():
    _stream_structured(
        make_completions_client(),
        "OpenAI completions strict structured stream",
        strict=True,
    )


def test_stream_structured_strict_responses():
    _stream_structured(
        make_responses_client(),
        "OpenAI responses strict structured stream",
        strict=True,
    )


async def test_astream_structured_completions():
    await _astream_structured(
        make_async_completions_client(),
        "OpenAI completions async structured stream",
    )


async def test_astream_structured_responses():
    await _astream_structured(
        make_async_responses_client(),
        "OpenAI responses async structured stream",
    )


async def test_astream_structured_strict_completions():
    await _astream_structured(
        make_async_completions_client(),
        "OpenAI completions async strict structured stream",
        strict=True,
    )


async def test_astream_structured_strict_responses():
    await _astream_structured(
        make_async_responses_client(),
        "OpenAI responses async strict structured stream",
        strict=True,
    )


def _stream_tool_calls(client: OpenAIClient, label: str):
    print(label)
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


async def _astream_tool_calls(client: AsyncOpenAIClient, label: str):
    print(label)
    async with client:
        async for chunk in client.agenerate(
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


def test_stream_tool_calls_completions():
    _stream_tool_calls(
        make_completions_client(),
        "OpenAI completions tool-call stream",
    )


def test_stream_tool_calls_responses():
    _stream_tool_calls(
        make_responses_client(),
        "OpenAI responses tool-call stream",
    )


async def test_astream_tool_calls_completions():
    await _astream_tool_calls(
        make_async_completions_client(),
        "OpenAI completions async tool-call stream",
    )


async def test_astream_tool_calls_responses():
    await _astream_tool_calls(
        make_async_responses_client(),
        "OpenAI responses async tool-call stream",
    )


def test_stream_web_search_responses():
    client = make_responses_client()

    print("OpenAI responses web-search stream")
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


async def test_astream_web_search_responses():
    print("OpenAI responses async web-search stream")
    async with make_async_responses_client() as client:
        async for chunk in client.agenerate(
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


def _generation_loop(client: OpenAIClient, label: str):
    messages = [
        UserMessage(
            content=(
                "Think as long as you want to define who is better at math AI "
                "or Human? You must think and answer"
            )
        )
    ]
    for _ in range(3):
        response = client.generate(
            model=MODEL,
            messages=messages,
        )
        messages = response.messages
        messages.append(UserMessage(content="Think more"))
        print(response.content)
        print("-" * 50)
    print(label)
    print(response)
    print("-" * 50)


async def _ageneration_loop(client: AsyncOpenAIClient, label: str):
    messages = [
        UserMessage(
            content="Think as long as you want to define who is better at math AI or Human? You must think and answer"
        )
    ]
    async with client:
        for _ in range(3):
            response = await client.agenerate(
                model=MODEL,
                messages=messages,
            )
            messages = response.messages
            messages.append(UserMessage(content="Think more"))
            print(response.content)
            print("-" * 50)
    print(label)
    print(response)
    print("-" * 50)


def test_generation_loop_completions():
    _generation_loop(
        make_completions_client(),
        "OpenAI completions generation loop",
    )


def test_generation_loop_responses():
    _generation_loop(
        make_responses_client(),
        "OpenAI responses generation loop",
    )


async def test_ageneration_loop_completions():
    await _ageneration_loop(
        make_async_completions_client(),
        "OpenAI completions async generation loop",
    )


async def test_ageneration_loop_responses():
    await _ageneration_loop(
        make_async_responses_client(),
        "OpenAI responses async generation loop",
    )


def _generate_reasoning(client: OpenAIClient, label: str):
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


async def _agenerate_reasoning(client: AsyncOpenAIClient, label: str):
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
        "OpenAI completions reasoning generation",
    )


def test_generate_reasoning_responses():
    _generate_reasoning(
        make_responses_client(),
        "OpenAI responses reasoning generation",
    )


async def test_agenerate_reasoning_completions():
    await _agenerate_reasoning(
        make_async_completions_client(),
        "OpenAI completions async reasoning generation",
    )


async def test_agenerate_reasoning_responses():
    await _agenerate_reasoning(
        make_async_responses_client(),
        "OpenAI responses async reasoning generation",
    )


def _stream_reasoning(client: OpenAIClient, label: str):
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


async def _astream_reasoning(client: AsyncOpenAIClient, label: str):
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
        "OpenAI completions reasoning stream",
    )


def test_stream_reasoning_responses():
    _stream_reasoning(
        make_responses_client(),
        "OpenAI responses reasoning stream",
    )


async def test_astream_reasoning_completions():
    await _astream_reasoning(
        make_async_completions_client(),
        "OpenAI completions async reasoning stream",
    )


async def test_astream_reasoning_responses():
    await _astream_reasoning(
        make_async_responses_client(),
        "OpenAI responses async reasoning stream",
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
# test_generate_web_search_responses()
# asyncio.run(test_agenerate_web_search_responses())
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
# test_stream_tool_calls_completions()
# test_stream_tool_calls_responses()
# asyncio.run(test_astream_tool_calls_completions())
# asyncio.run(test_astream_tool_calls_responses())
# test_stream_web_search_responses()
# asyncio.run(test_astream_web_search_responses())
# test_generation_loop_completions()
# test_generation_loop_responses()
# asyncio.run(test_ageneration_loop_completions())
# asyncio.run(test_ageneration_loop_responses())
# test_generate_reasoning_completions()
# test_generate_reasoning_responses()
# asyncio.run(test_agenerate_reasoning_completions())
# asyncio.run(test_agenerate_reasoning_responses())
# test_stream_reasoning_completions()
# test_stream_reasoning_responses()
# asyncio.run(test_astream_reasoning_completions())
# asyncio.run(test_astream_reasoning_responses())
