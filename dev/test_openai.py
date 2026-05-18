import os

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


def make_completions_client() -> OpenAIClient:
    return make_client(OpenAIApiType.COMPLETIONS)


def make_responses_client() -> OpenAIClient:
    return make_client(OpenAIApiType.RESPONSES)


def make_reasoning_effort() -> ReasoningEffort:
    return ReasoningEffort(
        effort=ReasoningEffortValue.LOW,
        summary=ReasoningSummary.DETAILED,
    )


def make_response_format(*, strict: bool = False) -> JSONSchemaResponse:
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


def test_generate_completions():
    _generate(make_completions_client(), "OpenAI completions plain generation")


def test_generate_responses():
    _generate(make_responses_client(), "OpenAI responses plain generation")


def _generate_structured(client: OpenAIClient, label: str, *, strict: bool = False):
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


def test_stream_completions():
    _stream(make_completions_client(), "OpenAI completions plain stream")


def test_stream_responses():
    _stream(make_responses_client(), "OpenAI responses plain stream")


def _stream_structured(client: OpenAIClient, label: str, *, strict: bool = False):
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


def _generation_loop(client: OpenAIClient, label: str):
    messages = [
        UserMessage(
            content="Think as long as you want to define who is better at math AI or Human? You must think and answer"
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


# test_generate_completions()
# test_generate_responses()
# test_generate_structured_completions()
# test_generate_structured_responses()
# test_generate_structured_strict_completions()
# test_generate_structured_strict_responses()
# test_generate_tool_calls_completions()
# test_generate_tool_calls_responses()
# test_generate_web_search_responses()
# test_stream_completions()
# test_stream_responses()
# test_stream_structured_completions()
# test_stream_structured_responses()
# test_stream_structured_strict_completions()
# test_stream_structured_strict_responses()
# test_stream_tool_calls_completions()
# test_stream_tool_calls_responses()
# test_stream_web_search_responses()
# test_generation_loop_completions()
# test_generation_loop_responses()
# test_generate_reasoning_completions()
# test_generate_reasoning_responses()
# test_stream_reasoning_completions()
# test_stream_reasoning_responses()
