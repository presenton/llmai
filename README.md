# llmai

`llmai` is a Python library for working with multiple LLM providers through a shared set of message, tool, schema, and response primitives.

Today the repository includes adapters for:

- OpenAI
- Anthropic
- Google Gemini

Each provider client exposes the same core entrypoints:

- `generate(...)`
- `stream(...)`

## Why This Exists

Provider SDKs differ in how they represent messages, tool calls, structured output, and streaming events. `llmai` smooths those differences out so application code can stay closer to one mental model.

## Installation

Install the project locally with `uv`:

```bash
uv sync
```

Or install it in editable mode with `pip`:

```bash
pip install -e .
```

## Quick Start

```python
from llmai import OpenAIClient
from llmai.shared import UserMessage

client = OpenAIClient(api_key="OPENAI_API_KEY")

result = client.generate(
    model="your-openai-model",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

If you want to swap providers, the overall call shape stays the same. In most cases you only need to change the client class, credentials, and model name.

## Structured Output

```python
from pydantic import BaseModel

from llmai import GoogleClient
from llmai.shared import JSONSchemaResponse, UserMessage


class Summary(BaseModel):
    title: str
    bullets: list[str]


client = GoogleClient(api_key="GOOGLE_API_KEY")

result = client.generate(
    model="your-google-model",
    messages=[
        UserMessage(content="Summarize retrieval-augmented generation in simple terms."),
    ],
    response_format=JSONSchemaResponse(json_schema=Summary),
)

print(result.content)
```

Use `JSONSchemaResponse`, `JSONObjectResponse`, or `TextResponse` to request different response shapes.

## Tool Calling

```python
from pydantic import BaseModel

from llmai import OpenAIClient
from llmai.shared import Tool, ToolChoice, ToolResponseMessage, UserMessage


class WeatherArgs(BaseModel):
    city: str


weather_tool = Tool(
    name="get_weather",
    description="Look up the weather for a city.",
    schema=WeatherArgs,
)

client = OpenAIClient(api_key="OPENAI_API_KEY")

first = client.generate(
    model="your-openai-model",
    messages=[
        UserMessage(content="What is the weather in Kathmandu?"),
    ],
    tools=[weather_tool],
    tool_choice=ToolChoice(optional=["get_weather"]),
)

for tool_call in first.tool_calls:
    if tool_call.name != "get_weather":
        continue

    follow_up = client.generate(
        model="your-openai-model",
        messages=[
            *first.messages,
            ToolResponseMessage(
                id=tool_call.id,
                content="It is sunny in Kathmandu.",
            ),
        ],
        tools=[weather_tool],
    )
    print(follow_up.content)
```

`llmai` returns tool calls in `first.tool_calls` and leaves execution to the caller.

## Streaming

```python
from llmai import AnthropicClient
from llmai.shared import UserMessage

client = AnthropicClient(api_key="ANTHROPIC_API_KEY")

for chunk in client.stream(
    model="your-anthropic-model",
    messages=[
        UserMessage(content="Explain recursion in one paragraph."),
    ],
):
    if chunk.type == "stream_content":
        print(chunk.chunk, end="")
```

`stream(...)` yields incremental `ResponseStreamContentChunk` values while the model is responding. Some clients also emit a `ResponseStreamCompletionChunk` when the stream finishes.

## Package Layout

- `llmai/openai`: OpenAI adapter
- `llmai/anthropic`: Anthropic adapter
- `llmai/google`: Google Gemini adapter
- `llmai/shared`: common message, tool, schema, and response models

## Core Types

The shared layer includes the main primitives you will use across providers:

- `UserMessage`, `SystemMessage`, `AssistantMessage`
- `Tool`, `ToolChoice`, `ToolResponseMessage`
- `JSONSchemaResponse`, `JSONObjectResponse`, `TextResponse`
- `ResponseContent`, `ResponseStreamContentChunk`, `ResponseStreamCompletionChunk`
