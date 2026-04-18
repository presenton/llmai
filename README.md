# llmai

`llmai` is a Python library for working with multiple LLM providers through a shared set of message, tool, schema, and response primitives.

Today the repository includes adapters for:

- OpenAI
- DeepSeek
- Anthropic
- Google Gemini
- Amazon Bedrock

Each provider client exposes the same core entrypoint:

- `generate(..., stream=False)`

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
from llmai.shared import TextContentPart, UserMessage

client = OpenAIClient(api_key="OPENAI_API_KEY")

result = client.generate(
    model="your-openai-model",
    messages=[
        UserMessage(content=[TextContentPart(text="Write a two-line poem about clean interfaces.")]),
    ],
)

print(result.content)
print(result.usage)
print(result.duration_seconds)
```

If you want to swap providers, the overall call shape stays the same. In most cases you only need to change the client class, credentials, and model name.

## DeepSeek

```python
from llmai import DeepSeekClient
from llmai.shared import JSONSchemaResponse, TextContentPart, UserMessage


client = DeepSeekClient()

result = client.generate(
    model="deepseek-chat",
    messages=[
        UserMessage(content=[TextContentPart(text="Return a JSON object with one field named answer.")]),
    ],
    response_format=JSONSchemaResponse(
        name="final_answer",
        json_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
        },
    ),
    stream=True,
)
```

`DeepSeekClient` uses the OpenAI SDK against DeepSeek's OpenAI-compatible API and reads `DEEPSEEK_API_KEY` by default. For structured output, it always uses an internal function-tool schema because DeepSeek does not support `response_format={"type":"json_schema"}`. During streaming, the internal response tool is surfaced as incremental JSON `content` chunks, and the stream still ends with parsed JSON in `ResponseStreamCompletionChunk.content`. If you need DeepSeek's server-side `strict` tool enforcement, point `base_url` at `https://api.deepseek.com/beta`.

## Amazon Bedrock

```python
from llmai import BedrockClient
from llmai.shared import TextContentPart, UserMessage


client = BedrockClient(
    region="us-east-1",
    aws_access_key_id="AWS_ACCESS_KEY_ID",
    aws_secret_access_key="AWS_SECRET_ACCESS_KEY",
)

# Or use Bedrock API-key auth:
# client = BedrockClient(region="us-east-1", api_key="BEDROCK_API_KEY")

result = client.generate(
    model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    messages=[
        UserMessage(content=[TextContentPart(text="Say hello.")]),
    ],
)

print(result.content)
```

## Structured Output

```python
from pydantic import BaseModel

from llmai import GoogleClient
from llmai.shared import JSONSchemaResponse, TextContentPart, UserMessage


class Summary(BaseModel):
    title: str
    bullets: list[str]


client = GoogleClient(api_key="GOOGLE_API_KEY")

result = client.generate(
    model="your-google-model",
    messages=[
        UserMessage(
            content=[
                TextContentPart(
                    text="Summarize retrieval-augmented generation in simple terms."
                )
            ]
        ),
    ],
    response_format=JSONSchemaResponse(json_schema=Summary),
)

print(result.content)
```

Use `JSONSchemaResponse`, `JSONObjectResponse`, or `TextResponse` to request different response shapes.

## Multimodal Content

```python
from llmai import GoogleClient
from llmai.shared import ImageContentPart, TextContentPart, UserMessage


client = GoogleClient(api_key="GOOGLE_API_KEY")

result = client.generate(
    model="your-google-model",
    messages=[
        UserMessage(
            content=[
                TextContentPart(text="Describe this image."),
                ImageContentPart(url="https://example.com/cat.png"),
            ]
        ),
    ],
)

print(result.content)
print(result.messages[-1].thinking)
```

Request messages can mix text and image parts. Normal completion content is surfaced as `list[TextContentPart | ImageContentPart]` when the provider returns message content, including text-only replies.

## Tool Calling

```python
from pydantic import BaseModel

from llmai import OpenAIClient
from llmai.shared import TextContentPart, Tool, ToolResponseMessage, UserMessage


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
        UserMessage(
            content=[TextContentPart(text="What is the weather in Kathmandu?")]
        ),
    ],
    tools=[weather_tool],
    tool_choice={"tools": ["get_weather"]},
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
                content=[TextContentPart(text="It is sunny in Kathmandu.")],
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
from llmai.shared import TextContentPart, UserMessage

client = AnthropicClient(api_key="ANTHROPIC_API_KEY")

for chunk in client.generate(
    model="your-anthropic-model",
    messages=[
        UserMessage(
            content=[TextContentPart(text="Explain recursion in one paragraph.")]
        ),
    ],
    stream=True,
):
    if chunk.type == "content":
        print(chunk.chunk, end="")
```

`generate(..., stream=True)` yields `ResponseStreamChunk` markers with `event="start"` and `event="end"` around each `content`, `thinking`, and `tool` section. `ResponseStreamCompletionChunk` is emitted bare at the end.

## Package Layout

- `llmai/openai`: OpenAI adapter
- `llmai/deepseek`: DeepSeek adapter
- `llmai/anthropic`: Anthropic adapter
- `llmai/google`: Google Gemini adapter
- `llmai/bedrock`: Amazon Bedrock adapter
- `llmai/shared`: common message, tool, schema, and response models

## Core Types

The shared layer includes the main primitives you will use across providers:

- `UserMessage`, `SystemMessage`, `AssistantMessage`
- `TextContentPart`, `ImageContentPart`
- `Tool`, `ToolResponseMessage`
- `JSONSchemaResponse`, `JSONObjectResponse`, `TextResponse`
- `ResponseContent`, `ResponseStreamChunk`, `ResponseStreamContentChunk`, `ResponseStreamThinkingChunk`, `ResponseStreamToolChunk`, `ResponseStreamCompletionChunk`
- `ResponseUsage`
