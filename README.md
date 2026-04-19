# llmai

`llmai` is a Python library for working with multiple LLM providers through a shared set of message, tool, schema, and response primitives.

Today the repository includes adapters for:

- ChatGPT
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
from llmai.shared import UserMessage

client = OpenAIClient(api_key="OPENAI_API_KEY")

result = client.generate(
    model="your-openai-model",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
print(result.usage)
print(result.duration_seconds)
```

For text-only prompts, `UserMessage(content="...")` is the simplest form. You can also pass explicit content parts like `TextContentPart` when you need mixed multimodal input or tighter control over message structure.

If you want to swap providers, the overall call shape stays the same. In most cases you only need to change the client class, credentials, and model name.

## ChatGPT

```python
from llmai import ChatGPTClient
from llmai.shared import UserMessage


client = ChatGPTClient(access_token="CHATGPT_ACCESS_TOKEN")

result = client.generate(
    model="chatgpt-4o-latest",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

`ChatGPTClient` targets ChatGPT's Codex backend at `https://chatgpt.com/backend-api/codex`. It always uses the Responses API internally, and reads `CHATGPT_ACCESS_TOKEN` or `CODEX_ACCESS_TOKEN` by default, with optional `CHATGPT_ACCOUNT_ID` or `CODEX_ACCOUNT_ID`.

## DeepSeek

```python
from llmai import DeepSeekClient
from llmai.shared import JSONSchemaResponse, UserMessage


client = DeepSeekClient()

result = client.generate(
    model="deepseek-chat",
    messages=[
        UserMessage(content="Return a JSON object with one field named answer."),
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
from llmai.shared import UserMessage


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
        UserMessage(content="Say hello."),
    ],
)

print(result.content)
```

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

Use explicit content parts when you need multimodal inputs or want to mix text with images in one message. Normal completion content is surfaced as `list[TextContentPart | ImageContentPart]` when the provider returns message content, including text-only replies. `AssistantMessage.thinking` is returned as `list[str]` when the provider exposes one or more reasoning blocks.

## Tool Calling

```python
from pydantic import BaseModel

from llmai import OpenAIClient
from llmai.shared import Tool, ToolResponseMessage, UserMessage


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
                content=["It is sunny in Kathmandu."],
            ),
        ],
        tools=[weather_tool],
    )
    print(follow_up.content)
```

`llmai` returns tool calls in `first.tool_calls` and leaves execution to the caller.

## Hosted Web Search

`llmai` also supports a provider-hosted web search tool that is not a function tool:

```python
from llmai import OpenAIClient
from llmai.shared import UserMessage, WebSearchTool

client = OpenAIClient(api_key="OPENAI_API_KEY")

result = client.generate(
    model="your-openai-model",
    messages=[
        UserMessage(content="What was a positive news story from today? Cite sources."),
    ],
    tools=[WebSearchTool()],
    api_type="responses",
)

print(result.content)
print(result.messages[-1].thinking)
```

You can also target it explicitly in `tool_choice`:

```python
tool_choice = {
    "mode": "required",
    "tools": ["web_search"],
}
```

Current `llmai` behavior:

- OpenAI Responses: attaches built-in `web_search`
- ChatGPT/Codex: attaches built-in `web_search`
- Anthropic: attaches Anthropic's hosted web-search tool
- Google Gemini: attaches `google_search`
- OpenAI Chat Completions: ignores hosted `web_search`
- DeepSeek: ignores hosted `web_search`
- Amazon Bedrock: ignores hosted `web_search`

`web_search` can be mixed with normal function tools in the same request.

## Streaming

```python
from llmai import AnthropicClient
from llmai.shared import UserMessage

client = AnthropicClient(api_key="ANTHROPIC_API_KEY")

for chunk in client.generate(
    model="your-anthropic-model",
    messages=[
        UserMessage(content="Explain recursion in one paragraph."),
    ],
    stream=True,
):
    if chunk.type == "content":
        print(chunk.chunk, end="")
```

`generate(..., stream=True)` yields `ResponseStreamChunk` markers with `event="start"` and `event="end"` around each `content`, `thinking`, and `tool` section. If a provider returns multiple reasoning blocks, each block gets its own `thinking` start/end pair. `ResponseStreamCompletionChunk` is emitted bare at the end.

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
- `Tool`, `WebSearchTool`, `ToolResponseMessage`
- `JSONSchemaResponse`, `JSONObjectResponse`, `TextResponse`
- `ResponseContent`, `ResponseStreamChunk`, `ResponseStreamContentChunk`, `ResponseStreamThinkingChunk`, `ResponseStreamToolChunk`, `ResponseStreamToolCompleteChunk`, `ResponseStreamCompletionChunk`
- `ResponseUsage`
