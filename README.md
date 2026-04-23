# llmai

`llmai` is a Python library for working with OpenAI, Azure OpenAI, Vertex AI, Anthropic, Google Gemini, DeepSeek, Bedrock, and ChatGPT through a shared set of message, tool, schema, and response primitives.

Today the repository includes adapters for:

- ChatGPT
- OpenAI
- Azure OpenAI
- Vertex AI
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
from llmai import OpenAIClient, OpenAIClientConfig
from llmai.shared import UserMessage

client = OpenAIClient(
    config=OpenAIClientConfig(api_key="<your-openai-api-key>"),
)

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

For text-only prompts, `UserMessage(content="...")` is the simplest form. `SystemMessage(content="...")` also takes a plain string. Use explicit content parts like `TextContentPart` only when you need mixed multimodal input or tighter control over user message structure.

If you want to swap providers, the overall call shape stays the same. In most cases you only need to change the client class, credentials, and model name.

## Azure OpenAI

```python
from llmai import AzureOpenAIClient, AzureOpenAIClientConfig
from llmai.shared import UserMessage


client = AzureOpenAIClient(
    config=AzureOpenAIClientConfig(
        api_key="<your-azure-openai-api-key>",
        endpoint="https://your-resource.openai.azure.com",
        api_version="2024-10-21",
    ),
)

result = client.generate(
    model="your-azure-deployment",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

`AzureOpenAIClient` uses the official OpenAI SDK's Azure client and requires an explicit `AzureOpenAIClientConfig`. The config supports API-key auth or Entra token auth and accepts `endpoint` or `base_url`, `api_version`, and optional `deployment`. Azure is always routed through chat completions.

## Vertex AI

```python
from llmai import VertexAIClient, VertexAIClientConfig
from llmai.shared import UserMessage


client = VertexAIClient(
    config=VertexAIClientConfig(
        project="your-gcp-project",
        location="us-central1",
    ),
)

result = client.generate(
    model="gemini-2.5-flash",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

`VertexAIClient` uses the `google-genai` Vertex AI path internally and requires an explicit `VertexAIClientConfig`. Use either `api_key` for Vertex express mode or `project`/`location`/`credentials` for standard Vertex auth; do not combine them. `base_url` remains optional.

## ChatGPT

```python
from llmai import ChatGPTClient, ChatGPTClientConfig
from llmai.shared import UserMessage


client = ChatGPTClient(
    config=ChatGPTClientConfig(access_token="<your-chatgpt-access-token>"),
)

result = client.generate(
    model="chatgpt-4o-latest",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

`ChatGPTClient` targets ChatGPT's Codex backend at `https://chatgpt.com/backend-api/codex` and always uses the Responses API internally. Credentials and optional overrides are passed through `ChatGPTClientConfig`, which uses `access_token`. When you include `SystemMessage` entries, ChatGPT collects them in order and sends them through the Responses API `instructions` field; otherwise it falls back to `instructions="Follow the prompt"`. The ChatGPT backend requires `stream=True`, so `generate(stream=False)` streams internally and returns the aggregated final response. It also does not support Responses `temperature` or `max_output_tokens`, so `temperature` and `max_tokens` are ignored for this client.

## DeepSeek

```python
from llmai import DeepSeekClient, DeepSeekClientConfig
from llmai.shared import JSONSchemaResponse, UserMessage


client = DeepSeekClient(
    config=DeepSeekClientConfig(api_key="<your-deepseek-api-key>"),
)

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
)

print(result.content)
```

`DeepSeekClient` uses the OpenAI SDK against DeepSeek's OpenAI-compatible API and requires an explicit `DeepSeekClientConfig`. For structured output, it always uses an internal function-tool schema because DeepSeek does not support `response_format={"type":"json_schema"}`. During streaming, the internal response tool is surfaced as incremental JSON `content` chunks, and the stream still ends with parsed JSON on the final completion chunk's `content`. If you need DeepSeek's server-side `strict` tool enforcement, point `base_url` at `https://api.deepseek.com/beta`.

## Amazon Bedrock

```python
from llmai import BedrockClient, BedrockClientConfig
from llmai.shared import UserMessage


client = BedrockClient(
    config=BedrockClientConfig(
        region="us-east-1",
        aws_access_key_id="<your-aws-access-key-id>",
        aws_secret_access_key="<your-aws-secret-access-key>",
    ),
)

# Or use Bedrock API-key auth:
# client = BedrockClient(
#     config=BedrockClientConfig(region="us-east-1", api_key="<your-bedrock-api-key>")
# )

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

from llmai import GoogleClient, GoogleClientConfig
from llmai.shared import JSONSchemaResponse, UserMessage


class Summary(BaseModel):
    title: str
    bullets: list[str]


client = GoogleClient(config=GoogleClientConfig(api_key="<your-google-api-key>"))

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
from llmai import GoogleClient, GoogleClientConfig
from llmai.shared import ImageContentPart, TextContentPart, UserMessage


client = GoogleClient(config=GoogleClientConfig(api_key="<your-google-api-key>"))

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
print(result.thinking)
```

Use explicit content parts when you need multimodal inputs or want to mix text with images in one message. Normal completion content is surfaced as `list[TextContentPart | ImageContentPart]` when the provider returns message content, including text-only replies. Reasoning is exposed on `ResponseContent.thinking` as `list[str]` when the provider returns one or more thinking blocks, and the same value is also available on the final `AssistantMessage`.

## Tool Calling

```python
from pydantic import BaseModel

from llmai import OpenAIClient, OpenAIClientConfig
from llmai.shared import Tool, ToolResponseMessage, UserMessage


class WeatherArgs(BaseModel):
    city: str


weather_tool = Tool(
    name="get_weather",
    description="Look up the weather for a city.",
    schema=WeatherArgs,
)

client = OpenAIClient(config=OpenAIClientConfig(api_key="<your-openai-api-key>"))

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
from llmai import OpenAIApiType, OpenAIClient, OpenAIClientConfig
from llmai.shared import UserMessage, WebSearchTool

client = OpenAIClient(
    config=OpenAIClientConfig(
        api_key="<your-openai-api-key>",
        api_type=OpenAIApiType.RESPONSES,
    )
)

result = client.generate(
    model="your-openai-model",
    messages=[
        UserMessage(content="What was a positive news story from today? Cite sources."),
    ],
    tools=[WebSearchTool()],
)

print(result.content)
print(result.thinking)
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
- Azure OpenAI: follows the same OpenAI adapter surface; service support depends on your Azure API version and deployment
- Vertex AI: attaches `google_search`
- ChatGPT/Codex: attaches built-in `web_search`
- Anthropic: attaches Anthropic's hosted web-search tool
- Google Gemini: attaches `google_search`
- OpenAI Chat Completions: ignores hosted `web_search`
- DeepSeek: ignores hosted `web_search`
- Amazon Bedrock: ignores hosted `web_search`

`web_search` can be mixed with normal function tools in the same request.

If you use `OpenAIClient` with `api_type=OpenAIApiType.RESPONSES`, `OpenAIClientConfig(provide_system_message_as_instructions=True)` lifts all `SystemMessage` values into the top-level Responses API `instructions` field. The default is `False`, which keeps system messages inline in `input`.

## Streaming

```python
from llmai import AnthropicClient, AnthropicClientConfig
from llmai.shared import UserMessage

client = AnthropicClient(
    config=AnthropicClientConfig(api_key="<your-anthropic-api-key>"),
)

for chunk in client.generate(
    model="your-anthropic-model",
    messages=[
        UserMessage(content="Explain recursion in one paragraph."),
    ],
    stream=True,
):
    if chunk.type == "content":
        print(chunk.chunk, end="")
    elif chunk.type == "completion":
        print("\nDone:", chunk.usage)
```

`generate(..., stream=True)` yields marker chunks with `type="event"` and `event="start"` / `event="end"` around each `content`, `thinking`, and `tool` section. If a provider returns multiple reasoning blocks, each block gets its own `thinking` start/end pair. The final chunk has `type="completion"` and includes top-level `content`, `thinking`, usage, and accumulated messages.

## Package Layout

- `llmai/openai`: OpenAI adapter
- `llmai/azure`: Azure OpenAI adapter
- `llmai/vertex`: Vertex AI adapter
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

`UserMessage.content` accepts either a plain string or explicit content parts. `SystemMessage.content` is always a plain string.
