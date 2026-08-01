# llmai

`llmai` is a Python library for working with OpenAI, Azure OpenAI, Vertex AI, Anthropic, Google Gemini, DeepSeek, OpenRouter, Cerebras, Fireworks, Together AI, LM Studio, Bedrock, LiteLLM, and ChatGPT through a shared set of message, tool, schema, and response primitives.

Today the repository includes adapters for:

- ChatGPT
- OpenAI
- Azure OpenAI
- Vertex AI
- DeepSeek
- OpenRouter
- Cerebras
- Fireworks
- Together AI
- LM Studio
- Anthropic
- Google Gemini
- Amazon Bedrock
- LiteLLM

Each synchronous provider client exposes the same core entrypoint:

- `generate(..., stream=False)`

Every provider also has a separate async client with:

- `await agenerate(..., stream=False)`
- `async for ... in agenerate(..., stream=True)`

The important boundary is provider-neutral: application code chooses an output
policy, reasoning policy, tools, and response shape. The adapter then translates
that request into the fields accepted by the selected provider and normalizes
the result back into shared response types.

## Compatibility at a Glance

The generation compatibility layer is designed for BYOK applications where the
provider and model are not known until a user supplies credentials.

| Concern | llmai behavior |
| --- | --- |
| Output limit | Resolves a safe default from the application policy and bundled model metadata. |
| Provider token field | Uses the provider's native field; OpenAI-compatible APIs can negotiate `max_completion_tokens` versus `max_tokens`. |
| Reasoning toggle | `True`, `False`, and provider/model default (`None`) are distinct choices. |
| Reasoning level | Normalizes `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`, then adapts to advertised model levels. |
| Reasoning budget | Chooses a metadata-aware default when a model requires manual budget tokens and keeps room for the visible answer. |
| Thinking trace | Returns only reasoning text or summaries the provider deliberately exposes. Hidden chain-of-thought is never reconstructed. |
| Thinking tokens | Keeps provider-reported billed tokens separate from a clearly marked estimate of visible trace tokens. |
| Tool support | Reports supported, unsupported, or unknown instead of treating missing metadata as false. |
| Continuation state | Preserves signatures, encrypted reasoning, and redacted blocks separately from displayable reasoning. |
| Validation | Adaptive mode warns and makes safe compatibility adjustments; strict mode raises; off mode delegates validation to the provider. |

This layer does not promise that every model accepts every feature. It gives the
application a stable API, makes model knowledge inspectable before a request,
and preserves provider errors when a capability cannot be known in advance.

## Local Model Metadata

The complete [`models.dev/api.json`](https://models.dev/api.json) response is
bundled at `llmai/data/models.json`. It currently contains thousands of model
records, including context limits, output limits, capabilities, modalities,
pricing, and release metadata.

```python
from llmai import (
    get_context_window,
    get_model_metadata,
    query_models,
    refresh_model_data,
)

# Provider-qualified references are unambiguous.
context = get_context_window("openai:gpt-4.1")
metadata = get_model_metadata("openai:gpt-4.1")
print(context)
print(metadata["max_output_tokens"], metadata["modalities"])

# Search names, IDs, references, families, and descriptions.
reasoning_models = query_models("reasoning", provider="anthropic")

# Re-download, validate, and atomically replace the bundled JSON.
refresh_model_data()
```

Each flattened record retains every upstream model field and includes
`provider_metadata` for the provider-level API, documentation, environment, and
package information. Bare model IDs and names are accepted when they identify exactly one record.
When multiple providers expose the same ID, use `provider:model-id`, the
`provider=` argument, or `query_models()` to inspect every match. Use
`load_model_data()` when you need the original provider-keyed JSON structure,
and `list_models()` for flattened records with `context_window` extracted. If
an upstream record has no valid context limit, `context_window` and
`get_context_window()` reliably fall back to `4000`. An unknown or ambiguous
model also returns this fallback from `get_context_window()`; pass
`default=...` to customize it. `get_model_metadata()` remains strict and raises
a lookup error when callers need to distinguish missing or ambiguous records.

## Generation Defaults and BYOK Compatibility

Every client config accepts the same nested `GenerationDefaults`. This is the
recommended place for an application such as Presenton to define one policy for
all user-supplied provider keys:

```python
from llmai import (
    GenerationDefaults,
    GenerationProfile,
    OpenAIClient,
    OpenAIClientConfig,
    ReasoningConfig,
    ReasoningEffortValue,
)

defaults = GenerationDefaults(
    profile=GenerationProfile.BALANCED,
    max_output_tokens_cap=32_768,
    reasoning=ReasoningConfig(enabled=None),  # provider/model default
)

client = OpenAIClient(
    config=OpenAIClientConfig(
        api_key="<user-supplied-key>",
        base_url="https://an-openai-compatible-service.example/v1",
        generation=defaults,
    )
)
```

In a BYOK service, create the provider config after identifying the key type and
attach the same `GenerationDefaults` to every config. Do not persist raw keys in
llmai responses or diagnostics; llmai only needs the key while constructing the
provider client.

The profiles resolve omitted output limits from bundled model metadata:

- `fast`: at most 4K output tokens and optional reasoning disabled
- `balanced` (default): advertised output limit capped at 32K, or 8K when unknown
- `deep`: advertised output limit capped at 64K, or 32K when unknown, with high reasoning
- `model_max`: the advertised maximum, or 32K when unknown

The output-limit precedence is:

1. Per-call `max_output_tokens=` or legacy `max_tokens=`
2. `GenerationDefaults.max_output_tokens`
3. The selected profile resolved against model metadata
4. The profile's documented fallback when metadata is unknown

`max_output_tokens_cap` is a final application safety ceiling, including for an
explicit call value. Passing both token argument names is an error.
OpenAI-compatible clients initially use
`max_completion_tokens`; when a provider returns a recognized pre-inference
400/422 “unsupported parameter” error, llmai retries the equivalent
`max_tokens` field once and caches that result for the model. It does not retry
arbitrary errors or remove explicitly requested behavior.

Use `ValidationMode.ADAPTIVE` (the default) for BYOK products: metadata
conflicts produce structured warnings and unsupported effort levels are mapped
to the nearest advertised level. `STRICT` raises instead, while `OFF` trusts
the caller and provider.

Call `client.prepare_generation(...)` to inspect the resolved output limit,
reasoning settings, capabilities, and structured warnings without spending
tokens:

```python
prepared = client.prepare_generation(
    model="user-selected-model",
    profile=GenerationProfile.DEEP,
    reasoning=ReasoningConfig(enabled=True, effort=ReasoningEffortValue.HIGH),
    tools_requested=True,
)

print(prepared.max_output_tokens)
print(prepared.capabilities.tool_call.support.status)
for warning in prepared.warnings:
    print(warning.code, warning.message)
```

Reasoning follows the same principle. Explicit fields in the per-call
`ReasoningConfig` override the legacy `ReasoningEffort`, profile behavior, and
config defaults. Fields omitted from a per-call config continue to inherit the
lower-precedence policy.

## Capabilities, Thinking, and Reasoning Tokens

Capabilities are tri-state because “not listed” is not the same as “not
supported”:

```python
from llmai import (
    get_model_capabilities,
    get_reasoning_levels,
    supports_thinking,
    supports_tool_call,
)

caps = get_model_capabilities("gpt-5", provider="openai")
print(caps.tool_call.support.status, caps.tool_call.support.source)
print(supports_tool_call("gpt-5", provider="openai"))  # True / False / None
print(supports_thinking("gpt-5", provider="openai"))
print(get_reasoning_levels("gpt-5", provider="openai"))
```

The same methods are available on sync and async clients. Client-side lookup
can lazily refresh stale or missing OpenRouter and Cerebras capabilities from
their live model APIs, with separate success and failure TTLs. Application
`capability_overrides` take precedence over live and bundled data.

Use an override when an enterprise gateway, fine-tuned model, or newly released
model behaves differently from bundled metadata:

```python
defaults = GenerationDefaults(
    capability_overrides={
        "my-provider:my-model": {
            "tool_call": True,
            "reasoning": True,
            "reasoning_levels": ["low", "high"],
            "max_output_tokens": 16_384,
        }
    }
)
```

Reasoning is configured without provider-specific request fields:

```python
result = client.generate(
    model="gpt-5",
    messages=[...],
    max_output_tokens=16_384,
    reasoning=ReasoningConfig(
        enabled=True,                       # False disables optional thinking
        effort=ReasoningEffortValue.HIGH,
        trace="auto",                      # auto/summary/full/none
        history="provider_default",
    ),
)

print(result.reasoning_trace.text if result.reasoning_trace else None)
if result.usage:
    print(result.usage.thinking_tokens)  # backwards-compatible convenience value
    print(result.usage.reasoning.billed_tokens if result.usage.reasoning else None)
    print(result.usage.reasoning.visible_tokens if result.usage.reasoning else None)
```

The trace setting is a request, not a guarantee. Some providers return a
summary, some return visible thought blocks, and some only report reasoning
token usage. A successful reasoning request can therefore have
`reasoning_trace=None`. That is expected when the provider does not expose a
trace for the model or request.

Provider translation currently follows these rules:

| Adapter | Output/reasoning translation |
| --- | --- |
| OpenAI / Azure | Responses uses `max_output_tokens` plus a reasoning object; chat uses reasoning effort and negotiates its token-limit field. |
| Anthropic | Uses adaptive thinking when available or manual `budget_tokens`; provider thinking/signature/redacted blocks are normalized. |
| Google Gemini / Vertex | Uses `max_output_tokens` plus `thinking_level` or `thinking_budget`, with `include_thoughts` controlled by trace policy. Gemini 2.5 budget models map normalized effort to a bounded budget. |
| Amazon Bedrock | Uses Converse inference limits and provider-specific thinking fields in additional model request data. |
| DeepSeek | Uses its `thinking` control. In adaptive mode, a forced tool choice on a thinking model becomes `auto` because DeepSeek rejects that combination; strict mode raises before sending. |
| OpenRouter / Cerebras | Uses the OpenAI-compatible request surface and augments bundled capabilities with live model metadata. |
| Fireworks / Together / LM Studio / LiteLLM | Uses the OpenAI-compatible surface while keeping application defaults and normalized responses consistent. |
| ChatGPT | Uses its Responses-style backend. The backend requires streaming and does not accept output-limit or temperature controls, so this adapter aggregates the required stream and documents the ignored fields. |

If a manual budget is required, llmai uses a metadata-aware default and reserves
at least 1,024 tokens for visible output. An explicit invalid budget raises
instead of silently changing caller intent.

`usage.reasoning.billed_tokens` is populated when a provider reports native
reasoning usage. A visible trace is also estimated as
`ceil(len(utf8_bytes) / 4)` and stored separately in
`usage.reasoning.visible_tokens`; it is never presented as an exact hidden or
billed count. Streaming thinking chunks carry a per-chunk estimate, and the
completion contains normalized final usage.

Displayable `reasoning_trace` is deliberately separate from opaque
`reasoning_state` (Anthropic signatures/redacted blocks, OpenAI encrypted
reasoning, Bedrock reasoning blocks, and Gemini thought signatures). Use
`safe_model_dump()` for logs or user-facing persistence and
`lossless_model_dump()` only when continuation state must be retained for tool
round trips.

Treat lossless dumps as sensitive application data. Opaque state is needed for
some multi-turn tool/reasoning continuations, but it is neither meaningful nor
appropriate to display as a thinking trace.

## Live Model Availability

The bundled metadata above describes known models; it does not prove that a
specific API key, account, or self-hosted endpoint can access them. Use the live
listing API for credential validation and model pickers:

```python
import asyncio

from llmai import OpenAIClientConfig, alist_available_models


async def main():
    models = await alist_available_models(
        config=OpenAIClientConfig(
            api_key="<your-openai-api-key>",
        )
    )
    print(models)


asyncio.run(main())
```

Synchronous applications can call `list_available_models(config=...)`.
Configured provider clients also expose `list_available_models()`, and async
clients expose `await alist_available_models()`. OpenAI, Anthropic, Google,
DeepSeek, OpenRouter, Cerebras, Fireworks, Together AI, LM Studio, and LiteLLM
use live provider APIs. An `OpenAIClientConfig` with a custom `base_url` also
supports OpenAI-compatible services such as Ollama and vLLM.

Failures use the same normalized `LLMAuthenticationError`,
`LLMRateLimitError`, `LLMConnectionError`, `LLMConfigurationError`, and
`LLMError` classes as generation calls. Their `status_code`, `message`, and
`provider` fields are suitable for API-layer translation. Providers without a
live listing implementation raise `LLMConfigurationError` instead of silently
falling back to the bundled metadata.

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

## Async Usage

Async clients are separate from their synchronous counterparts, so adding async
support does not change existing applications:

```python
import asyncio

from llmai import AsyncOpenAIClient, OpenAIClientConfig
from llmai.shared import UserMessage


async def main():
    async with AsyncOpenAIClient(
        config=OpenAIClientConfig(api_key="<your-openai-api-key>"),
    ) as client:
        result = await client.agenerate(
            model="your-openai-model",
            messages=[UserMessage(content="Explain async I/O briefly.")],
        )
        print(result.content)


asyncio.run(main())
```

Use `get_async_client()` when the provider is selected dynamically:

```python
from llmai import OpenAIClientConfig, get_async_client

client = get_async_client(
    config=OpenAIClientConfig(api_key="<your-openai-api-key>"),
)
```

Independent calls can run concurrently:

```python
first, second = await asyncio.gather(
    client.agenerate(model="your-model", messages=first_messages),
    client.agenerate(model="your-model", messages=second_messages),
)
await client.aclose()
```

OpenAI-compatible providers, Azure, Anthropic, Google, and Vertex use their
provider SDK's native async transport. Bedrock uses `asyncio.to_thread` because
boto3 does not provide an async client. This prevents Bedrock calls from blocking
the event loop, although an already-running boto3 operation cannot be forcibly
cancelled.

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

## OpenRouter

```python
from llmai import OpenRouterClient, OpenRouterClientConfig
from llmai.shared import UserMessage


client = OpenRouterClient(
    config=OpenRouterClientConfig(api_key="<your-openrouter-api-key>"),
)

result = client.generate(
    model="openai/gpt-5.4-mini",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

`OpenRouterClient` uses the OpenAI SDK against OpenRouter's OpenAI-compatible chat-completions API. The default base URL is `https://openrouter.ai/api/v1`.

## Fireworks

```python
from llmai import FireworksClient, FireworksClientConfig
from llmai.shared import UserMessage


client = FireworksClient(
    config=FireworksClientConfig(api_key="<your-fireworks-api-key>"),
)

result = client.generate(
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

`FireworksClient` uses the OpenAI SDK against Fireworks' OpenAI-compatible chat-completions API. The default base URL is `https://api.fireworks.ai/inference/v1`.

## Together AI

```python
from llmai import TogetherAIClient, TogetherAIClientConfig
from llmai.shared import UserMessage


client = TogetherAIClient(
    config=TogetherAIClientConfig(api_key="<your-together-api-key>"),
)

result = client.generate(
    model="openai/gpt-oss-20b",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

`TogetherAIClient` uses the OpenAI SDK against Together AI's OpenAI-compatible chat-completions API. The default base URL is `https://api.together.ai/v1`. Together AI does not support the OpenAI Responses API, so this adapter always uses chat completions.

## LM Studio

```python
from llmai import LMStudioClient, LMStudioClientConfig
from llmai.shared import UserMessage


client = LMStudioClient(
    config=LMStudioClientConfig(
        base_url="http://localhost:1234",
    ),
)

result = client.generate(
    model="openai/gpt-oss-20b",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

`LMStudioClient` uses the OpenAI SDK against LM Studio's OpenAI-compatible chat-completions endpoint. The default base URL is `http://localhost:1234/v1`, custom base URLs have `/v1` appended automatically when omitted, and `api_key` is optional. Schemas are reduced to LM Studio's grammar-friendly core fields, so unsupported regex `pattern` constraints are removed before sending.

## Cerebras

```python
from llmai import CerebrasClient, CerebrasClientConfig
from llmai.shared import UserMessage


client = CerebrasClient(
    config=CerebrasClientConfig(api_key="<your-cerebras-api-key>"),
)

result = client.generate(
    model="llama-4-scout-17b-16e-instruct",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

`CerebrasClient` uses the OpenAI SDK against Cerebras' OpenAI-compatible API. The default base URL is `https://api.cerebras.ai/v1`.

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

## LiteLLM

```python
from llmai import LiteLLMClient, LiteLLMClientConfig
from llmai.shared import UserMessage


client = LiteLLMClient(
    config=LiteLLMClientConfig(
        api_key="litellm-proxy-key",
        base_url="https://litellm.example/v1",
    )
)

result = client.generate(
    model="gpt-5.4-mini",
    messages=[
        UserMessage(content="Write a two-line poem about clean interfaces."),
    ],
)

print(result.content)
```

`LiteLLMClient` uses the OpenAI Python client against an OpenAI-compatible LiteLLM proxy. Set `LiteLLMClientConfig(api_type=OpenAIApiType.RESPONSES)` to call the proxy's Responses API instead of chat completions. Pass the proxy key and URL through `api_key` and `base_url`; config-level `extra_kwargs` and per-call `generate(..., extra_body={...})` are forwarded as request `extra_body`.

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

Use explicit content parts when you need multimodal inputs or want to mix text
with images in one message. Normal completion content is surfaced as
`list[TextContentPart | ImageContentPart]` when the provider returns message
content, including text-only replies. Provider-exposed reasoning is available
as `AssistantReasoningItem` entries on `ResponseContent.thinking` and through
the normalized `reasoning_trace`; it is also retained on the final
`AssistantMessage`.

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

support = client.supports_tool_call("your-openai-model")
if support is False:
    raise RuntimeError("Choose a model that supports tools")
# support=None means metadata is inconclusive; adaptive mode can still try the call.

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

Async clients expose the same chunks through direct async iteration:

```python
from llmai import AsyncAnthropicClient, AnthropicClientConfig
from llmai.shared import UserMessage

async with AsyncAnthropicClient(
    config=AnthropicClientConfig(api_key="<your-anthropic-api-key>"),
) as client:
    async for chunk in client.agenerate(
        model="your-anthropic-model",
        messages=[UserMessage(content="Explain recursion briefly.")],
        stream=True,
    ):
        if chunk.type == "content":
            print(chunk.chunk, end="")
```

## Verification and Reviewer Guide

The compatibility work is covered by deterministic tests in
`tests/test_generation.py` in addition to the existing client, async,
message, model, availability, and tool suites. The tests mock provider
transports and verify request translation as well as response normalization;
they do not require credentials or spend tokens.

Run the complete offline review gate from the repository root:

```bash
uv sync
uv run python -m unittest discover -s tests
uvx ruff check --select F llmai tests/test_generation.py
uv run python -m compileall -q llmai tests
git diff --check
```

At the time of this change, the result is **283 passing tests**. Coverage added
for this work includes:

- profile/default/call precedence and safety caps
- strict, adaptive, and off validation behavior
- tri-state tool and reasoning capabilities plus application overrides
- reasoning level adaptation and required default budgets
- reasoning enable/disable behavior, including models where thinking is mandatory
- exact provider usage versus approximate visible-trace token accounting
- streaming reasoning chunks and final normalized usage
- safe versus lossless serialization of opaque continuation state
- OpenAI-compatible output-token field negotiation and retry boundaries
- Google SDK `models/` prefix normalization and Gemini 2.5 effort-to-budget mapping
- DeepSeek thinking plus forced-tool compatibility in adaptive mode
- sync/async request parity and provider-specific parsers

The scoped Ruff command above passes for the package and the new generation
tests. A broader `ruff check --select F llmai tests` also sees six pre-existing
unused `marker_chunks` assignments in `tests/test_clients.py`; that unrelated
test file is not changed by this work.

Live BYOK smoke checks were also run with the locally configured credentials.
Vertex was explicitly excluded from this pass; Google below means the direct
Gemini API, not Vertex AI.

| Provider and tested model | Plain sync | Reasoning/accounting | Forced tool | Async stream |
| --- | --- | --- | --- | --- |
| OpenAI — `gpt-5.4-mini` | Pass | Pass | Pass | Pass |
| Google Gemini — `gemini-2.5-flash` | Pass | Pass, visible trace and accounting observed | Pass | Pass |
| Anthropic — `claude-sonnet-4-6` | Pass | Pass; visible trace remains provider/request dependent | Pass | Pass |
| OpenRouter — `openai/gpt-4.1-mini` | Pass | Pass; selected model did not emit reasoning | Pass | Pass |
| DeepSeek — `deepseek-v4-flash` | Pass | Pass, visible trace and accounting observed | Pass after documented adaptive choice handling | Pass |
| Cerebras — `gpt-oss-120b` | Pass | Pass, visible trace and accounting observed | Pass | Pass |

These live checks validate the selected account/model combinations, not every
model exposed by each provider. Availability, rollout, quotas, and accepted
features can differ by key. A reviewer should use `list_available_models()`
first and override the model constants used by the scripts in `dev/` rather
than assuming a bundled model is enabled for their account. The expected key
names are `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, and `CEREBRAS_API_KEY`.

Live checks should remain opt-in because they can consume quota and incur cost.
Never commit `.env`, print keys in test output, or confuse successful model
listing with successful generation—the latter can still fail because of model
permissions, quota, or feature-specific restrictions.

## Package Layout

- `llmai/openai`: OpenAI adapter
- `llmai/azure`: Azure OpenAI adapter
- `llmai/vertex`: Vertex AI adapter
- `llmai/deepseek`: DeepSeek adapter
- `llmai/openrouter`: OpenRouter adapter
- `llmai/cerebras`: Cerebras adapter
- `llmai/fireworks`: Fireworks adapter
- `llmai/togetherai`: Together AI adapter
- `llmai/lmstudio`: LM Studio adapter
- `llmai/chatgpt`: ChatGPT backend adapter
- `llmai/anthropic`: Anthropic adapter
- `llmai/google`: Google Gemini adapter
- `llmai/bedrock`: Amazon Bedrock adapter
- `llmai/litellm`: LiteLLM adapter
- `llmai/capabilities.py`: bundled/live capability normalization and lookup
- `llmai/shared/generation.py`: provider-neutral defaults, profiles, validation,
  and request preparation
- `llmai/shared`: common message, tool, schema, reasoning, and response models

## Core Types

The shared layer includes the main primitives you will use across providers:

- `UserMessage`, `SystemMessage`, `AssistantMessage`
- `TextContentPart`, `ImageContentPart`
- `Tool`, `WebSearchTool`, `ToolResponseMessage`
- `JSONSchemaResponse`, `JSONObjectResponse`, `TextResponse`
- `GenerationDefaults`, `GenerationProfile`, `ValidationMode`,
  `PreparedGeneration`
- `ReasoningConfig`, `ReasoningEffortValue`, `ReasoningTrace`,
  `ReasoningState`, `ReasoningUsage`
- `ModelCapabilities`, `CapabilityValue`, `CapabilityStatus`, `GenerationWarning`
- `ResponseContent`, `ResponseStreamChunk`, `ResponseStreamContentChunk`,
  `ResponseStreamThinkingChunk`, `ResponseStreamToolChunk`,
  `ResponseStreamToolCompleteChunk`, `ResponseStreamCompletionChunk`
- `AsyncBaseClient`, `AsyncResponseResult`
- `ResponseUsage`

`UserMessage.content` accepts either a plain string or explicit content parts.
`SystemMessage.content` is always a plain string.
