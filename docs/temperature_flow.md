# Temperature flow by provider

This document describes what happens when `temperature` is omitted or supplied
to each `llmai` provider.

## Common behavior

Temperature is intentionally much less abstracted than reasoning or output
limits:

- there is no temperature value in `GenerationDefaults`;
- generation profiles do not set or change temperature;
- capability metadata does not adapt temperature;
- `llmai` does not validate, clamp, or normalize its numeric range;
- the same behavior is used for streaming and non-streaming requests.

Therefore, `temperature=None` means “do not choose a value in `llmai`,” while a
non-`None` value is generally forwarded unchanged. The provider SDK or API may
still reject a value or apply model-specific restrictions.

## Summary matrix

| Provider | If temperature is not set | If temperature is set |
| --- | --- | --- |
| OpenAI Chat Completions | Use the SDK omit sentinel | Forward unchanged, including `0` |
| OpenAI Responses | Use the SDK omit sentinel | Forward unchanged, including `0` |
| Azure OpenAI | Same as the selected OpenAI API type | Same as the selected OpenAI API type |
| Together AI | Same as OpenAI Chat Completions | Forward unchanged, including `0` |
| OpenRouter | Same as OpenAI Chat Completions | Forward unchanged, including `0` |
| Cerebras | Same as OpenAI Chat Completions | Forward unchanged, including `0` |
| Fireworks | Same as OpenAI Chat Completions | Forward unchanged, including `0` |
| LM Studio | Same as OpenAI Chat Completions | Forward unchanged, including `0` |
| LiteLLM | Same as its selected OpenAI API type | Same as its selected OpenAI API type |
| ChatGPT | Always omitted | Ignored and omitted |
| DeepSeek | Use the SDK omit sentinel | Forward unchanged, including `0` |
| Anthropic | Use the SDK omit sentinel | Forward only truthy values; `0` is omitted |
| Google Gemini | Put `None` in generation config for SDK serialization | Put the supplied value in generation config, including `0` |
| Vertex AI | Same as Google Gemini | Same as Google Gemini |
| Amazon Bedrock | Leave temperature out of `inferenceConfig` | Add unchanged to `inferenceConfig`, including `0` |

## Provider details

### OpenAI

OpenAI behavior depends on `OpenAIApiType`:

- Chat Completions uses the OpenAI SDK's omit sentinel when the value is
  `None`, so the request leaves `temperature` out instead of sending JSON
  `null`. Any explicit value, including `0`, is passed unchanged.
- Responses explicitly uses the OpenAI SDK's omit sentinel when the value is
  `None`. Any explicit value, including `0`, is passed unchanged.

`llmai` does not automatically remove temperature for reasoning models. If a
particular model does not accept it, the provider returns the error.

### Azure OpenAI

Azure inherits the OpenAI behavior for its configured API type. Its default is
Responses, but Chat Completions can also be selected.

### Together AI, OpenRouter, Cerebras, Fireworks, and LM Studio

These adapters use OpenAI-compatible Chat Completions. An unset value uses the
SDK omit sentinel; an explicit value, including `0`, is forwarded unchanged.
The upstream provider or local model determines accepted ranges and whether
temperature is supported.

### LiteLLM

LiteLLM follows the selected OpenAI-compatible API type:

- Completions omits unset temperature and forwards explicit values unchanged;
- Responses omits `None` with the SDK sentinel and forwards explicit values.

The model routed behind LiteLLM determines actual support.

### ChatGPT

The ChatGPT backend used by this adapter does not accept temperature. The
public generation method accepts the argument for interface consistency, but
the adapter discards it and always sends the SDK omit sentinel.

This means `None`, `0`, and any other supplied value produce the same request.

### DeepSeek

DeepSeek uses the OpenAI-compatible Chat Completions path. An unset temperature
uses the SDK omit sentinel; any explicit value, including `0`, is forwarded
unchanged.

No reasoning-specific temperature adjustment is performed.

### Anthropic

Anthropic builds its SDK argument with `temperature or Omit()`:

- `None` is omitted;
- `0` and `0.0` are also omitted because they are falsy;
- nonzero numeric values are forwarded unchanged.

There is no range validation before the SDK call. This behavior applies even
when Anthropic thinking/reasoning is enabled; `llmai` does not force or replace
the value based on thinking mode.

### Google Gemini and Vertex AI

Vertex AI inherits Google Gemini's behavior. The adapter constructs a Google
generation config with `temperature` set to the Python value:

- unset temperature becomes `None` in that config, leaving final omission and
  default behavior to the Google SDK/provider;
- an explicit value, including `0`, is placed in the config unchanged.

### Amazon Bedrock

Bedrock adds `inferenceConfig.temperature` only when temperature is not `None`.
An explicit `0` is included because the check is against `None`, not truthiness.

No temperature key is added when it is unset. Other inference-config fields,
such as the resolved output-token limit, can still cause `inferenceConfig` to
be present.

## Source of truth

Temperature handling is implemented in each provider's
`llmai/<provider>/client.py` and corresponding async adapter. OpenAI-compatible
inheritance begins in
[`llmai/openai/client.py`](../llmai/openai/client.py).

When changing temperature handling, update this document together with the
implementation and tests.
