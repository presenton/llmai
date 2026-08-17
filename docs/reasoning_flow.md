# Reasoning flow by provider

This document describes what `llmai` does when reasoning is not set and when it
is explicitly configured. It covers the provider-neutral preparation step, the
request sent by every provider adapter, and the normalized reasoning returned
to the caller.

The current reasoning inputs are:

- `reasoning=ReasoningConfig(...)`, the preferred provider-neutral interface;
- `reasoning_effort=ReasoningEffort(...)`, the legacy compatibility interface;
- reasoning defaults inside the client's `GenerationDefaults`;
- `profile`, which can imply reasoning behavior.

## What “not set” means

With the default balanced profile, the default empty `ReasoningConfig`, and no
reasoning arguments on the generation call, the prepared reasoning value is
`None`. Most adapters then omit their thinking/reasoning control and let the
provider and model decide.

There are two wire-level exceptions:

- OpenAI's Responses API sends `reasoning: {"summary": "auto"}` even when
  reasoning was not set.
- ChatGPT also sends `reasoning: {"summary": "auto"}` even when reasoning was
  not set.

“Not set on the call” does not necessarily mean provider-default reasoning if
client defaults or a non-balanced profile are configured:

- `FAST` sets `enabled=false` when the configured default did not already
  choose an enabled state.
- `DEEP` sets `enabled=true` and defaults effort to `high` when the configured
  default did not already choose an enabled state.
- `MODEL_MAX` and `BALANCED` do not imply a reasoning change.

Explicit call fields are applied after the profile and client defaults, so they
can override profile-derived values.

## Provider-neutral preparation

Every synchronous and asynchronous provider goes through the same reasoning
preparation before building its request.

### Precedence

Values are resolved in this order, with later values winning:

1. `GenerationDefaults.reasoning`
2. the selected profile's `FAST` or `DEEP` implication
3. legacy `reasoning_effort`
4. fields explicitly set on `reasoning=ReasoningConfig(...)`

The explicit `ReasoningConfig` is merged field by field. An omitted field keeps
the earlier value rather than resetting it.

### Normalization and validation

The normalized fields have these meanings:

- `enabled=None`: let the provider/model decide;
- `enabled=false`: normalize the outgoing effort to `none`;
- `enabled=true`: request reasoning, with an effort or budget if available;
- `effort`: one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`,
  or `default`; `default` is normalized to no explicit effort;
- `budget_tokens`: request a manual reasoning budget;
- `trace`: request no trace, a summary, or fuller visible reasoning;
- `history`: choose whether prior assistant reasoning state is retained.

Model capability metadata can further alter or reject the request:

- In adaptive validation mode, an unsupported effort is changed to the nearest
  advertised effort and a warning is logged. Strict validation raises instead.
- If metadata says reasoning is mandatory, adaptive mode changes an attempted
  disable to the lowest supported effort or budget. Strict mode raises when
  the disable was explicit.
- Enabling reasoning on a model marked as not supporting it logs a warning in
  adaptive mode and raises in strict mode.
- When a manual-budget model needs a budget and none was supplied, `llmai`
  chooses a metadata-bounded budget: normally 4,096 tokens, or 16,384 for the
  `DEEP` profile.
- A reasoning budget must leave at least 1,024 tokens for visible output. An
  invalid explicit budget raises; an inferred/default budget can be clamped
  with a warning.

After preparation, a completely undecided configuration becomes `None`.
Otherwise it becomes a `ReasoningEffort` carrying the resolved effort, budget,
summary request, and trace-inclusion preference.

`trace` controls requested visibility; it does not enable or disable reasoning
by itself. Use `enabled=false` or effort `none` to request disabled reasoning.

## Summary matrix

| Provider | If reasoning is not set | If reasoning is set |
| --- | --- | --- |
| OpenAI Chat Completions | Omit `reasoning_effort` | Send only normalized effort as `reasoning_effort`; budget and trace controls are not sent |
| OpenAI Responses | Send `reasoning.summary=auto` | Send normalized effort and summary; manual budget is not sent |
| Azure OpenAI | Same as the selected OpenAI API type | Same as the selected OpenAI API type |
| Together AI | Same as OpenAI Chat Completions | Same as OpenAI Chat Completions |
| OpenRouter | Same as OpenAI Chat Completions | Same as OpenAI Chat Completions |
| Cerebras | Same as OpenAI Chat Completions | Same as OpenAI Chat Completions |
| Fireworks | Same as OpenAI Chat Completions | Same as OpenAI Chat Completions |
| LM Studio | Same as OpenAI Chat Completions | Same as OpenAI Chat Completions |
| LiteLLM | Same as its selected OpenAI API type | Same as its selected OpenAI API type |
| ChatGPT | Send `reasoning.summary=auto` | Send normalized effort/summary and always default summary to `auto` |
| DeepSeek | Omit `thinking`; model decides | Send only `thinking.type=enabled` or `disabled`; effort, budget size, and trace are not sent |
| Anthropic | Omit `thinking` and `output_config`; model decides | Send disabled, budgeted, or adaptive thinking, plus supported effort control |
| Google Gemini | Omit `thinking_config`; model decides | Send thinking level and/or budget plus visible-thought preference |
| Vertex AI | Same as Google Gemini | Same as Google Gemini |
| Amazon Bedrock | Omit generated `thinking`; model decides | Send disabled, budgeted, or adaptive thinking in additional model request fields |

The table describes normalized reasoning after defaults, profile behavior,
capability adaptation, and validation have run.

## Provider details

### OpenAI

OpenAI behavior depends on `OpenAIApiType`.

For Chat Completions:

- no prepared reasoning: omit `reasoning_effort`;
- prepared effort: send its string value as `reasoning_effort`;
- budget tokens, summary, and trace inclusion are not represented on this path;
- a configuration that only supplies a budget or trace policy can therefore
  produce no Chat Completions reasoning parameter.

For Responses:

- no prepared reasoning: send `reasoning: {"summary": "auto"}`;
- explicit effort: send it as `reasoning.effort`;
- a trace mode of `auto`, `summary`, or `full` maps to `auto`, `concise`, or
  `detailed` summary respectively;
- `trace=none` suppresses the automatically added summary, unless a summary
  was supplied through `extra_body`;
- manual budget tokens are not sent by this adapter.

If `extra_body` contains a `reasoning` dictionary, the Responses adapter merges
it with normalized values. Normalized effort and summary overwrite matching
dictionary fields. A non-dictionary raw reasoning value is forwarded only when
there is no normalized reasoning value.

Responses reasoning items, summaries, encrypted content, and reported token
usage are normalized into `AssistantMessage.thinking` and `ResponseUsage`.
Chat-compatible `reasoning_content` is also normalized when returned.

### Azure OpenAI

Azure inherits OpenAI's reasoning translation. Its default API type is
Responses, but it can be configured for either Responses or Chat Completions.
The corresponding OpenAI rules above apply.

### Together AI, OpenRouter, Cerebras, Fireworks, and LM Studio

These adapters use the inherited OpenAI Chat Completions reasoning path:

- not set: omit `reasoning_effort`;
- set with an effort: send the normalized effort string;
- budget and trace controls are not sent directly.

Provider/model capability validation still runs before this translation. Each
upstream service or selected model ultimately decides which effort values are
accepted and what reasoning content is returned.

Fireworks and LM Studio additionally recognize their provider-specific visible
reasoning fields when parsing responses. Other OpenAI-compatible adapters use
the base OpenAI-compatible reasoning parser.

### LiteLLM

LiteLLM can use either OpenAI-compatible API type:

- Completions follows the Chat Completions rules;
- Responses follows the Responses rules, including the automatic summary when
  reasoning is not set.

The selected model behind LiteLLM determines actual reasoning support.

### ChatGPT

ChatGPT uses its Responses-style backend but has a separate translator.

- It always sends at least `reasoning: {"summary": "auto"}`, including when
  reasoning is not set or `trace=none` was requested.
- A normalized effort is sent as `reasoning.effort`.
- An explicit or trace-derived summary replaces `auto`.
- Manual budget tokens and `include_trace` are not sent.
- A reasoning dictionary in `extra_body` is merged in the same general manner
  as OpenAI Responses, with normalized effort/summary taking precedence.

The backend always streams internally. Reasoning output items are collected
and normalized before a non-streaming result is returned.

### DeepSeek

When reasoning is not set, `llmai` does not add a `thinking` object and lets the
model decide.

When reasoning is set and `extra_body` does not already contain `thinking`:

- effort `none` or a zero budget becomes `thinking: {"type": "disabled"}`;
- every other prepared reasoning configuration becomes
  `thinking: {"type": "enabled"}`.

The specific effort, token count, summary, and trace preference are not sent.
An explicit `extra_body["thinking"]` takes precedence over the normalized
control.

For a model advertised as reasoning-capable, DeepSeek does not support forced
tool choice. Adaptive and off validation modes change a required tool choice to
automatic while preserving the selected tools; strict validation raises. This
check is based on model capability, even if the current request tries to
disable reasoning.

Returned `reasoning_content` is normalized as assistant thinking.

### Anthropic

When reasoning is not set, both `thinking` and reasoning `output_config` are
omitted, leaving behavior to the model.

When set:

- effort `none` or budget `0` sends `thinking: {"type": "disabled"}`;
- a manual budget sends enabled thinking with `budget_tokens`;
- otherwise it sends `thinking: {"type": "adaptive"}`;
- an effort without a manual budget is also sent through `output_config.effort`
  when it survives capability preparation.

Effort values are mapped for Anthropic's output config: `minimal`/`low` become
`low`, `xhigh`/`max` become `max`, and `medium`/`high` remain unchanged.

For budget-oriented models whose effort levels are absent or inferred,
`llmai` converts effort to a bounded budget: 1,024 for minimal, 4,096 for low,
8,192 for medium, 16,384 for high, and 32,768 for xhigh/max.

Anthropic thinking text, signatures, and redacted blocks are normalized and
retained as appropriate for display and continuation.

### Google Gemini and Vertex AI

Vertex AI inherits Google Gemini's reasoning translation.

When reasoning is not set, `thinking_config` is omitted.

When set:

- effort `none` becomes a zero thinking budget when no budget was supplied and
  disables included thoughts;
- minimal/low map to thinking level `LOW`;
- medium maps to `MEDIUM`;
- high/xhigh/max map to `HIGH`;
- a manual token budget is sent as `thinking_budget`;
- requested trace visibility is sent as `include_thoughts`.

For budget-oriented Gemini models whose reasoning levels are inferred, effort
is converted to a bounded budget: 128 for minimal, 1,024 for low, 4,096 for
medium, 8,192 for high, 16,384 for xhigh, and 24,576 for max.

Returned thought parts and thought signatures are normalized into assistant
thinking and continuation state.

### Amazon Bedrock

When reasoning is not set, no generated `thinking` value is added to
`additionalModelRequestFields`.

When set:

- effort `none` or budget `0` sends `thinking: {"type": "disabled"}`;
- a manual budget sends enabled thinking with `budget_tokens`;
- otherwise it sends adaptive thinking and includes the normalized effort when
  present.

An explicit `extra_body["thinking"]` takes precedence. Bedrock reasoning text,
signatures, and redacted content are normalized from Converse responses.

## Trace, usage, and history

A trace request is not a guarantee that a visible trace will be returned.
Providers may return visible thoughts, a summary, opaque continuation state,
only billed reasoning-token usage, or nothing visible.

`llmai` keeps these concepts separate:

- displayable reasoning is exposed through normalized assistant thinking and
  reasoning-trace helpers;
- opaque signatures, encrypted content, and redacted state are retained for
  lossless continuation where providers require them;
- provider-reported billed reasoning tokens and locally estimated visible
  tokens are recorded separately.

`ReasoningHistoryMode.DISABLED` removes prior assistant thinking and tool-call
thought signatures before sending messages. Other history modes preserve the
messages for provider-specific serialization.

## Source of truth

The behavior above is implemented in:

- [`llmai/shared/generation.py`](../llmai/shared/generation.py) for defaults,
  profiles, capability adaptation, and validation;
- [`llmai/shared/reasoning.py`](../llmai/shared/reasoning.py) for normalized
  reasoning models;
- each provider's `llmai/<provider>/client.py` for wire translation and response
  parsing.

When changing reasoning behavior, update this document together with the
implementation and tests.
