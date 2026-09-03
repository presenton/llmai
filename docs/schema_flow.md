# Schema flow by provider

This document describes how `llmai` changes a JSON Schema before sending it to
each provider. It covers both schema entry points:

- output schemas supplied through `JSONSchemaResponse(json_schema=..., strict=...)`
- function-tool input schemas supplied through
  `Tool(schema=..., strict=...)`

`strict` in this document is the value on the relevant `JSONSchemaResponse` or
`Tool`. Both classes default it to `false`. Output strictness and tool
strictness are independent.

## Common input handling

Before provider-specific handling, `llmai` turns the schema into a dictionary:

- a `dict` is deep-copied;
- a Pydantic model class or instance is converted with `model_json_schema()`;
- `None` becomes an empty schema (`{}`) where a schema is required.

This initial conversion does not change based on `strict`. Strictness only
affects the later provider-specific processing and, where supported, the
`strict` value sent on the wire.

The transformations named below mean:

- **Filter fields**: recursively remove schema keywords outside the provider's
  allowlist. Property names and names inside `$defs`/`definitions` are not
  treated as schema keywords and are preserved.
- **Filter string formats**: remove `format` from a schema whose `type` is
  `string` when the format is outside the provider's allowlist.
- **Flatten `allOf`**: merge object-valued `allOf` members into their parent.
  `properties` are combined and `required` values are deduplicated. Later
  values override earlier values for other keys.
- **Flatten refs**: resolve local JSON Pointers, inline their schemas, and
  normally remove `$defs`/`definitions`. Recursive or unresolved references
  are retained so the result remains representable.
- **Close objects**: set `additionalProperties: false` on every schema node
  whose `type` is `object`, including nested objects and objects in definitions.
  This overwrites an existing `additionalProperties` value.
- **Require properties**: replace every object's `required` array with all keys
  declared by that object's `properties`, including nested objects, array item
  objects, and objects in definitions.
- **Remove additional-properties keywords**: recursively remove both
  `additionalProperties` and `additional_properties`.
- **Collapse `anyOf`**: when every member has the same string-valued `type`,
  replace `anyOf` with that common type; if every member also has an `enum`,
  combine the enum values.

## Summary matrix

| Provider | Output schema, `strict=false` | Output schema, `strict=true` | Tool schema, `strict=false` | Tool schema, `strict=true` |
| --- | --- | --- | --- | --- |
| OpenAI | Unchanged; send `strict: false` | OpenAI strict processing; send `strict: true` | Unchanged; send `strict: false` | OpenAI strict processing; send `strict: true` |
| Azure OpenAI | Same as OpenAI | Same as OpenAI | Same as OpenAI | Same as OpenAI |
| Together AI | Same as OpenAI | Same as OpenAI | Same as OpenAI | Same as OpenAI |
| OpenRouter | Unchanged; send `strict: false` | OpenAI strict processing; send `strict: true` | Unchanged; send `strict: false` | OpenAI strict processing; send `strict: true` |
| LiteLLM | Unchanged; send `strict: false` | Unchanged; send `strict: true` | Unchanged; send `strict: false` | Unchanged; send `strict: true` |
| ChatGPT | Unchanged; send `strict: false` | Unchanged; send `strict: true` | Unchanged; send `strict: false` | Unchanged; send `strict: true` |
| Cerebras | Unchanged; send `strict: false` | Cerebras strict processing; send `strict: true` | Usually unchanged; see request-wide rule below | Cerebras strict processing; all tools become strict |
| Fireworks | Fireworks processing; no output `strict` field | Same processing; no output `strict` field | Fireworks processing; send `strict: false` | Same processing; send `strict: true` |
| LM Studio | LM Studio processing; send `strict: false` | Same processing; send `strict: true` | LM Studio processing; send `strict: false` | Same processing; send `strict: true` |
| DeepSeek | Unchanged internal response-tool schema; response tool is non-strict | Unchanged internal response-tool schema; strict only on the beta endpoint | Unchanged; send `strict: false` | Unchanged; send `strict: true` |
| Anthropic | Unchanged internal response-tool schema; send `strict: false` | Anthropic strict processing on the internal tool; send `strict: true` | Unchanged; send `strict: false` | Anthropic strict processing; send `strict: true` |
| Google Gemini | Unchanged; no strict marker | Unchanged; no strict marker | Google tool processing; no strict marker | Same processing; no strict marker |
| Vertex AI | Same as Google Gemini | Same as Google Gemini | Same as Google Gemini | Same as Google Gemini |
| Amazon Bedrock | Bedrock processing; no output strict marker | Same processing; no output strict marker | Bedrock processing; send `strict: false` | Same processing; send `strict: true` |

“Unchanged” means unchanged after the common conversion to a dictionary. The
provider SDK may still perform its own serialization or validation.

## Provider details

### OpenAI, Azure OpenAI, and Together AI

Azure OpenAI and Together AI inherit OpenAI's schema behavior without
overriding it. The same processing is used by both the Chat Completions and
Responses API paths; only their surrounding request envelopes differ.

With `strict=false`, output and tool schemas are forwarded unchanged and the
corresponding wire object contains `strict: false`.

With `strict=true`, output and tool schemas are processed as follows:

1. Convert `oneOf` to `anyOf` and convert `const` to a typed, single-value
   `enum`.
2. Flatten `allOf`, including resolving a `$ref` used inside an `allOf`.
3. Keep only these schema fields:
   `$defs`, `$ref`, `additionalProperties`, `anyOf`, `description`, `enum`,
   `exclusiveMaximum`, `exclusiveMinimum`, `format`, `items`, `maxItems`,
   `maximum`, `minItems`, `minimum`, `maxLength`, `minLength`, `multipleOf`,
   `pattern`, `properties`, `required`, and `type`.
4. For string schemas, keep only these formats: `date-time`, `time`, `date`,
   `duration`, `email`, `hostname`, `ipv4`, `ipv6`, and `uuid`.
5. Close every object with `additionalProperties: false`.
6. Require every declared object property.
7. Send `strict: true`.

OpenAI strict schemas cannot omit a declared property. Optional semantics must
instead be represented by a required nullable property, such as a union with
`null`.

General `$defs` and `$ref` values are preserved rather than flattened.

### OpenRouter and LiteLLM

Both adapters use the OpenAI-compatible request shape. LiteLLM deliberately
bypasses OpenAI's schema cleanup and forwards schemas unchanged in both modes.
OpenRouter forwards non-strict schemas unchanged, but applies the OpenAI strict
processing described above when `strict=true`. The requested `strict` boolean
is still included on the wire.

Consequently, the model/provider selected behind LiteLLM, or behind OpenRouter
in non-strict mode, is responsible for accepting the schema keywords and
enforcing strictness.

### ChatGPT

The ChatGPT adapter uses the Responses-style request shape. It forwards both
output and tool schemas unchanged in both modes and includes the requested
`strict` boolean on the corresponding format or function definition.

Unlike the OpenAI API adapter, it does not apply OpenAI's strict-schema field,
format, `allOf`, or `additionalProperties` transformations.

### Cerebras

Output schemas follow the OpenAI-compatible request shape:

- `strict=false`: forward the schema unchanged and send `strict: false`.
- `strict=true`: flatten `allOf`, filter fields, close all objects, and send
  `strict: true`.

The Cerebras strict field allowlist is: `$defs`, `$ref`,
`additionalProperties`, `anyOf`, `enum`, `exclusiveMaximum`,
`exclusiveMinimum`, `items`, `maximum`, `minimum`, `multipleOf`, `prefixItems`,
`properties`, `required`, and `type`. Cerebras does not apply a separate string
format allowlist because `format` itself is not in the field allowlist and is
therefore removed in strict mode.

Tool strictness is request-wide in this adapter. Before serializing tools,
`llmai` computes `effective_strict = any(tool.strict for tool in tools)`:

- if every tool has `strict=false`, every tool schema is unchanged and every
  tool is sent with `strict: false`;
- if any tool has `strict=true`, every tool schema receives Cerebras strict
  processing and every tool is sent with `strict: true`, including tools that
  were declared non-strict.

### Fireworks

Fireworks applies the same schema processing in both modes, for output schemas
and tool schemas:

1. Flatten local refs and `allOf`.
2. Remove all additional-properties keywords.
3. Keep only `$defs`, `$ref`, `anyOf`, `definitions`, `description`, `enum`,
   `items`, `properties`, `required`, and `type`.

Because refs are flattened, `$defs`/`definitions` usually disappear unless an
unresolved or recursive reference still needs them.

For JSON-schema output, Fireworks omits the `strict` field entirely, so
`strict=true` and `strict=false` produce the same output-schema request. For
function tools, the schema is the same in both modes but the tool's requested
`strict` boolean is still sent.

### LM Studio

LM Studio applies the same processing to output and tool schemas in both
modes:

1. Flatten `allOf`.
2. Keep only `$defs`, `$ref`, `additionalProperties`, `anyOf`, `description`,
   `enum`, `items`, `maxItems`, `minItems`, `properties`, `required`, and
   `type`.

Refs are not generally flattened. The requested `strict` boolean is included
on output formats and function tools, but it does not change schema processing.

### DeepSeek

DeepSeek does not send `JSONSchemaResponse` through OpenAI's native
`response_format={"type": "json_schema"}`. Instead, `llmai` creates an internal
function tool whose input schema is the requested output schema, steers the
model toward that tool, and converts the internal tool arguments back into the
response content. The internal tool is hidden from returned user tool calls.

Neither output schemas nor ordinary tool schemas are transformed in either
mode. They retain such keywords as `allOf`, `maxItems`, and their original
`additionalProperties` state.

For the internal output tool:

- `strict=false` sends `strict: false`;
- `strict=true` sends `strict: true` only when the configured base URL ends in
  `/beta`; otherwise it sends `strict: false`.

Ordinary tools carry their declared `strict` boolean. DeepSeek's server-side
strict-tool support requires its beta endpoint, so a strict ordinary tool sent
to the stable endpoint may be rejected or may not receive strict enforcement.

### Anthropic

Anthropic also implements `JSONSchemaResponse` as an internal function tool.
The internal tool arguments become final structured content and the tool is
not exposed as a user tool call.

For output and ordinary tool schemas:

- `strict=false`: forward the schema unchanged and send `strict: false`;
- `strict=true`: filter fields and string formats, close every object, and send
  `strict: true`.

Anthropic's strict field allowlist is: `$defs`, `$ref`,
`additionalProperties`, `allOf`, `anyOf`, `description`, `enum`, `format`,
`items`, `properties`, `required`, `title`, and `type`.

Its strict string-format allowlist is: `date-time`, `time`, `date`, `duration`,
`email`, `hostname`, `uri`, `ipv4`, `ipv6`, and `uuid`.

If Anthropic rejects a request because strict tools are unsupported, `llmai`
retries the whole request with `strict=false`. On that retry, both ordinary
tools and the internal response tool use the original, unprocessed schemas.

### Google Gemini and Vertex AI

Vertex AI inherits Google Gemini's schema behavior without overriding it.

Output schemas are passed unchanged as `response_json_schema` in both modes.
No strict marker is sent, so the output schema request is identical for
`strict=true` and `strict=false`.

Tool schemas always receive Google tool processing, regardless of strictness:

1. Flatten local refs and `allOf`.
2. Remove all additional-properties keywords.
3. Keep only `additionalProperties`, `anyOf`, `default`, `description`, `enum`,
   `example`, `format`, `items`, `maxItems`, `maxLength`, `maxProperties`,
   `maximum`, `minItems`, `minLength`, `minProperties`, `minimum`, `nullable`,
   `pattern`, `properties`, `propertyOrdering`, `required`, `title`, and
   `type`.

The allowlist contains `additionalProperties`, but the later removal step means
it is absent from the final tool schema. No per-tool strict marker is sent, so
tool requests are identical in both modes.

### Amazon Bedrock

Bedrock applies the same processing to output and tool schemas in both modes:

1. Flatten local refs and `allOf`.
2. Collapse compatible `anyOf` groups.
3. Keep only `$defs`, `$ref`, `additionalProperties`, `description`, `enum`,
   `items`, `properties`, `required`, and `type`.
4. Close every object with `additionalProperties: false`.

Output schemas are serialized into the native `outputConfig.textFormat`
JSON-schema structure. There is no output strict marker, so strictness does not
change either the processed schema or its envelope.

Tool schemas use the same processing in both modes. The tool specification
still includes the tool's declared `strict` boolean.

## Source of truth

The behavior above is implemented in:

- [`llmai/shared/schema.py`](../llmai/shared/schema.py) for common conversion and
  transformations;
- [`llmai/openai/client.py`](../llmai/openai/client.py) for OpenAI and inherited
  OpenAI-compatible behavior;
- each provider's `llmai/<provider>/client.py` for overrides and native request
  formats.

When changing schema processing, update this document together with the
provider implementation and its tests.
