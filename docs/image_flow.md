# Image flow by provider

This document describes the provider-neutral image input and the Google Gemini
and Vertex AI serialization behavior.

## Provider-neutral image input

Images are represented by `ImageContentPart` using exactly one source:

- `url` for a remote URI or an image data URI;
- `data` for raw bytes, together with a required `mime_type`.

Supplying both sources or neither source is rejected. Byte-backed images without
a MIME type are also rejected.

## Google Gemini and Vertex AI

Vertex AI inherits the Google Gemini image conversion.

Google image sources are serialized as follows:

| Input | Google part |
| --- | --- |
| HTTPS, File API, YouTube, or `gs://` URI | `file_data` via `Part.from_uri(...)` |
| `data:image/...;base64,...` URI | decoded `inline_data` via `Part.from_bytes(...)` |
| Percent-encoded `data:image/...,...` URI | decoded `inline_data` via `Part.from_bytes(...)` |
| Raw `data` plus `mime_type` | `inline_data` via `Part.from_bytes(...)` |

Image data URIs are decoded locally because Google's `file_uri` field accepts
remote/file URIs, not `data:` URIs. The MIME type is read from the data URI, or
from `ImageContentPart.mime_type` when the URI omits it.

Malformed base64, a missing image MIME type, a missing comma separator, or an
empty payload raises an `LLMConfigurationError` before a provider request is
made.

## Source of truth

Image input validation is implemented in
[`llmai/shared/messages.py`](../llmai/shared/messages.py). Google and Vertex AI
serialization is implemented in
[`llmai/google/client.py`](../llmai/google/client.py).
