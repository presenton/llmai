from __future__ import annotations

import json
import os
from logging import Logger
from time import perf_counter
from typing import Any

import boto3

from llmai.shared.base import BaseClient
from llmai.shared.errors import LLMError, configuration_error, raise_llm_error
from llmai.shared.messages import (
    AssistantMessage,
    AssistantToolCall,
    ImageContentPart,
    Message,
    MessageContent,
    SystemMessage,
    TextContentPart,
    ToolResponseMessage,
    UserMessage,
    collapse_content_parts,
    content_has_images,
    normalize_content_parts,
)
from llmai.shared.response_formats import (
    JSONSchemaResponse,
    JSONObjectResponse,
    ResponseFormat,
    get_response_schema,
)
from llmai.shared.responses import (
    ResponseContent,
    ResponseStreamCompletionChunk,
    ResponseStreamContentChunk,
    ResponseUsage,
)
from llmai.shared.schema import get_schema_as_dict
from llmai.shared.tools import Tool, ToolChoice, resolve_tools


class BedrockClient(BaseClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)

        explicit_aws_auth = any(
            value is not None
            for value in (
                aws_access_key_id,
                aws_secret_access_key,
                aws_session_token,
                profile_name,
            )
        )
        if api_key and explicit_aws_auth:
            raise configuration_error(
                "Provide either api_key or AWS credentials/profile, not both",
                provider="bedrock",
            )

        if (aws_access_key_id is None) != (aws_secret_access_key is None):
            raise configuration_error(
                "aws_access_key_id and aws_secret_access_key must be provided together",
                provider="bedrock",
            )

        try:
            if api_key is not None:
                # Amazon Bedrock's Boto3 API key flow relies on the bearer token env var.
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
                profile_name=profile_name,
            )
            self._client = session.client("bedrock-runtime", region_name=region_name)
        except Exception as exc:
            raise_llm_error(exc, provider="bedrock")

    def _parse_tool_arguments(self, arguments: str | None) -> dict:
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
        except Exception:
            return {}

        return parsed if isinstance(parsed, dict) else {}

    def _mime_type_to_bedrock_image_format(self, mime_type: str | None) -> str:
        if not mime_type:
            raise LLMError(400, "mime_type is required for Bedrock image content")

        normalized = mime_type.lower()
        if normalized == "image/jpg":
            normalized = "image/jpeg"

        mapping = {
            "image/png": "png",
            "image/jpeg": "jpeg",
            "image/gif": "gif",
            "image/webp": "webp",
        }
        if normalized not in mapping:
            raise LLMError(
                400,
                "Bedrock Converse only supports png, jpeg, gif, and webp images",
            )

        return mapping[normalized]

    def _bedrock_image_format_to_mime_type(self, image_format: str | None) -> str | None:
        if not image_format:
            return None
        return f"image/{image_format}"

    def _image_content_part_to_bedrock_image(
        self,
        part: ImageContentPart,
    ) -> dict[str, object]:
        image = {
            "format": self._mime_type_to_bedrock_image_format(part.mime_type),
        }

        if part.url is not None:
            if not part.url.startswith("s3://"):
                raise LLMError(
                    400,
                    "Bedrock Converse only supports image bytes or s3:// image URIs",
                )
            image["source"] = {"s3Location": {"uri": part.url}}
            return image

        image["source"] = {"bytes": part.data or b""}
        return image

    def _bedrock_image_to_content_part(
        self,
        image: dict[str, object] | None,
    ) -> ImageContentPart | None:
        if not image:
            return None

        mime_type = self._bedrock_image_format_to_mime_type(image.get("format"))
        source = image.get("source") or {}
        if not isinstance(source, dict):
            return None

        if source.get("bytes") is not None:
            return ImageContentPart(data=source["bytes"], mime_type=mime_type)

        s3_location = source.get("s3Location") or {}
        if isinstance(s3_location, dict) and s3_location.get("uri"):
            return ImageContentPart(url=s3_location["uri"], mime_type=mime_type)

        return None

    def _content_to_bedrock_blocks(
        self,
        content: MessageContent | None,
    ) -> list[dict[str, object]]:
        blocks: list[dict[str, object]] = []

        for part in normalize_content_parts(content):
            if isinstance(part, ImageContentPart):
                blocks.append({"image": self._image_content_part_to_bedrock_image(part)})
            else:
                blocks.append({"text": part.text})

        return blocks

    def _tool_response_to_bedrock_content_blocks(
        self,
        content: list[TextContentPart] | None,
    ) -> list[dict[str, object]]:
        return [{"text": part.text} for part in content or []]

    def _assistant_message_to_bedrock_message(
        self,
        message: AssistantMessage,
    ) -> dict[str, object]:
        if content_has_images(message.content):
            raise LLMError(
                400,
                "Bedrock Converse does not support assistant image content in conversation history",
            )

        content_blocks = self._content_to_bedrock_blocks(message.content)
        for tool_call in message.tool_calls:
            content_blocks.append(
                {
                    "toolUse": {
                        "toolUseId": tool_call.id,
                        "name": tool_call.name,
                        "input": self._parse_tool_arguments(tool_call.arguments),
                    }
                }
            )

        return {
            "role": "assistant",
            "content": content_blocks,
        }

    def _messages_to_bedrock_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, object]]:
        bedrock_messages: list[dict[str, object]] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                continue

            if isinstance(message, UserMessage):
                bedrock_messages.append(
                    {
                        "role": "user",
                        "content": self._content_to_bedrock_blocks(message.content),
                    }
                )
                continue

            if isinstance(message, AssistantMessage):
                bedrock_messages.append(
                    self._assistant_message_to_bedrock_message(message)
                )
                continue

            if isinstance(message, ToolResponseMessage):
                bedrock_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": message.id,
                                    "content": self._tool_response_to_bedrock_content_blocks(
                                        message.content
                                    ),
                                    "status": "success",
                                }
                            }
                        ],
                    }
                )

        return bedrock_messages

    def _get_system_blocks(self, messages: list[Message]) -> list[dict[str, object]] | None:
        system_blocks: list[dict[str, object]] = []

        for message in messages:
            if not isinstance(message, SystemMessage):
                continue

            for part in message.content:
                system_blocks.append({"text": part.text})

        return system_blocks or None

    def _llm_tools_to_bedrock_tools(self, tools: list[Tool]) -> list[dict[str, object]]:
        return [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": get_schema_as_dict(tool.input_schema)},
                    "strict": tool.strict,
                }
            }
            for tool in tools
        ]

    def _response_schema_tool(self, response_schema: dict) -> dict[str, object]:
        return {
            "toolSpec": {
                "name": "ResponseSchema",
                "description": "Provide the final response to the user",
                "inputSchema": {"json": response_schema},
                "strict": True,
            }
        }

    def _get_bedrock_tool_config(
        self,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> dict[str, object] | None:
        resolved = resolve_tools(tools, tool_choice)
        bedrock_tools = self._llm_tools_to_bedrock_tools(resolved.tools)

        response_schema = get_response_schema(response_format)
        if use_tools_for_structured_output and response_schema:
            bedrock_tools.append(self._response_schema_tool(response_schema))

        if not bedrock_tools:
            return None

        config: dict[str, object] = {"tools": bedrock_tools}

        if resolved.required_names:
            if len(resolved.required_names) == 1 and not resolved.optional_names:
                config["toolChoice"] = {"tool": {"name": resolved.required_names[0]}}
            else:
                config["toolChoice"] = {"any": {}}
        elif resolved.optional_names:
            config["toolChoice"] = {"auto": {}}
        elif use_tools_for_structured_output and response_schema:
            config["toolChoice"] = {"any": {}}

        return config

    def _get_output_config(
        self,
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> dict[str, object] | None:
        if use_tools_for_structured_output:
            return None

        response_schema = get_response_schema(response_format)
        if not response_schema:
            return None

        return {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "schema": json.dumps(response_schema),
                    }
                },
            }
        }

    def _final_content(
        self,
        content: list[TextContentPart | ImageContentPart] | None,
        response_schema_content: dict | None,
        user_tool_calls: list[AssistantToolCall],
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> object:
        if response_schema_content is not None:
            return response_schema_content

        text_content = "".join(
            part.text for part in (content or []) if isinstance(part, TextContentPart)
        )
        if (
            text_content
            and not use_tools_for_structured_output
            and isinstance(response_format, (JSONSchemaResponse, JSONObjectResponse))
        ):
            return json.loads(text_content)

        if content is None and not user_tool_calls:
            return None

        return content

    def _response_usage(self, usage: dict[str, object] | None) -> ResponseUsage | None:
        if not usage:
            return None

        details = dict(usage)
        details.pop("inputTokens", None)
        details.pop("outputTokens", None)
        details.pop("totalTokens", None)

        return ResponseUsage(
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            total_tokens=usage.get("totalTokens"),
            details=details,
        )

    def _append_generated_content_block(
        self,
        block: dict[str, object],
        *,
        content_parts: list[TextContentPart | ImageContentPart],
        thinking_chunks: list[str],
        user_tool_calls: list[AssistantToolCall],
        response_schema_holder: list[dict | None],
    ) -> None:
        text = block.get("text")
        if text:
            content_parts.append(TextContentPart(text=text))

        image_part = self._bedrock_image_to_content_part(block.get("image"))
        if image_part is not None:
            content_parts.append(image_part)

        reasoning_content = block.get("reasoningContent") or {}
        if isinstance(reasoning_content, dict):
            reasoning_text = reasoning_content.get("reasoningText") or {}
            if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
                thinking_chunks.append(reasoning_text["text"])

        citations_content = block.get("citationsContent") or {}
        if isinstance(citations_content, dict):
            for generated in citations_content.get("content") or []:
                if isinstance(generated, dict) and generated.get("text"):
                    content_parts.append(TextContentPart(text=generated["text"]))

        tool_use = block.get("toolUse") or {}
        if isinstance(tool_use, dict) and tool_use.get("name"):
            raw_input = tool_use.get("input")
            arguments = (
                raw_input if isinstance(raw_input, str) else json.dumps(raw_input or {})
            )

            tool_call = AssistantToolCall(
                id=tool_use.get("toolUseId") or tool_use["name"],
                name=tool_use["name"],
                arguments=arguments,
            )
            if tool_call.name == "ResponseSchema":
                response_schema_holder[0] = self._parse_tool_arguments(
                    tool_call.arguments
                )
            else:
                user_tool_calls.append(tool_call)

    def _response_message_to_assistant_message(
        self,
        message: dict[str, object] | None,
    ) -> tuple[AssistantMessage, dict | None, list[AssistantToolCall]]:
        content_parts: list[TextContentPart | ImageContentPart] = []
        thinking_chunks: list[str] = []
        response_schema_holder: list[dict | None] = [None]
        user_tool_calls: list[AssistantToolCall] = []

        for block in (message or {}).get("content") or []:
            if isinstance(block, dict):
                self._append_generated_content_block(
                    block,
                    content_parts=content_parts,
                    thinking_chunks=thinking_chunks,
                    user_tool_calls=user_tool_calls,
                    response_schema_holder=response_schema_holder,
                )

        assistant_message = AssistantMessage(
            content=collapse_content_parts(content_parts),
            thinking="".join(thinking_chunks) or None,
            tool_calls=user_tool_calls,
        )
        return assistant_message, response_schema_holder[0], user_tool_calls

    def _converse_kwargs(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        max_tokens: int | None,
        extra_body: dict | None,
        use_tools_for_structured_output: bool | None,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "modelId": model,
            "messages": self._messages_to_bedrock_messages(messages),
        }

        system_blocks = self._get_system_blocks(messages)
        if system_blocks:
            kwargs["system"] = system_blocks

        inference_config: dict[str, object] = {}
        if temperature is not None:
            inference_config["temperature"] = temperature
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        tool_config = self._get_bedrock_tool_config(
            tools,
            tool_choice,
            response_format,
            use_tools_for_structured_output,
        )
        if tool_config:
            kwargs["toolConfig"] = tool_config

        output_config = self._get_output_config(
            response_format,
            use_tools_for_structured_output,
        )
        if output_config:
            kwargs["outputConfig"] = output_config

        if extra_body is not None:
            kwargs["additionalModelRequestFields"] = extra_body

        return kwargs

    def _raise_for_stream_error(self, event: dict[str, object]) -> None:
        error_status = {
            "validationException": 400,
            "throttlingException": 429,
            "internalServerException": 500,
            "modelStreamErrorException": 500,
            "serviceUnavailableException": 503,
        }
        for key, status_code in error_status.items():
            details = event.get(key)
            if isinstance(details, dict):
                message = details.get("message") or key
                raise LLMError(status_code, message)

    def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        use_tools_for_structured_output: bool | None = None,
    ) -> ResponseContent:
        try:
            start_time = perf_counter()
            response = self._client.converse(
                **self._converse_kwargs(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    use_tools_for_structured_output=use_tools_for_structured_output,
                )
            )
            duration_seconds = perf_counter() - start_time

            response_message = ((response.get("output") or {}).get("message")) or {}
            assistant_message, response_schema_content, user_tool_calls = (
                self._response_message_to_assistant_message(response_message)
            )
            new_messages = [*messages, assistant_message]

            return ResponseContent(
                content=self._final_content(
                    assistant_message.content,
                    response_schema_content,
                    user_tool_calls,
                    response_format,
                    use_tools_for_structured_output,
                ),
                messages=new_messages,
                tool_calls=user_tool_calls,
                usage=self._response_usage(response.get("usage")),
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider="bedrock")

    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        use_tools_for_structured_output: bool | None = None,
    ):
        try:
            start_time = perf_counter()
            response = self._client.converse_stream(
                **self._converse_kwargs(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    use_tools_for_structured_output=use_tools_for_structured_output,
                )
            )
            stream_id = "0"
            usage: ResponseUsage | None = None
            content_blocks: dict[int, dict[str, Any]] = {}
            thinking_chunks: list[str] = []

            for event in response.get("stream") or []:
                if not isinstance(event, dict):
                    continue

                self._raise_for_stream_error(event)

                metadata = event.get("metadata")
                if isinstance(metadata, dict):
                    event_usage = self._response_usage(metadata.get("usage"))
                    if event_usage is not None:
                        usage = event_usage
                    continue

                content_block_start = event.get("contentBlockStart")
                if isinstance(content_block_start, dict):
                    index = content_block_start.get("contentBlockIndex")
                    start = content_block_start.get("start") or {}
                    if not isinstance(index, int) or not isinstance(start, dict):
                        continue

                    tool_use = start.get("toolUse")
                    if isinstance(tool_use, dict):
                        content_blocks[index] = {
                            "toolUse": {
                                "toolUseId": tool_use.get("toolUseId"),
                                "name": tool_use.get("name"),
                                "input": "",
                            }
                        }
                        continue

                    image = start.get("image")
                    if isinstance(image, dict):
                        content_blocks[index] = {
                            "image": {
                                "format": image.get("format"),
                                "source": {},
                            }
                        }
                    continue

                content_block_delta = event.get("contentBlockDelta")
                if not isinstance(content_block_delta, dict):
                    continue

                index = content_block_delta.get("contentBlockIndex")
                delta = content_block_delta.get("delta") or {}
                if not isinstance(index, int) or not isinstance(delta, dict):
                    continue

                if delta.get("text"):
                    current = content_blocks.setdefault(index, {"text": ""})
                    current["text"] = f"{current.get('text', '')}{delta['text']}"
                    yield ResponseStreamContentChunk(
                        id=stream_id,
                        source="direct",
                        chunk=delta["text"],
                    )

                reasoning_content = delta.get("reasoningContent") or {}
                if isinstance(reasoning_content, dict) and reasoning_content.get("text"):
                    thinking_chunks.append(reasoning_content["text"])

                tool_use_delta = delta.get("toolUse") or {}
                if isinstance(tool_use_delta, dict) and tool_use_delta.get("input") is not None:
                    current = content_blocks.setdefault(
                        index,
                        {
                            "toolUse": {
                                "toolUseId": None,
                                "name": None,
                                "input": "",
                            }
                        },
                    )
                    current_tool_use = current.setdefault("toolUse", {})
                    current_tool_use["input"] = (
                        f"{current_tool_use.get('input', '')}{tool_use_delta['input']}"
                    )

                    tool_name = current_tool_use.get("name")
                    if tool_name == "ResponseSchema":
                        yield ResponseStreamContentChunk(
                            id=stream_id,
                            source="direct",
                            chunk=tool_use_delta["input"],
                        )
                    else:
                        yield ResponseStreamContentChunk(
                            id=stream_id,
                            source="tool",
                            tool=tool_name,
                            chunk=tool_use_delta["input"],
                        )

                image_delta = delta.get("image") or {}
                if isinstance(image_delta, dict):
                    current = content_blocks.setdefault(
                        index,
                        {
                            "image": {
                                "format": None,
                                "source": {},
                            }
                        },
                    )
                    image_block = current.setdefault("image", {})
                    source = image_block.setdefault("source", {})
                    delta_source = image_delta.get("source") or {}
                    if isinstance(delta_source, dict):
                        if delta_source.get("bytes") is not None:
                            source["bytes"] = delta_source["bytes"]
                        if isinstance(delta_source.get("s3Location"), dict):
                            source["s3Location"] = delta_source["s3Location"]

            content_parts: list[TextContentPart | ImageContentPart] = []
            response_schema_content: dict | None = None
            user_tool_calls: list[AssistantToolCall] = []
            response_schema_holder: list[dict | None] = [None]
            for index in sorted(content_blocks):
                block = content_blocks[index]
                self._append_generated_content_block(
                    block,
                    content_parts=content_parts,
                    thinking_chunks=thinking_chunks,
                    user_tool_calls=user_tool_calls,
                    response_schema_holder=response_schema_holder,
                )

            response_schema_content = response_schema_holder[0]
            assistant_message = AssistantMessage(
                content=collapse_content_parts(content_parts),
                thinking="".join(thinking_chunks) or None,
                tool_calls=user_tool_calls,
            )
            new_messages = [*messages, assistant_message]
            duration_seconds = perf_counter() - start_time

            yield ResponseStreamCompletionChunk(
                id=stream_id,
                content=self._final_content(
                    assistant_message.content,
                    response_schema_content,
                    user_tool_calls,
                    response_format,
                    use_tools_for_structured_output,
                ),
                messages=new_messages,
                tool_calls=user_tool_calls,
                usage=usage,
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider="bedrock")
