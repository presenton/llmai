from __future__ import annotations

from collections.abc import AsyncIterator
from logging import Logger
from time import perf_counter
from typing import Any

import httpx
from botocore.awsrequest import AWSPreparedRequest, AWSRequest
from botocore.eventstream import EventStreamBuffer
from botocore.parsers import create_parser

from llmai.bedrock.client import BedrockClient
from llmai.shared.base import AsyncBaseClient
from llmai.shared.configs import BedrockClientConfig
from llmai.shared.errors import LLMError, raise_llm_error
from llmai.shared.messages import (
    AssistantMessage,
    AssistantToolCall,
    ImageContentPart,
    Message,
    TextContentPart,
    collapse_content_parts,
    collapse_thinking_blocks,
)
from llmai.shared.responses import (
    ResponseContent,
    ResponseStreamCompletionChunk,
    ResponseStreamContentChunk,
    ResponseStreamEvent,
    ResponseStreamThinkingChunk,
    ResponseStreamToolChunk,
    ResponseStreamToolCompleteChunk,
    ResponseUsage,
)


class AsyncBedrockClient(AsyncBaseClient):
    def __init__(
        self,
        *,
        config: BedrockClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = BedrockClient(config=config, logger=logger)
        self._api_key = config.api_key
        self._provider_client = sync_client._client
        self._service_model = self._provider_client.meta.service_model
        self._response_parser = create_parser(self._service_model.metadata["protocol"])
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=None,
                write=60.0,
                pool=60.0,
            )
        )
        super().__init__(
            sync_client=sync_client,
            async_close=self._close,
        )

    @property
    def _parser(self) -> BedrockClient:
        return self._sync_client

    async def _close(self) -> None:
        await self._http_client.aclose()
        close = getattr(self._provider_client, "close", None)
        if callable(close):
            close()

    def _prepare_request(
        self,
        operation_name: str,
        params: dict[str, object],
    ) -> AWSPreparedRequest:
        operation_model = self._service_model.operation_model(operation_name)
        request_dict = self._provider_client._convert_to_request_dict(
            params,
            operation_model,
            self._provider_client.meta.endpoint_url,
        )
        request = AWSRequest(
            method=request_dict["method"],
            url=request_dict["url"],
            data=request_dict["body"],
            headers=request_dict["headers"],
        )
        request.context.update(request_dict.get("context", {}))
        if self._api_key is not None:
            request.headers["Authorization"] = f"Bearer {self._api_key}"
        else:
            self._provider_client._request_signer.sign(operation_name, request)
        return request.prepare()

    async def _send_request(
        self,
        operation_name: str,
        params: dict[str, object],
    ) -> httpx.Response:
        request = self._prepare_request(operation_name, params)
        response = await self._http_client.request(
            request.method,
            request.url,
            headers=dict(request.headers),
            content=request.body,
        )
        response.raise_for_status()
        return response

    def _parse_response(
        self,
        operation_name: str,
        response: httpx.Response,
    ) -> dict[str, object]:
        operation_model = self._service_model.operation_model(operation_name)
        return self._response_parser.parse(
            {
                "headers": dict(response.headers),
                "status_code": response.status_code,
                "body": response.content,
            },
            operation_model.output_shape,
        )

    def _parse_stream_event_response(
        self,
        response_dict: dict[str, Any],
        output_shape: Any,
    ) -> dict[str, object]:
        parsed_response = self._response_parser.parse(response_dict, output_shape)
        if response_dict["status_code"] != 200:
            return parsed_response

        if any(key in parsed_response for key in output_shape.members):
            return parsed_response

        event_type = response_dict.get("headers", {}).get(":event-type")
        if not isinstance(event_type, str):
            return parsed_response

        event_shape = output_shape.members.get(event_type)
        if event_shape is None:
            return parsed_response

        event_payload = self._response_parser.parse(response_dict, event_shape)
        event_payload.pop("ResponseMetadata", None)
        if not event_payload:
            return parsed_response

        return {event_type: event_payload}

    async def _iter_stream_events(
        self,
        response: httpx.Response,
    ) -> AsyncIterator[dict[str, object]]:
        output_shape = (
            self._service_model.operation_model("ConverseStream")
            .output_shape.members["stream"]
        )
        event_stream_buffer = EventStreamBuffer()

        async for chunk in response.aiter_bytes():
            event_stream_buffer.add_data(chunk)
            for message in event_stream_buffer:
                response_dict = message.to_response_dict(
                    status_code=response.status_code
                )
                parsed_response = self._parse_stream_event_response(
                    response_dict,
                    output_shape,
                )
                if response_dict["status_code"] == 200:
                    if parsed_response:
                        yield parsed_response
                    continue

                error = parsed_response.get("Error") or {}
                message_text = (
                    error.get("Message")
                    or error.get("message")
                    or str(parsed_response)
                )
                raise LLMError(
                    response_dict["status_code"],
                    message_text,
                    provider="bedrock",
                )

    async def _agenerate_once(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any | None = None,
        response_format: Any | None = None,
        max_tokens: int | None = None,
        reasoning_effort: Any | None = None,
        extra_body: dict | None = None,
    ) -> ResponseContent:
        parser = self._parser
        try:
            start_time = perf_counter()
            response = await self._send_request(
                "Converse",
                parser._converse_kwargs(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    extra_body=extra_body,
                ),
            )
            parsed = self._parse_response("Converse", response)
            duration_seconds = perf_counter() - start_time

            response_message = ((parsed.get("output") or {}).get("message")) or {}
            assistant_message, user_tool_calls = (
                parser._response_message_to_assistant_message(response_message)
            )
            new_messages = [*messages, assistant_message]

            return ResponseContent(
                content=parser._final_content(
                    assistant_message.content,
                    user_tool_calls,
                    response_format,
                ),
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=user_tool_calls,
                usage=parser._response_usage(parsed.get("usage")),
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider="bedrock")

    async def _agenerate_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any | None = None,
        response_format: Any | None = None,
        max_tokens: int | None = None,
        reasoning_effort: Any | None = None,
        extra_body: dict | None = None,
    ) -> AsyncIterator[ResponseStreamEvent]:
        parser = self._parser
        response = None
        try:
            start_time = perf_counter()
            request = self._prepare_request(
                "ConverseStream",
                parser._converse_kwargs(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    extra_body=extra_body,
                ),
            )
            async with self._http_client.stream(
                request.method,
                request.url,
                headers=dict(request.headers),
                content=request.body,
            ) as stream_response:
                stream_response.raise_for_status()
                response = stream_response
                current_chunk_type = None
                current_tool = None
                active_thinking_index: int | None = None
                usage: ResponseUsage | None = None
                content_blocks: dict[int, dict[str, Any]] = {}
                thinking_blocks_by_index: dict[int, str] = {}
                thinking_order: list[int] = []

                async for event in self._iter_stream_events(stream_response):
                    if not isinstance(event, dict):
                        continue

                    parser._raise_for_stream_error(event)

                    metadata = event.get("metadata")
                    if isinstance(metadata, dict):
                        event_usage = parser._response_usage(metadata.get("usage"))
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
                                    "toolUseId": parser._tool_call_id(
                                        tool_use.get("toolUseId")
                                    ),
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

                    content_block_stop = event.get("contentBlockStop")
                    if isinstance(content_block_stop, dict):
                        index = content_block_stop.get("contentBlockIndex")
                        if (
                            isinstance(index, int)
                            and current_chunk_type == "thinking"
                            and active_thinking_index == index
                        ):
                            stream_chunk = parser._close_stream_chunk(
                                current_chunk_type=current_chunk_type,
                                current_tool=current_tool,
                            )
                            if stream_chunk is not None:
                                yield stream_chunk
                            current_chunk_type = None
                            current_tool = None
                            active_thinking_index = None
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
                        if current_chunk_type == "thinking":
                            active_thinking_index = None
                        current_chunk_type, current_tool, stream_chunks = (
                            parser._transition_stream_chunk(
                                current_chunk_type=current_chunk_type,
                                next_chunk_type="content",
                                current_tool=current_tool,
                            )
                        )
                        for stream_chunk in stream_chunks:
                            yield stream_chunk
                        yield ResponseStreamContentChunk(chunk=delta["text"])

                    reasoning_content = delta.get("reasoningContent") or {}
                    if (
                        isinstance(reasoning_content, dict)
                        and reasoning_content.get("text")
                    ):
                        if index not in thinking_blocks_by_index:
                            thinking_blocks_by_index[index] = ""
                            thinking_order.append(index)

                        if (
                            current_chunk_type == "thinking"
                            and active_thinking_index != index
                        ):
                            stream_chunk = parser._close_stream_chunk(
                                current_chunk_type=current_chunk_type,
                                current_tool=current_tool,
                            )
                            if stream_chunk is not None:
                                yield stream_chunk
                            current_chunk_type = None
                            current_tool = None

                        active_thinking_index = index
                        thinking_blocks_by_index[index] += reasoning_content["text"]
                        current_chunk_type, current_tool, stream_chunks = (
                            parser._transition_stream_chunk(
                                current_chunk_type=current_chunk_type,
                                next_chunk_type="thinking",
                                current_tool=current_tool,
                            )
                        )
                        for stream_chunk in stream_chunks:
                            yield stream_chunk
                        yield ResponseStreamThinkingChunk(
                            chunk=reasoning_content["text"],
                        )

                    tool_use_delta = delta.get("toolUse") or {}
                    if (
                        isinstance(tool_use_delta, dict)
                        and tool_use_delta.get("input") is not None
                    ):
                        current = content_blocks.setdefault(
                            index,
                            {
                                "toolUse": {
                                    "toolUseId": parser._tool_call_id(),
                                    "name": None,
                                    "input": "",
                                }
                            },
                        )
                        current_tool_use = current.setdefault("toolUse", {})
                        current_tool_use["input"] = (
                            f"{current_tool_use.get('input', '')}"
                            f"{tool_use_delta['input']}"
                        )

                        tool_name = current_tool_use.get("name")
                        if current_chunk_type == "thinking":
                            active_thinking_index = None
                        current_chunk_type, current_tool, stream_chunks = (
                            parser._transition_stream_chunk(
                                current_chunk_type=current_chunk_type,
                                next_chunk_type="tool",
                                current_tool=current_tool,
                                next_tool=tool_name,
                            )
                        )
                        for stream_chunk in stream_chunks:
                            yield stream_chunk
                        yield ResponseStreamToolChunk(
                            id=parser._tool_call_id(
                                current_tool_use.get("toolUseId")
                            ),
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
                user_tool_calls: list[AssistantToolCall] = []
                for index in sorted(content_blocks):
                    block = content_blocks[index]
                    parser._append_generated_content_block(
                        block,
                        content_parts=content_parts,
                        thinking_blocks=[],
                        user_tool_calls=user_tool_calls,
                    )

                assistant_message = AssistantMessage(
                    content=collapse_content_parts(content_parts),
                    thinking=collapse_thinking_blocks(
                        [
                            thinking_blocks_by_index[index]
                            for index in thinking_order
                            if thinking_blocks_by_index[index]
                        ]
                    ),
                    tool_calls=user_tool_calls,
                )
                new_messages = [*messages, assistant_message]
                duration_seconds = perf_counter() - start_time

                if current_chunk_type == "tool":
                    for tool_call in user_tool_calls:
                        yield ResponseStreamToolCompleteChunk(
                            id=tool_call.id,
                            tool=tool_call.name,
                            arguments=tool_call.arguments,
                        )

                stream_chunk = parser._close_stream_chunk(
                    current_chunk_type=current_chunk_type,
                    current_tool=current_tool,
                )
                if stream_chunk is not None:
                    yield stream_chunk

                if current_chunk_type != "tool":
                    for tool_call in user_tool_calls:
                        yield ResponseStreamToolCompleteChunk(
                            id=tool_call.id,
                            tool=tool_call.name,
                            arguments=tool_call.arguments,
                        )
                yield ResponseStreamCompletionChunk(
                    content=parser._final_content(
                        assistant_message.content,
                        user_tool_calls,
                        response_format,
                    ),
                    thinking=assistant_message.thinking,
                    messages=new_messages,
                    tool_calls=user_tool_calls,
                    usage=usage,
                    duration_seconds=duration_seconds,
                )
        except Exception as exc:
            raise_llm_error(exc, provider="bedrock")
        finally:
            if response is not None:
                await response.aclose()
