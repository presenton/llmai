from __future__ import annotations

import json
from collections.abc import AsyncIterator
from logging import Logger
from time import perf_counter
from typing import Any

from google.genai.types import GenerateContentConfig

from llmai.google.client import GoogleClient
from llmai.shared.async_utils import close_async_resource, resolve_async_result
from llmai.shared.base import AsyncBaseClient
from llmai.shared.configs import GoogleClientConfig
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
from llmai.shared.model_listing import model_ids
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


async def close_google_client(client: Any) -> None:
    await client.aio.aclose()
    client.close()


class AsyncGoogleCompatibleClient(AsyncBaseClient):
    def __init__(
        self,
        *,
        sync_client: GoogleClient,
        provider_client: Any,
    ):
        self._provider_client = provider_client
        super().__init__(
            sync_client=sync_client,
            async_close=lambda: close_google_client(provider_client),
        )

    @property
    def _parser(self) -> GoogleClient:
        return self._sync_client

    async def alist_available_models(self) -> list[str]:
        self._ensure_open()
        try:
            result = self._provider_client.aio.models.list(config={"page_size": 100})
            return model_ids(await resolve_async_result(result))
        except Exception as exc:
            raise_llm_error(exc, provider=self._parser.PROVIDER_NAME)

    def _config(
        self,
        *,
        messages: list[Message],
        temperature: float | None,
        tools: list[Any] | None,
        tool_choice: Any | None,
        response_format: Any | None,
        max_tokens: int | None,
        reasoning_effort: Any | None,
    ) -> GenerateContentConfig:
        parser = self._parser
        google_tools, tool_config = parser._get_google_tools_and_tool_config(
            tools,
            tool_choice,
        )
        return GenerateContentConfig(
            tools=google_tools,
            tool_config=tool_config,
            system_instruction=parser._get_system_prompt(messages),
            response_mime_type=parser._get_response_mime_type(response_format),
            response_json_schema=parser._get_response_json_schema(response_format),
            thinking_config=parser._get_google_thinking_config(reasoning_effort),
            max_output_tokens=max_tokens,
            temperature=temperature,
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
        del extra_body

        parser = self._parser
        config = self._config(
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

        try:
            start_time = perf_counter()
            response = self._provider_client.aio.models.generate_content(
                model=model,
                contents=parser._messages_to_google_messages(messages),
                config=config,
            )
            response = await resolve_async_result(response)
            duration_seconds = perf_counter() - start_time

            if not (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                raise LLMError(400, "No content returned from LLM")

            raw_assistant_message = parser._content_to_assistant_message(
                response.candidates[0].content
            )
            user_tool_calls = raw_assistant_message.tool_calls

            assistant_message = AssistantMessage(
                content=raw_assistant_message.content,
                thinking=raw_assistant_message.thinking,
                tool_calls=user_tool_calls,
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
                usage=parser._response_usage(getattr(response, "usage_metadata", None)),
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider=parser.PROVIDER_NAME)

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
        del extra_body

        parser = self._parser
        config = self._config(
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        response = None

        try:
            current_chunk_type = None
            current_tool = None
            active_thinking_part_index: int | None = None
            content_parts: list[TextContentPart | ImageContentPart] = []
            thinking_blocks_by_part_index: dict[int, str] = {}
            thinking_order: list[int] = []
            tool_calls_by_id: dict[str, AssistantToolCall] = {}
            tool_call_order: list[str] = []
            last_emitted_chunks: dict[str, str] = {}
            generated_tool_ids_by_part_index: dict[int, str] = {}
            seen_images: set[tuple[str, str, bytes | str]] = set()
            usage: ResponseUsage | None = None
            start_time = perf_counter()

            response = self._provider_client.aio.models.generate_content_stream(
                model=model,
                contents=parser._messages_to_google_messages(messages),
                config=config,
            )
            response = await resolve_async_result(response)

            async for event in response:
                event_usage = parser._response_usage(
                    getattr(event, "usage_metadata", None)
                )
                if event_usage is not None:
                    usage = event_usage

                if not (
                    event.candidates
                    and event.candidates[0].content
                    and event.candidates[0].content.parts
                ):
                    continue

                for part_index, each_part in enumerate(
                    event.candidates[0].content.parts
                ):
                    text = getattr(each_part, "text", None)
                    if text:
                        if getattr(each_part, "thought", False):
                            if part_index not in thinking_blocks_by_part_index:
                                thinking_blocks_by_part_index[part_index] = ""
                                thinking_order.append(part_index)

                            if (
                                current_chunk_type == "thinking"
                                and active_thinking_part_index != part_index
                            ):
                                stream_chunk = parser._close_stream_chunk(
                                    current_chunk_type=current_chunk_type,
                                    current_tool=current_tool,
                                )
                                if stream_chunk is not None:
                                    yield stream_chunk
                                current_chunk_type = None
                                current_tool = None

                            active_thinking_part_index = part_index
                            thinking_blocks_by_part_index[part_index] += text
                            current_chunk_type, current_tool, stream_chunks = (
                                parser._transition_stream_chunk(
                                    current_chunk_type=current_chunk_type,
                                    next_chunk_type="thinking",
                                    current_tool=current_tool,
                                )
                            )
                            for stream_chunk in stream_chunks:
                                yield stream_chunk
                            yield ResponseStreamThinkingChunk(chunk=text)
                        else:
                            if current_chunk_type == "thinking":
                                active_thinking_part_index = None
                            content_parts.append(TextContentPart(text=text))
                            current_chunk_type, current_tool, stream_chunks = (
                                parser._transition_stream_chunk(
                                    current_chunk_type=current_chunk_type,
                                    next_chunk_type="content",
                                    current_tool=current_tool,
                                )
                            )
                            for stream_chunk in stream_chunks:
                                yield stream_chunk
                            yield ResponseStreamContentChunk(chunk=text)

                    inline_data = getattr(each_part, "inline_data", None)
                    if (
                        inline_data
                        and inline_data.data is not None
                        and inline_data.mime_type
                        and inline_data.mime_type.startswith("image/")
                    ):
                        image_key = (
                            "inline",
                            inline_data.mime_type,
                            inline_data.data,
                        )
                        if image_key not in seen_images:
                            seen_images.add(image_key)
                            content_parts.append(
                                ImageContentPart(
                                    data=inline_data.data,
                                    mime_type=inline_data.mime_type,
                                )
                            )

                    file_data = getattr(each_part, "file_data", None)
                    if (
                        file_data
                        and file_data.file_uri
                        and file_data.mime_type
                        and file_data.mime_type.startswith("image/")
                    ):
                        image_key = (
                            "file",
                            file_data.mime_type,
                            file_data.file_uri,
                        )
                        if image_key not in seen_images:
                            seen_images.add(image_key)
                            content_parts.append(
                                ImageContentPart(
                                    url=file_data.file_uri,
                                    mime_type=file_data.mime_type,
                                )
                            )

                    function_call = getattr(each_part, "function_call", None)
                    if not function_call:
                        continue

                    tool_name = function_call.name
                    tool_id = (
                        function_call.id
                        or generated_tool_ids_by_part_index.setdefault(
                            part_index,
                            parser._tool_call_id(),
                        )
                    )
                    arguments = json.dumps(function_call.args or {})
                    thought_signature = getattr(
                        each_part,
                        "thought_signature",
                        None,
                    )
                    existing_tool_call = tool_calls_by_id.get(tool_id)
                    if thought_signature is None and existing_tool_call is not None:
                        thought_signature = existing_tool_call.thought_signature

                    tool_calls_by_id[tool_id] = AssistantToolCall(
                        id=tool_id,
                        name=tool_name,
                        arguments=arguments,
                        thought_signature=thought_signature,
                    )
                    if tool_id not in tool_call_order:
                        tool_call_order.append(tool_id)

                    if last_emitted_chunks.get(tool_id) != arguments:
                        last_emitted_chunks[tool_id] = arguments
                        if current_chunk_type == "thinking":
                            active_thinking_part_index = None
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
                            id=tool_id,
                            tool=tool_name,
                            chunk=arguments,
                        )

            user_tool_calls = [tool_calls_by_id[tool_id] for tool_id in tool_call_order]
            assistant_message = AssistantMessage(
                content=collapse_content_parts(content_parts),
                thinking=collapse_thinking_blocks(
                    [
                        thinking_blocks_by_part_index[part_index]
                        for part_index in thinking_order
                        if thinking_blocks_by_part_index[part_index]
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
            raise_llm_error(exc, provider=parser.PROVIDER_NAME)
        finally:
            await close_async_resource(response)


class AsyncGoogleClient(AsyncGoogleCompatibleClient):
    def __init__(
        self,
        *,
        config: GoogleClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = GoogleClient(config=config, logger=logger)
        super().__init__(
            sync_client=sync_client,
            provider_client=sync_client._client,
        )
