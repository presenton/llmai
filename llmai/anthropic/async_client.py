from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from logging import Logger
from time import perf_counter
from typing import Any

from anthropic import AsyncAnthropic
from anthropic import Omit as AnthropicOmit

from llmai.anthropic.client import AnthropicClient
from llmai.openai.async_client import create_async_provider_client
from llmai.shared.async_utils import close_sync_provider_client, resolve_async_result
from llmai.shared.base import AsyncBaseClient
from llmai.shared.configs import AnthropicClientConfig
from llmai.shared.errors import raise_llm_error
from llmai.shared.messages import (
    AssistantMessage,
    AssistantToolCall,
    Message,
    collapse_thinking_blocks,
    content_from_text,
)
from llmai.shared.model_listing import amodel_ids
from llmai.shared.response_formats import (
    get_response_format_name,
    get_response_format_strict,
    get_response_schema,
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


class AsyncAnthropicClient(AsyncBaseClient):
    def __init__(
        self,
        *,
        config: AnthropicClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = AnthropicClient(config=config, logger=logger)
        provider_client = create_async_provider_client(
            AsyncAnthropic,
            provider="anthropic",
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self._provider_client = provider_client
        close_sync_provider_client(sync_client, provider_client)
        super().__init__(
            sync_client=sync_client,
            async_close=provider_client.close,
        )

    @property
    def _parser(self) -> AnthropicClient:
        return self._sync_client

    async def alist_available_models(self) -> list[str]:
        self._ensure_open()
        try:
            page = await self._provider_client.models.list(limit=100)
            return await amodel_ids(page)
        except Exception as exc:
            raise_llm_error(exc, provider=self._parser.PROVIDER_NAME)

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
        anthropic_tools, anthropic_tool_choice = (
            parser._get_anthropic_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
                response_format,
            )
        )

        try:
            start_time = perf_counter()
            try:
                response = await self._provider_client.messages.create(
                    model=model,
                    system=parser._get_system_prompt(messages),
                    messages=parser._messages_to_anthropic_messages(messages),
                    tools=anthropic_tools,
                    tool_choice=anthropic_tool_choice,
                    thinking=parser._get_anthropic_thinking_or_omit(reasoning_effort),
                    output_config=parser._get_anthropic_output_config_or_omit(
                        reasoning_effort
                    ),
                    max_tokens=max_tokens or 8000,
                    temperature=temperature or AnthropicOmit(),
                    extra_body=extra_body,
                )
            except Exception as exc:
                if not parser._should_retry_without_strict_tools(
                    exc,
                    anthropic_tools,
                ):
                    raise

                parser._log_strict_tool_fallback()
                anthropic_tools, anthropic_tool_choice = (
                    parser._get_anthropic_tools_and_tool_choice_or_omit(
                        tools,
                        tool_choice,
                        response_format,
                        strict_override=False,
                    )
                )
                response = await self._provider_client.messages.create(
                    model=model,
                    system=parser._get_system_prompt(messages),
                    messages=parser._messages_to_anthropic_messages(messages),
                    tools=anthropic_tools,
                    tool_choice=anthropic_tool_choice,
                    thinking=parser._get_anthropic_thinking_or_omit(reasoning_effort),
                    output_config=parser._get_anthropic_output_config_or_omit(
                        reasoning_effort
                    ),
                    max_tokens=max_tokens or 8000,
                    temperature=temperature or AnthropicOmit(),
                    extra_body=extra_body,
                )
            duration_seconds = perf_counter() - start_time

            text_chunks: list[str] = []
            thinking_blocks: list[str] = []
            thinking_items = parser._anthropic_reasoning_items(response.content)
            response_schema_content: dict | None = None
            response_schema_tool_name = (
                get_response_format_name(response_format, default="response")
                if get_response_schema(
                    response_format,
                    strict=get_response_format_strict(
                        response_format,
                        default=False,
                    ),
                )
                else None
            )
            user_tool_calls: list[AssistantToolCall] = []
            for content in response.content:
                if content.type == "text":
                    text_chunks.append(content.text)
                elif content.type == "thinking":
                    thinking_blocks.append(content.thinking)
                elif content.type == "tool_use":
                    tool_call = AssistantToolCall(
                        id=content.id,
                        name=content.name,
                        arguments=json.dumps(content.input),
                    )
                    if tool_call.name == response_schema_tool_name:
                        response_schema_content = parser._parse_tool_arguments(
                            tool_call.arguments
                        )
                    else:
                        user_tool_calls.append(tool_call)

            assistant_message = AssistantMessage(
                content=content_from_text("".join(text_chunks) or None),
                thinking=thinking_items or collapse_thinking_blocks(thinking_blocks),
                tool_calls=user_tool_calls,
            )
            new_messages = [*messages, assistant_message]

            final_content: object = response_schema_content
            if final_content is None:
                final_content = assistant_message.content
            if final_content is None and not user_tool_calls:
                final_content = ""

            return ResponseContent(
                content=final_content,
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=user_tool_calls,
                usage=parser._response_usage(getattr(response, "usage", None)),
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
        parser = self._parser
        anthropic_tools, anthropic_tool_choice = (
            parser._get_anthropic_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
                response_format,
            )
        )

        current_chunk_type = None
        current_tool = None
        text_chunks: list[str] = []
        thinking_blocks: list[str] = []
        response_schema_content: dict | None = None
        response_schema_tool_name = (
            get_response_format_name(response_format, default="response")
            if get_response_schema(
                response_format,
                strict=get_response_format_strict(
                    response_format,
                    default=False,
                ),
            )
            else None
        )
        user_tool_calls: list[AssistantToolCall] = []
        active_tool_name: str | None = None
        active_tool_id: str | None = None
        active_thinking_block: list[str] | None = None
        start_time = perf_counter()
        usage: ResponseUsage | None = None
        final_thinking_items = []

        try:
            async with AsyncExitStack() as stack:
                try:
                    stream_response = await stack.enter_async_context(
                        self._provider_client.messages.stream(
                            model=model,
                            system=parser._get_system_prompt(messages),
                            messages=parser._messages_to_anthropic_messages(messages),
                            tools=anthropic_tools,
                            tool_choice=anthropic_tool_choice,
                            thinking=parser._get_anthropic_thinking_or_omit(
                                reasoning_effort
                            ),
                            output_config=parser._get_anthropic_output_config_or_omit(
                                reasoning_effort
                            ),
                            max_tokens=max_tokens or 8000,
                            temperature=temperature or AnthropicOmit(),
                            extra_body=extra_body,
                        )
                    )
                except Exception as exc:
                    if not parser._should_retry_without_strict_tools(
                        exc,
                        anthropic_tools,
                    ):
                        raise

                    parser._log_strict_tool_fallback()
                    anthropic_tools, anthropic_tool_choice = (
                        parser._get_anthropic_tools_and_tool_choice_or_omit(
                            tools,
                            tool_choice,
                            response_format,
                            strict_override=False,
                        )
                    )
                    stream_response = await stack.enter_async_context(
                        self._provider_client.messages.stream(
                            model=model,
                            system=parser._get_system_prompt(messages),
                            messages=parser._messages_to_anthropic_messages(messages),
                            tools=anthropic_tools,
                            tool_choice=anthropic_tool_choice,
                            thinking=parser._get_anthropic_thinking_or_omit(
                                reasoning_effort
                            ),
                            output_config=parser._get_anthropic_output_config_or_omit(
                                reasoning_effort
                            ),
                            max_tokens=max_tokens or 8000,
                            temperature=temperature or AnthropicOmit(),
                            extra_body=extra_body,
                        )
                    )

                async for event in stream_response:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            active_tool_name = event.content_block.name
                            active_tool_id = event.content_block.id
                        elif event.content_block.type == "thinking":
                            active_thinking_block = []
                        continue

                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            text_chunks.append(event.delta.text)
                            if response_schema_tool_name is None:
                                current_chunk_type, current_tool, stream_chunks = (
                                    parser._transition_stream_chunk(
                                        current_chunk_type=current_chunk_type,
                                        next_chunk_type="content",
                                        current_tool=current_tool,
                                    )
                                )
                                for stream_chunk in stream_chunks:
                                    yield stream_chunk
                                yield ResponseStreamContentChunk(
                                    chunk=event.delta.text,
                                )
                        elif event.delta.type == "thinking_delta":
                            if active_thinking_block is None:
                                active_thinking_block = []
                            active_thinking_block.append(event.delta.thinking)
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
                                chunk=event.delta.thinking,
                            )
                        elif (
                            event.delta.type == "input_json_delta" and active_tool_name
                        ):
                            chunk = event.delta.partial_json
                            if active_tool_name == response_schema_tool_name:
                                continue
                            current_chunk_type, current_tool, stream_chunks = (
                                parser._transition_stream_chunk(
                                    current_chunk_type=current_chunk_type,
                                    next_chunk_type="tool",
                                    current_tool=current_tool,
                                    next_tool=active_tool_name,
                                )
                            )
                            for stream_chunk in stream_chunks:
                                yield stream_chunk
                            yield ResponseStreamToolChunk(
                                id=active_tool_id or active_tool_name or "",
                                tool=active_tool_name,
                                chunk=chunk,
                            )
                        continue

                    if (
                        event.type == "content_block_stop"
                        and event.content_block.type == "thinking"
                    ):
                        if active_thinking_block:
                            thinking_blocks.append("".join(active_thinking_block))
                        active_thinking_block = None
                        if current_chunk_type == "thinking":
                            stream_chunk = parser._close_stream_chunk(
                                current_chunk_type=current_chunk_type,
                                current_tool=current_tool,
                            )
                            if stream_chunk is not None:
                                yield stream_chunk
                            current_chunk_type = None
                            current_tool = None
                        continue

                    if (
                        event.type == "content_block_stop"
                        and event.content_block.type == "tool_use"
                    ):
                        tool_call = AssistantToolCall(
                            id=event.content_block.id,
                            name=event.content_block.name,
                            arguments=json.dumps(event.content_block.input),
                        )
                        if tool_call.name == response_schema_tool_name:
                            response_schema_content = parser._parse_tool_arguments(
                                tool_call.arguments
                            )
                        else:
                            user_tool_calls.append(tool_call)
                            yield ResponseStreamToolCompleteChunk(
                                id=tool_call.id,
                                tool=tool_call.name,
                                arguments=tool_call.arguments,
                            )
                        active_tool_name = None
                        active_tool_id = None

                if hasattr(stream_response, "get_final_message"):
                    final_message = stream_response.get_final_message()
                    final_message = await resolve_async_result(final_message)
                    usage = parser._response_usage(
                        getattr(final_message, "usage", None)
                    )
                    final_thinking_items = parser._anthropic_reasoning_items(
                        list(getattr(final_message, "content", []) or [])
                    )

            assistant_message = AssistantMessage(
                content=content_from_text("".join(text_chunks) or None),
                thinking=final_thinking_items
                or collapse_thinking_blocks(
                    [
                        *thinking_blocks,
                        *(
                            ["".join(active_thinking_block)]
                            if active_thinking_block
                            else []
                        ),
                    ]
                ),
                tool_calls=user_tool_calls,
            )
            new_messages = [*messages, assistant_message]

            final_content: object = response_schema_content
            if final_content is None:
                final_content = assistant_message.content
            if final_content is None and not user_tool_calls:
                final_content = ""
            duration_seconds = perf_counter() - start_time

            stream_chunk = parser._close_stream_chunk(
                current_chunk_type=current_chunk_type,
                current_tool=current_tool,
            )
            if stream_chunk is not None:
                yield stream_chunk
            yield ResponseStreamCompletionChunk(
                content=final_content,
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=user_tool_calls,
                usage=usage,
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider=parser.PROVIDER_NAME)
