from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from logging import Logger
from time import perf_counter
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI, Omit as OpenAIOmit

from llmai.openai.client import OpenAIClient
from llmai.shared.async_utils import close_async_resource, close_sync_provider_client
from llmai.shared.base import AsyncBaseClient
from llmai.shared.configs import OpenAIApiType, OpenAIClientConfig
from llmai.shared.errors import LLMError, raise_llm_error
from llmai.shared.messages import (
    AssistantMessage,
    AssistantReasoningItem,
    AssistantToolCall,
    Message,
    content_from_text,
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


def create_async_provider_client(
    factory: Any,
    *,
    provider: str,
    **kwargs: Any,
) -> Any:
    try:
        return factory(**kwargs)
    except Exception as exc:
        raise_llm_error(exc, provider=provider)


class AsyncOpenAICompatibleClient(AsyncBaseClient):
    def __init__(
        self,
        *,
        sync_client: OpenAIClient,
        provider_client: AsyncOpenAI | AsyncAzureOpenAI,
    ):
        self._provider_client = provider_client
        close_sync_provider_client(sync_client, provider_client)
        super().__init__(
            sync_client=sync_client,
            async_close=provider_client.close,
        )

    @property
    def _parser(self) -> OpenAIClient:
        return self._sync_client

    def _prepare_extra_body(self, extra_body: dict | None) -> dict | None:
        return extra_body

    async def alist_available_models(self) -> list[str]:
        self._ensure_open()
        models = getattr(self._provider_client, "models", None)
        list_models = getattr(models, "list", None)
        if not callable(list_models):
            return await super().alist_available_models()

        try:
            result = list_models()
            if inspect.isawaitable(result):
                result = await result
            return model_ids(result)
        except Exception as exc:
            raise_llm_error(exc, provider=self._parser.PROVIDER_NAME)

    async def _agenerate_once(self, **kwargs: Any) -> ResponseContent:
        if self._parser._api_type == OpenAIApiType.RESPONSES:
            return await self._agenerate_responses_once(**kwargs)
        return await self._agenerate_completions_once(**kwargs)

    async def _agenerate_stream(
        self,
        **kwargs: Any,
    ) -> AsyncIterator[ResponseStreamEvent]:
        if self._parser._api_type == OpenAIApiType.RESPONSES:
            async for event in self._agenerate_responses_stream(**kwargs):
                yield event
            return

        async for event in self._agenerate_completions_stream(**kwargs):
            yield event

    async def _agenerate_completions_once(
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
        request_extra_body = self._prepare_extra_body(extra_body)
        openai_tools, openai_tool_choice = (
            parser._get_openai_tools_and_tool_choice_or_omit(tools, tool_choice)
        )

        try:
            start_time = perf_counter()
            response = await self._provider_client.chat.completions.create(
                model=model,
                messages=parser._messages_to_openai_messages(messages),
                temperature=temperature,
                response_format=parser._get_openai_response_format_or_omit(
                    response_format
                ),
                tools=openai_tools,
                tool_choice=openai_tool_choice,
                **parser._get_openai_chat_max_tokens_kwargs(max_tokens),
                reasoning_effort=parser._get_openai_chat_reasoning_effort_or_omit(
                    reasoning_effort
                ),
                extra_body=request_extra_body,
            )
            duration_seconds = perf_counter() - start_time

            if not response.choices:
                raise LLMError(400, "No content returned from LLM")

            assistant_message = parser._chat_completion_message_to_assistant_message(
                response.choices[0].message
            )
            new_messages = [*messages, assistant_message]

            return ResponseContent(
                content=parser._final_content(
                    assistant_message.content,
                    response_format,
                ),
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=assistant_message.tool_calls,
                usage=parser._response_usage(getattr(response, "usage", None)),
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider=parser.PROVIDER_NAME)

    async def _agenerate_completions_stream(
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
        request_extra_body = self._prepare_extra_body(extra_body)
        openai_tools, openai_tool_choice = (
            parser._get_openai_tools_and_tool_choice_or_omit(tools, tool_choice)
        )
        response = None

        try:
            start_time = perf_counter()
            response = await self._provider_client.chat.completions.create(
                model=model,
                messages=parser._messages_to_openai_messages(messages),
                temperature=temperature,
                response_format=parser._get_openai_response_format_or_omit(
                    response_format
                ),
                tools=openai_tools,
                tool_choice=openai_tool_choice,
                **parser._get_openai_chat_max_tokens_kwargs(max_tokens),
                reasoning_effort=parser._get_openai_chat_reasoning_effort_or_omit(
                    reasoning_effort
                ),
                extra_body=request_extra_body,
                stream=True,
                stream_options={"include_usage": True},
            )

            current_chunk_type = None
            current_tool = None
            content = ""
            thinking = ""
            partial_tool_calls: dict[int, dict[str, str | None]] = {}
            tool_order: list[int] = []
            usage: ResponseUsage | None = None

            async for event in response:
                event_usage = parser._response_usage(getattr(event, "usage", None))
                if event_usage is not None:
                    usage = event_usage

                if not getattr(event, "choices", None):
                    continue

                delta = event.choices[0].delta

                thinking_delta = parser._chat_completion_delta_to_thinking_text(delta)
                if thinking_delta:
                    thinking += thinking_delta
                    current_chunk_type, current_tool, stream_chunks = (
                        parser._transition_stream_chunk(
                            current_chunk_type=current_chunk_type,
                            next_chunk_type="thinking",
                            current_tool=current_tool,
                        )
                    )
                    for stream_chunk in stream_chunks:
                        yield stream_chunk
                    yield ResponseStreamThinkingChunk(chunk=thinking_delta)

                if delta.content:
                    content += delta.content
                    current_chunk_type, current_tool, stream_chunks = (
                        parser._transition_stream_chunk(
                            current_chunk_type=current_chunk_type,
                            next_chunk_type="content",
                            current_tool=current_tool,
                        )
                    )
                    for stream_chunk in stream_chunks:
                        yield stream_chunk
                    yield ResponseStreamContentChunk(chunk=delta.content)

                if not delta.tool_calls:
                    continue

                for tool_call_delta in delta.tool_calls:
                    current = partial_tool_calls.get(tool_call_delta.index)
                    if current is None:
                        current = {"id": None, "name": None, "arguments": None}
                        partial_tool_calls[tool_call_delta.index] = current
                        tool_order.append(tool_call_delta.index)

                    if tool_call_delta.id:
                        current["id"] = tool_call_delta.id

                    if tool_call_delta.function and tool_call_delta.function.name:
                        current["name"] = tool_call_delta.function.name

                    tool_arguments = (
                        tool_call_delta.function.arguments
                        if tool_call_delta.function
                        else None
                    )
                    if current["arguments"] is None:
                        current["arguments"] = tool_arguments
                    elif tool_arguments:
                        current["arguments"] += tool_arguments

                    if tool_arguments:
                        current_chunk_type, current_tool, stream_chunks = (
                            parser._transition_stream_chunk(
                                current_chunk_type=current_chunk_type,
                                next_chunk_type="tool",
                                current_tool=current_tool,
                                next_tool=current["name"],
                            )
                        )
                        for stream_chunk in stream_chunks:
                            yield stream_chunk
                        yield ResponseStreamToolChunk(
                            id=current["id"] or current["name"] or "",
                            tool=current["name"],
                            chunk=tool_arguments,
                        )

            tool_calls = [
                AssistantToolCall(
                    id=(
                        partial_tool_calls[index]["id"]
                        or partial_tool_calls[index]["name"]
                        or ""
                    ),
                    name=partial_tool_calls[index]["name"] or "",
                    arguments=partial_tool_calls[index]["arguments"],
                )
                for index in tool_order
                if partial_tool_calls[index]["name"]
            ]

            assistant_message = AssistantMessage(
                content=content_from_text(content or None),
                thinking=(
                    [AssistantReasoningItem(summary=[thinking])] if thinking else None
                ),
                tool_calls=tool_calls,
            )
            new_messages = [*messages, assistant_message]
            duration_seconds = perf_counter() - start_time

            if current_chunk_type == "tool":
                for tool_call in tool_calls:
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
                for tool_call in tool_calls:
                    yield ResponseStreamToolCompleteChunk(
                        id=tool_call.id,
                        tool=tool_call.name,
                        arguments=tool_call.arguments,
                    )
            yield ResponseStreamCompletionChunk(
                content=parser._final_content(
                    assistant_message.content,
                    response_format,
                ),
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=tool_calls,
                usage=usage,
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider=parser.PROVIDER_NAME)
        finally:
            await close_async_resource(response)

    def _responses_request_kwargs(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None,
        tools: list[Any] | None,
        tool_choice: Any | None,
        response_format: Any | None,
        max_tokens: int | None,
        reasoning_effort: Any | None,
        extra_body: dict | None,
        stream: bool,
    ) -> dict[str, object]:
        parser = self._parser
        request_extra_body = self._prepare_extra_body(extra_body)
        openai_tools, openai_tool_choice = (
            parser._get_openai_responses_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
            )
        )
        reasoning, request_extra_body = (
            parser._openai_responses_reasoning_and_extra_body(
                reasoning_effort,
                request_extra_body,
            )
        )
        request_kwargs: dict[str, object] = {
            "model": model,
            "input": parser._messages_to_openai_responses_input(messages),
            "temperature": temperature if temperature is not None else OpenAIOmit(),
            "text": parser._get_openai_responses_text_or_omit(response_format),
            "tools": openai_tools,
            "tool_choice": openai_tool_choice,
            "reasoning": reasoning,
            "max_output_tokens": max_tokens if max_tokens is not None else OpenAIOmit(),
            "extra_body": request_extra_body,
            "stream": stream,
        }
        instructions = parser._messages_to_openai_responses_instructions(messages)
        if instructions is not None:
            request_kwargs["instructions"] = instructions
        return request_kwargs

    async def _agenerate_responses_once(
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
        request_kwargs = self._responses_request_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            stream=False,
        )

        try:
            start_time = perf_counter()
            response = await self._provider_client.responses.create(**request_kwargs)
            duration_seconds = perf_counter() - start_time

            assistant_message = parser._responses_output_to_assistant_message(
                getattr(response, "output", []) or []
            )
            new_messages = [*messages, assistant_message]

            return ResponseContent(
                content=parser._final_content(
                    assistant_message.content,
                    response_format,
                ),
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=assistant_message.tool_calls,
                usage=parser._response_usage(getattr(response, "usage", None)),
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider=parser.PROVIDER_NAME)

    async def _agenerate_responses_stream(
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
        request_kwargs = self._responses_request_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            stream=True,
        )
        async for event in aiter_openai_responses_stream(
            parser=self._parser,
            provider_client=self._provider_client,
            request_kwargs=request_kwargs,
            messages=messages,
            response_format=response_format,
        ):
            yield event


async def aiter_openai_responses_stream(
    *,
    parser: Any,
    provider_client: Any,
    request_kwargs: dict[str, object],
    messages: list[Message],
    response_format: Any | None,
) -> AsyncIterator[ResponseStreamEvent]:
    response = None
    try:
        start_time = perf_counter()
        response = await provider_client.responses.create(**request_kwargs)

        current_chunk_type = None
        current_tool = None
        content = ""
        streamed_assistant_message_id: str | None = None
        active_thinking_key: tuple[str, int] | None = None
        thinking_blocks_by_key: dict[tuple[str, int], str] = {}
        thinking_order: list[tuple[str, int]] = []
        partial_tool_calls: dict[str, dict[str, str | None]] = {}
        tool_order: list[str] = []
        completed_tool_ids: set[str] = set()
        final_response = None

        async for event in response:
            event_type = getattr(event, "type", None)

            if event_type == "response.output_text.delta":
                streamed_assistant_message_id = (
                    streamed_assistant_message_id or getattr(event, "item_id", None)
                )
                if current_chunk_type == "thinking":
                    active_thinking_key = None
                content += event.delta
                current_chunk_type, current_tool, stream_chunks = (
                    parser._transition_stream_chunk(
                        current_chunk_type=current_chunk_type,
                        next_chunk_type="content",
                        current_tool=current_tool,
                    )
                )
                for stream_chunk in stream_chunks:
                    yield stream_chunk
                yield ResponseStreamContentChunk(chunk=event.delta)
                continue

            if event_type == "response.reasoning_summary_text.delta":
                thinking_key = (event.item_id, event.summary_index)
                if thinking_key not in thinking_blocks_by_key:
                    thinking_blocks_by_key[thinking_key] = ""
                    thinking_order.append(thinking_key)

                if (
                    current_chunk_type == "thinking"
                    and active_thinking_key != thinking_key
                ):
                    stream_chunk = parser._close_stream_chunk(
                        current_chunk_type=current_chunk_type,
                        current_tool=current_tool,
                    )
                    if stream_chunk is not None:
                        yield stream_chunk
                    current_chunk_type = None
                    current_tool = None

                active_thinking_key = thinking_key
                thinking_blocks_by_key[thinking_key] += event.delta
                current_chunk_type, current_tool, stream_chunks = (
                    parser._transition_stream_chunk(
                        current_chunk_type=current_chunk_type,
                        next_chunk_type="thinking",
                        current_tool=current_tool,
                    )
                )
                for stream_chunk in stream_chunks:
                    yield stream_chunk
                yield ResponseStreamThinkingChunk(chunk=event.delta)
                continue

            if event_type in {
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
            }:
                thinking_key = (event.item_id, event.summary_index)
                if thinking_key not in thinking_blocks_by_key:
                    thinking_blocks_by_key[thinking_key] = ""
                    thinking_order.append(thinking_key)

                text = getattr(event, "text", None)
                if text is not None:
                    thinking_blocks_by_key[thinking_key] = text

                if (
                    current_chunk_type == "thinking"
                    and active_thinking_key == thinking_key
                ):
                    stream_chunk = parser._close_stream_chunk(
                        current_chunk_type=current_chunk_type,
                        current_tool=current_tool,
                    )
                    if stream_chunk is not None:
                        yield stream_chunk
                    current_chunk_type = None
                    current_tool = None
                    active_thinking_key = None
                continue

            if event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) != "function_call":
                    continue

                tool_key = (
                    getattr(item, "id", None)
                    or getattr(item, "call_id", None)
                    or parser._response_item_id("tool")
                )
                current = partial_tool_calls.get(tool_key)
                if current is None:
                    current = {"id": None, "name": None, "arguments": None}
                    partial_tool_calls[tool_key] = current
                    tool_order.append(tool_key)

                current["id"] = getattr(item, "call_id", None) or getattr(
                    item, "id", None
                )
                current["name"] = getattr(item, "name", None)
                if current["arguments"] is None:
                    current["arguments"] = getattr(item, "arguments", None)
                continue

            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) != "function_call":
                    continue

                tool_key = getattr(item, "id", None) or getattr(item, "call_id", None)
                if tool_key is None:
                    continue

                current = partial_tool_calls.get(tool_key)
                if current is None:
                    current = {"id": None, "name": None, "arguments": None}
                    partial_tool_calls[tool_key] = current
                    tool_order.append(tool_key)

                current["id"] = getattr(item, "call_id", None) or getattr(
                    item, "id", None
                )
                current["name"] = getattr(item, "name", None)
                current["arguments"] = getattr(item, "arguments", None)
                completed_tool_id = current["id"] or current["name"] or tool_key
                if current["name"] and completed_tool_id not in completed_tool_ids:
                    completed_tool_ids.add(completed_tool_id)
                    yield ResponseStreamToolCompleteChunk(
                        id=completed_tool_id,
                        tool=current["name"],
                        arguments=current["arguments"],
                    )
                continue

            if event_type == "response.function_call_arguments.delta":
                tool_key = event.item_id
                current = partial_tool_calls.get(tool_key)
                if current is None:
                    current = {"id": None, "name": None, "arguments": None}
                    partial_tool_calls[tool_key] = current
                    tool_order.append(tool_key)

                if current["arguments"] is None:
                    current["arguments"] = event.delta
                else:
                    current["arguments"] += event.delta

                if current_chunk_type == "thinking":
                    active_thinking_key = None
                current_chunk_type, current_tool, stream_chunks = (
                    parser._transition_stream_chunk(
                        current_chunk_type=current_chunk_type,
                        next_chunk_type="tool",
                        current_tool=current_tool,
                        next_tool=current["name"],
                    )
                )
                for stream_chunk in stream_chunks:
                    yield stream_chunk
                yield ResponseStreamToolChunk(
                    id=current["id"] or current["name"] or tool_key,
                    tool=current["name"],
                    chunk=event.delta,
                )
                continue

            if event_type == "response.completed":
                final_response = event.response

        streamed_thinking_by_id: dict[str, AssistantReasoningItem] = {}
        streamed_thinking_order: list[str] = []
        for item_id, summary_index in thinking_order:
            thinking_text = thinking_blocks_by_key.get((item_id, summary_index))
            if not thinking_text:
                continue

            thinking_item = streamed_thinking_by_id.get(item_id)
            if thinking_item is None:
                thinking_item = AssistantReasoningItem(
                    id=item_id,
                    summary=[],
                )
                streamed_thinking_by_id[item_id] = thinking_item
                streamed_thinking_order.append(item_id)
            thinking_item.summary.append(thinking_text)

        streamed_thinking = [
            streamed_thinking_by_id[item_id] for item_id in streamed_thinking_order
        ]
        streamed_tool_calls = [
            AssistantToolCall(
                id=partial_tool_calls[tool_key]["id"]
                or partial_tool_calls[tool_key]["name"]
                or tool_key,
                name=partial_tool_calls[tool_key]["name"] or "",
                arguments=partial_tool_calls[tool_key]["arguments"],
            )
            for tool_key in tool_order
            if partial_tool_calls[tool_key]["name"]
        ]
        streamed_assistant_message = AssistantMessage(
            id=streamed_assistant_message_id,
            content=content_from_text(content or None),
            thinking=streamed_thinking or None,
            tool_calls=streamed_tool_calls,
        )

        if final_response is not None:
            response_assistant_message = parser._responses_output_to_assistant_message(
                getattr(final_response, "output", []) or []
            )
            thinking = response_assistant_message.thinking
            if streamed_assistant_message.thinking and not all(
                thinking_item.id is not None for thinking_item in (thinking or [])
            ):
                thinking = streamed_assistant_message.thinking
            assistant_message = AssistantMessage(
                id=(response_assistant_message.id or streamed_assistant_message.id),
                content=(
                    response_assistant_message.content
                    or streamed_assistant_message.content
                ),
                thinking=thinking or streamed_assistant_message.thinking,
                tool_calls=(
                    response_assistant_message.tool_calls
                    or streamed_assistant_message.tool_calls
                ),
            )
            tool_calls = assistant_message.tool_calls
            usage = parser._response_usage(getattr(final_response, "usage", None))
        else:
            assistant_message = streamed_assistant_message
            tool_calls = streamed_tool_calls
            usage = None

        new_messages = [*messages, assistant_message]
        duration_seconds = perf_counter() - start_time

        pending_tool_calls = [
            tool_call for tool_call in tool_calls if tool_call.id not in completed_tool_ids
        ]

        if current_chunk_type == "tool":
            for tool_call in pending_tool_calls:
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
            for tool_call in pending_tool_calls:
                yield ResponseStreamToolCompleteChunk(
                    id=tool_call.id,
                    tool=tool_call.name,
                    arguments=tool_call.arguments,
                )
        yield ResponseStreamCompletionChunk(
            content=parser._final_content(
                assistant_message.content,
                response_format,
            ),
            thinking=assistant_message.thinking,
            messages=new_messages,
            tool_calls=tool_calls,
            usage=usage,
            duration_seconds=duration_seconds,
        )
    except Exception as exc:
        raise_llm_error(exc, provider=parser.PROVIDER_NAME)
    finally:
        await close_async_resource(response)


class AsyncOpenAIClient(AsyncOpenAICompatibleClient):
    def __init__(
        self,
        *,
        config: OpenAIClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = OpenAIClient(config=config, logger=logger)
        provider_client = create_async_provider_client(
            AsyncOpenAI,
            provider="openai",
            base_url=config.base_url,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
        )
