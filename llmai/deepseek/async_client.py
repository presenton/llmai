from __future__ import annotations

from collections.abc import AsyncIterator
from logging import Logger
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI

from llmai.deepseek.client import DeepSeekClient
from llmai.openai.async_client import (
    AsyncOpenAICompatibleClient,
    create_async_provider_client,
)
from llmai.shared.async_utils import close_async_resource
from llmai.shared.configs import DeepSeekClientConfig
from llmai.shared.errors import LLMError, raise_llm_error
from llmai.shared.messages import (
    AssistantMessage,
    AssistantToolCall,
    Message,
    content_from_text,
)
from llmai.shared.responses import (
    ResponseContent,
    ResponseStreamCompletionChunk,
    ResponseStreamContentChunk,
    ResponseStreamEvent,
    ResponseStreamToolChunk,
    ResponseStreamToolCompleteChunk,
)


class AsyncDeepSeekClient(AsyncOpenAICompatibleClient):
    def __init__(
        self,
        *,
        config: DeepSeekClientConfig,
        logger: Logger | None = None,
    ):
        sync_client = DeepSeekClient(config=config, logger=logger)
        provider_client = create_async_provider_client(
            AsyncOpenAI,
            provider="deepseek",
            base_url=config.base_url or DeepSeekClient.DEFAULT_BASE_URL,
            api_key=config.api_key,
        )
        super().__init__(
            sync_client=sync_client,
            provider_client=provider_client,
        )

    @property
    def _parser(self) -> DeepSeekClient:
        return self._sync_client

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
        del reasoning_effort

        parser = self._parser
        deepseek_tools, deepseek_tool_choice, response_schema_tool_name = (
            parser._get_deepseek_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
                response_format,
            )
        )

        try:
            start_time = perf_counter()
            response = await self._provider_client.chat.completions.create(
                model=model,
                messages=parser._messages_to_openai_messages(messages),
                temperature=temperature,
                response_format=parser._get_deepseek_response_format_or_omit(
                    response_format
                ),
                tools=deepseek_tools,
                tool_choice=deepseek_tool_choice,
                max_completion_tokens=max_tokens,
                extra_body=extra_body,
            )
            duration_seconds = perf_counter() - start_time

            if not response.choices:
                raise LLMError(400, "No content returned from LLM")

            raw_assistant_message = parser._chat_completion_message_to_assistant_message(
                response.choices[0].message
            )
            response_schema_content: dict | None = None
            user_tool_calls: list[AssistantToolCall] = []
            for tool_call in raw_assistant_message.tool_calls:
                if tool_call.name == response_schema_tool_name:
                    response_schema_content = parser._parse_tool_arguments(
                        tool_call.arguments
                    )
                else:
                    user_tool_calls.append(tool_call)

            assistant_message = AssistantMessage(
                content=raw_assistant_message.content,
                tool_calls=user_tool_calls,
            )
            new_messages = [*messages, assistant_message]

            return ResponseContent(
                content=parser._final_content(
                    assistant_message.content,
                    response_format,
                    response_schema_content=response_schema_content,
                ),
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
        del reasoning_effort

        parser = self._parser
        deepseek_tools, deepseek_tool_choice, response_schema_tool_name = (
            parser._get_deepseek_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
                response_format,
            )
        )
        response = None

        try:
            start_time = perf_counter()
            response = await self._provider_client.chat.completions.create(
                model=model,
                messages=parser._messages_to_openai_messages(messages),
                temperature=temperature,
                response_format=parser._get_deepseek_response_format_or_omit(
                    response_format
                ),
                tools=deepseek_tools,
                tool_choice=deepseek_tool_choice,
                max_completion_tokens=max_tokens,
                extra_body=extra_body,
                stream=True,
                stream_options={"include_usage": True},
            )

            current_chunk_type = None
            current_tool = None
            content = ""
            partial_tool_calls: dict[int, dict[str, str | int | None]] = {}
            tool_order: list[int] = []
            usage = None

            async for event in response:
                event_usage = parser._response_usage(getattr(event, "usage", None))
                if event_usage is not None:
                    usage = event_usage

                if not getattr(event, "choices", None):
                    continue

                delta = event.choices[0].delta

                if delta.content:
                    content += delta.content
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
                        yield ResponseStreamContentChunk(chunk=delta.content)

                if not delta.tool_calls:
                    continue

                for tool_call_delta in delta.tool_calls:
                    current = partial_tool_calls.get(tool_call_delta.index)
                    if current is None:
                        current = {
                            "id": None,
                            "name": None,
                            "arguments": "",
                            "emitted_length": 0,
                        }
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
                    if tool_arguments:
                        current["arguments"] = f"{current['arguments']}{tool_arguments}"

                    tool_name = current["name"]
                    if not tool_name:
                        continue

                    arguments_text = str(current["arguments"] or "")
                    emitted_length = int(current["emitted_length"] or 0)
                    if emitted_length >= len(arguments_text):
                        continue
                    new_chunk = arguments_text[emitted_length:]

                    if tool_name == response_schema_tool_name:
                        content += new_chunk
                        current_chunk_type, current_tool, stream_chunks = (
                            parser._transition_stream_chunk(
                                current_chunk_type=current_chunk_type,
                                next_chunk_type="content",
                                current_tool=current_tool,
                            )
                        )
                        for stream_chunk in stream_chunks:
                            yield stream_chunk

                        yield ResponseStreamContentChunk(chunk=new_chunk)
                        current["emitted_length"] = len(arguments_text)
                        continue

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
                        id=str(current["id"] or tool_name),
                        tool=tool_name,
                        chunk=new_chunk,
                    )
                    current["emitted_length"] = len(arguments_text)

            response_schema_content: dict | None = None
            user_tool_calls: list[AssistantToolCall] = []
            for index in tool_order:
                partial_tool_call = partial_tool_calls[index]
                tool_name = partial_tool_call["name"]
                if not tool_name:
                    continue

                tool_call = AssistantToolCall(
                    id=str(partial_tool_call["id"] or tool_name),
                    name=tool_name,
                    arguments=str(partial_tool_call["arguments"] or "") or None,
                )
                if tool_call.name == response_schema_tool_name:
                    response_schema_content = parser._parse_tool_arguments(
                        tool_call.arguments
                    )
                else:
                    user_tool_calls.append(tool_call)

            assistant_message = AssistantMessage(
                content=content_from_text(content or None),
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
                    response_format,
                    response_schema_content=response_schema_content,
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
