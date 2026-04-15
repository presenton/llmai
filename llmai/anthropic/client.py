from __future__ import annotations

import base64
import json
from logging import Logger
from time import perf_counter

from anthropic import Anthropic, Omit
from anthropic.types import (
    ImageBlockParam,
    Message as AnthropicMessage,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from llmai.shared.base import BaseClient
from llmai.shared.messages import (
    AssistantMessage,
    AssistantToolCall,
    ImageContentPart,
    Message,
    SystemMessage,
    ToolResponseMessage,
    UserMessage,
    content_from_text,
    normalize_content_parts,
)
from llmai.shared.response_formats import (
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


class AnthropicClient(BaseClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)
        self._client = Anthropic(api_key=api_key)

    def _get_system_prompt(self, messages: list[Message]) -> str | Omit:
        for message in messages:
            if isinstance(message, SystemMessage):
                return "".join(part.text for part in message.content)
        return Omit()

    def _parse_tool_arguments(self, arguments: str | None) -> dict:
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
        except Exception:
            return {}

        return parsed if isinstance(parsed, dict) else {}

    def _assistant_message_to_message_param(
        self,
        message: AssistantMessage,
    ) -> MessageParam:
        content_blocks: list[TextBlockParam | ImageBlockParam | ToolUseBlockParam] = (
            self._content_to_anthropic_blocks(message.content)
        )

        for each in message.tool_calls:
            content_blocks.append(
                ToolUseBlockParam(
                    type="tool_use",
                    id=each.id,
                    name=each.name,
                    input=self._parse_tool_arguments(each.arguments),
                )
            )

        return MessageParam(role="assistant", content=content_blocks)

    def _content_to_anthropic_blocks(
        self,
        content: list[object] | None,
    ) -> list[TextBlockParam | ImageBlockParam]:
        blocks: list[TextBlockParam | ImageBlockParam] = []

        for part in normalize_content_parts(content):
            if isinstance(part, ImageContentPart):
                if part.url is not None:
                    source: dict[str, object] = {
                        "type": "url",
                        "url": part.url,
                    }
                else:
                    encoded = base64.b64encode(part.data or b"").decode("ascii")
                    source = {
                        "type": "base64",
                        "data": encoded,
                        "media_type": part.mime_type,
                    }

                blocks.append(
                    ImageBlockParam(
                        type="image",
                        source=source,
                    )
                )
            else:
                blocks.append(
                    TextBlockParam(
                        type="text",
                        text=part.text,
                    )
                )

        return blocks

    def _messages_to_anthropic_messages(
        self,
        messages: list[Message],
    ) -> list[MessageParam]:
        anthropic_messages: list[MessageParam] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                continue

            if isinstance(message, UserMessage):
                anthropic_messages.append(
                    MessageParam(
                        role="user",
                        content=self._content_to_anthropic_blocks(message.content),
                    )
                )
            elif isinstance(message, AssistantMessage):
                anthropic_messages.append(
                    self._assistant_message_to_message_param(message)
                )
            elif isinstance(message, ToolResponseMessage):
                anthropic_messages.append(
                    MessageParam(
                        role="user",
                        content=[
                            ToolResultBlockParam(
                                type="tool_result",
                                tool_use_id=message.id,
                                content="".join(
                                    part.text for part in (message.content or [])
                                ),
                                is_error=False,
                            )
                        ],
                    )
                )

        return anthropic_messages

    def _llm_tools_to_anthropic_tools(self, tools: list[Tool]) -> list[ToolParam]:
        return [
            ToolParam(
                name=tool.name,
                description=tool.description,
                strict=tool.strict,
                input_schema=get_schema_as_dict(tool.input_schema),
            )
            for tool in tools
        ]

    def _response_schema_tool(self, response_schema: dict) -> dict[str, object]:
        return {
            "name": "ResponseSchema",
            "description": "Provide the final response to the user",
            "input_schema": response_schema,
        }

    def _get_anthropic_tools_and_tool_choice_or_omit(
        self,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> tuple[list[dict[str, object]] | Omit, dict[str, object] | Omit]:
        resolved = resolve_tools(tools, tool_choice)
        anthropic_tools: list[dict[str, object]] = list(
            self._llm_tools_to_anthropic_tools(resolved.tools)
        )

        response_schema = get_response_schema(response_format)
        if response_schema and use_tools_for_structured_output is not False:
            anthropic_tools.append(self._response_schema_tool(response_schema))

        anthropic_tool_choice: dict[str, object] | Omit = Omit()
        if resolved.required_names:
            if len(resolved.required_names) == 1 and not resolved.optional_names:
                anthropic_tool_choice = {
                    "type": "tool",
                    "name": resolved.required_names[0],
                }
            else:
                anthropic_tool_choice = {"type": "any"}

        return anthropic_tools or Omit(), anthropic_tool_choice

    def _response_usage(self, usage: object | None) -> ResponseUsage | None:
        raw_usage = self._dump_model(usage)
        if not raw_usage:
            return None

        direct_input_tokens = raw_usage.get("input_tokens")
        cache_creation_input_tokens = raw_usage.get("cache_creation_input_tokens") or 0
        cache_read_input_tokens = raw_usage.get("cache_read_input_tokens") or 0
        output_tokens = raw_usage.get("output_tokens")

        input_tokens = None
        if (
            direct_input_tokens is not None
            or cache_creation_input_tokens
            or cache_read_input_tokens
        ):
            input_tokens = (
                (direct_input_tokens or 0)
                + cache_creation_input_tokens
                + cache_read_input_tokens
            )

        total_tokens = None
        if input_tokens is not None or output_tokens is not None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        details = dict(raw_usage)
        details.pop("input_tokens", None)
        details.pop("output_tokens", None)

        return ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            details=details,
        )

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
        anthropic_tools, anthropic_tool_choice = (
            self._get_anthropic_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
                response_format,
                use_tools_for_structured_output,
            )
        )

        start_time = perf_counter()
        response: AnthropicMessage = self._client.messages.create(
            model=model,
            system=self._get_system_prompt(messages),
            messages=self._messages_to_anthropic_messages(messages),
            tools=anthropic_tools,
            tool_choice=anthropic_tool_choice,
            max_tokens=max_tokens or 8000,
            temperature=temperature or Omit(),
            extra_body=extra_body,
        )
        duration_seconds = perf_counter() - start_time

        text_chunks: list[str] = []
        thinking_chunks: list[str] = []
        response_schema_content: dict | None = None
        user_tool_calls: list[AssistantToolCall] = []
        for content in response.content:
            if content.type == "text":
                text_chunks.append(content.text)
            elif content.type == "thinking":
                thinking_chunks.append(content.thinking)
            elif content.type == "tool_use":
                tool_call = AssistantToolCall(
                    id=content.id,
                    name=content.name,
                    arguments=json.dumps(content.input),
                )
                if tool_call.name == "ResponseSchema":
                    response_schema_content = self._parse_tool_arguments(
                        tool_call.arguments
                    )
                else:
                    user_tool_calls.append(tool_call)

        assistant_message = AssistantMessage(
            content=content_from_text("".join(text_chunks) or None),
            thinking="".join(thinking_chunks) or None,
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
            messages=new_messages,
            tool_calls=user_tool_calls,
            usage=self._response_usage(getattr(response, "usage", None)),
            duration_seconds=duration_seconds,
        )

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
        anthropic_tools, anthropic_tool_choice = (
            self._get_anthropic_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
                response_format,
                use_tools_for_structured_output,
            )
        )

        stream_id = "0"
        text_chunks: list[str] = []
        thinking_chunks: list[str] = []
        response_schema_content: dict | None = None
        user_tool_calls: list[AssistantToolCall] = []
        active_tool_name: str | None = None
        start_time = perf_counter()
        usage: ResponseUsage | None = None

        with self._client.messages.stream(
            model=model,
            system=self._get_system_prompt(messages),
            messages=self._messages_to_anthropic_messages(messages),
            tools=anthropic_tools,
            tool_choice=anthropic_tool_choice,
            max_tokens=max_tokens or 8000,
            temperature=temperature or Omit(),
            extra_body=extra_body,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        active_tool_name = event.content_block.name
                    continue

                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        text_chunks.append(event.delta.text)
                        if not use_tools_for_structured_output:
                            yield ResponseStreamContentChunk(
                                id=stream_id,
                                source="direct",
                                chunk=event.delta.text,
                            )
                    elif event.delta.type == "thinking_delta":
                        thinking_chunks.append(event.delta.thinking)
                    elif event.delta.type == "input_json_delta" and active_tool_name:
                        chunk = event.delta.partial_json
                        if active_tool_name == "ResponseSchema":
                            yield ResponseStreamContentChunk(
                                id=stream_id,
                                source="direct",
                                chunk=chunk,
                            )
                        else:
                            yield ResponseStreamContentChunk(
                                id=stream_id,
                                source="tool",
                                tool=active_tool_name,
                                chunk=chunk,
                            )
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
                    if tool_call.name == "ResponseSchema":
                        response_schema_content = self._parse_tool_arguments(
                            tool_call.arguments
                        )
                    else:
                        user_tool_calls.append(tool_call)
                    active_tool_name = None

            if hasattr(stream, "get_final_message"):
                final_message = stream.get_final_message()
                usage = self._response_usage(getattr(final_message, "usage", None))

        assistant_message = AssistantMessage(
            content=content_from_text("".join(text_chunks) or None),
            thinking="".join(thinking_chunks) or None,
            tool_calls=user_tool_calls,
        )
        new_messages = [*messages, assistant_message]

        final_content: object = response_schema_content
        if final_content is None:
            final_content = assistant_message.content
        if final_content is None and not user_tool_calls:
            final_content = ""
        duration_seconds = perf_counter() - start_time

        yield ResponseStreamCompletionChunk(
            id=stream_id,
            content=final_content,
            messages=new_messages,
            tool_calls=user_tool_calls,
            usage=usage,
            duration_seconds=duration_seconds,
        )
