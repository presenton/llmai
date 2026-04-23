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
from llmai.shared.configs import AnthropicClientConfig
from llmai.shared.errors import raise_llm_error
from llmai.shared.messages import (
    AssistantMessage,
    AssistantToolCall,
    ImageContentPart,
    Message,
    SystemMessage,
    ToolResponseMessage,
    UserMessage,
    collapse_thinking_blocks,
    content_from_text,
    normalize_content_parts,
)
from llmai.shared.reasoning import ReasoningEffort
from llmai.shared.response_formats import (
    ResponseFormat,
    get_response_format_name,
    get_response_format_strict,
    get_response_schema,
)
from llmai.shared.responses import (
    ResponseContent,
    ResponseResult,
    ResponseStreamCompletionChunk,
    ResponseStreamContentChunk,
    ResponseStreamThinkingChunk,
    ResponseStreamToolCompleteChunk,
    ResponseStreamToolChunk,
    ResponseUsage,
)
from llmai.shared.schema import get_schema_as_dict
from llmai.shared.tools import (
    LLMTool,
    Tool,
    ToolChoice,
    WEB_SEARCH_TOOL_NAME,
    WebSearchTool,
    filter_resolved_tools_for_provider,
    resolve_tools,
)

ANTHROPIC_SUPPORTED_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "anyOf",
    "description",
    "enum",
    "items",
    "properties",
    "required",
    "type",
}


class AnthropicClient(BaseClient):
    def __init__(
        self,
        *,
        config: AnthropicClientConfig,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)
        try:
            self._client = Anthropic(
                api_key=config.api_key,
                base_url=config.base_url,
            )
        except Exception as exc:
            raise_llm_error(exc, provider="anthropic")

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

    def _get_anthropic_thinking_or_omit(
        self,
        reasoning_effort: ReasoningEffort | None,
    ) -> dict[str, object] | Omit:
        if reasoning_effort is None:
            return Omit()

        if reasoning_effort.effort == "none" or reasoning_effort.tokens == 0:
            return {"type": "disabled"}

        if reasoning_effort.tokens is not None:
            return {
                "type": "enabled",
                "budget_tokens": reasoning_effort.tokens,
            }

        return {"type": "adaptive"}

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
                input_schema=self._anthropic_input_schema(
                    tool.input_schema,
                    strict=tool.strict,
                ),
            )
            for tool in tools
        ]

    def _web_search_tool_to_anthropic_tool(
        self,
        tool: WebSearchTool,
    ) -> dict[str, object]:
        del tool
        return {
            "type": "web_search_20250305",
            "name": WEB_SEARCH_TOOL_NAME,
        }

    def _anthropic_input_schema(
        self,
        schema: object,
        *,
        strict: bool,
    ) -> dict:
        return get_schema_as_dict(
            schema,
            supported_keys=ANTHROPIC_SUPPORTED_SCHEMA_KEYS,
            supported_string_formats=None,
            strict=strict,
        )

    def _response_schema_tool(
        self,
        response_format: ResponseFormat | None,
        response_schema: dict,
    ) -> dict[str, object]:
        return {
            "name": get_response_format_name(response_format, default="response"),
            "description": "Provide the final response to the user",
            "strict": get_response_format_strict(response_format, default=True),
            "input_schema": response_schema,
        }

    def _get_anthropic_tools_and_tool_choice_or_omit(
        self,
        tools: list[LLMTool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
    ) -> tuple[list[dict[str, object]] | Omit, dict[str, object] | Omit]:
        resolved = filter_resolved_tools_for_provider(
            resolve_tools(tools, tool_choice),
            supports_web_search=True,
        )
        anthropic_tools: list[dict[str, object]] = list(
            self._llm_tools_to_anthropic_tools(resolved.function_tools)
        )
        if resolved.web_search_tool is not None:
            anthropic_tools.append(
                self._web_search_tool_to_anthropic_tool(resolved.web_search_tool)
            )

        response_schema = get_response_schema(
            response_format,
            supported_keys=ANTHROPIC_SUPPORTED_SCHEMA_KEYS,
            supported_string_formats=None,
            strict=get_response_format_strict(response_format, default=False),
        )
        response_schema_tool_name: str | None = None
        if response_schema:
            response_schema_tool_name = get_response_format_name(
                response_format,
                default="response",
            )
            anthropic_tools.append(
                self._response_schema_tool(response_format, response_schema)
            )

        anthropic_tool_choice: dict[str, object] | Omit = Omit()
        if resolved.requires_tool:
            if len(resolved.tools) == 1:
                anthropic_tool_choice = {
                    "type": "tool",
                    "name": resolved.tool_names[0],
                }
            else:
                anthropic_tool_choice = {"type": "any"}
        elif response_schema_tool_name:
            if resolved.is_explicit and resolved.tools:
                anthropic_tool_choice = {"type": "any"}
            else:
                anthropic_tool_choice = {
                    "type": "tool",
                    "name": response_schema_tool_name,
                }

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
        tools: list[LLMTool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
        stream: bool = False,
    ) -> ResponseResult:
        if stream:
            return self._generate_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                extra_body=extra_body,
            )

        return self._generate_once(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        )

    def _generate_once(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[LLMTool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
    ) -> ResponseContent:
        anthropic_tools, anthropic_tool_choice = (
            self._get_anthropic_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
                response_format,
            )
        )

        try:
            start_time = perf_counter()
            response: AnthropicMessage = self._client.messages.create(
                model=model,
                system=self._get_system_prompt(messages),
                messages=self._messages_to_anthropic_messages(messages),
                tools=anthropic_tools,
                tool_choice=anthropic_tool_choice,
                thinking=self._get_anthropic_thinking_or_omit(reasoning_effort),
                max_tokens=max_tokens or 8000,
                temperature=temperature or Omit(),
                extra_body=extra_body,
            )
            duration_seconds = perf_counter() - start_time

            text_chunks: list[str] = []
            thinking_blocks: list[str] = []
            response_schema_content: dict | None = None
            response_schema_tool_name = (
                get_response_format_name(response_format, default="response")
                if get_response_schema(
                    response_format,
                    supported_keys=ANTHROPIC_SUPPORTED_SCHEMA_KEYS,
                    supported_string_formats=None,
                    strict=get_response_format_strict(response_format, default=False),
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
                        response_schema_content = self._parse_tool_arguments(
                            tool_call.arguments
                        )
                    else:
                        user_tool_calls.append(tool_call)

            assistant_message = AssistantMessage(
                content=content_from_text("".join(text_chunks) or None),
                thinking=collapse_thinking_blocks(thinking_blocks),
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
                usage=self._response_usage(getattr(response, "usage", None)),
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider="anthropic")

    def _generate_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[LLMTool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
    ):
        anthropic_tools, anthropic_tool_choice = (
            self._get_anthropic_tools_and_tool_choice_or_omit(
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
                supported_keys=ANTHROPIC_SUPPORTED_SCHEMA_KEYS,
                supported_string_formats=None,
                strict=get_response_format_strict(response_format, default=False),
            )
            else None
        )
        user_tool_calls: list[AssistantToolCall] = []
        active_tool_name: str | None = None
        active_tool_id: str | None = None
        active_thinking_block: list[str] | None = None
        start_time = perf_counter()
        usage: ResponseUsage | None = None

        try:
            with self._client.messages.stream(
                model=model,
                system=self._get_system_prompt(messages),
                messages=self._messages_to_anthropic_messages(messages),
                tools=anthropic_tools,
                tool_choice=anthropic_tool_choice,
                thinking=self._get_anthropic_thinking_or_omit(reasoning_effort),
                max_tokens=max_tokens or 8000,
                temperature=temperature or Omit(),
                extra_body=extra_body,
            ) as stream_response:
                for event in stream_response:
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
                                    self._transition_stream_chunk(
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
                                self._transition_stream_chunk(
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
                        elif event.delta.type == "input_json_delta" and active_tool_name:
                            chunk = event.delta.partial_json
                            if active_tool_name == response_schema_tool_name:
                                continue
                            current_chunk_type, current_tool, stream_chunks = (
                                self._transition_stream_chunk(
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
                            stream_chunk = self._close_stream_chunk(
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
                            response_schema_content = self._parse_tool_arguments(
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
                    usage = self._response_usage(getattr(final_message, "usage", None))

            assistant_message = AssistantMessage(
                content=content_from_text("".join(text_chunks) or None),
                thinking=collapse_thinking_blocks(
                    [
                        *thinking_blocks,
                        *(["".join(active_thinking_block)] if active_thinking_block else []),
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

            stream_chunk = self._close_stream_chunk(
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
            raise_llm_error(exc, provider="anthropic")
