from __future__ import annotations

import base64
import json
from logging import Logger
from time import perf_counter
from uuid import uuid4

from openai import Omit, OpenAI

from llmai.shared.base import BaseClient
from llmai.shared.configs import ChatGPTClientConfig
from llmai.shared.errors import LLMError, configuration_error, raise_llm_error
from llmai.shared.messages import (
    AssistantContent,
    AssistantMessage,
    AssistantReasoningItem,
    AssistantToolCall,
    ImageContentPart,
    Message,
    MessageContent,
    SystemMessage,
    ToolResponseMessage,
    UserMessage,
    content_from_text,
    content_has_images,
    normalize_content_parts,
)
from llmai.shared.reasoning import ReasoningEffort
from llmai.shared.response_formats import (
    JSONSchemaResponse,
    JSONObjectResponse,
    ResponseFormat,
    TextResponse,
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

CHATGPT_DEFAULT_INSTRUCTIONS = "Follow the prompt"
OPENAI_SUPPORTED_STRING_FORMATS = {
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "uuid",
}
OPENAI_SUPPORTED_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "anyOf",
    "const",
    "description",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "items",
    "maximum",
    "maxItems",
    "minimum",
    "minItems",
    "multipleOf",
    "pattern",
    "properties",
    "required",
    "type",
}


class ChatGPTClient(BaseClient):
    PROVIDER_NAME = "chatgpt"
    DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"

    def __init__(
        self,
        *,
        config: ChatGPTClientConfig,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)
        self._base_url = config.base_url or self.DEFAULT_BASE_URL
        resolved_access_token = self._resolve_access_token(config.access_token)
        resolved_account_id = _strip_or_none(config.account_id)

        default_headers = {
            "OpenAI-Beta": "responses=experimental",
            "originator": "pi",
        }
        if resolved_account_id is not None:
            default_headers["chatgpt-account-id"] = resolved_account_id

        try:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=resolved_access_token,
                default_headers=default_headers,
                timeout=120.0,
            )
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

        if self._logger:
            self._logger.info("ChatGPT client created")
            self._logger.info("Base URL: %s", self._base_url)

    def _response_item_id(self, prefix: str = "item") -> str:
        return f"{prefix}_{uuid4().hex}"

    def _assistant_content_to_openai_content(
        self,
        content: AssistantContent,
    ) -> str | None:
        if content is None:
            return None

        if isinstance(content, str):
            return content

        if content_has_images(content):
            raise LLMError(
                400,
                "ChatGPT conversation history does not support assistant image content",
                provider=self.PROVIDER_NAME,
            )

        return "".join(part.text for part in normalize_content_parts(content))

    def _text_content_to_string(
        self,
        content: list[object] | None,
    ) -> str:
        return "".join(
            part.text
            for part in normalize_content_parts(content)
            if hasattr(part, "text")
        )

    def _image_content_part_to_openai_image_url(
        self,
        part: ImageContentPart,
    ) -> str:
        if part.url is not None:
            return part.url

        encoded = base64.b64encode(part.data or b"").decode("ascii")
        return f"data:{part.mime_type};base64,{encoded}"

    def _message_content_to_responses_content(
        self,
        content: MessageContent,
    ) -> list[dict[str, object]]:
        openai_content: list[dict[str, object]] = []
        for part in normalize_content_parts(content):
            if isinstance(part, ImageContentPart):
                openai_content.append(
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": self._image_content_part_to_openai_image_url(part),
                    }
                )
            else:
                openai_content.append(
                    {
                        "type": "input_text",
                        "text": part.text,
                    }
                )

        return openai_content

    def _assistant_message_to_responses_input_items(
        self,
        message: AssistantMessage,
    ) -> list[dict[str, object]]:
        input_items: list[dict[str, object]] = []

        for reasoning_item in message.thinking or []:
            if reasoning_item.id is None:
                continue

            serialized_reasoning_item: dict[str, object] = {
                "id": reasoning_item.id,
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": summary_text,
                    }
                    for summary_text in reasoning_item.summary
                ],
            }
            if reasoning_item.encrypted_content is not None:
                serialized_reasoning_item["encrypted_content"] = (
                    reasoning_item.encrypted_content
                )
            input_items.append(serialized_reasoning_item)

        text_content = self._assistant_content_to_openai_content(message.content)
        if text_content is not None:
            message_item: dict[str, object] = {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text_content,
                        "annotations": [],
                    }
                ],
            }
            if message.id is not None:
                message_item["id"] = message.id
            input_items.append(message_item)

        for tool_call in message.tool_calls:
            input_items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments or "",
                }
            )

        return input_items

    def _messages_to_responses_input(
        self,
        messages: list[Message],
    ) -> list[dict[str, object]]:
        responses_input: list[dict[str, object]] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            if isinstance(message, UserMessage):
                responses_input.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": self._message_content_to_responses_content(
                            message.content
                        ),
                    }
                )
                continue
            if isinstance(message, AssistantMessage):
                responses_input.extend(
                    self._assistant_message_to_responses_input_items(message)
                )
                continue
            if isinstance(message, ToolResponseMessage):
                responses_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.id,
                        "output": self._text_content_to_string(message.content),
                    }
                )

        return responses_input

    def _messages_to_responses_instructions(
        self,
        messages: list[Message],
    ) -> str:
        system_messages = [
            message.content for message in messages if isinstance(message, SystemMessage)
        ]
        if not system_messages:
            return CHATGPT_DEFAULT_INSTRUCTIONS

        return "\n\n".join(system_messages)

    def _get_responses_text_or_omit(
        self,
        response_format: ResponseFormat | None,
    ) -> dict[str, object] | Omit:
        if isinstance(response_format, JSONSchemaResponse):
            return {
                "format": {
                    "type": "json_schema",
                    "name": get_response_format_name(
                        response_format,
                        default="response",
                    ),
                    "schema": get_response_schema(
                        response_format,
                        supported_keys=OPENAI_SUPPORTED_SCHEMA_KEYS,
                        supported_string_formats=OPENAI_SUPPORTED_STRING_FORMATS,
                        strict=get_response_format_strict(
                            response_format,
                            default=False,
                        ),
                    )
                    or {},
                    "strict": get_response_format_strict(response_format, default=True),
                }
            }

        if isinstance(response_format, JSONObjectResponse):
            return {
                "format": {"type": "json_object"},
            }

        if isinstance(response_format, TextResponse):
            return {
                "format": {"type": "text"},
            }

        return Omit()

    def _llm_tools_to_responses_tools(
        self,
        tools: list[Tool],
    ) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": get_schema_as_dict(
                    tool.input_schema,
                    supported_keys=OPENAI_SUPPORTED_SCHEMA_KEYS,
                    supported_string_formats=OPENAI_SUPPORTED_STRING_FORMATS,
                    strict=tool.strict,
                ),
                "strict": tool.strict,
            }
            for tool in tools
        ]

    def _web_search_tool_to_responses_tool(
        self,
        tool: WebSearchTool,
    ) -> dict[str, object]:
        del tool
        return {"type": WEB_SEARCH_TOOL_NAME}

    def _get_responses_tools_and_tool_choice_or_omit(
        self,
        tools: list[LLMTool] | None,
        tool_choice: ToolChoice | None,
    ) -> tuple[list[dict[str, object]] | Omit, object | Omit]:
        resolved = filter_resolved_tools_for_provider(
            resolve_tools(tools, tool_choice),
            supports_web_search=True,
        )
        responses_tools = self._llm_tools_to_responses_tools(
            resolved.function_tools
        )
        if resolved.web_search_tool is not None:
            responses_tools.append(
                self._web_search_tool_to_responses_tool(resolved.web_search_tool)
            )
        if not responses_tools:
            return Omit(), Omit()

        if resolved.has_web_search and (resolved.is_explicit or resolved.requires_tool):
            return responses_tools, {
                "type": "allowed_tools",
                "mode": "required" if resolved.requires_tool else "auto",
                "tools": responses_tools,
            }

        if not resolved.requires_tool:
            return responses_tools, Omit()

        if len(resolved.function_tools) == 1:
            return responses_tools, {
                "type": "function",
                "name": resolved.function_tools[0].name,
            }

        return responses_tools, "required"

    def _final_content(
        self,
        content: AssistantContent,
        response_format: ResponseFormat | None,
    ) -> object:
        text_content = self._assistant_content_to_openai_content(content)
        if text_content and isinstance(
            response_format, (JSONSchemaResponse, JSONObjectResponse)
        ):
            return json.loads(text_content)

        return content

    def _response_usage(self, usage: object | None) -> ResponseUsage | None:
        raw_usage = self._dump_model(usage)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        if not raw_usage and all(
            value is None for value in (input_tokens, output_tokens, total_tokens)
        ):
            return None

        details = dict(raw_usage)
        details.pop("input_tokens", None)
        details.pop("output_tokens", None)
        details.pop("total_tokens", None)

        return ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            details=details,
        )

    def _responses_reasoning_and_extra_body(
        self,
        reasoning_effort: ReasoningEffort | None,
        extra_body: dict | None,
    ) -> tuple[dict[str, object] | Omit, dict | None]:
        request_extra_body = dict(extra_body or {})
        raw_reasoning = request_extra_body.pop("reasoning", None)

        if raw_reasoning is None:
            reasoning: dict[str, object] = {}
        elif isinstance(raw_reasoning, dict):
            reasoning = dict(raw_reasoning)
        elif reasoning_effort is None:
            return raw_reasoning, request_extra_body or None
        else:
            reasoning = {}

        if reasoning_effort is not None:
            if reasoning_effort.effort is not None:
                reasoning["effort"] = reasoning_effort.effort
            if reasoning_effort.summary is not None:
                reasoning["summary"] = reasoning_effort.summary

        reasoning.setdefault("summary", "auto")
        return reasoning or Omit(), request_extra_body or None

    def _responses_output_to_assistant_message(
        self,
        output: list[object],
    ) -> AssistantMessage:
        text_chunks: list[str] = []
        thinking_items: list[AssistantReasoningItem] = []
        tool_calls: list[AssistantToolCall] = []
        assistant_message_id: str | None = None

        for item in output:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                assistant_message_id = assistant_message_id or getattr(item, "id", None)
                for content in getattr(item, "content", []) or []:
                    content_type = getattr(content, "type", None)
                    if content_type == "output_text" and getattr(content, "text", None):
                        text_chunks.append(content.text)
                    elif content_type == "refusal" and getattr(
                        content, "refusal", None
                    ):
                        text_chunks.append(content.refusal)
            elif item_type == "reasoning":
                summary_texts: list[str] = []
                for summary in getattr(item, "summary", []) or []:
                    if getattr(summary, "text", None):
                        summary_texts.append(summary.text)
                if summary_texts or getattr(item, "id", None) is not None:
                    thinking_items.append(
                        AssistantReasoningItem(
                            id=getattr(item, "id", None),
                            summary=summary_texts,
                            encrypted_content=getattr(item, "encrypted_content", None),
                        )
                    )
            elif item_type == "function_call":
                tool_calls.append(
                    AssistantToolCall(
                        id=getattr(item, "call_id", None) or getattr(item, "id", None) or "",
                        name=getattr(item, "name", None) or "",
                        arguments=getattr(item, "arguments", None),
                    )
                )

        return AssistantMessage(
            id=assistant_message_id,
            content=content_from_text("".join(text_chunks) or None),
            thinking=thinking_items or None,
            tool_calls=tool_calls,
        )

    def _responses_request_kwargs(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[LLMTool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        reasoning_effort: ReasoningEffort | None,
        extra_body: dict | None,
        stream: bool,
    ) -> dict[str, object]:
        responses_tools, responses_tool_choice = (
            self._get_responses_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
            )
        )
        reasoning, request_extra_body = self._responses_reasoning_and_extra_body(
            reasoning_effort,
            extra_body,
        )
        return {
            "model": model,
            "input": self._messages_to_responses_input(messages),
            "temperature": Omit(),
            "text": self._get_responses_text_or_omit(response_format),
            "tools": responses_tools,
            "tool_choice": responses_tool_choice,
            "reasoning": reasoning,
            "max_output_tokens": Omit(),
            "extra_body": request_extra_body,
            "instructions": self._messages_to_responses_instructions(messages),
            "stream": stream,
        }

    def _generate_responses_once(
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
        del temperature
        del max_tokens

        completion_chunk = None
        for chunk in self._generate_responses_stream(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        ):
            if getattr(chunk, "type", None) == "completion":
                completion_chunk = chunk

        if completion_chunk is None:
            raise LLMError(
                500,
                "No completion returned from streamed ChatGPT response",
                provider=self.PROVIDER_NAME,
            )

        return ResponseContent(
            content=completion_chunk.content,
            thinking=completion_chunk.thinking,
            messages=completion_chunk.messages,
            tool_calls=completion_chunk.tool_calls,
            usage=completion_chunk.usage,
            duration_seconds=completion_chunk.duration_seconds,
        )

    def _generate_responses_stream(
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
        del temperature
        del max_tokens

        request_kwargs = self._responses_request_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            stream=True,
        )

        try:
            start_time = perf_counter()
            response = self._client.responses.create(**request_kwargs)

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

            for event in response:
                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    streamed_assistant_message_id = (
                        streamed_assistant_message_id
                        or getattr(event, "item_id", None)
                    )
                    if current_chunk_type == "thinking":
                        active_thinking_key = None
                    content += event.delta
                    current_chunk_type, current_tool, stream_chunks = (
                        self._transition_stream_chunk(
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
                        stream_chunk = self._close_stream_chunk(
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
                        self._transition_stream_chunk(
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
                        stream_chunk = self._close_stream_chunk(
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

                    tool_key = getattr(item, "id", None) or getattr(
                        item, "call_id", None
                    ) or self._response_item_id("tool")
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

                    tool_key = getattr(item, "id", None) or getattr(
                        item, "call_id", None
                    )
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
                        self._transition_stream_chunk(
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

            thinking_blocks = [
                thinking_blocks_by_key[key]
                for key in thinking_order
                if thinking_blocks_by_key[key]
            ]
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
                streamed_thinking_by_id[item_id]
                for item_id in streamed_thinking_order
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
                response_assistant_message = self._responses_output_to_assistant_message(
                    getattr(final_response, "output", []) or []
                )
                thinking = response_assistant_message.thinking
                if streamed_assistant_message.thinking and not all(
                    thinking_item.id is not None
                    for thinking_item in (thinking or [])
                ):
                    thinking = streamed_assistant_message.thinking
                assistant_message = AssistantMessage(
                    id=(
                        response_assistant_message.id
                        or streamed_assistant_message.id
                    ),
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
                usage = self._response_usage(getattr(final_response, "usage", None))
            else:
                assistant_message = streamed_assistant_message
                tool_calls = streamed_tool_calls
                usage = None

            new_messages = [*messages, assistant_message]
            duration_seconds = perf_counter() - start_time

            pending_tool_calls = [
                tool_call
                for tool_call in tool_calls
                if tool_call.id not in completed_tool_ids
            ]

            if current_chunk_type == "tool":
                for tool_call in pending_tool_calls:
                    yield ResponseStreamToolCompleteChunk(
                        id=tool_call.id,
                        tool=tool_call.name,
                        arguments=tool_call.arguments,
                    )

            stream_chunk = self._close_stream_chunk(
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
                content=self._final_content(
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
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

    def _resolve_access_token(self, access_token: str | None) -> str:
        resolved_access_token = _strip_or_none(access_token)
        if resolved_access_token is not None:
            return resolved_access_token

        raise configuration_error(
            "Missing ChatGPT access token. Provide config.access_token",
            provider=self.PROVIDER_NAME,
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
        request_extra_body = {
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": True,
            **(extra_body or {}),
        }
        if response_format is None and "text" not in request_extra_body:
            request_extra_body["text"] = {"verbosity": "medium"}

        if stream:
            return self._generate_responses_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                extra_body=request_extra_body,
            )

        return self._generate_responses_once(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra_body=request_extra_body,
        )


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None
