from __future__ import annotations

import base64
from copy import deepcopy
import json
from logging import Logger
from time import perf_counter
from uuid import uuid4

from openai import Omit, OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.completion_create_params import (
    ResponseFormat as OpenAIResponseFormat,
)
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.shared_params.response_format_json_object import (
    ResponseFormatJSONObject,
)
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)
from openai.types.shared_params.response_format_text import ResponseFormatText

from llmai.shared.base import BaseClient
from llmai.shared.configs import OpenAIApiType, OpenAIClientConfig
from llmai.shared.errors import LLMError, configuration_error, raise_llm_error
from llmai.shared.generation import GenerationProfile
from llmai.shared.logs import LogLevel
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
from llmai.shared.model_listing import model_ids
from llmai.shared.reasoning import ReasoningConfig, ReasoningEffort
from llmai.shared.response_formats import (
    JSONObjectResponse,
    JSONSchemaResponse,
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
    ResponseStreamToolChunk,
    ResponseStreamToolCompleteChunk,
    ResponseUsage,
)
from llmai.shared.schema import get_schema_as_dict, process_schema
from llmai.shared.tools import (
    WEB_SEARCH_TOOL_NAME,
    LLMTool,
    Tool,
    ToolChoice,
    WebSearchTool,
    filter_resolved_tools_for_provider,
    resolve_tools,
)


def _json_type_for_const(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(
        f"Unsupported JSON Schema const value type: {type(value).__name__}"
    )


def _normalize_openai_strict_schema(schema: dict) -> dict:
    """Translate valid JSON Schema constructs to OpenAI's strict subset."""

    def normalize(value: object, *, in_named_schema_map: bool = False) -> object:
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if not isinstance(value, dict):
            return value

        if in_named_schema_map:
            return {key: normalize(item) for key, item in value.items()}

        normalized = {
            key: (
                deepcopy(item)
                if key in {"const", "enum", "default", "examples", "example"}
                else normalize(
                    item,
                    in_named_schema_map=key
                    in {"properties", "$defs", "definitions"},
                )
            )
            for key, item in value.items()
        }

        one_of = normalized.get("oneOf")
        if isinstance(one_of, list):
            if "anyOf" in normalized:
                raise ValueError(
                    "OpenAI strict schema node cannot contain both oneOf and anyOf"
                )
            normalized.pop("oneOf")
            normalized["anyOf"] = one_of

        if "const" in normalized:
            const_value = normalized.pop("const")
            enum_values = normalized.get("enum")
            if isinstance(enum_values, list) and const_value not in enum_values:
                raise ValueError(
                    "OpenAI strict schema const must be included in its enum"
                )
            normalized["enum"] = [const_value]
            normalized.setdefault("type", _json_type_for_const(const_value))

        return normalized

    normalized = normalize(schema)
    if not isinstance(normalized, dict):
        raise TypeError("JSON Schema root must be an object")
    return normalized


class OpenAIClient(BaseClient):
    PROVIDER_NAME = "openai"
    PROVIDER_LABEL = "OpenAI"
    STRICT_SUPPORTED_STRING_FORMATS = [
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "uuid",
    ]
    STRICT_SUPPORTED_SCHEMA_FIELDS = [
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "description",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "items",
        "maxItems",
        "maximum",
        "minItems",
        "minimum",
        "maxLength",
        "minLength",
        "multipleOf",
        "pattern",
        "properties",
        "required",
        "type",
    ]

    def __init__(
        self,
        *,
        config: OpenAIClientConfig,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger, generation_defaults=config.generation)
        self._api_type = self._coerce_api_type(config.api_type)
        if self._api_type is None:
            raise configuration_error(
                f"Unsupported OpenAI api_type: {config.api_type}",
                provider=self.PROVIDER_NAME,
            )
        self._provide_system_message_as_instructions = (
            config.provide_system_message_as_instructions
        )
        try:
            self._client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
            )
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

        if self._logger:
            self._logger.info("%s client created", self.PROVIDER_LABEL)
            self._logger.info("Base URL: %s", config.base_url)

    def list_available_models(self) -> list[str]:
        try:
            return model_ids(self._client.models.list())
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

    def _chat_completion_message_to_assistant_message(
        self,
        message: ChatCompletionMessage,
    ) -> AssistantMessage:
        return AssistantMessage(
            content=content_from_text(message.content),
            thinking=self._chat_completion_message_to_thinking_items(message) or None,
            tool_calls=[
                AssistantToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in (message.tool_calls or [])
            ],
        )

    def _chat_completion_message_to_thinking_items(
        self,
        message: ChatCompletionMessage,
    ) -> list[AssistantReasoningItem]:
        items: list[AssistantReasoningItem] = []
        details = getattr(message, "reasoning_details", None) or []
        for detail in details:
            raw = self._dump_model(detail)
            text = raw.get("text") or raw.get("reasoning")
            summary = raw.get("summary")
            if isinstance(summary, str):
                summaries = [summary]
            elif isinstance(summary, list):
                summaries = [str(value) for value in summary]
            else:
                summaries = [str(text)] if text else []
            items.append(
                AssistantReasoningItem(
                    id=raw.get("id"),
                    summary=summaries,
                    encrypted_content=raw.get("encrypted_content"),
                    provider=self.PROVIDER_NAME,
                    raw=raw,
                )
            )
        if items:
            return items
        text = getattr(message, "reasoning_content", None)
        if not text:
            raw_reasoning = getattr(message, "reasoning", None)
            text = raw_reasoning if isinstance(raw_reasoning, str) else None
        return (
            [AssistantReasoningItem(summary=[text], provider=self.PROVIDER_NAME)]
            if text
            else []
        )

    def _chat_completion_delta_to_thinking_text(self, delta: object) -> str | None:
        text = getattr(delta, "reasoning_content", None)
        if not text:
            reasoning = getattr(delta, "reasoning", None)
            text = reasoning if isinstance(reasoning, str) else None
        return text or None

    def _response_item_id(self, prefix: str = "item") -> str:
        return f"{prefix}_{uuid4().hex}"

    def _assistant_message_to_chat_completion_assistant_message_param(
        self,
        message: AssistantMessage,
    ) -> ChatCompletionAssistantMessageParam:
        tool_calls = [
            ChatCompletionMessageFunctionToolCallParam(
                id=tool_call.id,
                type="function",
                function=Function(
                    name=tool_call.name,
                    arguments=tool_call.arguments or "",
                ),
            )
            for tool_call in message.tool_calls
        ]

        result = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=self._assistant_content_to_openai_content(message.content),
        )
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    def _responses_finish_reason(self, response: object) -> str | None:
        """Normalize Responses API termination metadata to chat finish reasons."""

        if getattr(response, "status", None) == "completed":
            return "stop"
        incomplete = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None)
        if reason in {"max_output_tokens", "max_tokens"}:
            return "length"
        return str(reason) if reason else None

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
                "OpenAI chat completions does not support assistant message image content in conversation history",
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

    def _user_content_to_openai_content(
        self,
        content: MessageContent,
    ) -> list[dict[str, object]]:
        openai_content: list[dict[str, object]] = []
        for part in normalize_content_parts(content):
            if isinstance(part, ImageContentPart):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_content_part_to_openai_image_url(part),
                        },
                    }
                )
            else:
                openai_content.append(
                    {
                        "type": "text",
                        "text": part.text,
                    }
                )

        return openai_content

    def _message_content_to_openai_responses_content(
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

    def _assistant_message_to_openai_responses_input_items(
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

    def _messages_to_openai_responses_input(
        self,
        messages: list[Message],
    ) -> list[dict[str, object]]:
        openai_input: list[dict[str, object]] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                if self._provide_system_message_as_instructions:
                    continue
                openai_input.append(
                    {
                        "type": "message",
                        "role": "system",
                        "content": self._message_content_to_openai_responses_content(
                            message.content
                        ),
                    }
                )
            elif isinstance(message, UserMessage):
                openai_input.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": self._message_content_to_openai_responses_content(
                            message.content
                        ),
                    }
                )
            elif isinstance(message, AssistantMessage):
                openai_input.extend(
                    self._assistant_message_to_openai_responses_input_items(message)
                )
            elif isinstance(message, ToolResponseMessage):
                openai_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.id,
                        "output": self._text_content_to_string(message.content),
                    }
                )

        return openai_input

    def _messages_to_openai_responses_instructions(
        self,
        messages: list[Message],
    ) -> str | None:
        if not self._provide_system_message_as_instructions:
            return None

        system_messages = [
            message.content
            for message in messages
            if isinstance(message, SystemMessage)
        ]
        if not system_messages:
            return None

        return "\n\n".join(system_messages)

    def _messages_to_openai_messages(
        self,
        messages: list[Message],
    ) -> list[ChatCompletionMessageParam]:
        openai_messages: list[ChatCompletionMessageParam] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                openai_messages.append(
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content=self._text_content_to_string(message.content),
                    )
                )
            elif isinstance(message, UserMessage):
                openai_messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=self._user_content_to_openai_content(message.content),
                    )
                )
            elif isinstance(message, AssistantMessage):
                openai_messages.append(
                    self._assistant_message_to_chat_completion_assistant_message_param(
                        message
                    )
                )
            elif isinstance(message, ToolResponseMessage):
                openai_messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        content=self._text_content_to_string(message.content),
                        tool_call_id=message.id,
                    )
                )

        return openai_messages

    def _get_openai_response_format_or_omit(
        self,
        response_format: ResponseFormat | None,
    ) -> OpenAIResponseFormat | Omit:
        if isinstance(response_format, JSONSchemaResponse):
            strict = get_response_format_strict(response_format, default=True)
            return ResponseFormatJSONSchema(
                type="json_schema",
                json_schema={
                    "name": get_response_format_name(
                        response_format, default="response"
                    ),
                    "schema": self._openai_schema(
                        get_response_schema(response_format, strict=strict) or {},
                        strict=strict,
                    ),
                    "strict": strict,
                },
            )

        if isinstance(response_format, JSONObjectResponse):
            return ResponseFormatJSONObject(type="json_object")

        if isinstance(response_format, TextResponse):
            return ResponseFormatText(type="text")

        return Omit()

    def _get_openai_responses_text_or_omit(
        self,
        response_format: ResponseFormat | None,
    ) -> dict[str, object] | Omit:
        if isinstance(response_format, JSONSchemaResponse):
            strict = get_response_format_strict(response_format, default=True)
            return {
                "format": {
                    "type": "json_schema",
                    "name": get_response_format_name(
                        response_format,
                        default="response",
                    ),
                    "schema": self._openai_schema(
                        get_response_schema(response_format, strict=strict) or {},
                        strict=strict,
                    ),
                    "strict": strict,
                }
            }

        if isinstance(response_format, JSONObjectResponse):
            return {
                "format": ResponseFormatJSONObject(type="json_object"),
            }

        if isinstance(response_format, TextResponse):
            return {
                "format": ResponseFormatText(type="text"),
            }

        return Omit()

    def _openai_schema(
        self,
        schema: dict,
        *,
        strict: bool,
    ) -> dict:
        if not strict:
            return schema

        return process_schema(
            _normalize_openai_strict_schema(schema),
            flatten_refs=False,
            flatten_allof=True,
            ensure_additional_properties=True,
            ensure_required_properties=True,
            supported_string_types=self.STRICT_SUPPORTED_STRING_FORMATS,
            supported_schema_fields=self.STRICT_SUPPORTED_SCHEMA_FIELDS,
        )

    def _llm_tools_to_openai_tools(
        self,
        tools: list[Tool],
    ) -> list[ChatCompletionFunctionToolParam]:
        return [
            ChatCompletionFunctionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=self._openai_schema(
                        get_schema_as_dict(tool.input_schema, strict=tool.strict),
                        strict=tool.strict,
                    ),
                    strict=tool.strict,
                ),
            )
            for tool in tools
        ]

    def _llm_tools_to_openai_responses_tools(
        self,
        tools: list[Tool],
    ) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": self._openai_schema(
                    get_schema_as_dict(tool.input_schema, strict=tool.strict),
                    strict=tool.strict,
                ),
                "strict": tool.strict,
            }
            for tool in tools
        ]

    def _web_search_tool_to_openai_responses_tool(
        self,
        tool: WebSearchTool,
    ) -> dict[str, object]:
        del tool
        return {"type": WEB_SEARCH_TOOL_NAME}

    def _get_openai_tools_and_tool_choice_or_omit(
        self,
        tools: list[LLMTool] | None,
        tool_choice: ToolChoice | None,
    ) -> tuple[list[ChatCompletionFunctionToolParam] | Omit, object | Omit]:
        resolved = filter_resolved_tools_for_provider(
            resolve_tools(tools, tool_choice),
            supports_web_search=False,
        )
        openai_tools = self._llm_tools_to_openai_tools(resolved.function_tools)
        if not openai_tools:
            return Omit(), Omit()

        if not resolved.requires_tool:
            return openai_tools, Omit()

        if len(resolved.function_tools) == 1:
            return openai_tools, {
                "type": "function",
                "function": {"name": resolved.function_tools[0].name},
            }

        return openai_tools, "required"

    def _get_openai_responses_tools_and_tool_choice_or_omit(
        self,
        tools: list[LLMTool] | None,
        tool_choice: ToolChoice | None,
    ) -> tuple[list[dict[str, object]] | Omit, object | Omit]:
        resolved = filter_resolved_tools_for_provider(
            resolve_tools(tools, tool_choice),
            supports_web_search=True,
        )
        openai_tools = self._llm_tools_to_openai_responses_tools(
            resolved.function_tools
        )
        if resolved.web_search_tool is not None:
            openai_tools.append(
                self._web_search_tool_to_openai_responses_tool(resolved.web_search_tool)
            )
        if not openai_tools:
            return Omit(), Omit()

        if resolved.has_web_search and (resolved.is_explicit or resolved.requires_tool):
            return openai_tools, {
                "type": "allowed_tools",
                "mode": "required" if resolved.requires_tool else "auto",
                "tools": openai_tools,
            }

        if not resolved.requires_tool:
            return openai_tools, Omit()

        if len(resolved.function_tools) == 1:
            return openai_tools, {
                "type": "function",
                "name": resolved.function_tools[0].name,
            }

        return openai_tools, "required"

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
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", None)

        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", None)

        total_tokens = getattr(usage, "total_tokens", None)

        if not raw_usage and all(
            value is None for value in (input_tokens, output_tokens, total_tokens)
        ):
            return None

        details = dict(raw_usage)
        details.pop("input_tokens", None)
        details.pop("output_tokens", None)
        details.pop("prompt_tokens", None)
        details.pop("completion_tokens", None)
        details.pop("total_tokens", None)

        return ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            details=details,
        )

    def _openai_responses_reasoning_and_extra_body(
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
                reasoning["effort"] = reasoning_effort.effort.value
            if reasoning_effort.summary is not None:
                reasoning["summary"] = reasoning_effort.summary.value

        if reasoning_effort is None or reasoning_effort.include_trace is not False:
            reasoning.setdefault("summary", "auto")
        return reasoning or Omit(), request_extra_body or None

    def _get_openai_chat_reasoning_effort_or_omit(
        self,
        reasoning_effort: ReasoningEffort | None,
    ) -> str | Omit:
        if reasoning_effort is None or reasoning_effort.effort is None:
            return Omit()

        return reasoning_effort.effort.value

    def _get_openai_temperature_or_omit(
        self,
        temperature: float | None,
    ) -> float | Omit:
        if temperature is None:
            return Omit()

        return temperature

    def _get_openai_chat_max_tokens_kwargs(
        self,
        max_tokens: int | None,
        *,
        model: str | None = None,
    ) -> dict[str, int | None]:
        field = self._compatibility_cache.get(
            (model or "*", "output_token_field"), "max_completion_tokens"
        )
        return {field: max_tokens}

    def _is_unsupported_token_parameter(self, error: Exception, field: str) -> bool:
        status = getattr(error, "status_code", None)
        if status not in {400, 422}:
            return False
        message = str(error).casefold()
        return field.casefold() in message and any(
            marker in message
            for marker in (
                "unsupported",
                "unrecognized",
                "unknown parameter",
                "unexpected keyword",
                "extra inputs",
                "not permitted",
            )
        )

    def _create_chat_completion(
        self,
        *,
        model: str,
        max_tokens: int | None,
        **kwargs: object,
    ):
        token_kwargs = self._get_openai_chat_max_tokens_kwargs(max_tokens, model=model)
        field = next(iter(token_kwargs))
        try:
            return self._client.chat.completions.create(
                model=model,
                **kwargs,
                **token_kwargs,
            )
        except Exception as exc:
            if not self._is_unsupported_token_parameter(exc, field):
                raise
            alternate = (
                "max_tokens"
                if field == "max_completion_tokens"
                else "max_completion_tokens"
            )
            self._compatibility_cache[(model, "output_token_field")] = alternate
            self.log(
                LogLevel.WARNING,
                {
                    "code": "output_token_parameter_negotiated",
                    "provider": self.PROVIDER_NAME,
                    "model": model,
                    "unsupported_parameter": field,
                    "replacement_parameter": alternate,
                    "retry_performed": True,
                },
            )
            return self._client.chat.completions.create(
                model=model,
                **kwargs,
                **{alternate: max_tokens},
            )

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
                encrypted_content = getattr(item, "encrypted_content", None)
                if (
                    summary_texts
                    or getattr(item, "id", None) is not None
                    or encrypted_content is not None
                ):
                    thinking_items.append(
                        AssistantReasoningItem(
                            id=getattr(item, "id", None),
                            summary=summary_texts,
                            encrypted_content=encrypted_content,
                            provider=self.PROVIDER_NAME,
                            raw=self._dump_model(item),
                        )
                    )
            elif item_type == "function_call":
                tool_calls.append(
                    AssistantToolCall(
                        id=getattr(item, "call_id", None)
                        or getattr(item, "id", None)
                        or "",
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
        max_output_tokens: int | None = None,
        profile: GenerationProfile | str | None = None,
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
        stream: bool = False,
    ) -> ResponseResult:
        prepared = self.prepare_generation(
            model=model,
            profile=profile,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            tools_requested=bool(tools),
        )
        max_tokens = prepared.max_output_tokens
        reasoning_effort = prepared.reasoning_effort
        messages = self._prepare_reasoning_history(messages, prepared.reasoning.history)
        if stream:
            if self._api_type == OpenAIApiType.RESPONSES:
                return self._generate_responses_stream(
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

            return self._generate_completions_stream(
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

        if self._api_type == OpenAIApiType.RESPONSES:
            return self._generate_responses_once(
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

        return self._generate_completions_once(
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

    def _coerce_api_type(
        self,
        api_type: OpenAIApiType | str,
    ) -> OpenAIApiType | None:
        if isinstance(api_type, OpenAIApiType):
            return api_type

        try:
            return OpenAIApiType(api_type)
        except ValueError:
            return None

    def _generate_completions_once(
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
        openai_tools, openai_tool_choice = (
            self._get_openai_tools_and_tool_choice_or_omit(tools, tool_choice)
        )

        try:
            start_time = perf_counter()
            response = self._create_chat_completion(
                model=model,
                max_tokens=max_tokens,
                messages=self._messages_to_openai_messages(messages),
                temperature=self._get_openai_temperature_or_omit(temperature),
                response_format=self._get_openai_response_format_or_omit(
                    response_format
                ),
                tools=openai_tools,
                tool_choice=openai_tool_choice,
                reasoning_effort=self._get_openai_chat_reasoning_effort_or_omit(
                    reasoning_effort
                ),
                extra_body=extra_body,
            )
            duration_seconds = perf_counter() - start_time

            if not response.choices:
                raise LLMError(400, "No content returned from LLM")

            choice = response.choices[0]
            assistant_message = self._chat_completion_message_to_assistant_message(
                choice.message
            )
            new_messages = [*messages, assistant_message]

            return ResponseContent(
                content=self._final_content(
                    assistant_message.content,
                    response_format,
                ),
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=assistant_message.tool_calls,
                usage=self._response_usage(getattr(response, "usage", None)),
                duration_seconds=duration_seconds,
                finish_reason=getattr(choice, "finish_reason", None),
            )
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

    def _generate_completions_stream(
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
        openai_tools, openai_tool_choice = (
            self._get_openai_tools_and_tool_choice_or_omit(tools, tool_choice)
        )

        try:
            start_time = perf_counter()
            response = self._create_chat_completion(
                model=model,
                max_tokens=max_tokens,
                messages=self._messages_to_openai_messages(messages),
                temperature=self._get_openai_temperature_or_omit(temperature),
                response_format=self._get_openai_response_format_or_omit(
                    response_format
                ),
                tools=openai_tools,
                tool_choice=openai_tool_choice,
                reasoning_effort=self._get_openai_chat_reasoning_effort_or_omit(
                    reasoning_effort
                ),
                extra_body=extra_body,
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
            finish_reason: str | None = None

            for event in response:
                event_usage = self._response_usage(getattr(event, "usage", None))
                if event_usage is not None:
                    usage = event_usage

                if not getattr(event, "choices", None):
                    continue

                choice = event.choices[0]
                if getattr(choice, "finish_reason", None) is not None:
                    finish_reason = choice.finish_reason
                delta = choice.delta

                thinking_delta = self._chat_completion_delta_to_thinking_text(delta)
                if thinking_delta:
                    thinking += thinking_delta
                    current_chunk_type, current_tool, stream_chunks = (
                        self._transition_stream_chunk(
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
                        self._transition_stream_chunk(
                            current_chunk_type=current_chunk_type,
                            next_chunk_type="content",
                            current_tool=current_tool,
                        )
                    )
                    for stream_chunk in stream_chunks:
                        yield stream_chunk
                    yield ResponseStreamContentChunk(
                        chunk=delta.content,
                    )

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

            stream_chunk = self._close_stream_chunk(
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
                content=self._final_content(
                    assistant_message.content,
                    response_format,
                ),
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=tool_calls,
                usage=usage,
                duration_seconds=duration_seconds,
                finish_reason=finish_reason,
            )
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

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
        openai_tools, openai_tool_choice = (
            self._get_openai_responses_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
            )
        )
        reasoning, request_extra_body = self._openai_responses_reasoning_and_extra_body(
            reasoning_effort, extra_body
        )
        response_input = self._messages_to_openai_responses_input(messages)
        request_kwargs = {
            "model": model,
            "input": response_input,
            "temperature": temperature if temperature is not None else Omit(),
            "text": self._get_openai_responses_text_or_omit(response_format),
            "tools": openai_tools,
            "tool_choice": openai_tool_choice,
            "reasoning": reasoning,
            "max_output_tokens": max_tokens if max_tokens is not None else Omit(),
            "extra_body": request_extra_body,
        }
        instructions = self._messages_to_openai_responses_instructions(messages)
        if instructions is not None:
            request_kwargs["instructions"] = instructions

        try:
            start_time = perf_counter()
            response = self._client.responses.create(**request_kwargs)
            duration_seconds = perf_counter() - start_time

            assistant_message = self._responses_output_to_assistant_message(
                getattr(response, "output", []) or []
            )
            new_messages = [*messages, assistant_message]

            return ResponseContent(
                content=self._final_content(
                    assistant_message.content,
                    response_format,
                ),
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=assistant_message.tool_calls,
                usage=self._response_usage(getattr(response, "usage", None)),
                duration_seconds=duration_seconds,
                finish_reason=self._responses_finish_reason(response),
            )
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)

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
        openai_tools, openai_tool_choice = (
            self._get_openai_responses_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
            )
        )
        reasoning, request_extra_body = self._openai_responses_reasoning_and_extra_body(
            reasoning_effort, extra_body
        )
        response_input = self._messages_to_openai_responses_input(messages)
        request_kwargs = {
            "model": model,
            "input": response_input,
            "temperature": temperature if temperature is not None else Omit(),
            "text": self._get_openai_responses_text_or_omit(response_format),
            "tools": openai_tools,
            "tool_choice": openai_tool_choice,
            "reasoning": reasoning,
            "max_output_tokens": max_tokens if max_tokens is not None else Omit(),
            "extra_body": request_extra_body,
            "stream": True,
        }
        instructions = self._messages_to_openai_responses_instructions(messages)
        if instructions is not None:
            request_kwargs["instructions"] = instructions

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
                        streamed_assistant_message_id or getattr(event, "item_id", None)
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

                    tool_key = (
                        getattr(item, "id", None)
                        or getattr(item, "call_id", None)
                        or self._response_item_id("tool")
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

                if event_type in {"response.completed", "response.incomplete"}:
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
                response_assistant_message = (
                    self._responses_output_to_assistant_message(
                        getattr(final_response, "output", []) or []
                    )
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
                finish_reason=(
                    self._responses_finish_reason(final_response)
                    if final_response is not None
                    else None
                ),
            )
        except Exception as exc:
            raise_llm_error(exc, provider=self.PROVIDER_NAME)
