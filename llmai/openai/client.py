from __future__ import annotations

import base64
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

from llmai.shared.configs import OpenAIApiType, OpenAIClientConfig
from llmai.shared.base import BaseClient
from llmai.shared.errors import LLMError, configuration_error, raise_llm_error
from llmai.shared.messages import (
    AssistantContent,
    AssistantMessage,
    AssistantToolCall,
    ImageContentPart,
    Message,
    MessageContent,
    SystemMessage,
    ToolResponseMessage,
    UserMessage,
    collapse_thinking_blocks,
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


class OpenAIClient(BaseClient):
    PROVIDER_NAME = "openai"
    PROVIDER_LABEL = "OpenAI"

    def __init__(
        self,
        *,
        config: OpenAIClientConfig,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)
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

    def _chat_completion_message_to_assistant_message(
        self,
        message: ChatCompletionMessage,
    ) -> AssistantMessage:
        return AssistantMessage(
            content=content_from_text(message.content),
            tool_calls=[
                AssistantToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in (message.tool_calls or [])
            ],
        )

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

        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=self._assistant_content_to_openai_content(message.content),
            tool_calls=tool_calls or None,
        )

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

        for thinking_block in message.thinking or []:
            input_items.append(
                {
                    "id": self._response_item_id("reasoning"),
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": thinking_block,
                        }
                    ],
                }
            )

        text_content = self._assistant_content_to_openai_content(message.content)
        if text_content is not None:
            input_items.append(
                {
                    "id": self._response_item_id("message"),
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text_content,
                            "annotations": [],
                        }
                    ],
                }
            )

        for tool_call in message.tool_calls:
            input_items.append(
                {
                    "type": "function_call",
                    "status": "completed",
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
            message.content for message in messages if isinstance(message, SystemMessage)
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
            return ResponseFormatJSONSchema(
                type="json_schema",
                json_schema={
                    "name": get_response_format_name(
                        response_format, default="response"
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
                "format": ResponseFormatJSONObject(type="json_object"),
            }

        if isinstance(response_format, TextResponse):
            return {
                "format": ResponseFormatText(type="text"),
            }

        return Omit()

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
                    parameters=get_schema_as_dict(
                        tool.input_schema,
                        supported_keys=OPENAI_SUPPORTED_SCHEMA_KEYS,
                        supported_string_formats=OPENAI_SUPPORTED_STRING_FORMATS,
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
                self._web_search_tool_to_openai_responses_tool(
                    resolved.web_search_tool
                )
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
                reasoning["effort"] = reasoning_effort.effort
            if reasoning_effort.summary is not None:
                reasoning["summary"] = reasoning_effort.summary

        reasoning.setdefault("summary", "auto")
        return reasoning or Omit(), request_extra_body or None

    def _get_openai_chat_reasoning_effort_or_omit(
        self,
        reasoning_effort: ReasoningEffort | None,
    ) -> str | Omit:
        if reasoning_effort is None or reasoning_effort.effort is None:
            return Omit()

        return reasoning_effort.effort

    def _get_openai_responses_temperature_or_omit(
        self,
        temperature: float | None,
    ) -> float | Omit:
        if temperature is None:
            return Omit()

        return temperature

    def _get_openai_responses_max_output_tokens_or_omit(
        self,
        max_tokens: int | None,
    ) -> int | Omit:
        if max_tokens is None:
            return Omit()

        return max_tokens

    def _responses_output_to_assistant_message(
        self,
        output: list[object],
    ) -> AssistantMessage:
        text_chunks: list[str] = []
        thinking_blocks: list[str] = []
        tool_calls: list[AssistantToolCall] = []

        for item in output:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                for content in getattr(item, "content", []) or []:
                    content_type = getattr(content, "type", None)
                    if content_type == "output_text" and getattr(content, "text", None):
                        text_chunks.append(content.text)
                    elif content_type == "refusal" and getattr(
                        content, "refusal", None
                    ):
                        text_chunks.append(content.refusal)
            elif item_type == "reasoning":
                block_parts: list[str] = []
                for summary in getattr(item, "summary", []) or []:
                    if getattr(summary, "text", None):
                        block_parts.append(summary.text)
                if block_parts:
                    thinking_blocks.append("".join(block_parts))
            elif item_type == "function_call":
                tool_calls.append(
                    AssistantToolCall(
                        id=getattr(item, "call_id", None) or getattr(item, "id", None) or "",
                        name=getattr(item, "name", None) or "",
                        arguments=getattr(item, "arguments", None),
                    )
                )

        return AssistantMessage(
            content=content_from_text("".join(text_chunks) or None),
            thinking=collapse_thinking_blocks(thinking_blocks),
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
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
        stream: bool = False,
    ) -> ResponseResult:
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
            response = self._client.chat.completions.create(
                model=model,
                messages=self._messages_to_openai_messages(messages),
                temperature=temperature,
                response_format=self._get_openai_response_format_or_omit(
                    response_format
                ),
                tools=openai_tools,
                tool_choice=openai_tool_choice,
                max_completion_tokens=max_tokens,
                reasoning_effort=self._get_openai_chat_reasoning_effort_or_omit(
                    reasoning_effort
                ),
                extra_body=extra_body,
            )
            duration_seconds = perf_counter() - start_time

            if not response.choices:
                raise LLMError(400, "No content returned from LLM")

            assistant_message = self._chat_completion_message_to_assistant_message(
                response.choices[0].message
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
            response = self._client.chat.completions.create(
                model=model,
                messages=self._messages_to_openai_messages(messages),
                temperature=temperature,
                response_format=self._get_openai_response_format_or_omit(
                    response_format
                ),
                tools=openai_tools,
                tool_choice=openai_tool_choice,
                max_completion_tokens=max_tokens,
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
            partial_tool_calls: dict[int, dict[str, str | None]] = {}
            tool_order: list[int] = []
            usage: ResponseUsage | None = None

            for event in response:
                event_usage = self._response_usage(getattr(event, "usage", None))
                if event_usage is not None:
                    usage = event_usage

                if not getattr(event, "choices", None):
                    continue

                delta = event.choices[0].delta

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
            reasoning_effort,
            extra_body
        )
        response_input = self._messages_to_openai_responses_input(messages)
        request_kwargs = {
            "model": model,
            "input": response_input,
            "temperature": self._get_openai_responses_temperature_or_omit(
                temperature
            ),
            "text": self._get_openai_responses_text_or_omit(response_format),
            "tools": openai_tools,
            "tool_choice": openai_tool_choice,
            "reasoning": reasoning,
            "max_output_tokens": self._get_openai_responses_max_output_tokens_or_omit(
                max_tokens
            ),
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
            reasoning_effort,
            extra_body
        )
        response_input = self._messages_to_openai_responses_input(messages)
        request_kwargs = {
            "model": model,
            "input": response_input,
            "temperature": self._get_openai_responses_temperature_or_omit(
                temperature
            ),
            "text": self._get_openai_responses_text_or_omit(response_format),
            "tools": openai_tools,
            "tool_choice": openai_tool_choice,
            "reasoning": reasoning,
            "max_output_tokens": self._get_openai_responses_max_output_tokens_or_omit(
                max_tokens
            ),
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

            if final_response is not None:
                assistant_message = self._responses_output_to_assistant_message(
                    getattr(final_response, "output", []) or []
                )
                tool_calls = assistant_message.tool_calls
                usage = self._response_usage(getattr(final_response, "usage", None))
            else:
                thinking_blocks = [
                    thinking_blocks_by_key[key]
                    for key in thinking_order
                    if thinking_blocks_by_key[key]
                ]
                tool_calls = [
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
                assistant_message = AssistantMessage(
                    content=content_from_text(content or None),
                    thinking=collapse_thinking_blocks(thinking_blocks),
                    tool_calls=tool_calls,
                )
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
