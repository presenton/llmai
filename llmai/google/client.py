from __future__ import annotations

import json
from logging import Logger
from time import perf_counter

from google import genai
from google.genai.types import (
    Content as GoogleContent,
    FunctionCallingConfig as GoogleFunctionCallingConfig,
    FunctionCallingConfigMode as GoogleFunctionCallingConfigMode,
    GenerateContentConfig,
    GoogleSearch,
    Part as GooglePart,
    ThinkingConfig as GoogleThinkingConfig,
    ThinkingLevel as GoogleThinkingLevel,
    Tool as GoogleTool,
    ToolConfig as GoogleToolConfig,
)

from llmai.shared.base import BaseClient
from llmai.shared.errors import LLMError, raise_llm_error
from llmai.shared.messages import (
    AssistantMessage,
    AssistantToolCall,
    ImageContentPart,
    Message,
    SystemMessage,
    TextContentPart,
    ToolResponseMessage,
    UserMessage,
    collapse_content_parts,
    collapse_thinking_blocks,
    normalize_content_parts,
)
from llmai.shared.reasoning import ReasoningEffort
from llmai.shared.response_formats import (
    JSONSchemaResponse,
    JSONObjectResponse,
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
    WebSearchTool,
    filter_resolved_tools_for_provider,
    resolve_tools,
)

GOOGLE_SUPPORTED_RESPONSE_SCHEMA_KEYS = {
    "$anchor",
    "$defs",
    "$id",
    "$ref",
    "additionalProperties",
    "anyOf",
    "description",
    "enum",
    "format",
    "items",
    "maximum",
    "maxItems",
    "minimum",
    "minItems",
    "oneOf",
    "prefixItems",
    "properties",
    "propertyOrdering",
    "required",
    "title",
    "type",
}
GOOGLE_SUPPORTED_TOOL_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "anyOf",
    "description",
    "enum",
    "format",
    "items",
    "maximum",
    "maxItems",
    "maxLength",
    "maxProperties",
    "minimum",
    "minItems",
    "minLength",
    "minProperties",
    "nullable",
    "pattern",
    "properties",
    "propertyOrdering",
    "required",
    "title",
    "type",
}


class GoogleClient(BaseClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)
        try:
            self._client = genai.Client(api_key=api_key)
        except Exception as exc:
            raise_llm_error(exc, provider="google")

    def _get_system_prompt(self, messages: list[Message]) -> str | None:
        for message in messages:
            if isinstance(message, SystemMessage):
                return "".join(part.text for part in message.content)
        return None

    def _parse_tool_arguments(self, arguments: str | None) -> dict:
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
        except Exception:
            return {}

        return parsed if isinstance(parsed, dict) else {}

    def _get_google_thinking_config(
        self,
        reasoning_effort: ReasoningEffort | None,
    ) -> GoogleThinkingConfig | None:
        if reasoning_effort is None:
            return None

        thinking_budget = reasoning_effort.tokens
        if reasoning_effort.effort == "none" and thinking_budget is None:
            thinking_budget = 0

        thinking_level = None
        if reasoning_effort.effort == "minimal":
            thinking_level = GoogleThinkingLevel.MINIMAL
        elif reasoning_effort.effort == "low":
            thinking_level = GoogleThinkingLevel.LOW
        elif reasoning_effort.effort == "medium":
            thinking_level = GoogleThinkingLevel.MEDIUM
        elif reasoning_effort.effort in {"high", "xhigh"}:
            thinking_level = GoogleThinkingLevel.HIGH

        include_thoughts = None
        if reasoning_effort.effort == "none":
            include_thoughts = False
        elif reasoning_effort.summary is not None:
            include_thoughts = True
        elif reasoning_effort.effort is not None or reasoning_effort.tokens is not None:
            include_thoughts = True

        return GoogleThinkingConfig(
            include_thoughts=include_thoughts,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        )

    def _content_to_assistant_message(self, content: GoogleContent) -> AssistantMessage:
        content_parts: list[TextContentPart | ImageContentPart] = []
        thinking_blocks: list[str] = []
        tool_calls: list[AssistantToolCall] = []

        for each_part in content.parts or []:
            text = getattr(each_part, "text", None)
            if text:
                if getattr(each_part, "thought", False):
                    thinking_blocks.append(text)
                else:
                    content_parts.append(TextContentPart(text=text))

            inline_data = getattr(each_part, "inline_data", None)
            if (
                inline_data
                and inline_data.data is not None
                and inline_data.mime_type
                and inline_data.mime_type.startswith("image/")
            ):
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
                content_parts.append(
                    ImageContentPart(
                        url=file_data.file_uri,
                        mime_type=file_data.mime_type,
                    )
                )

            function_call = getattr(each_part, "function_call", None)
            if function_call:
                tool_calls.append(
                    AssistantToolCall(
                        id=self._tool_call_id(function_call.id),
                        name=function_call.name,
                        arguments=json.dumps(function_call.args or {}),
                    )
                )

        return AssistantMessage(
            content=collapse_content_parts(content_parts),
            thinking=collapse_thinking_blocks(thinking_blocks),
            tool_calls=tool_calls,
        )

    def _content_to_google_parts(
        self,
        content: list[object] | None,
    ) -> list[GooglePart]:
        parts: list[GooglePart] = []

        for part in normalize_content_parts(content):
            if isinstance(part, ImageContentPart):
                if part.url is not None:
                    parts.append(
                        GooglePart.from_uri(
                            file_uri=part.url,
                            mime_type=part.mime_type,
                        )
                    )
                else:
                    parts.append(
                        GooglePart.from_bytes(
                            data=part.data or b"",
                            mime_type=part.mime_type or "",
                        )
                    )
            else:
                parts.append(GooglePart.from_text(text=part.text))

        return parts

    def _messages_to_google_messages(
        self,
        messages: list[Message],
    ) -> list[GoogleContent]:
        contents: list[GoogleContent] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                continue

            if isinstance(message, UserMessage):
                contents.append(
                    GoogleContent(
                        role="user",
                        parts=self._content_to_google_parts(message.content),
                    )
                )
                continue

            if isinstance(message, AssistantMessage):
                parts = self._content_to_google_parts(message.content)

                for tool_call in message.tool_calls:
                    parts.append(
                        GooglePart.from_function_call(
                            name=tool_call.name,
                            args=self._parse_tool_arguments(tool_call.arguments),
                        )
                    )

                if parts:
                    contents.append(GoogleContent(role="model", parts=parts))
                continue

            if isinstance(message, ToolResponseMessage):
                contents.append(
                    GoogleContent(
                        role="user",
                        parts=[
                            GooglePart.from_function_response(
                                name=message.id,
                                response={
                                    "result": "".join(
                                        part.text for part in (message.content or [])
                                    )
                                },
                            )
                        ],
                    )
                )

        return contents

    def _get_response_mime_type(
        self,
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> str | None:
        if not use_tools_for_structured_output and isinstance(
            response_format,
            (JSONSchemaResponse, JSONObjectResponse),
        ):
            return "application/json"

        return None

    def _get_response_json_schema(
        self,
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> dict | None:
        if use_tools_for_structured_output:
            return None

        return get_response_schema(
            response_format,
            supported_keys=GOOGLE_SUPPORTED_RESPONSE_SCHEMA_KEYS,
            supported_string_formats=None,
            strict=get_response_format_strict(response_format, default=False),
        )

    def _llm_tools_to_google_tools(self, tools: list[Tool]) -> list[GoogleTool]:
        return [
            GoogleTool(
                function_declarations=[
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": get_schema_as_dict(
                            tool.input_schema,
                            supported_keys=GOOGLE_SUPPORTED_TOOL_SCHEMA_KEYS,
                            supported_string_formats=None,
                            strict=tool.strict,
                        ),
                    }
                ]
            )
            for tool in tools
        ]

    def _web_search_tool_to_google_tool(
        self,
        tool: WebSearchTool,
    ) -> GoogleTool:
        del tool
        return GoogleTool(google_search=GoogleSearch())

    def _response_schema_tool(
        self,
        response_format: ResponseFormat | None,
        response_schema: dict,
    ) -> GoogleTool:
        return GoogleTool(
            function_declarations=[
                {
                    "name": get_response_format_name(response_format, default="response"),
                    "description": "Provide the final response to the user",
                    "parameters": response_schema,
                }
            ]
        )

    def _get_google_tools_and_tool_config(
        self,
        tools: list[LLMTool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> tuple[list[GoogleTool] | None, GoogleToolConfig | None]:
        resolved = filter_resolved_tools_for_provider(
            resolve_tools(tools, tool_choice),
            supports_web_search=True,
        )
        google_tools = self._llm_tools_to_google_tools(resolved.function_tools)
        if resolved.web_search_tool is not None:
            google_tools.append(
                self._web_search_tool_to_google_tool(resolved.web_search_tool)
            )

        response_schema = get_response_schema(
            response_format,
            supported_keys=(
                GOOGLE_SUPPORTED_TOOL_SCHEMA_KEYS
                if use_tools_for_structured_output
                else GOOGLE_SUPPORTED_RESPONSE_SCHEMA_KEYS
            ),
            supported_string_formats=None,
            strict=get_response_format_strict(response_format, default=False),
        )
        if use_tools_for_structured_output and response_schema:
            google_tools.append(
                self._response_schema_tool(response_format, response_schema)
            )

        if not google_tools:
            return None, None

        function_tools_present = bool(
            resolved.function_tools
            or (use_tools_for_structured_output and response_schema)
        )
        if not function_tools_present:
            return google_tools, None

        mode = GoogleFunctionCallingConfigMode.AUTO
        allowed_function_names: list[str] | None = None
        if use_tools_for_structured_output or resolved.requires_tool:
            mode = GoogleFunctionCallingConfigMode.ANY
            if resolved.requires_tool and len(resolved.function_tools) == 1:
                allowed_function_names = [resolved.function_tools[0].name]

        tool_config = GoogleToolConfig(
            function_calling_config=GoogleFunctionCallingConfig(
                mode=mode,
                allowed_function_names=allowed_function_names,
            )
        )
        return google_tools, tool_config

    def _final_content(
        self,
        content: list[TextContentPart | ImageContentPart] | None,
        response_schema_content: dict | None,
        user_tool_calls: list[AssistantToolCall],
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> object:
        if response_schema_content is not None:
            return response_schema_content

        text_content = "".join(
            part.text for part in (content or []) if isinstance(part, TextContentPart)
        )
        if (
            text_content
            and not use_tools_for_structured_output
            and isinstance(response_format, (JSONSchemaResponse, JSONObjectResponse))
        ):
            return json.loads(text_content)

        if content is None and not user_tool_calls:
            return None

        return content

    def _response_usage(self, usage_metadata: object | None) -> ResponseUsage | None:
        raw_usage = self._dump_model(usage_metadata)
        if not raw_usage:
            return None

        prompt_token_count = raw_usage.get("prompt_token_count")
        tool_use_prompt_token_count = raw_usage.get("tool_use_prompt_token_count") or 0
        input_tokens = None
        if prompt_token_count is not None or tool_use_prompt_token_count:
            input_tokens = (prompt_token_count or 0) + tool_use_prompt_token_count

        output_token_count = raw_usage.get("response_token_count")
        if output_token_count is None:
            output_token_count = raw_usage.get("candidates_token_count")
        thoughts_token_count = raw_usage.get("thoughts_token_count") or 0
        output_tokens = None
        if output_token_count is not None or thoughts_token_count:
            output_tokens = (output_token_count or 0) + thoughts_token_count

        total_tokens = raw_usage.get("total_token_count")
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        details = dict(raw_usage)
        details.pop("prompt_token_count", None)
        details.pop("response_token_count", None)
        details.pop("candidates_token_count", None)
        details.pop("total_token_count", None)

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
        use_tools_for_structured_output: bool | None = None,
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
                use_tools_for_structured_output=use_tools_for_structured_output,
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
            use_tools_for_structured_output=use_tools_for_structured_output,
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
        use_tools_for_structured_output: bool | None = None,
    ) -> ResponseContent:
        del extra_body

        google_tools, tool_config = self._get_google_tools_and_tool_config(
            tools,
            tool_choice,
            response_format,
            use_tools_for_structured_output,
        )
        config = GenerateContentConfig(
            tools=google_tools,
            tool_config=tool_config,
            system_instruction=self._get_system_prompt(messages),
            response_mime_type=self._get_response_mime_type(
                response_format,
                use_tools_for_structured_output,
            ),
            response_json_schema=self._get_response_json_schema(
                response_format,
                use_tools_for_structured_output,
            ),
            thinking_config=self._get_google_thinking_config(reasoning_effort),
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            start_time = perf_counter()
            response = self._client.models.generate_content(
                model=model,
                contents=self._messages_to_google_messages(messages),
                config=config,
            )
            duration_seconds = perf_counter() - start_time

            if not (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                raise LLMError(400, "No content returned from LLM")

            raw_assistant_message = self._content_to_assistant_message(
                response.candidates[0].content
            )
            response_schema_content: dict | None = None
            response_schema_tool_name = get_response_format_name(
                response_format,
                default="response",
            )
            user_tool_calls: list[AssistantToolCall] = []
            for each in raw_assistant_message.tool_calls:
                if each.name == response_schema_tool_name:
                    response_schema_content = self._parse_tool_arguments(each.arguments)
                else:
                    user_tool_calls.append(each)

            assistant_message = AssistantMessage(
                content=raw_assistant_message.content,
                thinking=raw_assistant_message.thinking,
                tool_calls=user_tool_calls,
            )
            new_messages = [*messages, assistant_message]

            return ResponseContent(
                content=self._final_content(
                    assistant_message.content,
                    response_schema_content,
                    user_tool_calls,
                    response_format,
                    use_tools_for_structured_output,
                ),
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=user_tool_calls,
                usage=self._response_usage(getattr(response, "usage_metadata", None)),
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider="google")

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
        use_tools_for_structured_output: bool | None = None,
    ):
        del extra_body

        google_tools, tool_config = self._get_google_tools_and_tool_config(
            tools,
            tool_choice,
            response_format,
            use_tools_for_structured_output,
        )
        config = GenerateContentConfig(
            tools=google_tools,
            tool_config=tool_config,
            system_instruction=self._get_system_prompt(messages),
            response_mime_type=self._get_response_mime_type(
                response_format,
                use_tools_for_structured_output,
            ),
            response_json_schema=self._get_response_json_schema(
                response_format,
                use_tools_for_structured_output,
            ),
            thinking_config=self._get_google_thinking_config(reasoning_effort),
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            current_chunk_type = None
            current_tool = None
            active_thinking_part_index: int | None = None
            content_parts: list[TextContentPart | ImageContentPart] = []
            thinking_blocks_by_part_index: dict[int, str] = {}
            thinking_order: list[int] = []
            response_schema_content: dict | None = None
            response_schema_tool_name = get_response_format_name(
                response_format,
                default="response",
            )
            tool_calls_by_id: dict[str, AssistantToolCall] = {}
            tool_call_order: list[str] = []
            last_emitted_chunks: dict[str, str] = {}
            generated_tool_ids_by_part_index: dict[int, str] = {}
            seen_images: set[tuple[str, str, bytes | str]] = set()
            usage: ResponseUsage | None = None
            start_time = perf_counter()

            for event in self._client.models.generate_content_stream(
                model=model,
                contents=self._messages_to_google_messages(messages),
                config=config,
            ):
                event_usage = self._response_usage(getattr(event, "usage_metadata", None))
                if event_usage is not None:
                    usage = event_usage

                if not (
                    event.candidates
                    and event.candidates[0].content
                    and event.candidates[0].content.parts
                ):
                    continue

                for part_index, each_part in enumerate(event.candidates[0].content.parts):
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
                                stream_chunk = self._close_stream_chunk(
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
                                self._transition_stream_chunk(
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
                            if not use_tools_for_structured_output:
                                current_chunk_type, current_tool, stream_chunks = (
                                    self._transition_stream_chunk(
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
                    tool_id = function_call.id or generated_tool_ids_by_part_index.setdefault(
                        part_index,
                        self._tool_call_id(),
                    )
                    arguments = json.dumps(function_call.args or {})

                    if tool_name == response_schema_tool_name:
                        response_schema_content = self._parse_tool_arguments(arguments)
                    else:
                        tool_calls_by_id[tool_id] = AssistantToolCall(
                            id=tool_id,
                            name=tool_name,
                            arguments=arguments,
                        )
                        if tool_id not in tool_call_order:
                            tool_call_order.append(tool_id)

                    if last_emitted_chunks.get(tool_id) != arguments:
                        last_emitted_chunks[tool_id] = arguments
                        if current_chunk_type == "thinking":
                            active_thinking_part_index = None
                        current_chunk_type, current_tool, stream_chunks = (
                            self._transition_stream_chunk(
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

            stream_chunk = self._close_stream_chunk(
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
                content=self._final_content(
                    assistant_message.content,
                    response_schema_content,
                    user_tool_calls,
                    response_format,
                    use_tools_for_structured_output,
                ),
                thinking=assistant_message.thinking,
                messages=new_messages,
                tool_calls=user_tool_calls,
                usage=usage,
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider="google")
