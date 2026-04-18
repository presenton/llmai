from __future__ import annotations

import json
import os
from logging import Logger
from time import perf_counter

from openai import Omit, OpenAI
from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

from llmai.openai.client import (
    OPENAI_SUPPORTED_SCHEMA_KEYS,
    OPENAI_SUPPORTED_STRING_FORMATS,
    OpenAIClient,
)
from llmai.shared.errors import LLMError, configuration_error, raise_llm_error
from llmai.shared.messages import (
    AssistantContent,
    AssistantMessage,
    AssistantToolCall,
    Message,
    content_from_text,
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
    ResponseStreamToolCompleteChunk,
    ResponseStreamToolChunk,
)
from llmai.shared.tools import Tool, ToolChoice, resolve_tools


class DeepSeekClient(OpenAIClient):
    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        logger: Logger | None = None,
    ):
        self._logger = logger
        self._base_url = base_url or self.DEFAULT_BASE_URL

        try:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._resolve_api_key(api_key),
            )
        except Exception as exc:
            raise_llm_error(exc, provider="deepseek")

        if self._logger:
            self._logger.info("DeepSeek client created")
            self._logger.info("Base URL: %s", self._base_url)

    def _resolve_api_key(self, api_key: str | None) -> str:
        if api_key is not None:
            stripped = api_key.strip()
            if stripped:
                return stripped

        env_api_key = os.getenv("DEEPSEEK_API_KEY")
        if env_api_key is not None:
            stripped_env_api_key = env_api_key.strip()
            if stripped_env_api_key:
                return stripped_env_api_key

        raise configuration_error(
            "Missing DeepSeek API key. Pass api_key or set DEEPSEEK_API_KEY",
            provider="deepseek",
        )

    def _supports_strict_tools(self) -> bool:
        return self._base_url.rstrip("/").endswith("/beta")

    def _parse_tool_arguments(self, arguments: str | None) -> dict:
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
        except Exception:
            return {}

        return parsed if isinstance(parsed, dict) else {}

    def _get_deepseek_response_schema(
        self,
        response_format: ResponseFormat | None,
    ) -> dict | None:
        return get_response_schema(
            response_format,
            supported_keys=OPENAI_SUPPORTED_SCHEMA_KEYS,
            supported_string_formats=OPENAI_SUPPORTED_STRING_FORMATS,
            strict=get_response_format_strict(response_format, default=False),
        )

    def _response_schema_tool(
        self,
        response_format: ResponseFormat | None,
        response_schema: dict,
    ) -> ChatCompletionFunctionToolParam:
        return ChatCompletionFunctionToolParam(
            type="function",
            function=FunctionDefinition(
                name=get_response_format_name(response_format, default="response"),
                description="Provide the final response to the user",
                parameters=response_schema,
                strict=(
                    get_response_format_strict(response_format, default=True)
                    if self._supports_strict_tools()
                    else False
                ),
            ),
        )

    def _get_deepseek_response_format_or_omit(
        self,
        response_format: ResponseFormat | None,
    ):
        if self._get_deepseek_response_schema(response_format) is not None:
            return Omit()

        return self._get_openai_response_format_or_omit(response_format)

    def _get_deepseek_tools_and_tool_choice_or_omit(
        self,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
    ) -> tuple[list[ChatCompletionFunctionToolParam] | Omit, object | Omit, str | None]:
        resolved = resolve_tools(tools, tool_choice)
        deepseek_tools = self._llm_tools_to_openai_tools(resolved.tools)

        response_schema_tool_name: str | None = None
        response_schema = self._get_deepseek_response_schema(response_format)
        if response_schema:
            response_schema_tool_name = get_response_format_name(
                response_format,
                default="response",
            )
            deepseek_tools.append(
                self._response_schema_tool(response_format, response_schema)
            )

        if not deepseek_tools:
            return Omit(), Omit(), response_schema_tool_name

        deepseek_tool_choice: object | Omit = Omit()
        if resolved.requires_tool:
            if len(resolved.tools) == 1:
                deepseek_tool_choice = {
                    "type": "function",
                    "function": {"name": resolved.tools[0].name},
                }
            else:
                deepseek_tool_choice = "required"
        elif response_schema_tool_name:
            if resolved.is_explicit and resolved.tools:
                deepseek_tool_choice = "required"
            else:
                deepseek_tool_choice = {
                    "type": "function",
                    "function": {"name": response_schema_tool_name},
                }

        return deepseek_tools, deepseek_tool_choice, response_schema_tool_name

    def _final_content(
        self,
        content: AssistantContent,
        response_format: ResponseFormat | None,
        *,
        response_schema_content: dict | None,
    ) -> object:
        if response_schema_content is not None:
            return response_schema_content

        text_content = self._assistant_content_to_openai_content(content)
        if text_content and isinstance(
            response_format,
            (JSONSchemaResponse, JSONObjectResponse),
        ):
            try:
                return json.loads(text_content)
            except Exception:
                pass

        if isinstance(response_format, TextResponse) and text_content is not None:
            return text_content

        return content

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
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
    ) -> ResponseContent:
        del reasoning_effort

        deepseek_tools, deepseek_tool_choice, response_schema_tool_name = (
            self._get_deepseek_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
                response_format,
            )
        )

        try:
            start_time = perf_counter()
            response = self._client.chat.completions.create(
                model=model,
                messages=self._messages_to_openai_messages(messages),
                temperature=temperature,
                response_format=self._get_deepseek_response_format_or_omit(response_format),
                tools=deepseek_tools,
                tool_choice=deepseek_tool_choice,
                max_completion_tokens=max_tokens,
                extra_body=extra_body,
            )
            duration_seconds = perf_counter() - start_time

            if not response.choices:
                raise LLMError(400, "No content returned from LLM")

            raw_assistant_message = self._chat_completion_message_to_assistant_message(
                response.choices[0].message
            )
            response_schema_content: dict | None = None
            user_tool_calls: list[AssistantToolCall] = []
            for tool_call in raw_assistant_message.tool_calls:
                if tool_call.name == response_schema_tool_name:
                    response_schema_content = self._parse_tool_arguments(
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
                content=self._final_content(
                    assistant_message.content,
                    response_format,
                    response_schema_content=response_schema_content,
                ),
                messages=new_messages,
                tool_calls=user_tool_calls,
                usage=self._response_usage(getattr(response, "usage", None)),
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider="deepseek")

    def _generate_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        extra_body: dict | None = None,
    ):
        del reasoning_effort

        deepseek_tools, deepseek_tool_choice, response_schema_tool_name = (
            self._get_deepseek_tools_and_tool_choice_or_omit(
                tools,
                tool_choice,
                response_format,
            )
        )

        try:
            start_time = perf_counter()
            response = self._client.chat.completions.create(
                model=model,
                messages=self._messages_to_openai_messages(messages),
                temperature=temperature,
                response_format=self._get_deepseek_response_format_or_omit(response_format),
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

            for event in response:
                event_usage = self._response_usage(getattr(event, "usage", None))
                if event_usage is not None:
                    usage = event_usage

                if not getattr(event, "choices", None):
                    continue

                delta = event.choices[0].delta

                if delta.content:
                    content += delta.content
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
                            self._transition_stream_chunk(
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
                    response_schema_content = self._parse_tool_arguments(
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
                    response_format,
                    response_schema_content=response_schema_content,
                ),
                messages=new_messages,
                tool_calls=user_tool_calls,
                usage=usage,
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            raise_llm_error(exc, provider="deepseek")
