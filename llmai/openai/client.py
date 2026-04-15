from __future__ import annotations

from logging import Logger

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
from llmai.shared.errors import LLMError
from llmai.shared.messages import (
    AssistantMessage,
    AssistantToolCall,
    Message,
    SystemMessage,
    ToolResponseMessage,
    UserMessage,
)
from llmai.shared.response_formats import (
    JSONSchemaResponse,
    JSONObjectResponse,
    ResponseFormat,
    TextResponse,
    get_response_schema,
)
from llmai.shared.responses import (
    ResponseContent,
    ResponseStreamCompletionChunk,
    ResponseStreamContentChunk,
)
from llmai.shared.schema import get_schema_as_dict
from llmai.shared.tools import Tool, ToolChoice, resolve_tools


class OpenAIClient(BaseClient):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)
        self._client = OpenAI(base_url=base_url, api_key=api_key)

        if self._logger:
            self._logger.info("OpenAI client created")
            self._logger.info("Base URL: %s", base_url)

    def _chat_completion_message_to_assistant_message(
        self,
        message: ChatCompletionMessage,
    ) -> AssistantMessage:
        return AssistantMessage(
            content=message.content,
            tool_calls=[
                AssistantToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in (message.tool_calls or [])
            ],
        )

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
            content=message.content,
            tool_calls=tool_calls or None,
        )

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
                        content=message.content,
                    )
                )
            elif isinstance(message, UserMessage):
                openai_messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=message.content,
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
                        content=message.content or "",
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
                    "name": "response",
                    "schema": get_response_schema(response_format) or {},
                    "strict": True,
                },
            )

        if isinstance(response_format, JSONObjectResponse):
            return ResponseFormatJSONObject(type="json_object")

        if isinstance(response_format, TextResponse):
            return ResponseFormatText(type="text")

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
                    parameters=get_schema_as_dict(tool.input_schema),
                    strict=tool.strict,
                ),
            )
            for tool in tools
        ]

    def _get_openai_tools_and_tool_choice_or_omit(
        self,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
    ) -> tuple[list[ChatCompletionFunctionToolParam] | Omit, dict[str, object] | Omit]:
        resolved = resolve_tools(tools, tool_choice)
        openai_tools = self._llm_tools_to_openai_tools(resolved.tools)
        if not openai_tools:
            return Omit(), Omit()

        if not resolved.required_names and not resolved.optional_names:
            return openai_tools, Omit()

        allowed_tools: list[dict[str, object]] = []
        if resolved.required_names:
            allowed_tools.append(
                {
                    "mode": "required",
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": name},
                        }
                        for name in resolved.required_names
                    ],
                }
            )

        if resolved.optional_names:
            allowed_tools.append(
                {
                    "mode": "auto",
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": name},
                        }
                        for name in resolved.optional_names
                    ],
                }
            )

        return openai_tools, {
            "type": "allowed_tools",
            "allowed_tools": allowed_tools,
        }

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
        del use_tools_for_structured_output

        openai_tools, openai_tool_choice = (
            self._get_openai_tools_and_tool_choice_or_omit(tools, tool_choice)
        )

        response = self._client.chat.completions.create(
            model=model,
            messages=self._messages_to_openai_messages(messages),
            temperature=temperature,
            response_format=self._get_openai_response_format_or_omit(response_format),
            tools=openai_tools,
            tool_choice=openai_tool_choice,
            max_completion_tokens=max_tokens,
            extra_body=extra_body,
        )

        if not response.choices:
            raise LLMError(400, "No content returned from LLM")

        assistant_message = self._chat_completion_message_to_assistant_message(
            response.choices[0].message
        )
        new_messages = [*messages, assistant_message]

        return ResponseContent(
            content=assistant_message.content,
            messages=new_messages,
            tool_calls=assistant_message.tool_calls,
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
        del use_tools_for_structured_output

        openai_tools, openai_tool_choice = (
            self._get_openai_tools_and_tool_choice_or_omit(tools, tool_choice)
        )

        response = self._client.chat.completions.create(
            model=model,
            messages=self._messages_to_openai_messages(messages),
            temperature=temperature,
            response_format=self._get_openai_response_format_or_omit(response_format),
            tools=openai_tools,
            tool_choice=openai_tool_choice,
            max_completion_tokens=max_tokens,
            extra_body=extra_body,
            stream=True,
        )

        stream_id = "0"
        content = ""
        partial_tool_calls: dict[int, dict[str, str | None]] = {}
        tool_order: list[int] = []

        for event in response:
            delta = event.choices[0].delta

            if delta.content:
                content += delta.content
                yield ResponseStreamContentChunk(
                    id=stream_id,
                    source="direct",
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
                    yield ResponseStreamContentChunk(
                        id=stream_id,
                        source="tool",
                        tool=current["name"],
                        chunk=tool_arguments,
                    )

        tool_calls = [
            AssistantToolCall(
                id=(partial_tool_calls[index]["id"] or partial_tool_calls[index]["name"] or ""),
                name=partial_tool_calls[index]["name"] or "",
                arguments=partial_tool_calls[index]["arguments"],
            )
            for index in tool_order
            if partial_tool_calls[index]["name"]
        ]

        assistant_message = AssistantMessage(
            content=content or None,
            tool_calls=tool_calls,
        )
        new_messages = [*messages, assistant_message]

        yield ResponseStreamCompletionChunk(
            id=stream_id,
            content=assistant_message.content,
            messages=new_messages,
            tool_calls=tool_calls,
        )
