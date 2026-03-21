import json
from logging import Logger

from openai import Omit, OpenAI
from openai.types.chat import (
    ChatCompletionAllowedToolChoiceParam,
    ChatCompletionAllowedToolsParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
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
from llmai.shared.tooling import ToolsManager
from llmai.shared.tools import Tool, ToolChoices


class OpenAIClient(BaseClient):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        tools_manager: ToolsManager | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._tools_manager = tools_manager

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
        tools: ToolChoices | None,
        depth: int,
    ) -> tuple[
        list[ChatCompletionFunctionToolParam] | Omit,
        ChatCompletionToolChoiceOptionParam | Omit,
    ]:
        if not tools:
            return Omit(), Omit()

        required_tools = self._llm_tools_to_openai_tools(
            tools.required_for_depth(depth)
        )
        optional_tools = self._llm_tools_to_openai_tools(
            tools.optional_for_depth(depth)
        )
        all_tools = [*required_tools, *optional_tools]

        if not all_tools:
            return Omit(), Omit()

        allowed_tools = []
        if required_tools:
            allowed_tools.append(
                ChatCompletionAllowedToolsParam(
                    mode="required",
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": each_tool.function.name},
                        }
                        for each_tool in required_tools
                    ],
                )
            )

        if optional_tools:
            allowed_tools.append(
                ChatCompletionAllowedToolsParam(
                    mode="auto",
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": each_tool.function.name},
                        }
                        for each_tool in optional_tools
                    ],
                )
            )

        return all_tools, ChatCompletionAllowedToolChoiceParam(
            type="allowed_tools",
            allowed_tools=allowed_tools,
        )

    def _handle_tool_calls(
        self,
        calls: list[AssistantToolCall],
    ) -> list[ToolResponseMessage]:
        tool_responses: list[ToolResponseMessage] = []

        for each in calls:
            arguments = None
            if each.arguments:
                try:
                    arguments = json.loads(each.arguments)
                except Exception:
                    arguments = None

            response = self._tools_manager.execute(each.name, arguments)
            tool_responses.append(
                ToolResponseMessage(
                    id=each.id,
                    content=response,
                )
            )

        return tool_responses

    def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: ToolChoices | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        use_tools_for_structured_output: bool | None = None,
        depth: int = 0,
    ) -> ResponseContent:
        del use_tools_for_structured_output

        openai_tools, tool_choice = self._get_openai_tools_and_tool_choice_or_omit(
            tools,
            depth,
        )

        response = self._client.chat.completions.create(
            model=model,
            messages=self._messages_to_openai_messages(messages),
            temperature=temperature,
            response_format=self._get_openai_response_format_or_omit(response_format),
            tools=openai_tools,
            tool_choice=tool_choice,
            max_completion_tokens=max_tokens,
            extra_body=extra_body,
        )

        if not response.choices:
            raise LLMError(400, "No content returned from LLM")

        assistant_message = self._chat_completion_message_to_assistant_message(
            response.choices[0].message
        )
        new_messages = [*messages, assistant_message]

        if assistant_message.tool_calls:
            for each in assistant_message.tool_calls:
                if tools and tools.stop_on and each.name in tools.stop_on:
                    return ResponseContent(
                        content=each.arguments,
                        messages=new_messages,
                    )

            if not self._tools_manager:
                raise LLMError(
                    400,
                    "Model requested tool calls but no tools manager is configured",
                )

            tool_responses = self._handle_tool_calls(assistant_message.tool_calls)
            new_messages = [*new_messages, *tool_responses]

            return self.generate(
                model=model,
                messages=new_messages,
                temperature=temperature,
                tools=tools,
                response_format=response_format,
                max_tokens=max_tokens,
                extra_body=extra_body,
                depth=depth + 1,
            )

        return ResponseContent(
            content=assistant_message.content,
            messages=new_messages,
        )

    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        tools: ToolChoices | None = None,
        response_format: ResponseFormat | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        use_tools_for_structured_output: bool | None = None,
        depth: int = 0,
    ):
        del use_tools_for_structured_output

        openai_tools, tool_choice = self._get_openai_tools_and_tool_choice_or_omit(
            tools,
            depth,
        )

        response = self._client.chat.completions.create(
            model=model,
            messages=self._messages_to_openai_messages(messages),
            temperature=temperature,
            response_format=self._get_openai_response_format_or_omit(response_format),
            tools=openai_tools,
            tool_choice=tool_choice,
            max_completion_tokens=max_tokens,
            extra_body=extra_body,
            stream=True,
        )

        content = ""
        yielded_tool_name: str | None = None

        tool_calls: list[AssistantToolCall] = []
        current_index = -1
        current_id: str | None = None
        current_name: str | None = None
        current_arguments: str | None = None
        has_stop_on_tool_call = False

        for event in response:
            delta = event.choices[0].delta

            if delta.content:
                content += delta.content
                yield ResponseStreamContentChunk(
                    id=str(depth),
                    source="direct",
                    chunk=delta.content,
                )

            if delta.tool_calls:
                tool_call_delta = delta.tool_calls[0]
                tool_index = tool_call_delta.index
                tool_id = tool_call_delta.id
                tool_name = (
                    tool_call_delta.function.name
                    if tool_call_delta.function
                    else None
                )
                tool_arguments = (
                    tool_call_delta.function.arguments
                    if tool_call_delta.function
                    else None
                )

                if current_index != tool_index:
                    if current_id is not None and current_name is not None:
                        tool_calls.append(
                            AssistantToolCall(
                                id=current_id,
                                name=current_name,
                                arguments=current_arguments,
                            )
                        )

                    current_index = tool_index
                    current_id = tool_id
                    current_name = tool_name
                    current_arguments = tool_arguments
                else:
                    current_id = tool_id or current_id
                    current_name = tool_name or current_name
                    if current_arguments is None:
                        current_arguments = tool_arguments
                    elif tool_arguments:
                        current_arguments += tool_arguments

                if tools and tools.stop_on and current_name in tools.stop_on:
                    if tool_arguments:
                        yielded_tool_name = yielded_tool_name or current_name
                        has_stop_on_tool_call = True
                        yield ResponseStreamContentChunk(
                            id=str(depth),
                            source="tool",
                            tool=current_name,
                            chunk=tool_arguments,
                        )

        if current_id is not None and current_name is not None:
            tool_calls.append(
                AssistantToolCall(
                    id=current_id,
                    name=current_name,
                    arguments=current_arguments,
                )
            )

        assistant_message = AssistantMessage(
            content=content or None,
            tool_calls=tool_calls,
        )
        new_messages = [*messages, assistant_message]

        if tool_calls and not has_stop_on_tool_call:
            if not self._tools_manager:
                raise LLMError(
                    400,
                    "Model requested tool calls but no tools manager is configured",
                )

            tool_responses = self._handle_tool_calls(tool_calls)
            new_messages = [*new_messages, *tool_responses]

            for chunk in self.stream(
                model=model,
                messages=new_messages,
                temperature=temperature,
                tools=tools,
                response_format=response_format,
                max_tokens=max_tokens,
                extra_body=extra_body,
                depth=depth + 1,
            ):
                yield chunk
            return

        if yielded_tool_name:
            for each in assistant_message.tool_calls:
                if each.name == yielded_tool_name:
                    content = each.arguments or content
                    break

        yield ResponseStreamCompletionChunk(
            id=str(depth),
            content=content,
            messages=new_messages,
        )
