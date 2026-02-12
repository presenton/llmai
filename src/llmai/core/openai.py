import json
from logging import Logger
from typing import List, Optional, Tuple

from openai import Omit, OpenAI
from openai.types.chat.completion_create_params import (
    ResponseFormat as OpenAIResponseFormat,
)
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)
from openai.types.shared_params.response_format_json_object import (
    ResponseFormatJSONObject,
)
from openai.types.shared_params.response_format_text import ResponseFormatText
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel

from core.base import BaseClient
from llmai.models.errors import LLMError
from llmai.models.response_formats import (
    JSONSchemaResponse,
    JSONObjectResponse,
    TextResponse,
    ResponseFormat,
)
from llmai.models.responses import ResponseContent, ResponseStreamContentChunk
from llmai.models.tools import LLMTool
from llmai.tools.manager import ToolsManager
from llmai.utils.schema import get_schema_as_dict
from models.messages import (
    AssistantMessage,
    AssistantToolCall,
    Message,
    SystemMessage,
    ToolResponseMessage,
    UserMessage,
)


class OpenAIClient(BaseClient):
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tools_manager: Optional[ToolsManager] = None,
        logger: Optional[Logger] = None,
    ):
        super().__init__(logger=logger)
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._tools_manager = tools_manager

    def _messages_to_openai_messages(
        self, messages: List[Message]
    ) -> List[ChatCompletionMessageParam]:
        openai_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                openai_messages.append(
                    ChatCompletionSystemMessageParam(content=message.content)
                )
            elif isinstance(message, UserMessage):
                openai_messages.append(
                    ChatCompletionUserMessageParam(content=message.content)
                )
            elif isinstance(message, AssistantMessage):
                msg = (
                    self._assistant_message_to_chat_completion_assistant_message_param(
                        message
                    )
                )
                openai_messages.append(msg)
            elif isinstance(message, ToolResponseMessage):
                openai_messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        content=message.content or "",
                        tool_call_id=message.id,
                    )
                )
        return openai_messages

    def _chat_completion_message_to_assistant_message(
        self, message: ChatCompletionMessage
    ) -> AssistantMessage:
        return AssistantMessage(
            content=message.content,
            tool_calls=[
                AssistantToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in message.tool_calls
            ],
        )

    def _assistant_message_to_chat_completion_assistant_message_param(
        self, message: AssistantMessage
    ) -> ChatCompletionAssistantMessageParam:
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ChatCompletionMessageFunctionToolCallParam(
                    id=tool_call.id,
                    type="function",
                    function=Function(
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    ),
                )
                for tool_call in message.tool_calls
            ]

        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=message.content,
            tool_calls=tool_calls,
        )

    def _llm_tools_to_openai_tools(
        self, tools: List[LLMTool]
    ) -> List[ChatCompletionToolUnionParam]:
        openai_tools = []
        for tool in tools:
            parameters = get_schema_as_dict(tool.schema)
            openai_tools.append(
                ChatCompletionFunctionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=parameters,
                        strict=tool.strict,
                    ),
                )
            )
        return openai_tools

    def _response_schema_tool(
        self, response_format: BaseModel | dict
    ) -> ChatCompletionToolUnionParam:
        return ChatCompletionFunctionToolParam(
            type="function",
            function=FunctionDefinition(
                name="ResponseSchema",
                description="Provide response to the user",
                parameters=response_format,
            ),
        )

    def _get_openai_response_format_or_omit(
        self,
        response_format: Optional[ResponseFormat],
        use_tools_for_structured_output: Optional[bool],
    ) -> OpenAIResponseFormat | Omit:
        if response_format:
            if isinstance(response_format, JSONSchemaResponse):
                if use_tools_for_structured_output:
                    return Omit()
                return ResponseFormatJSONSchema(
                    type="json_schema",
                    json_schema=get_schema_as_dict(response_format.json_schema),
                )
            elif isinstance(response_format, JSONObjectResponse):
                return ResponseFormatJSONObject(
                    type="json_object",
                )
            elif isinstance(response_format, TextResponse):
                return ResponseFormatText(
                    type="text",
                )

        return Omit()

    def _get_openai_tools_or_omit(
        self,
        tools: Optional[List[LLMTool]],
        use_tools_for_structured_output: Optional[bool],
        response_format: Optional[BaseModel | dict],
    ) -> List[ChatCompletionToolUnionParam] | Omit:
        openai_tools = []
        if tools:
            openai_tools = self._llm_tools_to_openai_tools(tools)
        if use_tools_for_structured_output:
            openai_tools.append(self._response_schema_tool(response_format))
        if not openai_tools:
            return Omit()
        return openai_tools

    def _handle_tool_calls(
        self, calls: List[AssistantToolCall]
    ) -> List[ToolResponseMessage]:
        tool_responses = []
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
        messages: List[Message],
        temperature: Optional[float] = None,
        tools: Optional[List[LLMTool]] = None,
        response_format: Optional[ResponseFormat] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        use_tools_for_structured_output: Optional[bool] = None,
        depth: int = 0,
    ) -> Tuple[ResponseContent, List[Message]]:

        response = self._client.chat.completions.create(
            model=model,
            messages=self._messages_to_openai_messages(messages),
            temperature=temperature,
            response_format=self._get_openai_response_format_or_omit(
                response_format, use_tools_for_structured_output
            ),
            tools=self._get_openai_tools_or_omit(
                tools, use_tools_for_structured_output, response_format
            ),
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
                if each.name == "ResponseSchema":
                    return (
                        ResponseContent(content=json.loads(each.arguments)),
                        new_messages,
                    )

            if self._tools_manager:
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
                use_tools_for_structured_output=use_tools_for_structured_output,
                depth=depth + 1,
            )

        if assistant_message.content and response_format:
            return (
                ResponseContent(content=json.loads(assistant_message.content)),
                new_messages,
            )
        
        if not assistant_message.content:
            raise LLMError(400, "No content returned from LLM")

        return ResponseContent(content=assistant_message.content), new_messages

    def stream(
        self,
        *,
        model: str,
        messages: List[Message],
        temperature: Optional[float] = None,
        tools: Optional[List[LLMTool]] = None,
        response_format: Optional[ResponseFormat] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        use_tools_for_structured_output: Optional[bool] = None,
        depth: int = 0,
    ):
        response = self._client.chat.completions.create(
            model=model,
            messages=self._messages_to_openai_messages(messages),
            temperature=temperature,
            response_format=self._get_openai_response_format_or_omit(
                response_format, use_tools_for_structured_output
            ),
            tools=self._get_openai_tools_or_omit(
                tools, use_tools_for_structured_output, response_format
            ),
            max_completion_tokens=max_tokens,
            extra_body=extra_body,
            stream=True,
        )

        tool_calls: List[AssistantToolCall] = []
        current_index = -1
        current_id = None
        current_name = None
        current_arguments = None
        has_response_schema_tool_call = False

        for event in response:
            content_chunk = event.choices[0].delta.content
            if content_chunk and not use_tools_for_structured_output:
                yield ResponseStreamContentChunk(
                    id=str(depth),
                    source="direct",
                    chunk=content_chunk,
                )

            tool_call_chunk = event.choices[0].delta.tool_calls
            if tool_call_chunk:
                tool_index = tool_call_chunk[0].index
                tool_id = tool_call_chunk[0].id
                tool_name = tool_call_chunk[0].function.name
                tool_arguments = tool_call_chunk[0].function.arguments

                if current_index != tool_index:
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
                    current_name = tool_name or current_name
                    current_id = tool_id or current_id
                    if current_arguments is None:
                        current_arguments = tool_arguments
                    elif tool_arguments:
                        current_arguments += tool_arguments

                if current_name == "ResponseSchema":
                    if tool_arguments:
                        yield ResponseStreamContentChunk(
                            id=str(depth),
                            source="tool",
                            chunk=tool_arguments,
                        )
                    has_response_schema_tool_call = True

        if current_id is not None:
            tool_calls.append(
                AssistantToolCall(
                    id=current_id,
                    name=current_name,
                    arguments=current_arguments,
                )
            )

        if tool_calls and not has_response_schema_tool_call:
            assistant_message = AssistantMessage(
                content=None,
                tool_calls=tool_calls,
            )
            new_messages = [*messages, assistant_message]

            if self._tools_manager:
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
                use_tools_for_structured_output=use_tools_for_structured_output,
                depth=depth + 1,
            ):
                yield chunk
