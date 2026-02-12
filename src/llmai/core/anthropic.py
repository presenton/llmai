import json
from logging import Logger
from typing import List, Optional

from anthropic import Anthropic, MessageStreamEvent, Omit
from anthropic.types import (
    Message as AnthropicMessage,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from pydantic import BaseModel

from core.base import BaseClient
from llmai.models.response_formats import ResponseFormat
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


class AnthropicClient(BaseClient):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tools_manager: Optional[ToolsManager] = None,
        logger: Optional[Logger] = None,
    ):
        super().__init__(logger=logger)
        self._client = Anthropic(api_key=api_key)
        self._tools_manager = tools_manager

    def _get_system_prompt(self, messages: List[Message]) -> str | Omit:
        for message in messages:
            if isinstance(message, SystemMessage):
                return message.content
        return Omit()

    def _messages_to_anthropic_messages(
        self, messages: List[Message]
    ) -> List[MessageParam]:
        anthropic_messages = []
        for message in messages:
            if isinstance(message, UserMessage):
                anthropic_messages.append(
                    MessageParam(role="user", content=message.content)
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
                                tool_use_id=response.id,
                                content=response.content or "",
                                is_error=False,
                            )
                            for response in message.responses
                        ],
                    )
                )
        return anthropic_messages

    def _assistant_message_to_message_param(
        self, message: AssistantMessage
    ) -> MessageParam:
        content_blocks = []
        if message.content:
            content_blocks.append(
                TextBlockParam(
                    type="text",
                    text=message.content,
                )
            )
        for each in message.tool_calls:
            content_blocks.append(
                ToolUseBlockParam(
                    type="tool_use",
                    id=each.id,
                    name=each.name,
                    input=each.arguments,
                )
            )
        return MessageParam(
            role="assistant",
            content=content_blocks,
        )

    def _llm_tools_to_anthropic_tools(self, tools: List[LLMTool]) -> List[ToolParam]:
        return [
            ToolParam(
                name=tool.name,
                description=tool.description,
                strict=tool.strict,
                input_schema=get_schema_as_dict(tool.schema),
            )
            for tool in tools
        ]

    def _response_schema_tool(self, response_format: BaseModel | dict) -> dict:
        return {
            "name": "ResponseSchema",
            "description": "Provide response to the user",
            "input_schema": response_format,
        }

    def _get_anthropic_tools(
        self,
        tools: Optional[List[LLMTool]],
        response_format: Optional[BaseModel | dict],
    ) -> Optional[List[dict]]:
        anthropic_tools: List[dict] = []
        if tools:
            anthropic_tools = self._llm_tools_to_anthropic_tools(tools)
        if response_format:
            anthropic_tools.append(self._response_schema_tool(response_format))
        return anthropic_tools or Omit()

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
    ):
        anthropic_tools = self._get_anthropic_tools(tools, response_format)
        response: AnthropicMessage = self._client.messages.create(
            model=model,
            system=self._get_system_prompt(messages),
            messages=self._messages_to_anthropic_messages(messages),
            tools=anthropic_tools,
            max_tokens=max_tokens or 8000,
            temperature=temperature or Omit(),
            extra_body=extra_body,
        )

        text_content = None
        tool_calls: List[AssistantToolCall] = []
        for content in response.content:
            if content.type == "text":
                text_content = content.text
            if content.type == "tool_use":
                tool_calls.append(
                    AssistantToolCall(
                        id=content.id,
                        name=content.name,
                        arguments=json.dumps(content.input),
                    )
                )

        if tool_calls:
            for each in tool_calls:
                if each.name == "ResponseSchema":
                    return ResponseContent(content=json.loads(each.arguments))

            if self._tools_manager:
                tool_responses = self._handle_tool_calls(tool_calls)
                new_messages = [
                    *messages,
                    AssistantMessage(content=text_content, tool_calls=tool_calls),
                    *tool_responses,
                ]
            else:
                new_messages = [
                    *messages,
                    AssistantMessage(content=text_content, tool_calls=tool_calls),
                ]

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

        if text_content and response_format:
            return ResponseContent(content=json.loads(text_content))

        return ResponseContent(content=text_content or "")

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
        anthropic_tools = self._get_anthropic_tools(tools, response_format)

        tool_calls: List[AssistantToolCall] = []
        has_response_schema_tool_call = False
        is_response_schema_tool_call_started = False
        with self._client.messages.stream(
            model=model,
            system=self._get_system_prompt(messages),
            messages=self._messages_to_anthropic_messages(messages),
            tools=anthropic_tools,
            max_tokens=max_tokens or 8000,
            temperature=temperature or Omit(),
            extra_body=extra_body,
        ) as stream:
            for event in stream:
                event: MessageStreamEvent = event
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                    and not use_tools_for_structured_output
                ):
                    yield ResponseStreamContentChunk(
                        id=str(depth),
                        source="direct",
                        chunk=event.delta.text,
                    )

                if (
                    event.type == "content_block_start"
                    and event.content_block.type == "tool_use"
                    and event.content_block.name == "ResponseSchema"
                ):
                    has_response_schema_tool_call = True
                    is_response_schema_tool_call_started = True

                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "input_json_delta"
                    and is_response_schema_tool_call_started
                ):
                    yield ResponseStreamContentChunk(
                        id=str(depth),
                        source="tool",
                        chunk=event.delta.partial_json,
                    )

                if (
                    event.type == "content_block_stop"
                    and event.content_block.type == "tool_use"
                ):
                    tool_calls.append(
                        AssistantToolCall(
                            id=event.content_block.id,
                            name=event.content_block.name,
                            arguments=json.dumps(event.content_block.input)
                            if event.content_block.input is not None
                            else None,
                        )
                    )
                    if is_response_schema_tool_call_started:
                        is_response_schema_tool_call_started = False

        if tool_calls and not has_response_schema_tool_call:
            new_messages = [
                *messages,
                AssistantMessage(content=None, tool_calls=tool_calls),
            ]

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
