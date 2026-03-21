import json
from logging import Logger

from anthropic import Anthropic, MessageStreamEvent, Omit
from anthropic.types import (
    Message as AnthropicMessage,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

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
    ResponseFormat,
    get_response_schema,
)
from llmai.shared.responses import ResponseContent, ResponseStreamContentChunk
from llmai.shared.schema import get_schema_as_dict
from llmai.shared.tooling import ToolsManager
from llmai.shared.tools import Tool, ToolChoices


class AnthropicClient(BaseClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        tools_manager: ToolsManager | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)
        self._client = Anthropic(api_key=api_key)
        self._tools_manager = tools_manager

    def _get_system_prompt(self, messages: list[Message]) -> str | Omit:
        for message in messages:
            if isinstance(message, SystemMessage):
                return message.content
        return Omit()

    def _parse_tool_arguments(self, arguments: str | None) -> dict:
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
        except Exception:
            return {}

        return parsed if isinstance(parsed, dict) else {}

    def _assistant_message_to_message_param(
        self,
        message: AssistantMessage,
    ) -> MessageParam:
        content_blocks: list[TextBlockParam | ToolUseBlockParam] = []

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
                    input=self._parse_tool_arguments(each.arguments),
                )
            )

        return MessageParam(role="assistant", content=content_blocks)

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
                                tool_use_id=message.id,
                                content=message.content or "",
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
                input_schema=get_schema_as_dict(tool.input_schema),
            )
            for tool in tools
        ]

    def _response_schema_tool(self, response_schema: dict) -> dict:
        return {
            "name": "ResponseSchema",
            "description": "Provide the final response to the user",
            "input_schema": response_schema,
        }

    def _get_anthropic_tools(
        self,
        tools: ToolChoices | None,
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
        depth: int,
    ) -> list[dict] | Omit:
        anthropic_tools = (
            self._llm_tools_to_anthropic_tools(tools.all_for_depth(depth))
            if tools
            else []
        )

        response_schema = get_response_schema(response_format)
        if response_schema and use_tools_for_structured_output is not False:
            anthropic_tools.append(self._response_schema_tool(response_schema))

        return anthropic_tools or Omit()

    def _handle_tool_calls(
        self,
        calls: list[AssistantToolCall],
    ) -> list[ToolResponseMessage]:
        tool_responses: list[ToolResponseMessage] = []

        for each in calls:
            response = self._tools_manager.execute(
                each.name,
                self._parse_tool_arguments(each.arguments),
            )
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
        response: AnthropicMessage = self._client.messages.create(
            model=model,
            system=self._get_system_prompt(messages),
            messages=self._messages_to_anthropic_messages(messages),
            tools=self._get_anthropic_tools(
                tools,
                response_format,
                use_tools_for_structured_output,
                depth,
            ),
            max_tokens=max_tokens or 8000,
            temperature=temperature or Omit(),
            extra_body=extra_body,
        )

        text_content = None
        tool_calls: list[AssistantToolCall] = []
        for content in response.content:
            if content.type == "text":
                text_content = content.text
            elif content.type == "tool_use":
                tool_calls.append(
                    AssistantToolCall(
                        id=content.id,
                        name=content.name,
                        arguments=json.dumps(content.input),
                    )
                )

        assistant_message = AssistantMessage(
            content=text_content,
            tool_calls=tool_calls,
        )
        new_messages = [*messages, assistant_message]

        if tool_calls:
            for each in tool_calls:
                if each.name == "ResponseSchema":
                    return ResponseContent(
                        content=self._parse_tool_arguments(each.arguments),
                        messages=new_messages,
                    )

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

            tool_responses = self._handle_tool_calls(tool_calls)
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

        return ResponseContent(
            content=text_content or "",
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
        tool_calls: list[AssistantToolCall] = []
        has_response_schema_tool_call = False
        is_response_schema_tool_call_started = False

        with self._client.messages.stream(
            model=model,
            system=self._get_system_prompt(messages),
            messages=self._messages_to_anthropic_messages(messages),
            tools=self._get_anthropic_tools(
                tools,
                response_format,
                use_tools_for_structured_output,
                depth,
            ),
            max_tokens=max_tokens or 8000,
            temperature=temperature or Omit(),
            extra_body=extra_body,
        ) as stream:
            for event in stream:
                event = event  # Helps static readers; Anthropic yields union events.
                event = event if isinstance(event, MessageStreamEvent) else event

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
                        tool="ResponseSchema",
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
                            arguments=json.dumps(event.content_block.input),
                        )
                    )
                    if is_response_schema_tool_call_started:
                        is_response_schema_tool_call_started = False

        if tool_calls and not has_response_schema_tool_call:
            new_messages = [
                *messages,
                AssistantMessage(content=None, tool_calls=tool_calls),
            ]

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
                use_tools_for_structured_output=use_tools_for_structured_output,
                depth=depth + 1,
            ):
                yield chunk
