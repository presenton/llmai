from __future__ import annotations

import json
from logging import Logger

from google import genai
from google.genai.types import (
    Content as GoogleContent,
    FunctionCallingConfig as GoogleFunctionCallingConfig,
    FunctionCallingConfigMode as GoogleFunctionCallingConfigMode,
    GenerateContentConfig,
    Part as GooglePart,
    Tool as GoogleTool,
    ToolConfig as GoogleToolConfig,
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
    JSONSchemaResponse,
    JSONObjectResponse,
    ResponseFormat,
    get_response_schema,
)
from llmai.shared.responses import (
    ResponseContent,
    ResponseStreamCompletionChunk,
    ResponseStreamContentChunk,
)
from llmai.shared.schema import get_schema_as_dict
from llmai.shared.tools import Tool, ToolChoice, resolve_tools


class GoogleClient(BaseClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)
        self._client = genai.Client(api_key=api_key)

    def _get_system_prompt(self, messages: list[Message]) -> str | None:
        for message in messages:
            if isinstance(message, SystemMessage):
                return message.content
        return None

    def _parse_tool_arguments(self, arguments: str | None) -> dict:
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
        except Exception:
            return {}

        return parsed if isinstance(parsed, dict) else {}

    def _content_to_assistant_message(self, content: GoogleContent) -> AssistantMessage:
        text_chunks: list[str] = []
        tool_calls: list[AssistantToolCall] = []

        for each_part in content.parts or []:
            if each_part.text:
                text_chunks.append(each_part.text)

            if each_part.function_call:
                tool_calls.append(
                    AssistantToolCall(
                        id=each_part.function_call.id or each_part.function_call.name,
                        name=each_part.function_call.name,
                        arguments=json.dumps(each_part.function_call.args or {}),
                    )
                )

        return AssistantMessage(
            content="".join(text_chunks) or None,
            tool_calls=tool_calls,
        )

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
                        parts=[GooglePart(text=message.content)],
                    )
                )
                continue

            if isinstance(message, AssistantMessage):
                parts: list[GooglePart] = []

                if message.content:
                    parts.append(GooglePart(text=message.content))

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
                                response={"result": message.content},
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

        return get_response_schema(response_format)

    def _llm_tools_to_google_tools(self, tools: list[Tool]) -> list[GoogleTool]:
        return [
            GoogleTool(
                function_declarations=[
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": get_schema_as_dict(tool.input_schema),
                    }
                ]
            )
            for tool in tools
        ]

    def _response_schema_tool(self, response_schema: dict) -> GoogleTool:
        return GoogleTool(
            function_declarations=[
                {
                    "name": "ResponseSchema",
                    "description": "Provide the final response to the user",
                    "parameters": response_schema,
                }
            ]
        )

    def _get_google_tools_and_tool_config(
        self,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> tuple[list[GoogleTool] | None, GoogleToolConfig | None]:
        resolved = resolve_tools(tools, tool_choice)
        google_tools = self._llm_tools_to_google_tools(resolved.tools)

        response_schema = get_response_schema(response_format)
        if use_tools_for_structured_output and response_schema:
            google_tools.append(self._response_schema_tool(response_schema))

        if not google_tools:
            return None, None

        mode = GoogleFunctionCallingConfigMode.AUTO
        allowed_function_names: list[str] | None = None
        if use_tools_for_structured_output or resolved.required_names:
            mode = GoogleFunctionCallingConfigMode.ANY
            allowed_function_names = resolved.required_names or None

        tool_config = GoogleToolConfig(
            function_calling_config=GoogleFunctionCallingConfig(
                mode=mode,
                allowed_function_names=allowed_function_names,
            )
        )
        return google_tools, tool_config

    def _final_content(
        self,
        content: str | None,
        response_schema_content: dict | None,
        user_tool_calls: list[AssistantToolCall],
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> object:
        if response_schema_content is not None:
            return response_schema_content

        if (
            content
            and not use_tools_for_structured_output
            and isinstance(response_format, (JSONSchemaResponse, JSONObjectResponse))
        ):
            return json.loads(content)

        if content is None and not user_tool_calls:
            return ""

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
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        response = self._client.models.generate_content(
            model=model,
            contents=self._messages_to_google_messages(messages),
            config=config,
        )

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
        user_tool_calls: list[AssistantToolCall] = []
        for each in raw_assistant_message.tool_calls:
            if each.name == "ResponseSchema":
                response_schema_content = self._parse_tool_arguments(each.arguments)
            else:
                user_tool_calls.append(each)

        assistant_message = AssistantMessage(
            content=raw_assistant_message.content,
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
            messages=new_messages,
            tool_calls=user_tool_calls,
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
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        stream_id = "0"
        text_chunks: list[str] = []
        response_schema_content: dict | None = None
        tool_calls_by_id: dict[str, AssistantToolCall] = {}
        tool_call_order: list[str] = []
        last_emitted_chunks: dict[str, str] = {}

        for event in self._client.models.generate_content_stream(
            model=model,
            contents=self._messages_to_google_messages(messages),
            config=config,
        ):
            if not (
                event.candidates
                and event.candidates[0].content
                and event.candidates[0].content.parts
            ):
                continue

            for each_part in event.candidates[0].content.parts:
                if each_part.text:
                    text_chunks.append(each_part.text)
                    if not use_tools_for_structured_output:
                        yield ResponseStreamContentChunk(
                            id=stream_id,
                            source="direct",
                            chunk=each_part.text,
                        )

                if not each_part.function_call:
                    continue

                tool_id = each_part.function_call.id or each_part.function_call.name
                tool_name = each_part.function_call.name
                arguments = json.dumps(each_part.function_call.args or {})

                if tool_name == "ResponseSchema":
                    response_schema_content = self._parse_tool_arguments(arguments)
                    if last_emitted_chunks.get(tool_id) != arguments:
                        last_emitted_chunks[tool_id] = arguments
                        yield ResponseStreamContentChunk(
                            id=stream_id,
                            source="direct",
                            chunk=arguments,
                        )
                    continue

                tool_calls_by_id[tool_id] = AssistantToolCall(
                    id=tool_id,
                    name=tool_name,
                    arguments=arguments,
                )
                if tool_id not in tool_call_order:
                    tool_call_order.append(tool_id)

                if last_emitted_chunks.get(tool_id) != arguments:
                    last_emitted_chunks[tool_id] = arguments
                    yield ResponseStreamContentChunk(
                        id=stream_id,
                        source="tool",
                        tool=tool_name,
                        chunk=arguments,
                    )

        user_tool_calls = [tool_calls_by_id[tool_id] for tool_id in tool_call_order]
        assistant_message = AssistantMessage(
            content="".join(text_chunks) or None,
            tool_calls=user_tool_calls,
        )
        new_messages = [*messages, assistant_message]

        yield ResponseStreamCompletionChunk(
            id=stream_id,
            content=self._final_content(
                assistant_message.content,
                response_schema_content,
                user_tool_calls,
                response_format,
                use_tools_for_structured_output,
            ),
            messages=new_messages,
            tool_calls=user_tool_calls,
        )
