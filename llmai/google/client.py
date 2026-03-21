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
from llmai.shared.responses import ResponseContent, ResponseStreamContentChunk
from llmai.shared.schema import get_schema_as_dict
from llmai.shared.tooling import ToolsManager
from llmai.shared.tools import Tool, ToolChoices


class GoogleClient(BaseClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        tools_manager: ToolsManager | None = None,
        logger: Logger | None = None,
    ):
        super().__init__(logger=logger)
        self._client = genai.Client(api_key=api_key)
        self._tools_manager = tools_manager

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

    def _get_google_tools(
        self,
        tools: list[Tool],
        response_format: ResponseFormat | None,
        use_tools_for_structured_output: bool | None,
    ) -> list[GoogleTool] | None:
        google_tools = self._llm_tools_to_google_tools(tools) if tools else []

        response_schema = get_response_schema(response_format)
        if use_tools_for_structured_output and response_schema:
            google_tools.append(self._response_schema_tool(response_schema))

        return google_tools or None

    def _get_tool_config(
        self,
        google_tools: list[GoogleTool] | None,
        tools: ToolChoices | None,
        depth: int,
        use_tools_for_structured_output: bool | None,
    ) -> GoogleToolConfig | None:
        if not google_tools:
            return None

        mode = GoogleFunctionCallingConfigMode.AUTO
        if use_tools_for_structured_output or (
            tools and tools.required_for_depth(depth)
        ):
            mode = GoogleFunctionCallingConfigMode.ANY

        return GoogleToolConfig(
            function_calling_config=GoogleFunctionCallingConfig(mode=mode)
        )

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
                    id=each.name,
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
        del extra_body

        available_tools = tools.all_for_depth(depth) if tools else []
        google_tools = self._get_google_tools(
            available_tools,
            response_format,
            use_tools_for_structured_output,
        )
        config = GenerateContentConfig(
            tools=google_tools,
            tool_config=self._get_tool_config(
                google_tools,
                tools,
                depth,
                use_tools_for_structured_output,
            ),
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

        assistant_message = self._content_to_assistant_message(
            response.candidates[0].content
        )
        new_messages = [*messages, assistant_message]

        if assistant_message.tool_calls:
            for each in assistant_message.tool_calls:
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

            tool_responses = self._handle_tool_calls(assistant_message.tool_calls)
            new_messages = [*new_messages, *tool_responses]

            return self.generate(
                model=model,
                messages=new_messages,
                temperature=temperature,
                tools=tools,
                response_format=response_format,
                max_tokens=max_tokens,
                use_tools_for_structured_output=use_tools_for_structured_output,
                depth=depth + 1,
            )

        content = assistant_message.content
        if content and isinstance(response_format, (JSONSchemaResponse, JSONObjectResponse)):
            return ResponseContent(
                content=json.loads(content),
                messages=new_messages,
            )

        return ResponseContent(
            content=content or "",
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
        del extra_body

        available_tools = tools.all_for_depth(depth) if tools else []
        google_tools = self._get_google_tools(
            available_tools,
            response_format,
            use_tools_for_structured_output,
        )
        config = GenerateContentConfig(
            tools=google_tools,
            tool_config=self._get_tool_config(
                google_tools,
                tools,
                depth,
                use_tools_for_structured_output,
            ),
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

        tool_calls: list[AssistantToolCall] = []
        has_response_schema_tool_call = False

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
                if each_part.text and not use_tools_for_structured_output:
                    yield ResponseStreamContentChunk(
                        id=str(depth),
                        source="direct",
                        chunk=each_part.text,
                    )

                if each_part.function_call:
                    tool_call = AssistantToolCall(
                        id=each_part.function_call.id or each_part.function_call.name,
                        name=each_part.function_call.name,
                        arguments=json.dumps(each_part.function_call.args or {}),
                    )
                    tool_calls.append(tool_call)

                    if each_part.function_call.name == "ResponseSchema":
                        has_response_schema_tool_call = True
                        yield ResponseStreamContentChunk(
                            id=str(depth),
                            source="tool",
                            tool=each_part.function_call.name,
                            chunk=json.dumps(each_part.function_call.args or {}),
                        )

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
                use_tools_for_structured_output=use_tools_for_structured_output,
                depth=depth + 1,
            ):
                yield chunk
