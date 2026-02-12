import json
from logging import Logger
from typing import List, Optional

from google import genai
from google.genai.types import (
    Content as GoogleContent,
    Part as GoogleContentPart,
    GenerateContentConfig,
    Tool as GoogleTool,
    ToolConfig as GoogleToolConfig,
    FunctionCallingConfig as GoogleFunctionCallingConfig,
    FunctionCallingConfigMode as GoogleFunctionCallingConfigMode,
)
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


class GoogleClient(BaseClient):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tools_manager: Optional[ToolsManager] = None,
        logger: Optional[Logger] = None,
    ):
        super().__init__(logger=logger)
        self._client = genai.Client(api_key=api_key)
        self._tools_manager = tools_manager

    def _get_system_prompt(self, messages: List[Message]) -> Optional[str]:
        for message in messages:
            if isinstance(message, SystemMessage):
                return message.content
        return None

    def _messages_to_google_messages(
        self, messages: List[Message]
    ) -> List[GoogleContent]:
        contents: List[GoogleContent] = []
        for message in messages:
            if isinstance(message, UserMessage):
                contents.append(
                    GoogleContent(
                        role="user",
                        parts=[GoogleContentPart(text=message.content)],
                    )
                )
            elif isinstance(message, AssistantMessage):
                if message.content:
                    contents.append(
                        GoogleContent(
                            role="model",
                            parts=[GoogleContentPart(text=message.content)],
                        )
                    )
            elif isinstance(message, ToolResponseMessage):
                contents.append(
                    GoogleContent(
                        role="user",
                        parts=[
                            GoogleContentPart.from_function_response(
                                name=message.id,
                                response={"result": message.content},
                            )
                        ],
                    )
                )
        return contents

    def _llm_tools_to_google_tools(self, tools: List[LLMTool]) -> List[GoogleTool]:
        return [
            GoogleTool(
                function_declarations=[
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": get_schema_as_dict(tool.schema),
                    }
                ]
            )
            for tool in tools
        ]

    def _response_schema_tool(self, response_format: BaseModel | dict) -> GoogleTool:
        return GoogleTool(
            function_declarations=[
                {
                    "name": "ResponseSchema",
                    "description": "Provide response to the user",
                    "parameters": response_format,
                }
            ]
        )

    def _get_google_tools(
        self,
        tools: Optional[List[LLMTool]],
        use_tools_for_structured_output: Optional[bool],
        response_format: Optional[BaseModel | dict],
    ) -> Optional[List[GoogleTool]]:
        google_tools: List[GoogleTool] = []
        if tools:
            google_tools = self._llm_tools_to_google_tools(tools)
        if use_tools_for_structured_output:
            google_tools.append(self._response_schema_tool(response_format))
        return google_tools or None

    def _get_response_mime_type(
        self,
        response_format: Optional[ResponseFormat],
        use_tools_for_structured_output: Optional[bool],
    ) -> Optional[str]:
        if response_format and not use_tools_for_structured_output:
            if isinstance(response_format, (JSONSchemaResponse, JSONObjectResponse)):
                return "application/json"
            if isinstance(response_format, TextResponse):
                return "text/plain"
        return None

    def _get_response_json_schema(
        self,
        response_format: Optional[ResponseFormat],
        use_tools_for_structured_output: Optional[bool],
    ) -> Optional[dict]:
        if (
            isinstance(response_format, JSONSchemaResponse)
            and not use_tools_for_structured_output
        ):
            return get_schema_as_dict(response_format.json_schema)
        return None

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
                    id=each.name,
                    content=response,
                )
            )
        return tool_responses

    def _content_to_assistant_message(self, content: GoogleContent) -> AssistantMessage:
        text_content = None
        tool_calls: List[AssistantToolCall] = []
        for each_part in content.parts or []:
            if each_part.function_call:
                tool_calls.append(
                    AssistantToolCall(
                        id=each_part.function_call.id or "",
                        name=each_part.function_call.name,
                        arguments=json.dumps(each_part.function_call.args)
                        if each_part.function_call.args
                        else None,
                    )
                )
            if each_part.text:
                text_content = each_part.text
        return AssistantMessage(content=text_content, tool_calls=tool_calls)

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
        google_tools = self._get_google_tools(
            tools, use_tools_for_structured_output, response_format
        )
        config = GenerateContentConfig(
            tools=google_tools,
            system_instruction=self._get_system_prompt(messages),
            response_mime_type=self._get_response_mime_type(
                response_format, use_tools_for_structured_output
            ),
            response_json_schema=self._get_response_json_schema(
                response_format, use_tools_for_structured_output
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

        if assistant_message.tool_calls:
            for each in assistant_message.tool_calls:
                if each.name == "ResponseSchema":
                    return ResponseContent(
                        content=json.loads(each.arguments) if each.arguments else {}
                    )

            if self._tools_manager:
                tool_responses = self._handle_tool_calls(assistant_message.tool_calls)
                new_messages = [
                    *messages,
                    assistant_message,
                    *tool_responses,
                ]
            else:
                new_messages = [
                    *messages,
                    assistant_message,
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

        if assistant_message.content and response_format:
            return ResponseContent(content=json.loads(assistant_message.content))

        return ResponseContent(content=assistant_message.content or "")

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
        google_tools = self._get_google_tools(
            tools, use_tools_for_structured_output, response_format
        )
        config = GenerateContentConfig(
            tools=google_tools,
            tool_config=(
                GoogleToolConfig(
                    function_calling_config=GoogleFunctionCallingConfig(
                        mode=GoogleFunctionCallingConfigMode.ANY,
                    )
                )
                if google_tools
                else None
            ),
            system_instruction=self._get_system_prompt(messages),
            response_mime_type=self._get_response_mime_type(
                response_format, use_tools_for_structured_output
            ),
            response_json_schema=self._get_response_json_schema(
                response_format, use_tools_for_structured_output
            ),
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        tool_calls: List[AssistantToolCall] = []
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
                    tool_calls.append(
                        AssistantToolCall(
                            id=each_part.function_call.id or "",
                            name=each_part.function_call.name,
                            arguments=json.dumps(each_part.function_call.args)
                            if each_part.function_call.args
                            else None,
                        )
                    )
                    if each_part.function_call.name == "ResponseSchema":
                        has_response_schema_tool_call = True
                        if each_part.function_call.args:
                            yield ResponseStreamContentChunk(
                                id=str(depth),
                                source="tool",
                                chunk=json.dumps(each_part.function_call.args),
                            )

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
