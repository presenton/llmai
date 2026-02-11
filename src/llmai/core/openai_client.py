import json
from logging import Logger
from typing import List, Optional, Tuple

from openai import AsyncOpenAI, Omit
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
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel

from core.base_client import BaseClient
from llmai.models.errors import LLMError
from llmai.models.tools import LLMTool
from llmai.tools.manager import ToolsManager
from models.messages import (
    AssistantMessage,
    AssistantToolCall,
    Message,
    SystemMessage,
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
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
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
            if isinstance(tool.schema, BaseModel):
                parameters = tool.schema.model_dump(mode="json")
            elif isinstance(tool.schema, dict):
                parameters = tool.schema
            else:
                parameters = {}
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
    

    async def _handle_tool_calls(self, calls: List[AssistantToolCall]):
        for each in calls:
            # response = await 
            pass



    async def generate(
        self,
        *,
        model: str,
        messages: List[Message],
        temperature: Optional[float] = None,
        tools: Optional[List[LLMTool]] = None,
        response_format: Optional[BaseModel | dict] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        use_tools_for_structured_output: Optional[bool] = None,
        depth: int = 0,
    ) -> Tuple[str | dict | None, List[Message]]:

        response = await self._client.chat.completions.create(
            model=model,
            messages=self._messages_to_openai_messages(messages),
            max_completion_tokens=max_tokens,
            tools=self._get_openai_tools_or_omit(
                tools, use_tools_for_structured_output, response_format
            ),
            extra_body=extra_body,
            temperature=temperature,
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
                    return json.loads(each.arguments), new_messages
            
            if self._tools_manager:


            return await self.generate(
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
            return json.loads(assistant_message.content), new_messages

        return assistant_message.content, new_messages