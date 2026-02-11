import asyncio
import json
from logging import Logger
from typing import AsyncGenerator, List, Optional
import dirtyjson
from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)
from anthropic._types import NOT_GIVEN as ANTHROPIC_NOT_GIVEN
from google import genai
from google.genai.types import Content as GoogleContent, Part as GoogleContentPart
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    ToolConfig as GoogleToolConfig,
    FunctionCallingConfig as GoogleFunctionCallingConfig,
    FunctionCallingConfigMode as GoogleFunctionCallingConfigMode,
)
from google.genai.types import Tool as GoogleTool
from anthropic import AsyncAnthropic
from anthropic.types import Message as AnthropicMessage
from anthropic import MessageStreamEvent as AnthropicMessageStreamEvent
from core.base_client import BaseClient
from models.llm_provider import LLMProvider
from models.llm_message import (
    AnthropicAssistantMessage,
    AnthropicUserMessage,
    GoogleAssistantMessage,
    GoogleToolCallMessage,
    OpenAIAssistantMessage,
    LLMMessage,
    LLMSystemMessage,
    LLMUserMessage,
)
from models.llm_tool_call import (
    AnthropicToolCall,
    GoogleToolCall,
    LLMToolCall,
    OpenAIToolCall,
    OpenAIToolCallFunction,
)
from models.llm_tools import LLMDynamicTool, LLMTool
from tools.llm_tool_calls_handler import LLMToolCallsHandler
from utils.async_iterator import iterator_to_async
from utils.dummy_functions import do_nothing_async
from utils.get_env import (
    get_anthropic_api_key_env,
    get_custom_llm_api_key_env,
    get_custom_llm_url_env,
    get_disable_thinking_env,
    get_google_api_key_env,
    get_ollama_url_env,
    get_openai_api_key_env,
    get_tool_calls_env,
    get_web_grounding_env,
)
from utils.llm_provider import get_llm_provider, get_model
from utils.parsers import parse_bool_or_none
from utils.schema_utils import (
    ensure_more_strict_json_schema,
    ensure_strict_json_schema,
    flatten_json_schema,
    remove_titles_from_schema,
)


class LLMClient(BaseClient):
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        logger: Optional[Logger] = None,
    ):
        self._logger = logger

        self.llm_provider = llm_provider or get_llm_provider()
        self._client = self._get_client()
        self.tool_calls_handler = LLMToolCallsHandler(self)

    # ? Use tool calls
    def use_tool_calls_for_structured_output(self) -> bool:
        if self.llm_provider != LLMProvider.CUSTOM:
            return False
        return parse_bool_or_none(get_tool_calls_env()) or False

    # ? Web Grounding
    def enable_web_grounding(self) -> bool:
        if (
            self.llm_provider == LLMProvider.OLLAMA
            or self.llm_provider == LLMProvider.CUSTOM
        ):
            return False
        return parse_bool_or_none(get_web_grounding_env()) or False

    # ? Disable thinking
    def disable_thinking(self) -> bool:
        return parse_bool_or_none(get_disable_thinking_env()) or False

    # ? Clients
    def _get_client(self):
        match self.llm_provider:
            case LLMProvider.OPENAI:
                if self._logger:
                    self._logger.info("Getting OpenAI client")
                return self._get_openai_client()
            case LLMProvider.GOOGLE:
                if self._logger:
                    self._logger.info("Getting Google client")
                return self._get_google_client()
            case LLMProvider.ANTHROPIC:
                if self._logger:
                    self._logger.info("Getting Anthropic client")
                return self._get_anthropic_client()
            case LLMProvider.OLLAMA:
                if self._logger:
                    self._logger.info("Getting Ollama client")
                return self._get_ollama_client()
            case LLMProvider.CUSTOM:
                if self._logger:
                    self._logger.info("Getting Custom client")
                return self._get_custom_client()
            case _:
                raise HTTPException(
                    status_code=500,
                    detail="LLM Provider must be either openai, google, anthropic, ollama, or custom",
                )

    def _get_openai_client(self):
        if not get_openai_api_key_env():
            raise HTTPException(
                status_code=500,
                detail="OpenAI API Key is not set",
            )
        return AsyncOpenAI()

    def _get_google_client(self):
        if not get_google_api_key_env():
            raise HTTPException(
                status_code=500,
                detail="Google API Key is not set",
            )
        return genai.Client()

    def _get_anthropic_client(self):
        if not get_anthropic_api_key_env():
            raise HTTPException(
                status_code=500,
                detail="Anthropic API Key is not set",
            )
        return AsyncAnthropic()

    def _get_ollama_client(self):
        return AsyncOpenAI(
            base_url=(get_ollama_url_env() or "http://localhost:11434") + "/v1",
            api_key="ollama",
        )

    def _get_custom_client(self):
        if not get_custom_llm_url_env():
            raise HTTPException(
                status_code=500,
                detail="Custom LLM URL is not set",
            )
        return AsyncOpenAI(
            base_url=get_custom_llm_url_env(),
            api_key=get_custom_llm_api_key_env() or "null",
        )

    # ? Prompts
    def _get_system_prompt(self, messages: List[LLMMessage]) -> str:
        for message in messages:
            if isinstance(message, LLMSystemMessage):
                return message.content
        return ""

    def _get_google_messages(self, messages: List[LLMMessage]) -> List[GoogleContent]:
        contents = []
        for message in messages:
            if isinstance(message, LLMUserMessage):
                contents.append(
                    GoogleContent(
                        role=message.role,
                        parts=[GoogleContentPart(text=message.content)],
                    )
                )
            elif isinstance(message, GoogleAssistantMessage):
                contents.append(message.content)
            elif isinstance(message, GoogleToolCallMessage):
                contents.append(
                    GoogleContent(
                        role="user",
                        parts=[
                            GoogleContentPart.from_function_response(
                                name=message.name,
                                response=message.response,
                            )
                        ],
                    )
                )

        return contents

    def _get_anthropic_messages(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        return [
            message for message in messages if not isinstance(message, LLMSystemMessage)
        ]

    # ? Generate Unstructured Content
    async def _generate_openai(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        extra_body: Optional[dict] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ) -> str | None:
        client: AsyncOpenAI = self._client
        if self._logger:
            self._logger.info(
                f"Generating Unstructured Content using OpenAI {model} at depth {depth}"
            )
            if tools:
                self._logger.info(f"{len(tools)} tools passed")

        response = await client.chat.completions.create(
            model=model,
            messages=[message.model_dump() for message in messages],
            max_completion_tokens=max_tokens,
            tools=tools,
            extra_body=extra_body,
            temperature=temperature,
        )
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            parsed_tool_calls = [
                OpenAIToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=OpenAIToolCallFunction(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
                for tool_call in tool_calls
            ]

            if self._logger:
                self._logger.info(f"Found {len(parsed_tool_calls)} tool calls")
                for index, each in enumerate(parsed_tool_calls):
                    self._logger.info(
                        f"Tool call {index}: {each.function.name} with arguments {each.function.arguments}"
                    )

            tool_call_messages = await self.tool_calls_handler.handle_tool_calls_openai(
                parsed_tool_calls
            )
            assistant_message = OpenAIAssistantMessage(
                role="assistant",
                content=response.choices[0].message.content,
                tool_calls=[tool_call.model_dump() for tool_call in parsed_tool_calls],
            )
            new_messages = [
                *messages,
                assistant_message,
                *tool_call_messages,
            ]
            return await self._generate_openai(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                tools=tools,
                extra_body=extra_body,
                temperature=temperature,
                depth=depth + 1,
            )

        return response.choices[0].message.content

    async def _generate_google(
        self,
        model: str,
        messages: List[LLMMessage],
        tools: Optional[List[dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ) -> str | None:
        client: genai.Client = self._client
        if self._logger:
            self._logger.info(
                f"Generating Structured Content using Google {model} at depth {depth}"
            )
        google_tools = None
        if tools:
            google_tools = [GoogleTool(function_declarations=[tool]) for tool in tools]

        if self._logger and google_tools:
            self._logger.info(f"{len(google_tools)} tools passed")

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=self._get_google_messages(messages),
            config=GenerateContentConfig(
                tools=google_tools,
                system_instruction=self._get_system_prompt(messages),
                response_mime_type="text/plain",
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        content = response.candidates[0].content
        response_parts = content.parts

        if not response_parts:
            return None

        text_content = None
        tool_calls: List[GoogleToolCall] = []
        for each_part in response_parts:
            if each_part.function_call:
                tool_calls.append(
                    GoogleToolCall(
                        id=each_part.function_call.id,
                        name=each_part.function_call.name,
                        arguments=each_part.function_call.args,
                    )
                )
            if each_part.text:
                text_content = each_part.text

        if tool_calls:
            tool_call_messages = await self.tool_calls_handler.handle_tool_calls_google(
                tool_calls
            )
            if self._logger:
                self._logger.info(f"Found {len(tool_calls)} tool calls")
                for index, each in enumerate(tool_calls):
                    self._logger.info(
                        f"Tool call {index}: {each.name} with arguments {each.arguments}"
                    )
            new_messages = [
                *messages,
                GoogleAssistantMessage(
                    role="assistant",
                    content=content,
                ),
                *tool_call_messages,
            ]
            return await self._generate_google(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                tools=tools,
                temperature=temperature,
                depth=depth + 1,
            )

        return text_content

    async def _generate_anthropic(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ) -> str | None:
        client: AsyncAnthropic = self._client
        if self._logger:
            self._logger.info(
                f"Generating Structured Content using Anthropic {model} at depth {depth}"
            )
            if tools:
                self._logger.info(f"{len(tools)} tools passed")

        response: AnthropicMessage = await client.messages.create(
            model=model,
            system=self._get_system_prompt(messages),
            messages=[
                message.model_dump()
                for message in self._get_anthropic_messages(messages)
            ],
            tools=tools,
            max_tokens=max_tokens or 8000,
            temperature=temperature or ANTHROPIC_NOT_GIVEN,
        )
        text_content = None
        tool_calls: List[AnthropicToolCall] = []
        for content in response.content:
            if content.type == "text" and isinstance(content.text, str):
                text_content = content.text

            if content.type == "tool_use":
                tool_calls.append(
                    AnthropicToolCall(
                        id=content.id,
                        type=content.type,
                        name=content.name,
                        input=content.input,
                    )
                )

        if tool_calls:
            tool_call_messages = (
                await self.tool_calls_handler.handle_tool_calls_anthropic(tool_calls)
            )
            if self._logger:
                self._logger.info(f"Found {len(tool_calls)} tool calls")
                for index, each in enumerate(tool_calls):
                    self._logger.info(
                        f"Tool call {index}: {each.name} with arguments {each.input}"
                    )
            new_messages = [
                *messages,
                AnthropicAssistantMessage(
                    role="assistant",
                    content=[each.model_dump() for each in tool_calls],
                ),
                AnthropicUserMessage(
                    role="user",
                    content=[each.model_dump() for each in tool_call_messages],
                ),
            ]
            return await self._generate_anthropic(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                tools=tools,
                temperature=temperature,
                depth=depth + 1,
            )

        return text_content

    async def _generate_ollama(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ):
        return await self._generate_openai(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            depth=depth,
        )

    async def _generate_custom(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ):
        extra_body = {"enable_thinking": False} if self.disable_thinking() else None
        return await self._generate_openai(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=extra_body,
            temperature=temperature,
            depth=depth,
        )

    async def generate(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None,
        temperature: Optional[float] = None,
    ):
        parsed_tools = self.tool_calls_handler.parse_tools(tools)

        content = None
        match self.llm_provider:
            case LLMProvider.OPENAI:
                content = await self._generate_openai(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    tools=parsed_tools,
                    temperature=temperature,
                )
            case LLMProvider.GOOGLE:
                content = await self._generate_google(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    tools=parsed_tools,
                    temperature=temperature,
                )
            case LLMProvider.ANTHROPIC:
                content = await self._generate_anthropic(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    tools=parsed_tools,
                    temperature=temperature,
                )
            case LLMProvider.OLLAMA:
                content = await self._generate_ollama(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case LLMProvider.CUSTOM:
                content = await self._generate_custom(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
        if content is None:
            raise HTTPException(
                status_code=500,
                detail="LLM did not return any content",
            )
        return content

    # ? Generate Structured Content
    async def _generate_openai_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        strict: bool = False,
        prune_length_constraints: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        extra_body: Optional[dict] = None,
        temperature: Optional[float] = None,
        tool_call_for_structured_output: Optional[bool] = None,
        depth: int = 0,
    ) -> dict | None:
        client: AsyncOpenAI = self._client
        response_schema = response_format
        all_tools = [*tools] if tools else None

        if self._logger:
            self._logger.info(
                f"Generating Structured Content using OpenAI {model} at depth {depth}"
            )

        if tool_call_for_structured_output is None:
            use_tool_calls_for_structured_output = (
                self.use_tool_calls_for_structured_output()
            )
        else:
            use_tool_calls_for_structured_output = tool_call_for_structured_output

        if self._logger and use_tool_calls_for_structured_output:
            self._logger.info("Using tool calls for structured output")

        if strict and depth == 0:
            if prune_length_constraints:
                response_schema = ensure_more_strict_json_schema(
                    response_schema,
                    path=(),
                    root=response_schema,
                )
            else:
                response_schema = ensure_strict_json_schema(
                    response_schema,
                    path=(),
                    root=response_schema,
                )
        if use_tool_calls_for_structured_output and depth == 0:
            if all_tools is None:
                all_tools = []
            all_tools.append(
                self.tool_calls_handler.parse_tool(
                    LLMDynamicTool(
                        name="ResponseSchema",
                        description="Provide response to the user",
                        parameters=response_schema,
                        handler=do_nothing_async,
                    ),
                    strict=strict,
                )
            )

        if self._logger:
            if all_tools:
                self._logger.info(f"{len(all_tools)} tools passed")

        response = await client.chat.completions.create(
            model=model,
            messages=[message.model_dump() for message in messages],
            response_format=(
                {
                    "type": "json_schema",
                    "json_schema": (
                        {
                            "name": "ResponseSchema",
                            "strict": strict,
                            "schema": response_schema,
                        }
                    ),
                }
                if not use_tool_calls_for_structured_output
                else None
            ),
            max_completion_tokens=max_tokens,
            tools=all_tools,
            stream=False,
            extra_body=extra_body,
            temperature=temperature,
        )

        content = response.choices[0].message.content

        tool_calls = response.choices[0].message.tool_calls
        has_response_schema = False

        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.function.name == "ResponseSchema":
                    content = tool_call.function.arguments
                    has_response_schema = True

            if self._logger:
                self._logger.info(f"Found {len(tool_calls)} tool calls")
                for index, each in enumerate(tool_calls):
                    self._logger.info(
                        f"Tool call {index}: {each.function.name} with arguments {each.function.arguments}"
                    )

            if not has_response_schema:
                parsed_tool_calls = [
                    OpenAIToolCall(
                        id=tool_call.id,
                        type=tool_call.type,
                        function=OpenAIToolCallFunction(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                    for tool_call in tool_calls
                ]
                tool_call_messages = (
                    await self.tool_calls_handler.handle_tool_calls_openai(
                        parsed_tool_calls
                    )
                )
                new_messages = [
                    *messages,
                    OpenAIAssistantMessage(
                        role="assistant",
                        content=response.choices[0].message.content,
                        tool_calls=[each.model_dump() for each in parsed_tool_calls],
                    ),
                    *tool_call_messages,
                ]
                content = await self._generate_openai_structured(
                    model=model,
                    messages=new_messages,
                    response_format=response_schema,
                    strict=strict,
                    prune_length_constraints=prune_length_constraints,
                    max_tokens=max_tokens,
                    tools=all_tools,
                    extra_body=extra_body,
                    temperature=temperature,
                    tool_call_for_structured_output=use_tool_calls_for_structured_output,
                    depth=depth + 1,
                )
        if content:
            if depth == 0:
                return dict(dirtyjson.loads(content))
            return content
        return None

    async def _generate_google_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        max_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ) -> dict | None:
        client: genai.Client = self._client
        if self._logger:
            self._logger.info(
                f"Generating Structured Content using Google {model} at depth {depth}"
            )

        google_tools = None
        if tools:
            google_tools = [GoogleTool(function_declarations=[tool]) for tool in tools]
            google_tools.append(
                GoogleTool(
                    function_declarations=[
                        {
                            "name": "ResponseSchema",
                            "description": "Provide response to the user",
                            "parameters": remove_titles_from_schema(
                                flatten_json_schema(response_format)
                            ),
                        }
                    ]
                )
            )

        if self._logger and google_tools:
            self._logger.info(f"{len(google_tools)} tools passed")

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=self._get_google_messages(messages),
            config=GenerateContentConfig(
                tools=google_tools,
                tool_config=(
                    GoogleToolConfig(
                        function_calling_config=GoogleFunctionCallingConfig(
                            mode=GoogleFunctionCallingConfigMode.ANY,
                        ),
                    )
                    if tools
                    else None
                ),
                system_instruction=self._get_system_prompt(messages),
                response_mime_type="application/json" if not tools else None,
                response_json_schema=response_format if not tools else None,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        content = response.candidates[0].content
        response_parts = content.parts
        text_content = None

        if not response_parts:
            return None

        tool_calls: List[GoogleToolCall] = []
        for each_part in response_parts:
            if each_part.function_call:
                tool_calls.append(
                    GoogleToolCall(
                        id=each_part.function_call.id,
                        name=each_part.function_call.name,
                        arguments=each_part.function_call.args,
                    )
                )

            if each_part.text:
                text_content = each_part.text

        for each in tool_calls:
            if each.name == "ResponseSchema":
                return each.arguments

        if tool_calls:
            tool_call_messages = await self.tool_calls_handler.handle_tool_calls_google(
                tool_calls
            )
            new_messages = [
                *messages,
                GoogleAssistantMessage(
                    role="assistant",
                    content=content,
                ),
                *tool_call_messages,
            ]
            return await self._generate_google_structured(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                temperature=temperature,
                depth=depth + 1,
            )

        if text_content:
            return dict(dirtyjson.loads(text_content))
        return None

    async def _generate_anthropic_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        tools: Optional[List[dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ):
        client: AsyncAnthropic = self._client
        if self._logger:
            self._logger.info(
                f"Generating Structured Content using Anthropic {model} at depth {depth}"
            )
        response: AnthropicMessage = await client.messages.create(
            model=model,
            system=self._get_system_prompt(messages),
            messages=[
                message.model_dump()
                for message in self._get_anthropic_messages(messages)
            ],
            # thinking=ThinkingConfigEnabledParam(budget_tokens=1024, type="enabled"),
            temperature=temperature or ANTHROPIC_NOT_GIVEN,
            max_tokens=max_tokens or 8000,
            tools=[
                {
                    "name": "ResponseSchema",
                    "description": "A response to the user's message",
                    "input_schema": response_format,
                },
                *(tools or []),
            ],
        )
        tool_calls: List[AnthropicToolCall] = []
        for content in response.content:
            if content.type == "tool_use":
                tool_calls.append(
                    AnthropicToolCall(
                        id=content.id,
                        type=content.type,
                        name=content.name,
                        input=content.input,
                    )
                )

        if tool_calls and self._logger:
            self._logger.info(f"Found {len(tool_calls)} tool calls")
            for index, each in enumerate(tool_calls):
                self._logger.info(
                    f"Tool call {index}: {each.name} with arguments {each.input}"
                )

        for each in tool_calls:
            if each.name == "ResponseSchema":
                return each.input

        if tool_calls:
            tool_call_messages = (
                await self.tool_calls_handler.handle_tool_calls_anthropic(tool_calls)
            )
            new_messages = [
                *messages,
                AnthropicAssistantMessage(
                    role="assistant",
                    content=[each.model_dump() for each in tool_calls],
                ),
                AnthropicUserMessage(
                    role="user",
                    content=[each.model_dump() for each in tool_call_messages],
                ),
            ]
            return await self._generate_anthropic_structured(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                temperature=temperature,
                depth=depth + 1,
            )

        return None

    async def _generate_ollama_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        strict: bool = False,
        prune_length_constraints: bool = False,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        temperature: Optional[float] = None,
        tool_call_for_structured_output: Optional[bool] = None,
        depth: int = 0,
    ):
        return await self._generate_openai_structured(
            model=model,
            messages=messages,
            response_format=response_format,
            strict=strict,
            prune_length_constraints=prune_length_constraints,
            max_tokens=max_tokens,
            extra_body=extra_body,
            temperature=temperature,
            tool_call_for_structured_output=tool_call_for_structured_output,
            depth=depth,
        )

    async def _generate_custom_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        strict: bool = False,
        prune_length_constraints: bool = False,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        temperature: Optional[float] = None,
        tool_call_for_structured_output: Optional[bool] = None,
        depth: int = 0,
    ):
        extra_body = {"enable_thinking": False} if self.disable_thinking() else None
        return await self._generate_openai_structured(
            model=model,
            messages=messages,
            response_format=response_format,
            strict=strict,
            prune_length_constraints=prune_length_constraints,
            max_tokens=max_tokens,
            extra_body=extra_body,
            temperature=temperature,
            tool_call_for_structured_output=tool_call_for_structured_output,
            depth=depth,
        )

    async def generate_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        strict: bool = False,
        prune_length_constraints: bool = False,
        tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        temperature: Optional[float] = None,
        tool_call_for_structured_output: Optional[bool] = None,
    ) -> dict:
        parsed_tools = self.tool_calls_handler.parse_tools(tools)

        content = None
        match self.llm_provider:
            case LLMProvider.OPENAI:
                content = await self._generate_openai_structured(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    strict=strict,
                    prune_length_constraints=prune_length_constraints,
                    tools=parsed_tools,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    temperature=temperature,
                    tool_call_for_structured_output=tool_call_for_structured_output,
                )
            case LLMProvider.GOOGLE:
                content = await self._generate_google_structured(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    tools=parsed_tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case LLMProvider.ANTHROPIC:
                content = await self._generate_anthropic_structured(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    tools=parsed_tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case LLMProvider.OLLAMA:
                content = await self._generate_ollama_structured(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    strict=strict,
                    prune_length_constraints=prune_length_constraints,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    temperature=temperature,
                    tool_call_for_structured_output=tool_call_for_structured_output,
                )
            case LLMProvider.CUSTOM:
                content = await self._generate_custom_structured(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    strict=strict,
                    prune_length_constraints=prune_length_constraints,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    temperature=temperature,
                    tool_call_for_structured_output=tool_call_for_structured_output,
                )
        if content is None:
            raise HTTPException(
                status_code=500,
                detail="LLM did not return any content",
            )
        return content

    # ? Stream Unstructured Content
    async def _stream_openai(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        extra_body: Optional[dict] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ) -> AsyncGenerator[str, None]:
        client: AsyncOpenAI = self._client
        if self._logger:
            self._logger.info(
                f"Streaming Unstructured Content using OpenAI {model} at depth {depth}"
            )
            if tools:
                self._logger.info(f"{len(tools)} tools passed")

        tool_calls: List[LLMToolCall] = []
        current_index = 0
        current_id = None
        current_name = None
        current_arguments = None
        async for event in await client.chat.completions.create(
            model=model,
            messages=[message.model_dump() for message in messages],
            max_completion_tokens=max_tokens,
            tools=tools,
            extra_body=extra_body,
            temperature=temperature,
            stream=True,
        ):
            event: OpenAIChatCompletionChunk = event
            content_chunk = event.choices[0].delta.content
            if content_chunk:
                yield content_chunk

            tool_call_chunk = event.choices[0].delta.tool_calls
            if tool_call_chunk:
                tool_index = tool_call_chunk[0].index
                tool_id = tool_call_chunk[0].id
                tool_name = tool_call_chunk[0].function.name
                tool_arguments = tool_call_chunk[0].function.arguments

                if current_index != tool_index:
                    tool_calls.append(
                        OpenAIToolCall(
                            id=current_id,
                            type="function",
                            function=OpenAIToolCallFunction(
                                name=current_name,
                                arguments=current_arguments,
                            ),
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

        if current_id is not None:
            tool_calls.append(
                OpenAIToolCall(
                    id=current_id,
                    type="function",
                    function=OpenAIToolCallFunction(
                        name=current_name,
                        arguments=current_arguments,
                    ),
                )
            )

        if tool_calls:
            if self._logger:
                self._logger.info(f"Found {len(tool_calls)} tool calls")
                for index, each in enumerate(tool_calls):
                    self._logger.info(
                        f"Tool call {index}: {each.function.name} with arguments {each.function.arguments}"
                    )
            tool_call_messages = await self.tool_calls_handler.handle_tool_calls_openai(
                tool_calls
            )
            new_messages = [
                *messages,
                OpenAIAssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[each.model_dump() for each in tool_calls],
                ),
                *tool_call_messages,
            ]
            async for event in self._stream_openai(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                tools=tools,
                extra_body=extra_body,
                temperature=temperature,
                depth=depth + 1,
            ):
                yield event

    async def _stream_google(
        self,
        model: str,
        messages: List[LLMMessage],
        tools: Optional[List[dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ) -> AsyncGenerator[str, None]:
        client: genai.Client = self._client
        if self._logger:
            self._logger.info(
                f"Streaming Unstructured Content using Google {model} at depth {depth}"
            )

        google_tools = None
        if tools:
            google_tools = [GoogleTool(function_declarations=[tool]) for tool in tools]
        if self._logger and google_tools:
            self._logger.info(f"{len(google_tools)} tools passed")

        generated_contents = []
        tool_calls: List[GoogleToolCall] = []
        async for event in iterator_to_async(client.models.generate_content_stream)(
            model=model,
            contents=self._get_google_messages(messages),
            config=GenerateContentConfig(
                system_instruction=self._get_system_prompt(messages),
                response_mime_type="text/plain",
                tools=google_tools,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        ):
            if not (
                event.candidates
                and event.candidates[0].content
                and event.candidates[0].content.parts
            ):
                continue

            generated_contents.append(event.candidates[0].content)

            for each_part in event.candidates[0].content.parts:
                if each_part.text:
                    yield each_part.text

                if each_part.function_call:
                    tool_calls.append(
                        GoogleToolCall(
                            id=each_part.function_call.id,
                            name=each_part.function_call.name,
                            arguments=each_part.function_call.args,
                        )
                    )

        if tool_calls:
            if self._logger:
                self._logger.info(f"Found {len(tool_calls)} tool calls")
                for index, each in enumerate(tool_calls):
                    self._logger.info(
                        f"Tool call {index}: {each.name} with arguments {each.arguments}"
                    )
            tool_call_messages = await self.tool_calls_handler.handle_tool_calls_google(
                tool_calls
            )
            new_messages = [
                *messages,
                *[
                    GoogleAssistantMessage(
                        role="assistant",
                        content=each,
                    )
                    for each in generated_contents
                ],
                *tool_call_messages,
            ]
            async for event in self._stream_google(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                tools=tools,
                temperature=temperature,
                depth=depth + 1,
            ):
                yield event

    async def _stream_anthropic(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ):
        client: AsyncAnthropic = self._client
        if self._logger:
            self._logger.info(
                f"Streaming Unstructured Content using Anthropic {model} at depth {depth}"
            )
            if tools:
                self._logger.info(f"{len(tools)} tools passed")

        tool_calls: List[AnthropicToolCall] = []
        async with client.messages.stream(
            model=model,
            system=self._get_system_prompt(messages),
            messages=[
                message.model_dump()
                for message in self._get_anthropic_messages(messages)
            ],
            max_tokens=max_tokens or 8000,
            tools=tools,
            temperature=temperature or ANTHROPIC_NOT_GIVEN,
        ) as stream:
            async for event in stream:
                event: AnthropicMessageStreamEvent = event

                if event.type == "text":
                    yield event.text

                if (
                    event.type == "content_block_stop"
                    and event.content_block.type == "tool_use"
                ):
                    tool_calls.append(
                        AnthropicToolCall(
                            id=event.content_block.id,
                            type=event.content_block.type,
                            name=event.content_block.name,
                            input=event.content_block.input,
                        )
                    )

        if tool_calls:
            if self._logger:
                self._logger.info(f"Found {len(tool_calls)} tool calls")
                for index, each in enumerate(tool_calls):
                    self._logger.info(
                        f"Tool call {index}: {each.name} with arguments {each.input}"
                    )
            tool_call_messages = (
                await self.tool_calls_handler.handle_tool_calls_anthropic(tool_calls)
            )
            new_messages = [
                *messages,
                AnthropicAssistantMessage(
                    role="assistant",
                    content=[each.model_dump() for each in tool_calls],
                ),
                AnthropicUserMessage(
                    role="user",
                    content=[each.model_dump() for each in tool_call_messages],
                ),
            ]
            async for event in self._stream_anthropic(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                tools=tools,
                temperature=temperature,
                depth=depth + 1,
            ):
                yield event

    def _stream_ollama(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ):
        return self._stream_openai(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            depth=depth,
        )

    def _stream_custom(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ):
        extra_body = {"enable_thinking": False} if self.disable_thinking() else None
        return self._stream_openai(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=extra_body,
            temperature=temperature,
            depth=depth,
        )

    def stream(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None,
        temperature: Optional[float] = None,
    ):
        parsed_tools = self.tool_calls_handler.parse_tools(tools)

        match self.llm_provider:
            case LLMProvider.OPENAI:
                return self._stream_openai(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    tools=parsed_tools,
                    temperature=temperature,
                )
            case LLMProvider.GOOGLE:
                return self._stream_google(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    tools=parsed_tools,
                    temperature=temperature,
                )
            case LLMProvider.ANTHROPIC:
                return self._stream_anthropic(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    tools=parsed_tools,
                    temperature=temperature,
                )
            case LLMProvider.OLLAMA:
                return self._stream_ollama(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case LLMProvider.CUSTOM:
                return self._stream_custom(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

    # ? Stream Structured Content
    async def _stream_openai_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        strict: bool = False,
        prune_length_constraints: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        extra_body: Optional[dict] = None,
        temperature: Optional[float] = None,
        tool_call_for_structured_output: Optional[bool] = None,
        depth: int = 0,
    ) -> AsyncGenerator[str, None]:
        client: AsyncOpenAI = self._client
        if self._logger:
            self._logger.info(
                f"Streaming Structured Content using OpenAI {model} at depth {depth}"
            )

        response_schema = response_format
        all_tools = [*tools] if tools else None

        if tool_call_for_structured_output is None:
            use_tool_calls_for_structured_output = (
                self.use_tool_calls_for_structured_output()
            )
        else:
            use_tool_calls_for_structured_output = tool_call_for_structured_output

        if self._logger and use_tool_calls_for_structured_output:
            self._logger.info("Using tool calls for structured output")

        if strict and depth == 0:
            if prune_length_constraints:
                response_schema = ensure_more_strict_json_schema(
                    response_schema,
                    path=(),
                    root=response_schema,
                )
            else:
                response_schema = ensure_strict_json_schema(
                    response_schema,
                    path=(),
                    root=response_schema,
                )

        if use_tool_calls_for_structured_output and depth == 0:
            if all_tools is None:
                all_tools = []
            all_tools.append(
                self.tool_calls_handler.parse_tool(
                    LLMDynamicTool(
                        name="ResponseSchema",
                        description="Provide response to the user",
                        parameters=response_schema,
                        handler=do_nothing_async,
                    ),
                    strict=strict,
                )
            )

        if self._logger:
            if all_tools:
                self._logger.info(f"{len(all_tools)} tools passed")

        tool_calls: List[LLMToolCall] = []
        current_index = 0
        current_id = None
        current_name = None
        current_arguments = None

        has_response_schema_tool_call = False
        async for event in await client.chat.completions.create(
            model=model,
            messages=[message.model_dump() for message in messages],
            max_completion_tokens=max_tokens,
            tools=all_tools,
            response_format=(
                {
                    "type": "json_schema",
                    "json_schema": (
                        {
                            "name": "ResponseSchema",
                            "strict": strict,
                            "schema": response_schema,
                        }
                    ),
                }
                if not use_tool_calls_for_structured_output
                else None
            ),
            extra_body=extra_body,
            temperature=temperature,
            stream=True,
        ):
            event: OpenAIChatCompletionChunk = event
            content_chunk = event.choices[0].delta.content
            if content_chunk and not use_tool_calls_for_structured_output:
                yield content_chunk

            tool_call_chunk = event.choices[0].delta.tool_calls
            if tool_call_chunk:
                tool_index = tool_call_chunk[0].index
                tool_id = tool_call_chunk[0].id
                tool_name = tool_call_chunk[0].function.name
                tool_arguments = tool_call_chunk[0].function.arguments

                if current_index != tool_index:
                    tool_calls.append(
                        OpenAIToolCall(
                            id=current_id,
                            type="function",
                            function=OpenAIToolCallFunction(
                                name=current_name,
                                arguments=current_arguments,
                            ),
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
                        yield tool_arguments
                    has_response_schema_tool_call = True

        if current_id is not None:
            tool_calls.append(
                OpenAIToolCall(
                    id=current_id,
                    type="function",
                    function=OpenAIToolCallFunction(
                        name=current_name,
                        arguments=current_arguments,
                    ),
                )
            )

        if tool_calls and not has_response_schema_tool_call:
            if self._logger:
                self._logger.info(f"Found {len(tool_calls)} tool calls")
                for index, each in enumerate(tool_calls):
                    self._logger.info(
                        f"Tool call {index}: {each.function.name} with arguments {each.function.arguments}"
                    )
            tool_call_messages = await self.tool_calls_handler.handle_tool_calls_openai(
                tool_calls
            )
            new_messages = [
                *messages,
                OpenAIAssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[each.model_dump() for each in tool_calls],
                ),
                *tool_call_messages,
            ]
            async for event in self._stream_openai_structured(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                strict=strict,
                prune_length_constraints=prune_length_constraints,
                tools=all_tools,
                response_format=response_schema,
                extra_body=extra_body,
                temperature=temperature,
                tool_call_for_structured_output=tool_call_for_structured_output,
                depth=depth + 1,
            ):
                yield event

    async def _stream_google_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        max_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ) -> AsyncGenerator[str, None]:
        client: genai.Client = self._client
        if self._logger:
            self._logger.info(
                f"Streaming Structured Content using Google {model} at depth {depth}"
            )

        google_tools = None
        if tools:
            google_tools = [GoogleTool(function_declarations=[tool]) for tool in tools]
            google_tools.append(
                GoogleTool(
                    function_declarations=[
                        {
                            "name": "ResponseSchema",
                            "description": "Provide response to the user",
                            "parameters": remove_titles_from_schema(
                                flatten_json_schema(response_format)
                            ),
                        }
                    ]
                )
            )

        if self._logger:
            self._logger.info(f"{len(google_tools)} tools passed")

        parsed_messages = self._get_google_messages(messages)

        generated_contents = []
        tool_calls: List[GoogleToolCall] = []
        has_response_schema_tool_call = False
        async for event in iterator_to_async(client.models.generate_content_stream)(
            model=model,
            contents=parsed_messages,
            config=GenerateContentConfig(
                tools=google_tools,
                tool_config=(
                    GoogleToolConfig(
                        function_calling_config=GoogleFunctionCallingConfig(
                            mode=GoogleFunctionCallingConfigMode.ANY,
                        ),
                    )
                    if tools
                    else None
                ),
                system_instruction=self._get_system_prompt(messages),
                response_mime_type="application/json" if not tools else None,
                response_json_schema=response_format if not tools else None,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        ):
            if not (
                event.candidates
                and event.candidates[0].content
                and event.candidates[0].content.parts
            ):
                continue

            generated_contents.append(event.candidates[0].content)

            for each_part in event.candidates[0].content.parts:
                if each_part.text and not google_tools:
                    yield each_part.text

                if each_part.function_call:
                    if each_part.function_call.name == "ResponseSchema":
                        has_response_schema_tool_call = True
                        if each_part.function_call.args:
                            yield json.dumps(each_part.function_call.args)

                    tool_calls.append(
                        GoogleToolCall(
                            id=each_part.function_call.id,
                            name=each_part.function_call.name,
                            arguments=each_part.function_call.args,
                        )
                    )

        if tool_calls and not has_response_schema_tool_call:
            if self._logger:
                self._logger.info(f"Found {len(tool_calls)} tool calls")
                for index, each in enumerate(tool_calls):
                    self._logger.info(
                        f"Tool call {index}: {each.name} with arguments {each.arguments}"
                    )
            tool_call_messages = await self.tool_calls_handler.handle_tool_calls_google(
                tool_calls
            )
            new_messages = [
                *messages,
                *[
                    GoogleAssistantMessage(
                        role="assistant",
                        content=each,
                    )
                    for each in generated_contents
                ],
                *tool_call_messages,
            ]
            async for event in self._stream_google_structured(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                temperature=temperature,
                depth=depth + 1,
            ):
                yield event

    async def _stream_anthropic_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        tools: Optional[List[dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        depth: int = 0,
    ) -> AsyncGenerator[str, None]:
        client: AsyncAnthropic = self._client
        if self._logger:
            self._logger.info(
                f"Streaming Structured Content using Anthropic {model} at depth {depth}"
            )

        tool_calls: List[AnthropicToolCall] = []
        has_response_schema_tool_call = False
        async with client.messages.stream(
            model=model,
            system=self._get_system_prompt(messages),
            messages=[
                message.model_dump()
                for message in self._get_anthropic_messages(messages)
            ],
            # thinking=ThinkingConfigEnabledParam(budget_tokens=1024, type="enabled"),
            temperature=temperature or ANTHROPIC_NOT_GIVEN,
            max_tokens=max_tokens or 8000,
            tools=[
                {
                    "name": "ResponseSchema",
                    "description": "A response to the user's message",
                    "input_schema": response_format,
                },
                *(tools or []),
            ],
        ) as stream:
            is_response_schema_tool_call_started = False
            async for event in stream:
                event: AnthropicMessageStreamEvent = event

                if (
                    event.type == "content_block_start"
                    and event.content_block.type == "tool_use"
                ):
                    if event.content_block.name == "ResponseSchema":
                        has_response_schema_tool_call = True
                        is_response_schema_tool_call_started = True

                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "input_json_delta"
                    and is_response_schema_tool_call_started
                ):
                    yield event.delta.partial_json

                if (
                    event.type == "content_block_stop"
                    and event.content_block.type == "tool_use"
                ):
                    tool_calls.append(
                        AnthropicToolCall(
                            id=event.content_block.id,
                            type=event.content_block.type,
                            name=event.content_block.name,
                            input=event.content_block.input,
                        )
                    )

        if tool_calls and not has_response_schema_tool_call:
            if self._logger:
                self._logger.info(f"Found {len(tool_calls)} tool calls")
                for index, each in enumerate(tool_calls):
                    self._logger.info(
                        f"Tool call {index}: {each.name} with arguments {each.input}"
                    )
            tool_call_messages = (
                await self.tool_calls_handler.handle_tool_calls_anthropic(tool_calls)
            )
            new_messages = [
                *messages,
                AnthropicAssistantMessage(
                    role="assistant",
                    content=[each.model_dump() for each in tool_calls],
                ),
                AnthropicUserMessage(
                    role="user",
                    content=[each.model_dump() for each in tool_call_messages],
                ),
            ]
            async for event in self._stream_anthropic_structured(
                model=model,
                messages=new_messages,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                temperature=temperature,
                depth=depth + 1,
            ):
                yield event

    def _stream_ollama_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        strict: bool = False,
        prune_length_constraints: bool = False,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        temperature: Optional[float] = None,
        tool_call_for_structured_output: Optional[bool] = None,
        depth: int = 0,
    ):
        return self._stream_openai_structured(
            model=model,
            messages=messages,
            response_format=response_format,
            strict=strict,
            prune_length_constraints=prune_length_constraints,
            max_tokens=max_tokens,
            extra_body=extra_body,
            temperature=temperature,
            tool_call_for_structured_output=tool_call_for_structured_output,
            depth=depth,
        )

    def _stream_custom_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        strict: bool = False,
        prune_length_constraints: bool = False,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        temperature: Optional[float] = None,
        tool_call_for_structured_output: Optional[bool] = None,
        depth: int = 0,
    ):
        extra_body = {"enable_thinking": False} if self.disable_thinking() else None
        return self._stream_openai_structured(
            model=model,
            messages=messages,
            response_format=response_format,
            strict=strict,
            prune_length_constraints=prune_length_constraints,
            max_tokens=max_tokens,
            extra_body=extra_body,
            temperature=temperature,
            tool_call_for_structured_output=tool_call_for_structured_output,
            depth=depth,
        )

    def stream_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        strict: bool = False,
        prune_length_constraints: bool = False,
        tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        temperature: Optional[float] = None,
        tool_call_for_structured_output: Optional[bool] = None,
    ):
        parsed_tools = self.tool_calls_handler.parse_tools(tools)

        match self.llm_provider:
            case LLMProvider.OPENAI:
                return self._stream_openai_structured(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    strict=strict,
                    prune_length_constraints=prune_length_constraints,
                    tools=parsed_tools,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    temperature=temperature,
                    tool_call_for_structured_output=tool_call_for_structured_output,
                )
            case LLMProvider.GOOGLE:
                return self._stream_google_structured(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    tools=parsed_tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case LLMProvider.ANTHROPIC:
                return self._stream_anthropic_structured(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    tools=parsed_tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case LLMProvider.OLLAMA:
                return self._stream_ollama_structured(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    strict=strict,
                    prune_length_constraints=prune_length_constraints,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    temperature=temperature,
                    tool_call_for_structured_output=tool_call_for_structured_output,
                )
            case LLMProvider.CUSTOM:
                return self._stream_custom_structured(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    strict=strict,
                    prune_length_constraints=prune_length_constraints,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    temperature=temperature,
                    tool_call_for_structured_output=tool_call_for_structured_output,
                )

    # ? Web search
    async def _search_openai(self, query: str) -> str:
        client: AsyncOpenAI = self._client
        if self._logger:
            self._logger.info("Performing web search using OpenAI")
        response = await client.responses.create(
            model=get_model(self.llm_provider),
            tools=[
                {
                    "type": "web_search_preview",
                }
            ],
            input=query,
        )
        return response.output_text

    async def _search_google(self, query: str) -> str:
        client: genai.Client = self._client
        if self._logger:
            self._logger.info("Performing web search using Google")
        grounding_tool = GoogleTool(google_search=GoogleSearch())
        config = GenerateContentConfig(tools=[grounding_tool])

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=get_model(self.llm_provider),
            contents=query,
            config=config,
        )
        return response.text

    async def _search_anthropic(self, query: str) -> str:
        client: AsyncAnthropic = self._client
        if self._logger:
            self._logger.info("Performing web search using Anthropic")

        response = await client.messages.create(
            model=get_model(self.llm_provider),
            max_tokens=4000,
            messages=[{"role": "user", "content": query}],
            tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
            ],
        )
        result = "\n".join(
            [each.text for each in response.content if each.type == "text"]
        )
        return result
