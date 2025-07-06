# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenAI API client wrapper with tool integration."""

import os
import json
import random
import time
import asyncio
import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolMessageParam, ChatCompletionMessageToolCallParam
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params.function_definition import FunctionDefinition
from typing import override

from ..tools.base import Tool, ToolCall, ToolResult
from ..utils.config import ModelParameters
from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, LLMUsage


class OpenAIClient(BaseLLMClient):
    """OpenAI client wrapper with tool schema generation."""

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)

        if self.api_key == "":
            self.api_key: str = os.getenv("OPENAI_API_KEY", "")

        if self.api_key == "":
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in environment variables or config file.")

        # Initialize OpenAI client with optional base_url
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.client: openai.OpenAI = openai.OpenAI(**client_kwargs)
        self.message_history: list[ChatCompletionMessageParam] = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    @override
    async def chat(self, messages: list[LLMMessage], model_parameters: ModelParameters, tools: list[Tool] | None = None, reuse_history: bool = True) -> LLMResponse:
        """Send chat messages to OpenAI with optional tool support."""
        openai_messages: list[ChatCompletionMessageParam] = self.parse_messages(messages)

        tool_schemas = None
        if tools:
            tool_schemas = [ChatCompletionToolParam(
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.get_input_schema()
                ),
                type="function"
            ) for tool in tools]

        if reuse_history:
            self.message_history = self.message_history + openai_messages
        else:
            self.message_history = openai_messages

        response = None
        error_message = ""
        for i in range(model_parameters.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_parameters.model,
                    messages=self.message_history,
                    tools=tool_schemas if tool_schemas else openai.NOT_GIVEN,
                    temperature=model_parameters.temperature,
                    top_p=model_parameters.top_p,
                    max_tokens=model_parameters.max_tokens,
                )
                break
            except Exception as e:
                error_message += f"Error {i + 1}: {str(e)}\n"
                # Randomly sleep for 3-30 seconds (async)
                await asyncio.sleep(random.randint(3, 30))
                continue

        if response is None:
            raise ValueError(f"Failed to get response from OpenAI after max retries: {error_message}")

        choice = response.choices[0]
        content = choice.message.content or ""
        
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls: list[ToolCall] | None = []
            for tool_call in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    name=tool_call.function.name,
                    call_id=tool_call.id,
                    arguments=json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                ))

        # Update message history
        if tool_calls:
            self.message_history.append(ChatCompletionAssistantMessageParam(
                role="assistant",
                content=content,
                tool_calls=[ChatCompletionMessageToolCallParam(
                    id=tool_call.call_id,
                    function=Function(
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.arguments)
                    ),
                    type="function"
                ) for tool_call in tool_calls]
            ))
        elif content:
            self.message_history.append(ChatCompletionAssistantMessageParam(
                content=content,
                role="assistant"
            ))

        usage = None
        if response.usage:
            # Safely extract cache and reasoning tokens
            cache_read_tokens = 0
            reasoning_tokens = 0
            
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                prompt_details = response.usage.prompt_tokens_details
                if hasattr(prompt_details, 'cached_tokens') and prompt_details.cached_tokens:
                    cache_read_tokens = prompt_details.cached_tokens
            
            if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                completion_details = response.usage.completion_tokens_details
                if hasattr(completion_details, 'reasoning_tokens') and completion_details.reasoning_tokens:
                    reasoning_tokens = completion_details.reasoning_tokens
            
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cache_read_input_tokens=cache_read_tokens,
                reasoning_tokens=reasoning_tokens
            )

        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls
        )

        # Record trajectory if recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="openai",
                model=model_parameters.model,
                tools=tools
            )

        return llm_response

    @override
    def supports_tool_calling(self, model_parameters: ModelParameters) -> bool:
        """Check if the current model supports tool calling."""

        if 'o1-mini' in model_parameters.model:
            return False

        tool_capable_models = [
            "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-4.1", "gpt-4.5",
            "o1", "o3", "o4"
        ]
        return any(model in model_parameters.model for model in tool_capable_models)

    def parse_messages(self, messages: list[LLMMessage]) -> list[ChatCompletionMessageParam]:
        """Parse the messages to OpenAI format."""
        openai_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if msg.tool_result:
                openai_messages.append(self.parse_tool_call_result(msg.tool_result))
            elif msg.tool_call:
                # Tool calls are handled in message history update, skip here
                continue
            else:
                if not msg.content:
                    raise ValueError("Message content is required")
                if msg.role == "system":
                    openai_messages.append(ChatCompletionSystemMessageParam(
                        role="system", 
                        content=msg.content
                    ))
                elif msg.role == "user":
                    openai_messages.append(ChatCompletionUserMessageParam(
                        role="user", 
                        content=msg.content
                    ))
                elif msg.role == "assistant":
                    openai_messages.append(ChatCompletionAssistantMessageParam(
                        role="assistant", 
                        content=msg.content
                    ))
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        return openai_messages

    def parse_tool_call_result(self, tool_call_result: ToolResult) -> ChatCompletionToolMessageParam:
        """Parse the tool call result from the LLM response."""
        result: str = ""
        if tool_call_result.result:
            result = result + tool_call_result.result + "\n"
        if tool_call_result.error:
            result += "Tool call failed with error:\n"
            result += tool_call_result.error
        result = result.strip()

        return ChatCompletionToolMessageParam(
            content=result,
            role="tool",
            tool_call_id=tool_call_result.call_id,
        )