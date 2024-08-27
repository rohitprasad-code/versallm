import inspect
import json
import logging
from typing import Optional, List, Dict, Any

import anthropic

from ..utils.response import Response, ToolUsed, Usage
from ..utils.memory import ConversationalMemory
from .base import VersaLLM


def _tool_def_conversion(
        tool_definition: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    anthropic_tool_def = []

    for item in tool_definition:
        function_name = item["function"]["name"]
        description = item["function"]["description"]
        input_schema = item["function"]["parameters"]

        converted_function = {
            "name": function_name,
            "description": description,
            "input_schema": input_schema,
        }

        anthropic_tool_def.append(converted_function)

    return anthropic_tool_def


class AnthropicClient(VersaLLM):

    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            temperature: int = 0,
            max_output_tokens: int = 1024,
            **kwargs: Any,
    ):
        super().__init__(model, api_key, temperature, max_output_tokens, **kwargs)
        self.system_prompt: Optional[str] = None  # Initialize system_prompt
        self.memory: ConversationalMemory = ConversationalMemory()  # Initialize chat_history

    def __repr__(self) -> str:
        return "Anthropic Client"

    def system_message(self, message: str) -> None:
        self.system_prompt = message

    def _execute_functions(self, tool_use) -> Optional[Any]:
        for func in self.functions:
            if func.__name__ == tool_use.name:
                sig = inspect.signature(func)
                try:
                    func_args = tool_use.input
                    sig.bind(**func_args)
                    result = func(**func_args)
                    return result
                except TypeError as e:
                    logging.error(f"TypeError: {e}")
                    return None
        return None

    def completion(
            self, user_prompt: str, tools=None, **kwargs: Any
    ) -> Response:
        if tools is None:
            tools = []
        client = anthropic.Anthropic()

        if "messages" in kwargs:
            message = kwargs["messages"]
            self.memory.chat_history = message
        else:
            user_message = {"role": "user", "content": user_prompt}
            self.memory.chat_history.append(user_message)

        while True:
            response = client.messages.create(
                model=self.model,
                system=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=self.memory.chat_history,
                tools=_tool_def_conversion(tools),
            )

            if response.stop_reason == "tool_use":
                self.memory.chat_history.append(
                    {"role": "assistant", "content": response.content}
                )

                tool_use = None
                if isinstance(response.content, list):
                    tool_use = next(
                        (
                            block
                            for block in response.content
                            if block.type == "tool_use"
                        ),
                        None,
                    )

                result = self._execute_functions(tool_use) if tool_use else None
                function_call_result_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id if tool_use else None,
                            "content": json.dumps(result),
                        }
                    ],
                }
                self.memory.chat_history.append(function_call_result_message)

                tool_used = ToolUsed(
                    name=tool_use.name if tool_use else None,
                    input=tool_use.input if tool_use else {},
                )

            else:
                self.memory.chat_history.append(
                    {"role": "assistant", "content": response.content}
                )
                tool_used = ToolUsed(
                    name=None,
                    input={},
                )

            message_content = (
                next(
                    (
                        block.text
                        for block in response.content
                        if hasattr(block, "text")
                    ),
                    None,
                )
                if isinstance(response.content, list)
                else response.content
            )

            yield Response(
                message=message_content,
                model=self.model,
                tool_used=tool_used,
                usage=Usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                ),
            )
            if response.stop_reason == "end_turn":
                break
