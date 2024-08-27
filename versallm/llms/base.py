import inspect
import json
from typing import Callable, Dict, List, Optional, Any

from ..utils.memory import ConversationalMemory
from ..utils.response import Response, Usage, ToolUsed


class VersaLLM:
    def __new__(cls, model: str, **kwargs: Any) -> "VersaLLM":
        if cls is VersaLLM:
            # Delayed imports to avoid circular dependencies
            if model in cls._get_groq_models():
                from .groq_client import GroqClient
                return GroqClient(model=model, **kwargs)
            elif model in cls._get_openai_models():
                from .openai_client import OpenAIClient
                return OpenAIClient(model=model, **kwargs)
            elif model in cls._get_anthropic_models():
                from .anthropic_client import AnthropicClient
                return AnthropicClient(model=model, **kwargs)
            else:
                raise ValueError(f"{model} not found!")
        return super().__new__(cls)

    @staticmethod
    def _get_groq_models():
        return [
            "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768",
            "gemma-7b-it", "gemma2-9b-it", "whisper-large-v3",
        ]

    @staticmethod
    def _get_openai_models():
        return [
            "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18", "gpt-4-turbo", "gpt-4-turbo-2024-04-09",
            "gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-1106-preview",
            "gpt-4-vision-preview", "gpt-4", "gpt-4-0314", "gpt-4-0613",
            "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-16k-0613",
        ]

    @staticmethod
    def _get_anthropic_models():
        return [
            "claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: int = 1024,
        functions: List[Callable] = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.memory = ConversationalMemory()
        self.functions = functions
        self.kwargs = kwargs

    def system_message(self, message: str) -> None:
        if len(self.memory.chat_history) == 0:
            system_message = {"role": "system", "content": message}
            self.memory.chat_history.append(system_message)

    def _execute_functions(self, tool_use):
        # Ensure tool_use is structured correctly before access
        if isinstance(tool_use, list) and len(tool_use) > 0 and hasattr(tool_use[0], "function"):
            func_name = tool_use[0].function.name
            for func in self.functions:
                if func.__name__ == func_name:
                    sig = inspect.signature(func)
                    try:
                        func_args = json.loads(tool_use[0].function.arguments)
                        sig.bind(**func_args)
                        result = func(**func_args)
                        return result
                    except TypeError as e:
                        print(f"TypeError: {e}")
                        return None
        return None

    def completion(
        self,
        user_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Response:
        client = self._get_client_instance()
        user_message = {"role": "user", "content": user_prompt}
        self.memory.chat_history.append(user_message)

        while True:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=self.memory.chat_history,
                tools=tools,
            )

            if response.choices[0].finish_reason == "tool_calls" and hasattr(response.choices[0].message, 'tool_calls'):
                self.memory.chat_history.append(response.choices[0].message)
                result = self._execute_functions(response.choices[0].message.tool_calls)

                function_call_result_message = {
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": response.choices[0].message.tool_calls[0].id,
                }

                tool_used = ToolUsed(
                    name=response.choices[0].message.tool_calls[0].function.name,
                    input=response.choices[0].message.tool_calls[0].function.arguments,
                )

                self.memory.chat_history.append(function_call_result_message)

            else:
                self.memory.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                )

                tool_used = ToolUsed(name=None, input={})

            yield Response(
                message=response.choices[0].message.content,
                model=self.model,
                tool_used=tool_used,
                usage=Usage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                ),
            )

            if response.choices[0].finish_reason == "stop":
                break

    def _get_client_instance(self) -> Any:
        raise NotImplementedError(
            "This method should be implemented by derived classes."
        )
