from openai import OpenAI

from .base import VersaLLM


class OpenAIClient(VersaLLM):

    def __init__(
        self,
        model=None,
        api_key=None,
        temperature: int = 0,
        max_output_tokens: int = 1024,
        **kwargs
    ):
        super().__init__(model, api_key, temperature, max_output_tokens, **kwargs)

    def __repr__(self) -> str:
        return "OpenAI Client"

    def _get_client_instance(self):
        return OpenAI()
