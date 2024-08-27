from typing import Optional

from groq import Groq

from .base import VersaLLM


class GroqClient(VersaLLM):

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: int = 0,
        max_output_tokens: int = 1024,
        **kwargs
    ):
        super().__init__(model, api_key, temperature, max_output_tokens, **kwargs)

    def __repr__(self) -> str:
        return "Groq Client"

    def _get_client_instance(self):
        return Groq()
