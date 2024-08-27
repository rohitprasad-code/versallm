from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ToolUsed:
    name: Optional[str]
    input: Dict[str, Any]


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class Response:
    message: str
    model: str
    tool_used: ToolUsed
    usage: Usage
