from dataclasses import dataclass

from pydantic import BaseModel, Field
from typing import Literal, Optional

@dataclass
class ChatPrompt:
    """
    Minimal chat prompt for vLLM / OpenAI-style models.
    """
    user_text: str
    system_text: Optional[str] = None
    def to_messages(self) -> list[dict[str, str]]:
        msgs = []
        if self.system_text:
            msgs.append({"role": "system", "content": self.system_text})
        msgs.append({"role": "user", "content": self.user_text})
        return msgs

class LikertScore(BaseModel):
    score: Literal[1, 2, 3, 4, 5] = Field(
        description="Likert score from 1 (lowest) to 5 (highest)"
    )
    justification: str

class FreeResponse(BaseModel):
    response: str