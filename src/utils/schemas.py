from pydantic import BaseModel, Field
from typing import Literal

class LikertScore(BaseModel):
    score: Literal[1, 2, 3, 4, 5] = Field(
        description="Likert score from 1 (lowest) to 5 (highest)"
    )
    justification: str

class FreeResponse(BaseModel):
    response: str