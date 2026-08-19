from pydantic import BaseModel, Field
from typing import Literal

class LikertScore(BaseModel):
    score: Literal[1, 2, 3, 4, 5] = Field(
        description="Likert score from 1 (lowest) to 5 (highest)"
    )
    justification: str

class FreeResponse(BaseModel):
    response: str

class GeneratedQuestion(BaseModel):
    scenario_type: str = Field(
        description=(
            "Short label describing the underlying situation. "
            "Examples: peer_pressure, party_aftermath, "
            "family_conflict, health_scare, rumor, "
            "social_media_influence, friend_in_trouble, "
            "relationship_issue, trying_to_quit."
        )
    )

    situation: str = Field(
        description=(
            "One-sentence description of the specific situation "
            "connecting the message to the health topic."
        )
    )

    response: str = Field(
        description="Realistic child-authored message."
    )