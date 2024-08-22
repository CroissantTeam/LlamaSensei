from pydantic import BaseModel, Field


class Question(BaseModel):
    question: str = Field(..., description="Query content")
    course: str = Field(..., description="Course to ask")


class ChatResponse(BaseModel):
    answer: str
    evidence: dict
