from typing import Dict, List

from pydantic import BaseModel, Field


class Question(BaseModel):
    question: str = Field(..., description="Query content")
    course: str = Field(..., description="Course to ask")
    indb: bool = Field(default=True, description="Search internal vectorDB")
    internet: bool = Field(default=False, description="Search on internet")


class ChatResponse(BaseModel):
    answer: str
    evidence: dict


class EvaluationRequest(BaseModel):
    query: str = Field(..., description="Query content")
    answer: str = Field(..., description="LLM answer")
    contexts: List[Dict] = Field(..., description="Retrieved contexts")
    course_name: str = Field(..., description="Course context")


class EvaluationResponse(BaseModel):
    f_score: str
    ar_score: str
