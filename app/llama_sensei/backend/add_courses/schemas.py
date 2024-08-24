from typing import Dict, List

from pydantic import BaseModel, Field


class AddCourseRequest(BaseModel):
    playlist_url: str = Field(..., description="Youtube course playlist")
    course_name: str = Field(..., description="Collection to create")


class SearchQuery(BaseModel):
    course_name: str = Field(..., description="Collection to search")
    text: str = Field(..., description="Query content")
    top_k: int = Field(default=5, gt=0)


class SearchResponse(BaseModel):
    documents: List[str]
    metadatas: List[Dict]
    embeddings: List[float]
