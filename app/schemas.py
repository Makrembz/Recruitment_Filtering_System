from datetime import datetime

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    title: str = Field(default="Untitled role", max_length=200)
    description: str = Field(min_length=30)


class JobRead(BaseModel):
    id: int
    title: str
    description: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CandidateRead(BaseModel):
    id: int
    job_id: int
    name: str
    filename: str
    semantic_score: float
    llm_score: float
    final_score: float
    interview_score: float | None = None
    combined_final_score: float | None = None
    recommendation: str | None = None
    llm_reasoning: str

    model_config = {"from_attributes": True}


class RankedCandidate(BaseModel):
    rank: int
    candidate: CandidateRead


class FinalRankedCandidate(BaseModel):
    rank: int
    id: int
    name: str
    cv_score: float
    interview_score: float
    combined_final_score: float
    recommendation: str | None


class InterviewMessage(BaseModel):
    answer: str = Field(min_length=1)


class InterviewRead(BaseModel):
    id: int
    candidate_id: int
    status: str
    transcript: list[dict[str, str]]
    summary: str
