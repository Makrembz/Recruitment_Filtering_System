from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Candidate, Job
from app.schemas import FinalRankedCandidate, JobCreate, JobRead

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobRead)
def create_job(payload: JobCreate, db: Session = Depends(get_db)) -> Job:
    job = Job(title=payload.title, description=payload.description)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@router.get("/{job_id}/final-ranking", response_model=list[FinalRankedCandidate])
def final_ranking(job_id: int, db: Session = Depends(get_db)) -> list[FinalRankedCandidate]:
    candidates = (
        db.query(Candidate)
        .filter(
            Candidate.job_id == job_id,
            Candidate.interview_score.is_not(None),
            Candidate.combined_final_score.is_not(None),
        )
        .order_by(Candidate.combined_final_score.desc())
        .all()
    )
    return [
        FinalRankedCandidate(
            rank=index + 1,
            id=candidate.id,
            name=candidate.name,
            cv_score=candidate.final_score,
            interview_score=candidate.interview_score or 0.0,
            combined_final_score=candidate.combined_final_score or 0.0,
            recommendation=candidate.recommendation,
        )
        for index, candidate in enumerate(candidates)
    ]
