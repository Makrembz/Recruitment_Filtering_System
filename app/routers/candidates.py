from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Candidate, Job
from app.schemas import CandidateRead, RankedCandidate
from app.services.parser import parse_cv
from app.services.scoring import infer_candidate_name, score_candidate
from app.utils.files import safe_upload_path

router = APIRouter(prefix="/candidates", tags=["candidates"])


@router.post("/upload/{job_id}", response_model=CandidateRead)
async def upload_candidate(
    job_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Candidate:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        path = safe_upload_path(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    content = await file.read()
    Path(path).write_bytes(content)
    cv_text = parse_cv(path)
    if not cv_text:
        raise HTTPException(status_code=422, detail="Could not extract text from CV")

    score = score_candidate(cv_text=cv_text, job_description=job.description)
    candidate = Candidate(
        job_id=job.id,
        name=infer_candidate_name(cv_text, file.filename or path.name),
        filename=file.filename or path.name,
        cv_text=cv_text,
        semantic_score=score.semantic_score,
        llm_score=score.llm_score,
        final_score=score.final_score,
        llm_reasoning=f"[{score.provider}] {score.reasoning}",
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return candidate


@router.get("/ranked/{job_id}", response_model=list[RankedCandidate])
def ranked_candidates(job_id: int, db: Session = Depends(get_db)) -> list[dict]:
    candidates = (
        db.query(Candidate)
        .filter(Candidate.job_id == job_id)
        .order_by(Candidate.final_score.desc())
        .all()
    )
    return [{"rank": index + 1, "candidate": candidate} for index, candidate in enumerate(candidates)]
