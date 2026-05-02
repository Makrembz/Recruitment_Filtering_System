from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Candidate, InterviewSession
from app.schemas import InterviewMessage, InterviewRead
from app.services.interview import (
    answer_interview,
    close_interview,
    serialize_session,
    start_interview,
)

router = APIRouter(prefix="/interviews", tags=["interviews"])


@router.post("/start/{candidate_id}", response_model=InterviewRead)
def create_interview(candidate_id: int, db: Session = Depends(get_db)) -> dict:
    candidate = db.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return serialize_session(start_interview(db, candidate))


@router.post("/{session_id}/message", response_model=InterviewRead)
def send_message(
    session_id: int, payload: InterviewMessage, db: Session = Depends(get_db)
) -> dict:
    session = db.get(InterviewSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Interview is not active")
    return serialize_session(answer_interview(db, session, payload.answer))


@router.post("/{session_id}/close", response_model=InterviewRead)
def finish_interview(session_id: int, db: Session = Depends(get_db)) -> dict:
    session = db.get(InterviewSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    return serialize_session(close_interview(db, session))
