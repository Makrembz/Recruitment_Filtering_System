import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.config import groq_client
from src.chatbot.interview_bot import InterviewBot, InterviewState
from src.evaluation.explainer import generate_candidate_breakdown
from src.parsers.cv_parser import parse_cv_file
from src.scoring.scorer import score_candidate


DATABASE_URL = "sqlite:///./data/recruitment_api.db"
UPLOAD_DIR = Path("./data/uploads")
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "api_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(240))
    description: Mapped[str] = mapped_column(Text)
    requirements_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    candidates: Mapped[list["Candidate"]] = relationship(back_populates="job")


class Candidate(Base):
    __tablename__ = "api_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("api_jobs.id"), index=True)
    name: Mapped[str] = mapped_column(String(240))
    filename: Mapped[str] = mapped_column(String(500))
    parsed_cv_json: Mapped[str] = mapped_column(Text)
    similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    llm_score: Mapped[float] = mapped_column(Float, default=0.0)
    weighted_score: Mapped[float] = mapped_column(Float, default=0.0)
    passed_filter: Mapped[bool] = mapped_column(Boolean, default=False)
    interview_score: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    combined_final_score: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    recommendation: Mapped[str | None] = mapped_column(String(20), nullable=True, default=None)
    justification_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    job: Mapped["Job"] = relationship(back_populates="candidates")
    interview_sessions: Mapped[list["InterviewSession"]] = relationship(back_populates="candidate")


class InterviewSession(Base):
    __tablename__ = "api_interview_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("api_candidates.id"), index=True)
    state_json: Mapped[str] = mapped_column(Text)
    final_report_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    candidate: Mapped["Candidate"] = relationship(back_populates="interview_sessions")


Path("./data").mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

app = FastAPI(
    title="Recruitment Filtering API",
    description="CV parsing, candidate scoring, shortlist ranking, and chatbot interviews.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobCreate(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1)
    requirements: dict[str, Any] | list[Any] | str | None = None


class JobCreated(BaseModel):
    job_id: int


class JobSummary(BaseModel):
    job_id: int
    title: str
    description: str
    requirements: dict[str, Any] | list[Any] | str | None = None


class CandidateUploadResult(BaseModel):
    candidate_id: int
    name: str | None
    score: float
    similarity_score: float | None = None
    llm_score: float | None = None
    weighted_score: float | None = None
    passed_filter: bool
    llm_justification: list[str] = []


class ShortlistCandidate(BaseModel):
    candidate_id: int
    name: str | None
    similarity_score: float
    llm_score: float
    weighted_score: float
    passed_filter: bool
    llm_justification: list[str]


class FinalRankingCandidate(BaseModel):
    candidate_id: int
    name: str | None
    cv_filter_score: float
    interview_score: float | None
    final_ranking_score: float | None
    status: str
    recommendation: str | None
    llm_justification: list[str]


class InterviewStartResponse(BaseModel):
    session_id: str
    first_question: str


class InterviewAnswer(BaseModel):
    answer: str = Field(min_length=1)


class InterviewAnswerResponse(BaseModel):
    next_question: str | None
    is_complete: bool
    follow_up: str | None


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_candidate_interview_columns()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobCreated)
async def create_job(payload: JobCreate, db: Session = Depends(get_db)) -> JobCreated:
    job = Job(
        title=payload.title,
        description=payload.description,
        requirements_json=_json_dumps(payload.requirements or {}),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return JobCreated(job_id=job.id)


@app.get("/jobs", response_model=list[JobSummary])
def list_jobs(db: Session = Depends(get_db)) -> list[JobSummary]:
    jobs = db.query(Job).order_by(Job.created_at.desc()).all()
    result: list[JobSummary] = []
    for job in jobs:
        result.append(
            JobSummary(
                job_id=job.id,
                title=job.title,
                description=job.description,
                requirements=_json_loads(job.requirements_json, {}),
            )
        )
    return result


@app.get("/jobs/{job_id}/interviews")
async def get_job_interviews(
    job_id: int,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    _get_job_or_404(db, job_id)
    sessions = (
        db.query(InterviewSession)
        .join(Candidate)
        .filter(Candidate.job_id == job_id)
        .order_by(InterviewSession.created_at.desc())
        .all()
    )
    return [
        {
            "session_id": session.id,
            "candidate_id": session.candidate_id,
            "candidate_name": session.candidate.name if session.candidate else None,
            "complete": session.is_complete,
            "messages": _messages_from_state(_json_loads(session.state_json, {})),
            "report": _json_loads(session.final_report_json, {}) if session.final_report_json else None,
            "state": _json_loads(session.state_json, {}),
            "created_at": session.created_at.isoformat() if session.created_at else None,
        }
        for session in sessions
    ]


@app.get("/jobs/{job_id}/candidates", response_model=list[CandidateUploadResult])
async def get_job_candidates(
    job_id: int,
    db: Session = Depends(get_db),
) -> list[CandidateUploadResult]:
    _get_job_or_404(db, job_id)
    candidates = db.query(Candidate).filter(Candidate.job_id == job_id).order_by(Candidate.created_at.desc()).all()
    result: list[CandidateUploadResult] = []
    for candidate in candidates:
        result.append(
            CandidateUploadResult(
                candidate_id=candidate.id,
                name=candidate.name,
                score=candidate.weighted_score,
                similarity_score=candidate.similarity_score,
                llm_score=candidate.llm_score,
                weighted_score=candidate.weighted_score,
                passed_filter=candidate.passed_filter,
                llm_justification=_json_loads(candidate.justification_json, []),
            )
        )
    return result


@app.post("/jobs/{job_id}/candidates/upload", response_model=list[CandidateUploadResult])
async def upload_candidates(
    job_id: int,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> list[CandidateUploadResult]:
    job = _get_job_or_404(db, job_id)
    results: list[CandidateUploadResult] = []

    for upload in files:
        path = await _save_upload(upload)
        parsed_cv = await asyncio.to_thread(parse_cv_file, path, groq_client)
        candidate = Candidate(
            job_id=job.id,
            name=parsed_cv.get("full_name") or Path(upload.filename or path.name).stem,
            filename=upload.filename or path.name,
            parsed_cv_json=_json_dumps(parsed_cv),
        )
        db.add(candidate)
        db.flush()

        parsed_cv["candidate_id"] = candidate.id
        parsed_cv["name"] = candidate.name
        score = await asyncio.to_thread(
            score_candidate,
            parsed_cv,
            _job_payload(job),
            groq_client,
        )
        candidate.parsed_cv_json = _json_dumps(parsed_cv)
        candidate.similarity_score = score["similarity_score"]
        candidate.llm_score = score["llm_score"]
        candidate.weighted_score = score["weighted_score"]
        candidate.passed_filter = score["passed"]
        candidate.justification_json = _json_dumps(score["justification"])
        db.commit()
        db.refresh(candidate)

        results.append(
            CandidateUploadResult(
                candidate_id=candidate.id,
                name=candidate.name,
                score=candidate.weighted_score,
                similarity_score=candidate.similarity_score,
                llm_score=candidate.llm_score,
                weighted_score=candidate.weighted_score,
                passed_filter=candidate.passed_filter,
                llm_justification=_json_loads(candidate.justification_json, []),
            )
        )

    return results


@app.get("/jobs/{job_id}/shortlist", response_model=list[ShortlistCandidate])
async def shortlist(job_id: int, db: Session = Depends(get_db)) -> list[ShortlistCandidate]:
    _get_job_or_404(db, job_id)
    candidates = (
        db.query(Candidate)
        .filter(Candidate.job_id == job_id, Candidate.passed_filter.is_(True))
        .order_by(Candidate.weighted_score.desc())
        .all()
    )
    return [
        ShortlistCandidate(
            candidate_id=candidate.id,
            name=candidate.name,
            similarity_score=candidate.similarity_score,
            llm_score=candidate.llm_score,
            weighted_score=candidate.weighted_score,
            passed_filter=candidate.passed_filter,
            llm_justification=_json_loads(candidate.justification_json, []),
        )
        for candidate in candidates
    ]


@app.get("/jobs/{job_id}/final-ranking", response_model=list[FinalRankingCandidate])
async def final_ranking(job_id: int, db: Session = Depends(get_db)) -> list[FinalRankingCandidate]:
    _get_job_or_404(db, job_id)
    candidates = db.query(Candidate).filter(Candidate.job_id == job_id).all()

    rows = [
        FinalRankingCandidate(
            candidate_id=candidate.id,
            name=candidate.name,
            cv_filter_score=round(_clamp(float(candidate.weighted_score or 0.0), 0.0, 100.0), 2),
            interview_score=round(_clamp(float(candidate.interview_score), 0.0, 100.0), 2)
            if candidate.interview_score is not None
            else None,
            final_ranking_score=round(_clamp(float(candidate.combined_final_score), 0.0, 100.0), 2)
            if candidate.combined_final_score is not None
            else None,
            status="not started" if candidate.interview_score is None else str(candidate.recommendation or "hold"),
            recommendation=candidate.recommendation,
            llm_justification=_json_loads(candidate.justification_json, []),
        )
        for candidate in candidates
    ]

    rows.sort(
        key=lambda item: (
            item.final_ranking_score is None,
            -(item.final_ranking_score or 0.0),
            -(item.cv_filter_score or 0.0),
        )
    )
    return rows


@app.post("/candidates/{candidate_id}/interview/start", response_model=InterviewStartResponse)
async def start_interview(
    candidate_id: int,
    db: Session = Depends(get_db),
) -> InterviewStartResponse:
    candidate = _get_candidate_or_404(db, candidate_id)
    bot = InterviewBot(client=groq_client)
    snapshot = await asyncio.to_thread(
        bot.start_session,
        _json_loads(candidate.parsed_cv_json, {}),
        _job_payload(candidate.job),
        candidate.id,
    )
    session_id = uuid4().hex
    record = InterviewSession(
        id=session_id,
        candidate_id=candidate.id,
        state_json=_json_dumps(_state_to_dict(bot.state)),
        is_complete=bool(snapshot["complete"]),
    )
    db.add(record)
    db.commit()
    first_question = snapshot["next_question"]
    return InterviewStartResponse(session_id=session_id, first_question=first_question)


@app.post(
    "/candidates/{candidate_id}/interview/{session_id}/answer",
    response_model=InterviewAnswerResponse,
)
async def answer_interview(
    candidate_id: int,
    session_id: str,
    payload: InterviewAnswer,
    db: Session = Depends(get_db),
) -> InterviewAnswerResponse:
    candidate = _get_candidate_or_404(db, candidate_id)
    session = _get_interview_or_404(db, session_id, candidate_id)
    bot = _restore_bot(session, candidate)

    before_index = bot.state.question_index if bot.state else 0
    result = await asyncio.to_thread(bot.answer, payload.answer)
    follow_up = _follow_up_question(bot, before_index)

    session.state_json = _json_dumps(_state_to_dict(bot.state))
    session.is_complete = bool(result.get("recommendation") or result.get("complete"))
    if session.is_complete:
        session.final_report_json = _json_dumps(result)
        _apply_interview_result(candidate, result)
    db.commit()

    return InterviewAnswerResponse(
        next_question=None if session.is_complete else result.get("next_question"),
        is_complete=session.is_complete,
        follow_up=follow_up,
    )


@app.get("/candidates/{candidate_id}/interview/{session_id}/report")
async def interview_report(
    candidate_id: int,
    session_id: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    candidate = _get_candidate_or_404(db, candidate_id)
    session = _get_interview_or_404(db, session_id, candidate.id)
    if not session.final_report_json:
        if not session.is_complete:
            raise HTTPException(status_code=409, detail="Interview is not complete yet")
    report = _json_loads(session.final_report_json, {})
    if candidate.interview_score is None and report:
        _apply_interview_result(candidate, report)
        db.commit()
    return report


@app.get("/candidates/{candidate_id}")
async def get_candidate(
    candidate_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    candidate = _get_candidate_or_404(db, candidate_id)
    return {
        "candidate_id": candidate.id,
        "name": candidate.name,
        "parsed_cv": _json_loads(candidate.parsed_cv_json, {}),
        "similarity_score": candidate.similarity_score,
        "llm_score": candidate.llm_score,
        "weighted_score": candidate.weighted_score,
        "passed_filter": candidate.passed_filter,
    }


@app.get("/candidates/{candidate_id}/explain")
async def explain_candidate(
    candidate_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    candidate = _get_candidate_or_404(db, candidate_id)
    payload = {
        **_json_loads(candidate.parsed_cv_json, {}),
        "candidate_id": candidate.id,
        "name": candidate.name,
        "similarity_score": candidate.similarity_score,
        "llm_score": candidate.llm_score,
        "weighted_score": candidate.weighted_score,
        "interview_score": candidate.interview_score,
        "combined_final_score": candidate.combined_final_score,
        "recommendation": candidate.recommendation,
        "llm_justification": _json_loads(candidate.justification_json, []),
    }
    return await asyncio.to_thread(
        generate_candidate_breakdown,
        payload,
        _job_payload(candidate.job),
        groq_client,
    )


def _ensure_candidate_interview_columns() -> None:
    inspector = inspect(engine)
    if "api_candidates" not in inspector.get_table_names():
        return
    existing_columns = {column["name"] for column in inspector.get_columns("api_candidates")}
    column_sql = {
        "interview_score": "ALTER TABLE api_candidates ADD COLUMN interview_score FLOAT",
        "combined_final_score": "ALTER TABLE api_candidates ADD COLUMN combined_final_score FLOAT",
        "recommendation": "ALTER TABLE api_candidates ADD COLUMN recommendation VARCHAR(20)",
    }
    with engine.begin() as connection:
        for column_name, statement in column_sql.items():
            if column_name not in existing_columns:
                connection.execute(text(statement))


async def _save_upload(upload: UploadFile) -> Path:
    filename = Path(upload.filename or "cv").name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")
    path = UPLOAD_DIR / f"{uuid4().hex}_{filename}"
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {filename}")
    path.write_bytes(content)
    return path


def _restore_bot(session: InterviewSession, candidate: Candidate) -> InterviewBot:
    state_data = _json_loads(session.state_json, {})
    bot = InterviewBot(client=groq_client)
    bot.state = InterviewState(
        candidate_id=state_data.get("candidate_id"),
        candidate_cv=state_data.get("candidate_cv") or _json_loads(candidate.parsed_cv_json, {}),
        job_description=state_data.get("job_description") or _job_payload(candidate.job),
        planned_questions=state_data.get("planned_questions") or [],
        question_index=int(state_data.get("question_index") or 0),
        questions_asked=state_data.get("questions_asked") or [],
        answers_given=state_data.get("answers_given") or [],
        per_question_scores=state_data.get("per_question_scores") or [],
        complete=bool(state_data.get("complete")),
    )
    _rebuild_memory(bot)
    return bot


def _rebuild_memory(bot: InterviewBot) -> None:
    state = bot.state
    if state is None:
        return
    bot.memory.clear()
    for index, question in enumerate(state.questions_asked):
        bot.memory.chat_memory.add_ai_message(question)
        if index < len(state.answers_given):
            bot.memory.chat_memory.add_user_message(state.answers_given[index])


def _messages_from_state(state_data: dict[str, Any]) -> list[dict[str, str]]:
    questions = state_data.get("questions_asked") or []
    answers = state_data.get("answers_given") or []
    messages: list[dict[str, str]] = []
    for index, question in enumerate(questions):
        if question:
            messages.append({"role": "assistant", "content": str(question)})
        if index < len(answers) and answers[index]:
            messages.append({"role": "user", "content": str(answers[index])})
    return messages


def _state_to_dict(state: InterviewState | None) -> dict[str, Any]:
    if state is None:
        return {}
    return {
        "candidate_id": state.candidate_id,
        "candidate_cv": state.candidate_cv,
        "job_description": state.job_description,
        "planned_questions": state.planned_questions,
        "question_index": state.question_index,
        "questions_asked": state.questions_asked,
        "answers_given": state.answers_given,
        "per_question_scores": state.per_question_scores,
        "complete": state.complete,
    }


def _follow_up_question(bot: InterviewBot, before_index: int) -> str | None:
    state = bot.state
    if state is None:
        return None
    next_index = before_index + 1
    if next_index >= len(state.planned_questions):
        return None
    next_question = state.planned_questions[next_index]
    if next_question.get("type") == "follow_up":
        return next_question.get("question")
    return None


def _apply_interview_result(candidate: Candidate, report: dict[str, Any]) -> None:
    interview_score = float(report.get("interview_score", report.get("final_chat_score", 0)) or 0)
    if 0.0 <= interview_score <= 1.0:
        interview_score *= 100.0
    interview_score = round(_clamp(interview_score, 0.0, 100.0), 2)

    cv_filter_score = round(_clamp(float(candidate.weighted_score or 0), 0.0, 100.0), 2)

    candidate.interview_score = interview_score
    candidate.combined_final_score = round((0.4 * cv_filter_score) + (0.6 * interview_score), 2)
    recommendation = str(report.get("recommendation") or "").lower().strip()
    if recommendation not in {"advance", "hold", "reject"}:
        recommendation = _recommendation_from_interview_score(interview_score)
    candidate.recommendation = recommendation


def _recommendation_from_interview_score(interview_score: float) -> str:
    if interview_score >= 80:
        return "advance"
    if interview_score >= 60:
        return "hold"
    return "reject"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _get_job_or_404(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _get_candidate_or_404(db: Session, candidate_id: int) -> Candidate:
    candidate = db.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


def _get_interview_or_404(db: Session, session_id: str, candidate_id: int) -> InterviewSession:
    session = db.get(InterviewSession, session_id)
    if not session or session.candidate_id != candidate_id:
        raise HTTPException(status_code=404, detail="Interview session not found")
    return session


def _job_payload(job: Job) -> dict[str, Any]:
    return {
        "title": job.title,
        "description": job.description,
        "requirements": _json_loads(job.requirements_json, {}),
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
