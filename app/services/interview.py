import json
import re

from langchain_core.runnables import RunnableLambda
from sqlalchemy.orm import Session

from app.config import llm_client
from app.models import Candidate, InterviewSession


def start_interview(db: Session, candidate: Candidate) -> InterviewSession:
    first_question = _question_chain().invoke(
        {
            "cv_text": candidate.cv_text,
            "job_description": candidate.job.description,
            "transcript": "[]",
        }
    )
    session = InterviewSession(
        candidate_id=candidate.id,
        transcript=json.dumps([{"role": "assistant", "content": first_question}]),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def answer_interview(db: Session, session: InterviewSession, answer: str) -> InterviewSession:
    transcript = json.loads(session.transcript)
    transcript.append({"role": "candidate", "content": answer})
    next_question = _question_chain().invoke(
        {
            "cv_text": session.candidate.cv_text,
            "job_description": session.candidate.job.description,
            "transcript": json.dumps(transcript),
        }
    )
    transcript.append({"role": "assistant", "content": next_question})
    session.transcript = json.dumps(transcript)
    db.commit()
    db.refresh(session)
    return session


def close_interview(db: Session, session: InterviewSession) -> InterviewSession:
    summary = _summary_chain().invoke(
        {
            "cv_text": session.candidate.cv_text,
            "job_description": session.candidate.job.description,
            "transcript": session.transcript,
        }
    )
    session.status = "completed"
    session.summary = summary
    _apply_legacy_interview_score(session)
    db.commit()
    db.refresh(session)
    return session


def serialize_session(session: InterviewSession) -> dict:
    return {
        "id": session.id,
        "candidate_id": session.candidate_id,
        "status": session.status,
        "transcript": json.loads(session.transcript),
        "summary": session.summary,
    }


def _apply_legacy_interview_score(session: InterviewSession) -> None:
    score_match = re.search(r"\b(\d{1,3})(?:/100|%)\b", session.summary)
    if not score_match:
        return
    final_chat_score = max(0, min(100, int(score_match.group(1))))
    interview_score = final_chat_score / 100
    candidate = session.candidate
    candidate.interview_score = interview_score
    candidate.combined_final_score = round((0.4 * candidate.final_score) + (0.6 * interview_score), 4)
    lowered = session.summary.lower()
    if "advance" in lowered:
        candidate.recommendation = "advance"
    elif "reject" in lowered:
        candidate.recommendation = "reject"
    else:
        candidate.recommendation = "hold"


def _question_chain():
    prompt = RunnableLambda(
        lambda inputs: [
            {
                "role": "system",
                "content": (
                    "You run concise screening interviews. Ask exactly one targeted question. "
                    "Adapt to the CV, job description, and prior transcript. Do not repeat yourself."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"JOB DESCRIPTION:\n{inputs['job_description'][:5000]}\n\n"
                    f"CV:\n{inputs['cv_text'][:7000]}\n\n"
                    f"TRANSCRIPT:\n{inputs['transcript'][:5000]}\n\n"
                    "Ask the next best interview question."
                ),
            },
        ]
    )
    model = RunnableLambda(
        lambda messages: llm_client.chat_completion(
            messages=messages, temperature=0.4, max_tokens=180
        ).content.strip()
    )
    return prompt | model


def _summary_chain():
    prompt = RunnableLambda(
        lambda inputs: [
            {
                "role": "system",
                "content": (
                    "Summarize the interview for a recruiter with a hiring recommendation. "
                    "Include exactly one line in this format: Final chat score: NN/100. "
                    "Also include exactly one recommendation word: advance, hold, or reject."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"JOB DESCRIPTION:\n{inputs['job_description'][:5000]}\n\n"
                    f"CV:\n{inputs['cv_text'][:7000]}\n\n"
                    f"TRANSCRIPT:\n{inputs['transcript'][:7000]}"
                ),
            },
        ]
    )
    model = RunnableLambda(
        lambda messages: llm_client.chat_completion(
            messages=messages, temperature=0.2, max_tokens=500
        ).content.strip()
    )
    return prompt | model
