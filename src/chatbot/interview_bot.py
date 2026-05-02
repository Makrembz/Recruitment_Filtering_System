import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.runnables import RunnableLambda
from openai import APIStatusError, RateLimitError

from app.config import groq_client

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory


GROQ_MODEL = "llama-3.3-70b-versatile"
Recommendation = Literal["advance", "reject", "hold"]

QUESTION_GENERATION_SYSTEM_PROMPT = """
You are a multilingual recruitment interview designer.

Generate 5 to 7 targeted interview questions from the candidate's structured CV and the job description.

Question mix:
- Exactly 2 technical questions based on skills listed in the CV.
- Exactly 2 behavioral questions based on gaps between the candidate's experience and the job requirements.
- 1 to 2 motivation or culture-fit questions.
- Add one extra question only if it is clearly useful for the role.

Language rule:
- Detect the main language of the CV.
- Write every question in the same language as the CV.
- Supported languages are French, English, and Arabic.

Quality rules:
- Ask concise, interview-ready questions.
- Each question must be specific to the CV and role.
- Do not ask generic questions that could apply to any candidate.
- Do not invent facts not present in the CV or job description.
- Respond ONLY in valid JSON with no preamble, markdown, comments, or trailing text.

Required JSON schema:
{
  "language": "English",
  "questions": [
    {
      "type": "technical",
      "question": ""
    }
  ]
}
""".strip()

ANSWER_EVALUATION_SYSTEM_PROMPT = """
You are a strict but fair recruitment interview evaluator.

Evaluate the candidate's latest answer against the current interview question, their structured CV, the job description, and the conversation history.

Return:
- answer_score: integer from 0 to 10
- decision: "follow_up" if the answer is vague, incomplete, contradictory, or reveals a risk that needs clarification
- decision: "next" if the answer is sufficient and the interview should continue to the next planned question
- follow_up_question: one concise follow-up question in the same language as the CV when decision is "follow_up"; otherwise null
- rationale: short recruiter-facing explanation

Rules:
- Be fair to French, English, and Arabic answers.
- Do not penalize grammar unless it blocks meaning.
- Do not invent evidence.
- Respond ONLY in valid JSON with no preamble, markdown, comments, or trailing text.

Required JSON schema:
{
  "answer_score": 0,
  "decision": "next",
  "follow_up_question": null,
  "rationale": ""
}
""".strip()

FINAL_EVALUATION_SYSTEM_PROMPT = """
You are a senior recruiter generating a final chatbot interview evaluation.

Use the structured CV, job description, all interview questions and answers, and per-question scores.

Return:
- interview_score: integer from 0 to 100
- final_chat_score: integer from 0 to 100
- overall_impression: concise recruiter-facing summary
- recommendation: one of "advance", "reject", or "hold"

Recommendation rules:
- 80-100: strong fit, clear confident answers, advance
- 60-79: good fit with some gaps, hold
- 0-59: weak fit, unclear answers, reject

Respond ONLY in valid JSON with no preamble, markdown, comments, or trailing text.

Required JSON schema:
{
    "interview_score": 0,
  "final_chat_score": 0,
  "overall_impression": "",
  "recommendation": "hold"
}
""".strip()


@dataclass
class InterviewState:
    candidate_id: str | int | None
    candidate_cv: dict[str, Any]
    job_description: str | dict[str, Any]
    planned_questions: list[dict[str, str]] = field(default_factory=list)
    question_index: int = 0
    questions_asked: list[str] = field(default_factory=list)
    answers_given: list[str] = field(default_factory=list)
    per_question_scores: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = False


class InterviewBot:
    """LangChain LCEL interview bot backed by Groq's OpenAI-compatible API."""

    def __init__(
        self,
        client: Any = groq_client,
        model: str = GROQ_MODEL,
        max_retries: int = 3,
    ) -> None:
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
        self.state: InterviewState | None = None

    def start_session(
        self,
        candidate_cv: dict[str, Any],
        job_description: str | dict[str, Any],
        candidate_id: str | int | None = None,
    ) -> dict[str, Any]:
        """Generate planned questions and return the first question."""
        questions = self._question_generation_chain().invoke(
            {"candidate_cv": candidate_cv, "job_description": job_description}
        )
        self.memory.clear()
        self.state = InterviewState(
            candidate_id=candidate_id or candidate_cv.get("candidate_id") or candidate_cv.get("id"),
            candidate_cv=candidate_cv,
            job_description=job_description,
            planned_questions=questions,
        )
        first_question = self._current_question()
        self.state.questions_asked.append(first_question)
        self.memory.chat_memory.add_ai_message(first_question)
        return self.session_snapshot(next_question=first_question)

    def answer(self, answer_text: str) -> dict[str, Any]:
        """Record an answer, score it, and return a follow-up, next question, or final report."""
        state = self._require_state()
        if state.complete:
            return self.final_evaluation()

        current_question = self._current_question()
        state.answers_given.append(answer_text)
        self.memory.chat_memory.add_user_message(answer_text)

        evaluation = self._answer_evaluation_chain().invoke(
            {
                "candidate_cv": state.candidate_cv,
                "job_description": state.job_description,
                "history": self._memory_history(),
                "question": current_question,
                "answer": answer_text,
            }
        )
        state.per_question_scores.append(
            {
                "question": current_question,
                "answer": answer_text,
                "score": evaluation["answer_score"],
                "rationale": evaluation["rationale"],
            }
        )

        if evaluation["decision"] == "follow_up" and evaluation.get("follow_up_question"):
            follow_up = evaluation["follow_up_question"]
            state.planned_questions.insert(
                state.question_index + 1,
                {"type": "follow_up", "question": follow_up},
            )

        state.question_index += 1
        if state.question_index >= len(state.planned_questions):
            state.complete = True
            return self.final_evaluation()

        next_question = self._current_question()
        state.questions_asked.append(next_question)
        self.memory.chat_memory.add_ai_message(next_question)
        return self.session_snapshot(next_question=next_question)

    def final_evaluation(self) -> dict[str, Any]:
        """Generate the final chatbot interview evaluation."""
        state = self._require_state()
        final = self._final_evaluation_chain().invoke(
            {
                "candidate_cv": state.candidate_cv,
                "job_description": state.job_description,
                "questions_answers": self.questions_answers,
                "per_question_scores": state.per_question_scores,
            }
        )
        state.complete = True
        return {
            "candidate_id": state.candidate_id,
            "questions_answers": self.questions_answers,
            "per_question_scores": state.per_question_scores,
            "interview_score": final["interview_score"],
            "final_chat_score": final["final_chat_score"],
            "overall_impression": final["overall_impression"],
            "recommendation": final["recommendation"],
        }

    def session_snapshot(self, next_question: str | None = None) -> dict[str, Any]:
        state = self._require_state()
        return {
            "candidate_id": state.candidate_id,
            "next_question": next_question,
            "questions_asked": state.questions_asked,
            "answers_given": state.answers_given,
            "per_question_scores": state.per_question_scores,
            "complete": state.complete,
        }

    @property
    def questions_answers(self) -> list[dict[str, str]]:
        state = self._require_state()
        pairs = []
        for index, answer in enumerate(state.answers_given):
            question = state.questions_asked[index] if index < len(state.questions_asked) else ""
            pairs.append({"question": question, "answer": answer})
        return pairs

    def _question_generation_chain(self):
        prompt = RunnableLambda(
            lambda inputs: [
                {"role": "system", "content": QUESTION_GENERATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "STRUCTURED CV:\n"
                        f"{json.dumps(inputs['candidate_cv'], ensure_ascii=False)}\n\n"
                        "JOB DESCRIPTION:\n"
                        f"{_json_or_text(inputs['job_description'])}"
                    ),
                },
            ]
        )
        model = RunnableLambda(
            lambda messages: self._chat_json(messages=messages, max_tokens=1200)
        )
        parser = RunnableLambda(lambda content: _normalize_questions(_load_json_object(content)))
        return prompt | model | parser

    def _answer_evaluation_chain(self):
        prompt = RunnableLambda(
            lambda inputs: [
                {"role": "system", "content": ANSWER_EVALUATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "STRUCTURED CV:\n"
                        f"{json.dumps(inputs['candidate_cv'], ensure_ascii=False)}\n\n"
                        "JOB DESCRIPTION:\n"
                        f"{_json_or_text(inputs['job_description'])}\n\n"
                        "CONVERSATION HISTORY:\n"
                        f"{inputs['history']}\n\n"
                        "CURRENT QUESTION:\n"
                        f"{inputs['question']}\n\n"
                        "LATEST ANSWER:\n"
                        f"{inputs['answer']}"
                    ),
                },
            ]
        )
        model = RunnableLambda(lambda messages: self._chat_json(messages=messages, max_tokens=700))
        parser = RunnableLambda(lambda content: _normalize_answer_eval(_load_json_object(content)))
        return prompt | model | parser

    def _final_evaluation_chain(self):
        prompt = RunnableLambda(
            lambda inputs: [
                {"role": "system", "content": FINAL_EVALUATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "STRUCTURED CV:\n"
                        f"{json.dumps(inputs['candidate_cv'], ensure_ascii=False)}\n\n"
                        "JOB DESCRIPTION:\n"
                        f"{_json_or_text(inputs['job_description'])}\n\n"
                        "QUESTIONS AND ANSWERS:\n"
                        f"{json.dumps(inputs['questions_answers'], ensure_ascii=False)}\n\n"
                        "PER QUESTION SCORES:\n"
                        f"{json.dumps(inputs['per_question_scores'], ensure_ascii=False)}"
                    ),
                },
            ]
        )
        model = RunnableLambda(lambda messages: self._chat_json(messages=messages, max_tokens=700))
        parser = RunnableLambda(lambda content: _normalize_final_eval(_load_json_object(content)))
        return prompt | model | parser

    def _chat_json(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or ""
            except (RateLimitError, APIStatusError) as exc:
                if not _is_rate_limit(exc) or attempt >= self.max_retries:
                    raise
                time.sleep(10)
        raise RuntimeError("Groq interview request failed after retries")

    def _current_question(self) -> str:
        state = self._require_state()
        return state.planned_questions[state.question_index]["question"]

    def _memory_history(self) -> str:
        return str(self.memory.load_memory_variables({}).get("history", ""))

    def _require_state(self) -> InterviewState:
        if self.state is None:
            raise RuntimeError("Interview session has not been started.")
        return self.state


def _normalize_questions(payload: dict[str, Any]) -> list[dict[str, str]]:
    questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
    normalized = []
    for item in questions:
        if isinstance(item, dict):
            question = str(item.get("question") or "").strip()
            question_type = str(item.get("type") or "general").strip()
        else:
            question = str(item).strip()
            question_type = "general"
        if question:
            normalized.append({"type": question_type, "question": question})
    if len(normalized) < 5:
        normalized.extend(_fallback_questions()[len(normalized) :])
    return normalized[:7]


def _normalize_answer_eval(payload: dict[str, Any]) -> dict[str, Any]:
    decision = str(payload.get("decision") or "next").strip().lower()
    if decision not in {"follow_up", "next"}:
        decision = "next"
    follow_up = payload.get("follow_up_question")
    return {
        "answer_score": _clamp_number(payload.get("answer_score"), 0, 10),
        "decision": decision,
        "follow_up_question": str(follow_up).strip() if follow_up else None,
        "rationale": str(payload.get("rationale") or "").strip(),
    }


def _normalize_final_eval(payload: dict[str, Any]) -> dict[str, Any]:
    interview_score = _clamp_number(
        payload.get("interview_score", payload.get("final_chat_score")),
        0,
        100,
    )
    recommendation = _recommendation_from_score(interview_score)
    return {
        "interview_score": interview_score,
        "final_chat_score": _clamp_number(payload.get("final_chat_score", interview_score), 0, 100),
        "overall_impression": str(payload.get("overall_impression") or "").strip(),
        "recommendation": recommendation,
    }


def _load_json_object(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def _clamp_number(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        number = minimum
    return max(minimum, min(maximum, number))


def _is_rate_limit(exc: Exception) -> bool:
    return isinstance(exc, RateLimitError) or getattr(exc, "status_code", None) == 429


def _json_or_text(value: str | dict[str, Any]) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _fallback_questions() -> list[dict[str, str]]:
    return [
        {"type": "technical", "question": "Which CV skill is most relevant to this role, and how have you applied it?"},
        {"type": "technical", "question": "Describe a technical problem from your CV that resembles this job's work."},
        {"type": "behavioral", "question": "Tell me about a time you had to close a gap in your experience quickly."},
        {"type": "behavioral", "question": "How do you handle unclear requirements or missing information at work?"},
        {"type": "motivation", "question": "Why are you interested in this role and organization?"},
    ]


def _recommendation_from_score(interview_score: int) -> Recommendation:
    if interview_score >= 80:
        return "advance"
    if interview_score >= 60:
        return "hold"
    return "reject"
