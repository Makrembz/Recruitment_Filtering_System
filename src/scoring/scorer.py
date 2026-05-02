import json
import re
import time
from functools import lru_cache
from typing import Any

import numpy as np
from openai import APIStatusError, RateLimitError
from sentence_transformers import SentenceTransformer


GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 60.0

LLM_SCORING_SYSTEM_PROMPT = """
You are a senior recruitment scoring engine.

Evaluate the candidate against the job description using only the supplied structured CV and job text.

Score exactly four dimensions, each from 0 to 25:
- technical_skills: match between required/preferred skills and candidate skills
- experience: relevance, seniority, domain fit, and duration of prior work
- education: required education, degrees, institutions, and role relevance
- potential: learning ability, adjacent experience, multilingual value, certifications, and growth indicators

Rules:
- total_score must equal the sum of the four dimension scores and must be between 0 and 100.
- Be strict for missing must-have requirements.
- Be fair to French, English, and Arabic CV content.
- Do not invent facts not supported by the CV.
- Return exactly three concise justification bullets.
- Respond ONLY in valid JSON with no preamble, markdown, comments, or trailing text.

Required JSON schema:
{
  "total_score": 0,
  "dimensions": {
    "technical_skills": 0,
    "experience": 0,
    "education": 0,
    "potential": 0
  },
  "justification_bullets": ["", "", ""]
}
""".strip()


def score_candidate(
    cv: dict[str, Any],
    job_description: str | dict[str, Any],
    client: Any,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = GROQ_MODEL,
) -> dict[str, Any]:
    """Score one structured CV with local semantic similarity plus LLM reasoning."""
    raw_text = str(cv.get("raw_text") or "")
    similarity_score = semantic_similarity_score(raw_text, _job_text(job_description))
    llm_payload = llm_reasoning_score(
        cv=cv,
        job_description=job_description,
        client=client,
        model=model,
    )
    llm_score = _clamp_score(llm_payload["total_score"])
    weighted_score = round((0.4 * similarity_score) + (0.6 * llm_score), 2)
    return {
        "candidate_id": cv.get("candidate_id") or cv.get("id"),
        "name": cv.get("full_name") or cv.get("name"),
        "similarity_score": similarity_score,
        "llm_score": llm_score,
        "weighted_score": weighted_score,
        "passed": weighted_score >= threshold,
        "justification": llm_payload["justification_bullets"],
    }


def batch_score(
    cvs: list[dict[str, Any]],
    job_description: str | dict[str, Any],
    client: Any,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = GROQ_MODEL,
) -> list[dict[str, Any]]:
    """Score all CVs and return candidates sorted by weighted score descending."""
    scored = [
        score_candidate(
            cv=cv,
            job_description=job_description,
            client=client,
            threshold=threshold,
            model=model,
        )
        for cv in cvs
    ]
    return sorted(scored, key=lambda item: item["weighted_score"], reverse=True)


def semantic_similarity_score(cv_raw_text: str, job_description: str) -> float:
    """Return cosine similarity between CV and job description normalized to 0-100."""
    if not cv_raw_text.strip() or not job_description.strip():
        return 0.0
    model = _embedding_model()
    vectors = model.encode(
        [cv_raw_text, job_description],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    cosine = float(np.dot(vectors[0], vectors[1]))
    normalized = max(0.0, min(1.0, cosine)) * 100
    return round(normalized, 2)


def llm_reasoning_score(
    cv: dict[str, Any],
    job_description: str | dict[str, Any],
    client: Any,
    model: str = GROQ_MODEL,
) -> dict[str, Any]:
    """Ask Groq/Llama for dimension-level recruitment reasoning."""
    messages = [
        {"role": "system", "content": LLM_SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "JOB DESCRIPTION:\n"
                f"{_json_or_text(job_description)}\n\n"
                "STRUCTURED CV:\n"
                f"{json.dumps(cv, ensure_ascii=False)}"
            ),
        },
    ]
    try:
        content = _chat_completion_with_rate_limit_retry(
            client=client,
            model=model,
            messages=messages,
            max_retries=3,
        )
        return _normalize_llm_payload(_load_json_object(content))
    except Exception:
        return fallback_llm_score(cv=cv, job_description=job_description)


def fallback_llm_score(
    cv: dict[str, Any],
    job_description: str | dict[str, Any],
) -> dict[str, Any]:
    """Deterministic fallback when the LLM is unavailable or returns invalid JSON."""
    job_text = _job_text(job_description).lower()
    skills = [str(skill).lower() for skill in cv.get("skills", [])]
    skill_hits = sum(1 for skill in skills if skill and skill in job_text)
    technical = min(25, skill_hits * 5)

    experiences = cv.get("experience") or []
    experience = min(25, len(experiences) * 6)
    if any((item.get("duration_months") or 0) >= 24 for item in experiences if isinstance(item, dict)):
        experience = min(25, experience + 5)

    education_entries = cv.get("education") or []
    education = 18 if education_entries else 5
    certifications = cv.get("certifications") or []
    potential = min(25, 8 + len(certifications) * 3 + len(cv.get("languages") or []) * 2)

    total = technical + experience + education + potential
    return _normalize_llm_payload(
        {
            "total_score": total,
            "dimensions": {
                "technical_skills": technical,
                "experience": experience,
                "education": education,
                "potential": potential,
            },
            "justification_bullets": [
                "Fallback score used because LLM scoring was unavailable.",
                f"Matched {skill_hits} listed skills against the job description.",
                "Score is based on structured skills, experience, education, languages, and certifications.",
            ],
        }
    )


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def _chat_completion_with_rate_limit_retry(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_retries: int,
) -> str:
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIStatusError) as exc:
            if not _is_rate_limit(exc) or attempt >= max_retries:
                raise
            time.sleep(10)
    raise RuntimeError("Groq scoring request failed after rate-limit retries")


def _is_rate_limit(exc: Exception) -> bool:
    return isinstance(exc, RateLimitError) or getattr(exc, "status_code", None) == 429


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


def _normalize_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    dimensions = payload.get("dimensions") if isinstance(payload.get("dimensions"), dict) else {}
    normalized_dimensions = {
        "technical_skills": _clamp_dimension(dimensions.get("technical_skills")),
        "experience": _clamp_dimension(dimensions.get("experience")),
        "education": _clamp_dimension(dimensions.get("education")),
        "potential": _clamp_dimension(dimensions.get("potential")),
    }
    dimension_sum = sum(normalized_dimensions.values())
    total_score = _clamp_score(payload.get("total_score"))
    if total_score == 0 and dimension_sum > 0:
        total_score = dimension_sum
    bullets = _string_list(payload.get("justification_bullets"))[:3]
    while len(bullets) < 3:
        bullets.append("No additional justification provided.")
    return {
        "total_score": _clamp_score(total_score),
        "dimensions": normalized_dimensions,
        "justification_bullets": bullets,
    }


def _clamp_dimension(value: Any) -> float:
    return round(max(0.0, min(25.0, _as_float(value))), 2)


def _clamp_score(value: Any) -> float:
    return round(max(0.0, min(100.0, _as_float(value))), 2)


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _job_text(job_description: str | dict[str, Any]) -> str:
    if isinstance(job_description, dict):
        return json.dumps(job_description, ensure_ascii=False)
    return str(job_description)


def _json_or_text(value: str | dict[str, Any]) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
