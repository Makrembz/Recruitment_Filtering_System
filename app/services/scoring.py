import json
import re
from dataclasses import dataclass

from app.config import llm_client
from app.services.embeddings import cosine_similarity


@dataclass(frozen=True)
class ScoreResult:
    semantic_score: float
    llm_score: float
    final_score: float
    reasoning: str
    provider: str


def score_candidate(cv_text: str, job_description: str) -> ScoreResult:
    semantic = cosine_similarity(cv_text, job_description)
    llm_result = llm_client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior technical recruiter. Score CV fit against the job "
                    "description using evidence only from the provided text. Return compact JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return JSON with keys score, reasoning, strengths, gaps. "
                    "score must be a number from 0 to 100.\n\n"
                    f"JOB DESCRIPTION:\n{job_description[:6000]}\n\nCV:\n{cv_text[:9000]}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=700,
    )
    payload = _extract_json(llm_result.content)
    llm_score = float(payload.get("score", 0)) / 100.0
    llm_score = max(0.0, min(1.0, llm_score))
    final_score = round((semantic * 0.55) + (llm_score * 0.45), 4)
    reasoning = payload.get("reasoning") or llm_result.content
    return ScoreResult(
        semantic_score=semantic,
        llm_score=round(llm_score, 4),
        final_score=final_score,
        reasoning=reasoning,
        provider=llm_result.provider,
    )


def infer_candidate_name(cv_text: str, filename: str) -> str:
    for line in cv_text.splitlines()[:8]:
        cleaned = line.strip()
        if 2 <= len(cleaned.split()) <= 5 and not any(char.isdigit() for char in cleaned):
            return cleaned[:200]
    return filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"score": 0, "reasoning": text}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"score": 0, "reasoning": text}
