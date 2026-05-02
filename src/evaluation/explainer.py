import json
import re
import statistics
import time
from collections import defaultdict
from typing import Any, Iterable, Literal, Sequence

import plotly.graph_objects as go
from openai import APIStatusError, RateLimitError

from src.scoring.scorer import semantic_similarity_score


GROQ_MODEL = "llama-3.3-70b-versatile"
DimensionName = Literal[
    "Technical Skills",
    "Experience",
    "Education",
    "Culture Fit",
    "Interview Score",
]

EXPLAINABILITY_DISCLAIMER = (
    "Bias audit results are heuristic screening indicators, not proof of bias or fairness. "
    "Name-origin grouping is approximate and must not be used for hiring decisions; it is only "
    "a prompt for human review, dataset improvement, and formal fairness testing."
)

DIMENSION_SCORING_SYSTEM_PROMPT = """
You are an explainability assistant for an academic recruitment filtering demo.

Score the candidate from 0 to 10 on these dimensions:
- Technical Skills
- Experience
- Education
- Culture Fit
- Interview Score

Use the structured CV, job description, candidate filtering score, LLM justification, and interview report if available.
If interview evidence is missing, estimate Interview Score conservatively from the available filter evidence.

Return ONLY valid JSON with no preamble, markdown, comments, or trailing text.

Required JSON schema:
{
  "dimensions": {
    "Technical Skills": 0,
    "Experience": 0,
    "Education": 0,
    "Culture Fit": 0,
    "Interview Score": 0
  },
  "explanation": ["", "", ""]
}
""".strip()

SYNTHETIC_TEST_SET = [
    {
        "cv_text": "Senior NLP engineer with Python, FastAPI, SQL, transformers, evaluation, and 5 years building production ML APIs.",
        "job_description": "We need a Python NLP engineer with FastAPI, SQL, transformer models, evaluation, and 3+ years experience.",
        "human_label": "hire",
        "interview_score": 86,
    },
    {
        "cv_text": "Machine learning engineer, 4 years, Python, scikit-learn, sentence-transformers, REST APIs, Docker, PostgreSQL.",
        "job_description": "Hiring ML engineer for CV filtering: Python, embeddings, API design, SQL, Docker, and model evaluation.",
        "human_label": "hire",
        "interview_score": 81,
    },
    {
        "cv_text": "Data scientist with 6 years in NLP classification, information extraction, FastAPI services, and recruiter analytics.",
        "job_description": "NLP recruitment product requiring information extraction, semantic search, FastAPI, analytics, and production ownership.",
        "human_label": "hire",
        "interview_score": 88,
    },
    {
        "cv_text": "Backend Python developer with 5 years in APIs, SQLAlchemy, SQLite, async processing, and LLM integrations.",
        "job_description": "Build a Python FastAPI backend with SQLAlchemy, SQLite, LLM scoring, CV parsing, and async upload workflows.",
        "human_label": "hire",
        "interview_score": 79,
    },
    {
        "cv_text": "AI engineer fluent in French and Arabic, experienced in document parsing, OCR QA, embeddings, and human evaluation.",
        "job_description": "Recruitment AI system needs multilingual CV parsing, embeddings, score explanations, and evaluation metrics.",
        "human_label": "hire",
        "interview_score": 84,
    },
    {
        "cv_text": "Graphic designer with portfolio branding, illustration, typography, and social media campaign assets.",
        "job_description": "We need a Python NLP engineer with FastAPI, SQL, transformer models, evaluation, and 3+ years experience.",
        "human_label": "reject",
        "interview_score": 31,
    },
    {
        "cv_text": "Junior sales associate with retail experience, customer support, CRM updates, and inventory coordination.",
        "job_description": "Hiring ML engineer for CV filtering: Python, embeddings, API design, SQL, Docker, and model evaluation.",
        "human_label": "reject",
        "interview_score": 28,
    },
    {
        "cv_text": "Accountant with payroll, reconciliation, tax filing, Excel reporting, and vendor invoice tracking.",
        "job_description": "NLP recruitment product requiring information extraction, semantic search, FastAPI, analytics, and production ownership.",
        "human_label": "reject",
        "interview_score": 24,
    },
    {
        "cv_text": "Recent biology graduate with lab reports, microscopy, field research, and academic poster presentations.",
        "job_description": "Build a Python FastAPI backend with SQLAlchemy, SQLite, LLM scoring, CV parsing, and async upload workflows.",
        "human_label": "reject",
        "interview_score": 35,
    },
    {
        "cv_text": "Hospitality coordinator with event scheduling, guest relations, booking tools, and team shift planning.",
        "job_description": "Recruitment AI system needs multilingual CV parsing, embeddings, score explanations, and evaluation metrics.",
        "human_label": "reject",
        "interview_score": 26,
    },
]


def run_bias_audit(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Flag score anomalies by approximate name-origin buckets and education levels."""
    scored = [candidate for candidate in candidates if _score(candidate) is not None]
    origin_groups: dict[str, list[float]] = defaultdict(list)
    education_groups: dict[str, list[float]] = defaultdict(list)

    for candidate in scored:
        score = _score(candidate)
        if score is None:
            continue
        origin_groups[classify_name_origin(candidate.get("name") or "")].append(score)
        education_groups[infer_education_level(candidate)].append(score)

    warnings = []
    origin_stats = _group_stats(origin_groups)
    education_stats = _group_stats(education_groups)

    origin_warning = _mean_gap_warning(origin_stats, "name-origin bucket", min_gap=15)
    if origin_warning:
        warnings.append(origin_warning)
    education_warning = _mean_gap_warning(education_stats, "education level", min_gap=20)
    if education_warning:
        warnings.append(education_warning)

    if education_stats and max(item["variance"] for item in education_stats.values()) > 250:
        warnings.append("High score variance detected inside at least one education-level group.")

    return {
        "disclaimer": EXPLAINABILITY_DISCLAIMER,
        "sample_size": len(scored),
        "warnings": warnings,
        "name_origin_stats": origin_stats,
        "education_level_stats": education_stats,
    }


def classify_name_origin(name: str) -> str:
    """Very small heuristic bucket for audit prompting, not identity inference."""
    first = (name or "").strip().split(" ")[0].lower()
    if not first:
        return "unknown"
    arabic_names = {
        "ahmed",
        "mohamed",
        "mohammed",
        "amina",
        "fatma",
        "youssef",
        "omar",
        "karim",
        "nour",
        "sarra",
    }
    french_names = {
        "jean",
        "pierre",
        "camille",
        "julien",
        "marie",
        "luc",
        "claire",
        "antoine",
        "nicolas",
    }
    english_names = {
        "john",
        "jane",
        "michael",
        "sarah",
        "david",
        "emily",
        "james",
        "anna",
        "robert",
    }
    if first in arabic_names or re.search(r"[\u0600-\u06ff]", name):
        return "arabic-associated"
    if first in french_names:
        return "french-associated"
    if first in english_names:
        return "english-associated"
    return "unknown"


def infer_education_level(candidate: dict[str, Any]) -> str:
    text = json.dumps(candidate, ensure_ascii=False).lower()
    if any(token in text for token in ["phd", "doctorate", "doctorat", "دكتوراه"]):
        return "doctorate"
    if any(token in text for token in ["master", "msc", "maîtrise", "maitrise", "ماجستير"]):
        return "master"
    if any(token in text for token in ["bachelor", "licence", "bs", "ba", "بكالوريوس"]):
        return "bachelor"
    if any(token in text for token in ["engineer", "ingénieur", "ingenieur"]):
        return "engineering"
    return "unknown"


def generate_candidate_breakdown(
    candidate: dict[str, Any],
    job_description: str | dict[str, Any],
    client: Any | None = None,
    model: str = GROQ_MODEL,
) -> dict[str, Any]:
    """Return 0-10 dimension scores and explanation for radar visualization."""
    if client is None:
        return heuristic_candidate_breakdown(candidate)
    messages = [
        {"role": "system", "content": DIMENSION_SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "CANDIDATE:\n"
                f"{json.dumps(candidate, ensure_ascii=False)}\n\n"
                "JOB DESCRIPTION:\n"
                f"{_json_or_text(job_description)}"
            ),
        },
    ]
    try:
        content = _chat_completion_with_retry(client, model, messages)
        return _normalize_breakdown(_load_json_object(content))
    except Exception:
        return heuristic_candidate_breakdown(candidate)


def heuristic_candidate_breakdown(candidate: dict[str, Any]) -> dict[str, Any]:
    weighted = float(candidate.get("weighted_score") or candidate.get("score") or 0)
    llm = float(candidate.get("llm_score") or weighted)
    similarity = float(candidate.get("similarity_score") or weighted)
    report = candidate.get("interview_report") or {}
    interview_score = float(report.get("final_chat_score") or 0)
    dimensions = {
        "Technical Skills": _clamp_10(similarity / 10),
        "Experience": _clamp_10(llm / 10),
        "Education": _clamp_10((llm * 0.8 + weighted * 0.2) / 10),
        "Culture Fit": _clamp_10((weighted * 0.7 + (interview_score or weighted) * 0.3) / 10),
        "Interview Score": _clamp_10((interview_score or weighted) / 10),
    }
    return {
        "dimensions": dimensions,
        "explanation": [
            "Heuristic breakdown used because LLM dimension scoring is unavailable in the UI context.",
            "Technical and experience values are derived from similarity and LLM scores.",
            "Interview score uses the final chatbot score when available.",
        ],
    }


def build_radar_chart(
    breakdown: dict[str, Any],
    title: str = "Candidate Score Breakdown",
) -> go.Figure:
    dimensions = breakdown.get("dimensions", {})
    labels = [
        "Technical Skills",
        "Experience",
        "Education",
        "Culture Fit",
        "Interview Score",
    ]
    values = [_clamp_10(dimensions.get(label, 0)) for label in labels]
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]
    figure = go.Figure(
        data=[
            go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill="toself",
                line_color="#2563eb",
                fillcolor="rgba(37, 99, 235, 0.24)",
                name="Score",
            )
        ]
    )
    figure.update_layout(
        title=title,
        polar={"radialaxis": {"visible": True, "range": [0, 10]}},
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        height=420,
    )
    return figure


def evaluate_filtering_system(
    labeled_test_set: Iterable[dict[str, Any] | tuple[str, str, str]],
    threshold: float = 60.0,
) -> dict[str, Any]:
    """Evaluate filtering stage against human hire/reject labels."""
    y_true = []
    y_pred = []
    hire_interview_scores = []
    reject_interview_scores = []

    for item in labeled_test_set:
        if isinstance(item, dict):
            cv_text = str(item["cv_text"])
            job_description = str(item["job_description"])
            human_label = str(item["human_label"])
            interview_score = item.get("interview_score")
        else:
            cv_text, job_description, human_label = item
            interview_score = None

        score = semantic_similarity_score(cv_text, job_description)
        prediction = "hire" if score >= threshold else "reject"
        y_true.append(human_label)
        y_pred.append(prediction)

        if interview_score is not None:
            if human_label == "hire":
                hire_interview_scores.append(float(interview_score))
            else:
                reject_interview_scores.append(float(interview_score))

    tp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == "hire" and pred == "hire")
    fp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == "reject" and pred == "hire")
    fn = sum(1 for actual, pred in zip(y_true, y_pred) if actual == "hire" and pred == "reject")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "mean_interview_score_hire": _safe_mean(hire_interview_scores),
        "mean_interview_score_reject": _safe_mean(reject_interview_scores),
        "threshold": threshold,
        "sample_size": len(y_true),
    }


def _group_stats(groups: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    stats = {}
    for group, scores in groups.items():
        stats[group] = {
            "count": len(scores),
            "mean": round(statistics.mean(scores), 2) if scores else 0.0,
            "variance": round(statistics.pvariance(scores), 2) if len(scores) > 1 else 0.0,
        }
    return stats


def _mean_gap_warning(
    stats: dict[str, dict[str, float | int]],
    label: str,
    min_gap: float,
) -> str | None:
    eligible = {key: value for key, value in stats.items() if value["count"] >= 2}
    if len(eligible) < 2:
        return None
    means = [float(value["mean"]) for value in eligible.values()]
    gap = max(means) - min(means)
    if gap >= min_gap:
        return f"Mean score gap of {gap:.1f} points detected across {label}s."
    return None


def _score(candidate: dict[str, Any]) -> float | None:
    value = candidate.get("weighted_score", candidate.get("score"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _chat_completion_with_retry(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIStatusError) as exc:
            if not _is_rate_limit(exc) or attempt >= max_retries:
                raise
            time.sleep(10)
    raise RuntimeError("Explanation request failed after retries")


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


def _normalize_breakdown(payload: dict[str, Any]) -> dict[str, Any]:
    dimensions = payload.get("dimensions") if isinstance(payload.get("dimensions"), dict) else {}
    labels = [
        "Technical Skills",
        "Experience",
        "Education",
        "Culture Fit",
        "Interview Score",
    ]
    return {
        "dimensions": {label: _clamp_10(dimensions.get(label, 0)) for label in labels},
        "explanation": _string_list(payload.get("explanation"))[:3],
    }


def _is_rate_limit(exc: Exception) -> bool:
    return isinstance(exc, RateLimitError) or getattr(exc, "status_code", None) == 429


def _json_or_text(value: str | dict[str, Any]) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _clamp_10(value: Any) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        number = 0.0
    return round(max(0.0, min(10.0, number)), 2)


def _safe_mean(values: Sequence[float]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]
