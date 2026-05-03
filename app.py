from typing import Any

import pandas as pd
import requests
import streamlit as st

from src.evaluation.explainer import (
    build_radar_chart,
    heuristic_candidate_breakdown,
)


st.set_page_config(
    page_title="Automated CV Filtering",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; max-width: 1280px; }
    h1, h2, h3 { letter-spacing: 0; }
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .status-pass { color: #047857; font-weight: 700; }
    .status-reject { color: #b91c1c; font-weight: 700; }
    .small-muted { color: #64748b; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


PAGES = [
    "Job Setup",
    "Upload & Filter CVs",
    "Chatbot Interview",
    "Recruiter Dashboard",
]


def init_state() -> None:
    defaults = {
        "api_url": "http://localhost:8000",
        "job_id": None,
        "job_title": "",
        "job_description": "",
        "required_skills": "",
        "candidates": [],
        "shortlist": [],
        "final_ranking": [],
        "interviews": {},
        "active_session_id": None,
        "active_candidate_id": None,
        "candidate_breakdowns": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def api_url(path: str) -> str:
    return f"{st.session_state.api_url.rstrip('/')}{path}"


def request_json(method: str, path: str, **kwargs: Any) -> Any:
    response = requests.request(method, api_url(path), timeout=240, **kwargs)
    response.raise_for_status()
    return response.json()


def refresh_shortlist() -> list[dict[str, Any]]:
    job_id = st.session_state.job_id
    if not job_id:
        return []
    shortlist = request_json("GET", f"/jobs/{job_id}/shortlist")
    st.session_state.shortlist = shortlist
    by_id = {item["candidate_id"]: item for item in shortlist}
    for candidate in st.session_state.candidates:
        enriched = by_id.get(candidate["candidate_id"])
        if enriched:
            candidate.update(
                {
                    "similarity_score": enriched.get("similarity_score"),
                    "llm_score": enriched.get("llm_score"),
                    "weighted_score": enriched.get("weighted_score"),
                    "llm_justification": enriched.get("llm_justification", []),
                }
            )
    return shortlist


def refresh_final_ranking() -> list[dict[str, Any]]:
    job_id = st.session_state.job_id
    if not job_id:
        return []
    final_ranking = request_json("GET", f"/jobs/{job_id}/final-ranking")
    st.session_state.final_ranking = final_ranking
    by_id = {item["candidate_id"]: item for item in final_ranking}
    for candidate in st.session_state.candidates:
        enriched = by_id.get(candidate["candidate_id"])
        if enriched:
            candidate.update(
                {
                    "interview_score": enriched.get("interview_score"),
                    "final_ranking_score": enriched.get("final_ranking_score"),
                    "status": enriched.get("status"),
                    "recommendation": enriched.get("recommendation"),
                }
            )
    return final_ranking


def upsert_candidates(items: list[dict[str, Any]]) -> None:
    existing = {candidate["candidate_id"]: candidate for candidate in st.session_state.candidates}
    for item in items:
        candidate_id = item["candidate_id"]
        existing[candidate_id] = {
            **existing.get(candidate_id, {}),
            "candidate_id": candidate_id,
            "name": item.get("name"),
            "score": item.get("score", 0),
            "similarity_score": item.get("similarity_score"),
            "llm_score": item.get("llm_score"),
            "weighted_score": item.get("weighted_score") or item.get("score", 0),
            "passed_filter": item.get("passed_filter", False),
            "llm_justification": item.get("llm_justification")
            or existing.get(candidate_id, {}).get("llm_justification", []),
        }
    st.session_state.candidates = list(existing.values())


def candidates_table(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        status = "\u2705 Passed" if candidate.get("passed_filter") else "\u274c Rejected"
        justification = candidate.get("llm_justification") or candidate.get("justification") or []
        rows.append(
            {
                "Name": candidate.get("name") or f"Candidate {candidate.get('candidate_id')}",
                "Score": round(float(candidate.get("weighted_score") or candidate.get("score") or 0), 2),
                "Status": status,
                "LLM Justification": " | ".join(justification) if justification else "Not available yet",
            }
        )
    return pd.DataFrame(rows)


def page_job_setup() -> None:
    st.title("Job Setup")
    st.caption("Create the role profile that CVs will be parsed and scored against.")
    # Load existing jobs from the backend so users can reuse stored jobs
    jobs: list[dict[str, Any]] = []
    try:
        jobs = request_json("GET", "/jobs")
    except requests.RequestException:
        jobs = []

    job_options = {f"#{j['job_id']} - {j['title']}": j for j in jobs}
    selected_job_label = st.selectbox("Load existing job (or leave blank to create new)", [""] + list(job_options.keys()))
    if selected_job_label:
        job = job_options[selected_job_label]
        if st.button("Use this job"):
            st.session_state.job_id = job.get("job_id")
            st.session_state.job_title = job.get("title") or ""
            st.session_state.job_description = job.get("description") or ""
            reqs = job.get("requirements") or {}
            if isinstance(reqs, dict) and reqs.get("required_skills"):
                st.session_state.required_skills = ", ".join(reqs.get("required_skills") or [])
            else:
                st.session_state.required_skills = str(reqs)
            st.success(f"Loaded job #{st.session_state.job_id}")

    with st.form("job_setup_form"):
        title = st.text_input("Job title", value=st.session_state.job_title)
        description = st.text_area(
            "Job description",
            value=st.session_state.job_description,
            height=220,
        )
        required_skills = st.text_area(
            "Required skills",
            value=st.session_state.required_skills,
            placeholder="Python, FastAPI, NLP, SQL, model evaluation",
            height=100,
        )
        submitted = st.form_submit_button("Create job", type="primary")

    if submitted:
        requirements = {
            "required_skills": [
                skill.strip() for skill in required_skills.replace("\n", ",").split(",") if skill.strip()
            ]
        }
        try:
            payload = {
                "title": title,
                "description": description,
                "requirements": requirements,
            }
            result = request_json("POST", "/jobs", json=payload)
            st.session_state.job_id = result["job_id"]
            st.session_state.job_title = title
            st.session_state.job_description = description
            st.session_state.required_skills = required_skills
            st.session_state.candidates = []
            st.session_state.shortlist = []
            st.session_state.final_ranking = []
            st.session_state.interviews = {}
            st.session_state.candidate_breakdowns = {}
            st.success(f"Job created. Current job ID: {st.session_state.job_id}")
        except requests.RequestException as exc:
            st.error(f"Could not create job: {exc}")

    if st.session_state.job_id:
        st.info(f"Active job: #{st.session_state.job_id} - {st.session_state.job_title}")


def page_upload_filter() -> None:
    st.title("Upload & Filter CVs")
    require_job()

    files = st.file_uploader(
        "Upload PDF or DOCX CVs",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    if files and st.button("Parse and score CVs", type="primary"):
        overall_progress = st.progress(0)
        status = st.empty()
        per_file_progress = {file.name: st.progress(0, text=file.name) for file in files}
        uploaded_results: list[dict[str, Any]] = []
        for index, file in enumerate(files, start=1):
            status.write(f"Processing {file.name}")
            per_file_progress[file.name].progress(0.25, text=f"{file.name} - uploading")
            try:
                result = request_json(
                    "POST",
                    f"/jobs/{st.session_state.job_id}/candidates/upload",
                    files=[("files", (file.name, file.getvalue(), file.type))],
                )
                uploaded_results.extend(result)
                per_file_progress[file.name].progress(1.0, text=f"{file.name} - complete")
            except requests.RequestException as exc:
                st.error(f"Failed to process {file.name}: {exc}")
                per_file_progress[file.name].progress(1.0, text=f"{file.name} - failed")
            overall_progress.progress(index / len(files))
        upsert_candidates(uploaded_results)
        try:
            refresh_shortlist()
        except requests.RequestException:
            pass
        status.write("Filtering complete")

    if st.session_state.candidates:
        sorted_candidates = sorted(
            st.session_state.candidates,
            key=lambda candidate: float(candidate.get("weighted_score") or candidate.get("score") or 0),
            reverse=True,
        )
        table = candidates_table(sorted_candidates)
        st.subheader("Filtering results")
        styled_table = table.style.map(
            lambda value: "color: #047857; font-weight: 700;"
            if value == "\u2705 Passed"
            else "color: #b91c1c; font-weight: 700;"
            if value == "\u274c Rejected"
            else "",
            subset=["Status"],
        )
        st.dataframe(styled_table, use_container_width=True, hide_index=True)

        chart_data = table[["Name", "Score"]].set_index("Name")
        st.subheader("Score distribution")
        st.bar_chart(chart_data)
    else:
        st.info("Upload CVs to see filtering results.")


def page_chatbot() -> None:
    st.title("Chatbot Interview")
    require_job()

    try:
        shortlist = refresh_shortlist()
    except requests.RequestException as exc:
        st.warning(f"Could not refresh shortlist: {exc}")
        shortlist = st.session_state.shortlist

    if not shortlist:
        st.info("No passed candidates in the shortlist yet.")
        return

    options = {
        f"{item['name']} - {item['weighted_score']:.2f}": item["candidate_id"]
        for item in shortlist
    }
    selected_label = st.selectbox("Select candidate", list(options.keys()))
    candidate_id = options[selected_label]
    interview_key = str(candidate_id)
    st.session_state.active_candidate_id = candidate_id

    interview = st.session_state.interviews.setdefault(
        interview_key,
        {"messages": [], "session_id": None, "complete": False, "report": None},
    )

    if not interview["session_id"]:
        if st.button("Start interview", type="primary"):
            try:
                result = request_json("POST", f"/candidates/{candidate_id}/interview/start")
                interview["session_id"] = result["session_id"]
                interview["messages"].append({"role": "assistant", "content": result["first_question"]})
                st.session_state.active_session_id = result["session_id"]
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Could not start interview: {exc}")
        return

    for message in interview["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if interview["complete"]:
        show_final_report(interview)
        return

    answer = st.chat_input("Candidate answer")
    if answer:
        interview["messages"].append({"role": "user", "content": answer})
        try:
            result = request_json(
                "POST",
                f"/candidates/{candidate_id}/interview/{interview['session_id']}/answer",
                json={"answer": answer},
            )
            next_question = result.get("follow_up") or result.get("next_question")
            if next_question:
                interview["messages"].append({"role": "assistant", "content": next_question})
            interview["complete"] = result["is_complete"]
            if interview["complete"]:
                report = request_json(
                    "GET",
                    f"/candidates/{candidate_id}/interview/{interview['session_id']}/report",
                )
                interview["report"] = report
                try:
                    refresh_final_ranking()
                except requests.RequestException:
                    pass
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not submit answer: {exc}")


def page_dashboard() -> None:
    st.title("Recruiter Dashboard")
    require_job()

    candidates = st.session_state.candidates
    total = len(candidates)
    passed_filter = sum(1 for candidate in candidates if candidate.get("passed_filter"))
    completed_reports = [
        interview.get("report")
        for interview in st.session_state.interviews.values()
        if interview.get("report")
    ]
    advanced = sum(1 for report in completed_reports if report.get("recommendation") == "advance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total CVs uploaded", total)
    col2.metric("% passed filter", f"{((passed_filter / total) * 100):.1f}%" if total else "0.0%")
    col3.metric(
        "% passed chatbot",
        f"{((advanced / total) * 100):.1f}%" if total else "0.0%",
    )

    try:
        refresh_shortlist()
    except requests.RequestException:
        pass
    try:
        refresh_final_ranking()
    except requests.RequestException:
        pass

    consolidated = consolidated_ranking_table(candidates, st.session_state.final_ranking)
    if not consolidated.empty:
        st.subheader("Candidate Rankings")
        st.dataframe(consolidated, use_container_width=True, hide_index=True)

        show_candidate_explanation_radar(candidates)

        csv_bytes = consolidated.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Candidate Rankings to CSV",
            data=csv_bytes,
            file_name="candidate_rankings.csv",
            mime="text/csv",
        )

        st.subheader("Candidate details")
        for candidate in candidates:
            interview = st.session_state.interviews.get(str(candidate["candidate_id"]), {})
            report = interview.get("report") or {}
            with st.expander(f"{candidate.get('name')} - {candidate.get('weighted_score') or candidate.get('score')}"):
                st.markdown("**LLM justification**")
                justification = candidate.get("llm_justification") or []
                st.write(justification or "Not available for this candidate in the current API response.")
                st.markdown("**Interview transcript**")
                messages = interview.get("messages") or []
                if messages:
                    for message in messages:
                        st.write(f"{message['role'].title()}: {message['content']}")
                else:
                    st.write("No interview started.")
                if report:
                    st.markdown("**Final report**")
                    st.json(report)

    else:
        st.info("No candidates uploaded yet.")
        show_candidate_explanation_radar(candidates)


def final_ranking_table(final_ranking: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rank, candidate in enumerate(final_ranking, start=1):
        interview_score = candidate.get("interview_score")
        final_score = candidate.get("final_ranking_score")
        rows.append(
            {
                "Rank": rank,
                "Name": candidate.get("name"),
                "CV Score": round(float(candidate.get("cv_filter_score") or 0), 2),
                "Status": str(candidate.get("status") or "not started").title(),
                "Interview Score": "—" if interview_score is None else round(float(interview_score), 2),
                "Final Ranking Score": "—" if final_score is None else round(float(final_score), 2),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    sortable = table["Final Ranking Score"].replace("—", float("-inf"))
    return table.assign(_sort=sortable).sort_values(by="_sort", ascending=False).drop(columns=["_sort"])


def consolidated_ranking_table(candidates: list[dict[str, Any]], final_ranking: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a single table containing embedding score, LLM CV score, CV combined score,
    interview score, final ranking score, status, rank, and name.
    Sort by final_ranking_score desc when available, otherwise by CV combined score."""
    rows = []
    by_final = {item["candidate_id"]: item for item in final_ranking or []}
    for candidate in candidates:
        fid = candidate.get("candidate_id")
        final = by_final.get(fid, {})
        emb = candidate.get("similarity_score")
        llm_cv = candidate.get("llm_score")
        cv_combined = candidate.get("weighted_score") or candidate.get("score") or 0.0
        interview = candidate.get("interview_score")
        final_score = candidate.get("final_ranking_score") or final.get("final_ranking_score")
        status = candidate.get("status") or final.get("status") or ("Not Started" if interview is None else (candidate.get("recommendation") or "hold"))
        rows.append(
            {
                "Name": candidate.get("name"),
                "Embedding Score": round(float(emb or 0.0), 2),
                "LLM CV Score": round(float(llm_cv or 0.0), 2),
                "CV Combined Score": round(float(cv_combined), 2),
                "Interview Score": "—" if interview is None else round(float(interview), 2),
                "Final Ranking Score": "—" if final_score is None else round(float(final_score), 2),
                "Status": str(status).title(),
            }
        )
    if not rows:
        return pd.DataFrame(rows)
    df = pd.DataFrame(rows)
    # Sort by Final Ranking Score when present, else by CV Combined Score
    df["_final_sort"] = df["Final Ranking Score"].apply(lambda v: float(v) if v != "—" else float("-inf"))
    df = df.sort_values(by=["_final_sort", "CV Combined Score"], ascending=[False, False]).drop(columns=["_final_sort"])
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def show_final_report(interview: dict[str, Any]) -> None:
    report = interview.get("report")
    if not report:
        st.info("Interview complete. Final report is not loaded yet.")
        return
    st.subheader("Final report")
    cols = st.columns(2)
    cols[0].metric("Chat score", report.get("final_chat_score", 0))
    cols[1].metric("Recommendation", str(report.get("recommendation", "hold")).title())
    st.write(report.get("overall_impression", ""))
    with st.expander("Questions and answers", expanded=True):
        st.json(report.get("questions_answers", []))


def require_job() -> None:
    if not st.session_state.job_id:
        st.warning("Create a job on the Job Setup page first.")
        st.stop()


def show_candidate_explanation_radar(candidates: list[dict[str, Any]]) -> None:
    st.subheader("Candidate explanation radar")
    if not candidates:
        st.info("Upload candidates to generate score breakdown charts.")
        return

    options = {
        f"{candidate.get('name')} - {candidate.get('weighted_score') or candidate.get('score')}": candidate
        for candidate in candidates
    }
    selected = st.selectbox("Explain candidate", list(options.keys()))
    candidate = options[selected]
    interview = st.session_state.interviews.get(str(candidate["candidate_id"]), {})
    candidate_with_report = {
        **candidate,
        "interview_report": interview.get("report") or {},
    }
    breakdown_key = str(candidate["candidate_id"])
    if st.button("Generate LLM breakdown", type="secondary"):
        try:
            st.session_state.candidate_breakdowns[breakdown_key] = request_json(
                "GET",
                f"/candidates/{candidate['candidate_id']}/explain",
            )
        except requests.RequestException as exc:
            st.warning(f"LLM breakdown unavailable, using local fallback: {exc}")
            st.session_state.candidate_breakdowns[breakdown_key] = heuristic_candidate_breakdown(
                candidate_with_report
            )
    breakdown = st.session_state.candidate_breakdowns.get(breakdown_key)
    if not breakdown:
        breakdown = heuristic_candidate_breakdown(candidate_with_report)
    st.plotly_chart(
        build_radar_chart(breakdown, title=f"Score Breakdown: {candidate.get('name')}"),
        use_container_width=True,
    )
    with st.expander("Breakdown explanation"):
        st.write(breakdown["explanation"])


def dashboard_table(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    ranked = sorted(
        candidates,
        key=lambda candidate: float(candidate.get("weighted_score") or candidate.get("score") or 0),
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, start=1):
        interview_score = candidate.get("interview_score")
        final_score = candidate.get("final_ranking_score")
        rows.append(
            {
                "Rank": rank,
                "Name": candidate.get("name"),
                "CV Score": round(float(candidate.get("weighted_score") or candidate.get("score") or 0), 2),
                "Status": str(candidate.get("status") or "not started").title(),
                "Interview Score": "—" if interview_score is None else round(float(interview_score), 2),
                "Final Ranking Score": "—" if final_score is None else round(float(final_score), 2),
            }
        )
    return pd.DataFrame(rows)


init_state()

with st.sidebar:
    st.title("CV Filtering")
    st.session_state.api_url = st.text_input("API URL", st.session_state.api_url)
    page = st.radio("Navigation", PAGES)
    st.divider()
    if st.session_state.job_id:
        st.caption(f"Active job #{st.session_state.job_id}")
    else:
        st.caption("No active job")


if page == "Job Setup":
    page_job_setup()
elif page == "Upload & Filter CVs":
    page_upload_filter()
elif page == "Chatbot Interview":
    page_chatbot()
else:
    page_dashboard()
