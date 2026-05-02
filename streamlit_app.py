import requests
import streamlit as st


st.set_page_config(page_title="CV Shortlist Dashboard", layout="wide")

API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

st.title("CV Shortlist Dashboard")

with st.sidebar:
    st.header("Role")
    title = st.text_input("Job title", "Machine Learning Engineer")
    description = st.text_area("Job description", height=240)
    if st.button("Create job", type="primary", disabled=len(description) < 30):
        response = requests.post(
            f"{API_URL}/jobs",
            json={"title": title, "description": description},
            timeout=30,
        )
        response.raise_for_status()
        st.session_state["job"] = response.json()
        st.success(f"Created job #{st.session_state['job']['id']}")

job = st.session_state.get("job")
if not job:
    st.info("Create a job, then upload PDF or DOCX CVs to build the ranked shortlist.")
    st.stop()

st.subheader(f"Role #{job['id']}: {job['title']}")
uploads = st.file_uploader("Upload CVs", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Parse and score CVs", disabled=not uploads):
    for upload in uploads:
        with st.spinner(f"Scoring {upload.name}"):
            response = requests.post(
                f"{API_URL}/candidates/upload/{job['id']}",
                files={"file": (upload.name, upload.getvalue(), upload.type)},
                timeout=180,
            )
            response.raise_for_status()
    st.success("Scoring complete")

ranked_response = requests.get(f"{API_URL}/candidates/ranked/{job['id']}", timeout=30)
ranked_response.raise_for_status()
ranked = ranked_response.json()
final_response = requests.get(f"{API_URL}/jobs/{job['id']}/final-ranking", timeout=30)
final_response.raise_for_status()
final_ranked = final_response.json()

left, right = st.columns([2, 1])
with left:
    st.subheader("After CV Filter")
    for row in ranked:
        candidate = row["candidate"]
        with st.container(border=True):
            st.write(f"#{row['rank']} - {candidate['name']}")
            st.progress(candidate["final_score"])
            st.caption(
                f"Semantic {candidate['semantic_score']:.2f} | "
                f"LLM {candidate['llm_score']:.2f} | Final {candidate['final_score']:.2f}"
            )
            st.write(candidate["llm_reasoning"])

    st.subheader("After Interview")
    if final_ranked:
        st.dataframe(
            [
                {
                    "Name": row["name"],
                    "CV Score": row["cv_score"],
                    "Interview Score": row["interview_score"],
                    "Combined Score": row["combined_final_score"],
                    "Recommendation": row["recommendation"],
                }
                for row in final_ranked
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No candidates have completed the interview yet.")

with right:
    st.subheader("Interview")
    candidate_ids = {f"{r['candidate']['name']} (#{r['candidate']['id']})": r["candidate"]["id"] for r in ranked}
    selected = st.selectbox("Candidate", list(candidate_ids.keys())) if candidate_ids else None
    if selected and st.button("Start interview"):
        response = requests.post(
            f"{API_URL}/interviews/start/{candidate_ids[selected]}",
            timeout=90,
        )
        response.raise_for_status()
        st.session_state["interview"] = response.json()

    interview = st.session_state.get("interview")
    if interview:
        for message in interview["transcript"]:
            with st.chat_message("assistant" if message["role"] == "assistant" else "user"):
                st.write(message["content"])
        answer = st.chat_input("Candidate answer")
        if answer:
            response = requests.post(
                f"{API_URL}/interviews/{interview['id']}/message",
                json={"answer": answer},
                timeout=90,
            )
            response.raise_for_status()
            st.session_state["interview"] = response.json()
            st.rerun()
        if st.button("Close interview"):
            response = requests.post(
                f"{API_URL}/interviews/{interview['id']}/close",
                timeout=90,
            )
            response.raise_for_status()
            st.session_state["interview"] = response.json()
            st.rerun()
        if interview.get("summary"):
            st.markdown("**Summary**")
            st.write(interview["summary"])
