# Automated CV Filtering And Interview System

This project is a recruitment filtering demo that parses uploaded CVs, scores candidates against a job description, runs a chatbot interview, and produces recruiter-facing rankings with explainability and evaluation tools.

The system is designed for an academic or prototype presentation. It uses free/local components where possible and keeps the architecture simple enough to run on one machine.

## Main Features

- Upload and parse CVs in PDF or DOCX format.
- Extract structured candidate information from CV text.
- Score candidates using two approaches:
  - local semantic similarity with `sentence-transformers`
  - LLM reasoning with Groq/Llama 3.3 70B
- Rank candidates after the CV filtering stage.
- Run a dynamic chatbot interview for shortlisted candidates.
- Update final ranking after interview completion.
- Show recruiter dashboard metrics, final recommendations, bias audit, radar chart explanations, and CSV export.

## Tech Stack

- Backend: FastAPI
- Frontend demo: Streamlit
- Database: SQLite with SQLAlchemy
- LLM: Groq API using OpenAI-compatible client
- LLM model: `llama-3.3-70b-versatile`
- Fallback provider in config: OpenRouter
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- PDF parsing: `pdfplumber`
- DOCX parsing: `python-docx`
- Chatbot orchestration: LangChain LCEL
- Explainability charts: Plotly

## Architecture

```text
Streamlit app
    |
    v
FastAPI backend
    |
    +-- CV parser
    |     +-- PDF/DOCX text extraction
    |     +-- LLM structured JSON extraction
    |
    +-- Scoring engine
    |     +-- semantic similarity score
    |     +-- LLM reasoning score
    |     +-- weighted CV filter score
    |
    +-- Interview bot
    |     +-- targeted questions
    |     +-- answer scoring
    |     +-- final chatbot report
    |
    +-- Evaluation module
          +-- bias audit
          +-- radar chart breakdown
          +-- synthetic precision/recall/F1 demo
```

## Folder Structure

```text
app.py                         # Main 4-page Streamlit demo
streamlit_app.py               # Earlier compact Streamlit demo
requirements.txt
.env.example

app/
  config.py                    # Groq/OpenRouter clients
  main.py                      # Earlier FastAPI backend
  models.py
  database.py
  routers/
  services/

src/
  api/main.py                  # Main FastAPI backend used by app.py
  parsers/cv_parser.py         # CV and job-description parsing
  scoring/scorer.py            # Semantic + LLM scoring engine
  chatbot/interview_bot.py     # LangChain chatbot interview module
  evaluation/explainer.py      # Bias audit, radar charts, metrics

data/
  uploads/                     # Uploaded CV files
  recruitment_api.db           # SQLite DB created at runtime
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

```bash
copy .env.example .env
```

Add your Groq key:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

OpenRouter is optional. Groq is the primary provider.

## Run The Project

Start the FastAPI backend:

```bash
uvicorn src.api.main:app --reload
```

Start the Streamlit demo in a second terminal:

```bash
streamlit run app.py
```

Use this API URL in the Streamlit sidebar:

```text
http://localhost:8000
```

Useful URLs:

- Streamlit app: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Streamlit Pages

### 1. Job Setup

Create a job by entering:

- job title
- job description
- required skills

The app calls:

```text
POST /jobs
```

### 2. Upload & Filter CVs

Upload multiple PDF/DOCX CVs. The backend parses each CV, scores it, and returns the first ranking.

The app calls:

```text
POST /jobs/{job_id}/candidates/upload
```

### 3. Chatbot Interview

Select a shortlisted candidate and start the chatbot interview. The bot generates targeted questions from the candidate CV and the job description.

The app calls:

```text
POST /candidates/{candidate_id}/interview/start
POST /candidates/{candidate_id}/interview/{session_id}/answer
GET  /candidates/{candidate_id}/interview/{session_id}/report
```

### 4. Recruiter Dashboard

Shows:

- total CVs uploaded
- percentage passed after CV filter
- percentage advanced after chatbot interview
- consolidated candidate rankings (embedding, LLM CV score, CV combined score, interview score, final ranking score)
- candidate details with extracted CV data
- CSV export
- candidate explanation radar chart (explainability panel retained; bias audit and synthetic evaluation panels were removed from the demo to keep the UI focused)

The final ranking uses:

```text
GET /jobs/{job_id}/final-ranking
```

## How Scoring Works

### CV Filter Score

The CV filtering stage combines two signals.

Semantic similarity:

```text
CV raw text -> embedding vector
Job description -> embedding vector
cosine similarity -> semantic score
```

LLM reasoning:

The structured CV and job description are sent to Groq/Llama. The model evaluates the candidate on:

- technical skills
- experience
- education
- potential

The current scoring engine combines them as:

```text
weighted_score = 0.4 * semantic_score + 0.6 * llm_score
```

Scores in `src/scoring/scorer.py` are normalized to `0-100`. Some earlier app modules use `0-1`, so the final-ranking endpoint safely normalizes scores before combining them with the interview score.

### Interview Score

After the chatbot interview finishes, the final report includes:

```text
interview_score: 0-100
final_chat_score: 0-100
recommendation: advance | hold | reject
```

The backend now keeps `interview_score` on a 0-100 scale and computes the final ranking score as:

```text
final_ranking_score = 0.4 * cv_filter_score + 0.6 * interview_score
```

Both `cv_filter_score` and `interview_score` are 0-100, so `final_ranking_score` is also 0-100. The interview therefore has greater influence in the final combined ranking.

## Why Groq

Groq was chosen because it provides very fast inference for Llama models using LPU hardware. For this project, Groq gives a practical free-tier way to use `llama-3.3-70b-versatile` without running a large model locally.

The project uses the OpenAI-compatible client pattern:

```python
from openai import OpenAI

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
```

## API Summary

```text
GET  /health
POST /jobs
GET  /jobs
GET  /jobs/{job_id}/candidates
POST /jobs/{job_id}/candidates/upload
GET  /jobs/{job_id}/shortlist
GET  /jobs/{job_id}/final-ranking
POST /candidates/{candidate_id}/interview/start
POST /candidates/{candidate_id}/interview/{session_id}/answer
GET  /candidates/{candidate_id}/interview/{session_id}/report
GET  /candidates/{candidate_id}
GET  /candidates/{candidate_id}/explain
```

## Demo Job Description

You can use this for a presentation:

```text
Senior Python Developer, 3+ years FastAPI, experience with ML pipelines.
The role requires building production APIs, working with SQL databases,
integrating ML models, writing maintainable Python code, and collaborating
with data scientists to deploy reliable machine learning workflows.
```

Required skills:

```text
Python, FastAPI, SQL, ML pipelines, APIs, Docker
```

## Explainability And Ethics

The dashboard includes a bias audit and disclaimer. The bias audit is heuristic and should not be treated as proof of fairness or unfairness. It only flags possible anomalies for human review.

Important production concerns:

- recruiter oversight is required
- CV data contains personal information
- GDPR-style consent and deletion workflows are needed
- storage should be encrypted
- access should be authenticated
- decisions should be logged and auditable
- larger human-labeled datasets are needed for real accuracy claims

## Known Limitations

- SQLite is fine for a local demo, but production should use PostgreSQL.
- Uploaded files are stored locally.
- Long CVs are embedded as one text block in the simple semantic pipeline.
- Real accuracy requires a larger labeled dataset.
- LLM output can vary, so human review remains necessary.
- Existing local SQLite databases may need restart/startup migration before new columns appear.

## Recommended Demo Flow

1. Start FastAPI.
2. Start Streamlit.
3. Create a job.
4. Upload one strong CV, one partial CV, and one weak CV.
5. Show the CV filter ranking.
6. Start an interview for a shortlisted candidate.
7. Answer the chatbot questions.
8. Show the final interview report.
9. Open the recruiter dashboard.
10. Compare:
    - after CV filter ranking
    - after interview ranking
11. Show the bias audit and radar chart.

## Notes

The main app to present is:

```bash
streamlit run app.py
```

The main backend to present is:

```bash
uvicorn src.api.main:app --reload
```
