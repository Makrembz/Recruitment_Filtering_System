from fastapi import FastAPI

from app.database import init_db
from app.routers import candidates, interviews, jobs

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

app = FastAPI(
    title="Automated CV Filtering API",
    description="Parse, score, interview, and rank candidates for recruiter review.",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(jobs.router)
app.include_router(candidates.router)
app.include_router(interviews.router)
