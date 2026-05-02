from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


if DATABASE_URL.startswith("sqlite:///"):
    db_path = DATABASE_URL.replace("sqlite:///", "", 1)
    if db_path and db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_candidate_interview_columns()


def _ensure_candidate_interview_columns() -> None:
    inspector = inspect(engine)
    if "candidates" not in inspector.get_table_names():
        return
    existing_columns = {column["name"] for column in inspector.get_columns("candidates")}
    column_sql = {
        "interview_score": "ALTER TABLE candidates ADD COLUMN interview_score FLOAT",
        "combined_final_score": "ALTER TABLE candidates ADD COLUMN combined_final_score FLOAT",
        "recommendation": "ALTER TABLE candidates ADD COLUMN recommendation VARCHAR(20)",
    }
    with engine.begin() as connection:
        for column_name, statement in column_sql.items():
            if column_name not in existing_columns:
                connection.execute(text(statement))
