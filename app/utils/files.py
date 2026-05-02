from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import UPLOAD_DIR


ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def safe_upload_path(file: UploadFile) -> Path:
    original = Path(file.filename or "upload").name
    extension = Path(original).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only PDF and DOCX CV uploads are supported.")
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir / f"{uuid4().hex}_{original}"
