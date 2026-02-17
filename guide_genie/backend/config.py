from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings

# backend/ lives one level inside the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    # ── CORS ──────────────────────────────────────────────────────
    # Override via .env:  CORS_ORIGINS=https://your-lovable-app.com
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
    ]

    # ── Storage ───────────────────────────────────────────────────
    UPLOAD_DIR: str = str(PROJECT_ROOT / "uploads")
    OUTPUT_DIR: str = str(PROJECT_ROOT / "outputs")

    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024          # 50 MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".md", ".mdx"]

    # ── LLM ───────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "case_sensitive": False,
        "extra": "ignore",   # silently ignore unknown .env vars (e.g. model, crewai_tracing_enabled)
    }


settings = Settings()