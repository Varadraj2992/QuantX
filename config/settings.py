from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings for local development and deployment."""

    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://quantx:quantx@localhost:5432/quantx",
        )
    )
    database_echo: bool = field(
        default_factory=lambda: os.getenv("DATABASE_ECHO", "false").lower() == "true"
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")).resolve()
    )
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    streamlit_host: str = field(default_factory=lambda: os.getenv("STREAMLIT_HOST", "0.0.0.0"))
    streamlit_port: int = field(default_factory=lambda: int(os.getenv("STREAMLIT_PORT", "8501")))
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "change-me-in-production"))


settings = Settings()
