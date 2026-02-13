"""Configuration management for the application."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "sqlite:///./pds_stress_test.db"

    # API
    api_v1_prefix: str = "/api/v1"
    project_name: str = "PDS Stress Test Engine"
    version: str = "0.1.0"

    # Application
    env: str = "development"
    log_level: str = "INFO"
    
    # GenAI — Google Gemini (free tier: 15 RPM, 1M tokens/day)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.7
    gemini_max_tokens: int = 8192

    # CORS (if needed for future UI)
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
