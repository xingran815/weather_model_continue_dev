"""Centralized configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from env vars and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_host: str = "weather_sql_container"
    mysql_port: int = 3306
    mysql_db: str = "weather_db"
    mlflow_tracking_uri: str = "http://localhost:8080"
    model_api_uri: str = "http://localhost:8000"


settings = Settings()
