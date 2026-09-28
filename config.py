"""
Application settings loaded from environment variables / `.env`.
This module is configuration-only: no business logic, no HTTP, no DB.
"""

import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    First try to Load settings from OS environment variables and if failed
    then Loads settings from .env file and if failed 
    then loads them from the classdefault values.
    Loading from .env file is for development purposes only and should not be used in production.
    In production, all settings should be loaded from environment variables .
    Environment variables is set by the deployment environment (e.g. Docker, systemd, or your cloud provider's secret manager).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application OAuth
    SECRET_KEY: SecretStr
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Google OAuth 2.0
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: SecretStr
    # Must be EXACTLY one of the "Authorized redirect URIs" registered for the
    # OAuth client in Google Cloud Console (e.g. /api/v1/auth/google/callback).
    GOOGLE_REDIRECT_URI: str
    # Reason: Added https://www.googleapis.com/auth/gmail.send to Google scopes so
    # the application can send outgoing emails via the Gmail API on behalf of authenticated users.
    GOOGLE_SCOPES: str = (
        "openid,email,profile,https://www.googleapis.com/auth/gmail.readonly,https://www.googleapis.com/auth/gmail.send"
    )

    # Reason: Email delivery settings (SMTP and fallback sender).
    # Necessary to enable sending emails via standard SMTP servers or automated fallback
    # when Google OAuth credentials are not used.
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: SecretStr = SecretStr("")
    SMTP_USE_TLS: bool = True
    DEFAULT_SENDER_EMAIL: str = "saifeldinyousry2005@gmail.com"


    # Groq / RAG
    GROQ_API_KEY: SecretStr = SecretStr("")
    GROQ_MODEL: str = "qwen/qwen3.6-27b"
    VECTOR_STORE_DIR: str

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: SecretStr
    POSTGRES_POOL_SIZE: int = 5

    # Discord OAuth 2.0
    DISCORD_CLIENT_ID: str = ""
    DISCORD_CLIENT_SECRET: SecretStr = SecretStr("")
    DISCORD_REDIRECT_URI: str = ""
    DISCORD_BOT_TOKEN: SecretStr = SecretStr("")
    DISCORD_BOT_SCOPES: str = "identify bot"
    DISCORD_BOT_PERMISSIONS: int = 8515702525261888

    # Slack OAuth 2.0
    SLACK_CLIENT_ID: str = ""
    SLACK_CLIENT_SECRET: SecretStr = SecretStr("")
    SLACK_REDIRECT_URI: str = ""
    SLACK_BOT_SCOPES: str = "channels:history,channels:read,groups:history,groups:read,im:history,im:read,mpim:history,mpim:read,users:read"
    SLACK_USER_SCOPES: str = ""
    SLACK_BOT_TOKEN: SecretStr = SecretStr("")

    # Telegram (Telethon MTProto)
    TELEGRAM_API_ID: int = 0
    TELEGRAM_API_HASH: SecretStr = SecretStr("")

    # HuggingFace
    HF_HUB_DISABLE_XET: str = "1"
    HF_HUB_OFFLINE: str = "1"

    @property
    def database_url(self) -> str:
        """SQLAlchemy connection string for PostgreSQL."""
        from urllib.parse import quote_plus
        password = quote_plus(self.POSTGRES_PASSWORD.get_secret_value())
        user = quote_plus(self.POSTGRES_USER)
        return (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def google_scopes(self) -> list[str]:
        """Parse comma-separated scopes into a list."""
        return [scope.strip() for scope in self.GOOGLE_SCOPES.split(",") if scope.strip()]

    @property
    def google_client_config(self) -> dict:
        """Dict in the shape google_auth_oauthlib expects for a Web client."""
        return {
            "web": {
                "client_id": self.GOOGLE_CLIENT_ID,
                "client_secret": self.GOOGLE_CLIENT_SECRET.get_secret_value(),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [self.GOOGLE_REDIRECT_URI],
            }
        }


settings = Settings()
# Set environment variables for HuggingFace Hub based on settings
os.environ.setdefault("HF_HUB_DISABLE_XET", settings.HF_HUB_DISABLE_XET)
os.environ.setdefault("HF_HUB_OFFLINE", settings.HF_HUB_OFFLINE)
