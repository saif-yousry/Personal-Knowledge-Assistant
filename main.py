"""
Personal Knowledge Assistant — a FastAPI application that ingests data from multiple platforms (Gmail, Slack, Discord, Telegram, PDFs), 
processes it through a RAG pipeline, 
and provides an agentic chat interface for semantic search over ingested content.
FastAPI application entry point. Configures the app, registers
routers, and runs database setup on startup via the lifespan handler.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from initializer import db
from services.gmail_sync_service import run_scheduled_sync
from routers import app_auth, base_route, chat, discord_auth, discord_ingest, email_agent, gmail_auth, gmail_ingest, pdf_ingest, slack_auth, slack_ingest, telegram_auth, telegram_ingest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMAIL_SYNC_INTERVAL = 60  # seconds


async def _sync_emails_periodically() -> None:
    """Background task: sync new emails for all linked Google accounts every minute."""
    while True:
        await asyncio.sleep(EMAIL_SYNC_INTERVAL)
        try:
            with db.session_context() as session:
                all_creds = db.find_all_google_credentials(session)
            for creds in all_creds:
                await asyncio.to_thread(
                    run_scheduled_sync,
                    creds.user_id,
                    {"access_token": creds.access_token, "refresh_token": creds.refresh_token, "token_expiry": creds.token_expiry},
                    gmail_ingest.ingest_status_by_user,
                )
        except Exception:
            logger.exception("Scheduled email sync failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """"STARTUP: runs once before the server accepts requests"""
    db.create_database()
    db.create_tables()
    sync_task = asyncio.create_task(_sync_emails_periodically())

    yield

    # SHUTDOWN: runs once after the server stops accepting requests
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)
app.include_router(base_route.base_router)
app.include_router(app_auth.router)
app.include_router(gmail_auth.router)
app.include_router(gmail_ingest.router)
app.include_router(pdf_ingest.router)
app.include_router(slack_auth.router)
app.include_router(slack_ingest.router)
app.include_router(discord_auth.router)
app.include_router(discord_ingest.router)
app.include_router(telegram_auth.router)
app.include_router(telegram_ingest.router)
app.include_router(chat.router)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
