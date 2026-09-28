"""
Personal Knowledge Assistant — a FastAPI application that ingests data from multiple platforms (Gmail, Slack, Discord, Telegram, PDFs), 
processes it through a RAG pipeline, 
and provides an agentic chat interface for semantic search over ingested content.
FastAPI application entry point. Configures the app, registers
routers, and runs database setup on startup via the lifespan handler.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from initializer import db

# Reason: Imported `email_agent` router to expose endpoints for processing incoming emails,
# running agent reasoning, sending responses, and storing resulting email conversations in the vector store.
from routers import app_auth, base_route, chat, discord_auth, discord_ingest, email_agent, gmail_auth, gmail_ingest, slack_auth, slack_ingest, telegram_auth, telegram_ingest

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """"STARTUP: runs once before the server accepts requests"""
    db.create_database()
    db.create_tables()

    yield

    # SHUTDOWN: runs once after the server stops accepting requests
    # (e.g., close connections, flush caches)

app = FastAPI(lifespan=lifespan)
app.include_router(base_route.base_router)
app.include_router(app_auth.router)
app.include_router(gmail_auth.router)
app.include_router(gmail_ingest.router)
app.include_router(slack_auth.router)
app.include_router(slack_ingest.router)
app.include_router(discord_auth.router)
app.include_router(discord_ingest.router)
app.include_router(telegram_auth.router)
app.include_router(telegram_ingest.router)
app.include_router(chat.router)
# Reason: Registered `email_agent.router` so HTTP clients and webhooks can invoke `/api/v1/email/incoming`.
app.include_router(email_agent.router)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
