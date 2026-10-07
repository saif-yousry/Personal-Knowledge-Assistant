# Personal Knowledge Assistant

Personal Knowledge Assistant is a FastAPI application that collects content from Gmail, Slack, Discord, Telegram, and PDF files. It normalizes the content, creates embeddings, and stores searchable chunks in ChromaDB. An LLM-powered agent can retrieve those chunks during chat and can send email through the linked user's Gmail API account.

This document describes the behavior currently implemented in the repository. Gmail history sync polls linked accounts every 60 seconds after ingestion is initialized. Automatic replies can be enabled or disabled per linked Google account in the UI.

## Contents

- [Capabilities and current behavior](#capabilities-and-current-behavior)
- [Architecture and data flow](#architecture-and-data-flow)
- [Requirements](#requirements)
- [Installation and startup](#installation-and-startup)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Email workflows](#email-workflows)
- [Data storage and privacy](#data-storage-and-privacy)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## Capabilities and current behavior

- Ingests Gmail, Slack, Discord, Telegram, and PDF content when the corresponding ingestion endpoint is called.
- Cleans and chunks source content, creates local SentenceTransformers embeddings, and persists them in ChromaDB.
- Provides authenticated agent chat with semantic search over indexed content.
- Supports app accounts and platform authentication flows.
- Can process a supplied incoming email, have the agent compose and send a reply, and index both the incoming email and reply.
- Can monitor synchronized Gmail messages for manual replies and optionally send agent-generated replies.
- Serves the frontend from `static/` and interactive API documentation from FastAPI.

Current behavior and limitations:

- **Gmail sync is opt-in.** `/api/v1/ingest` initializes historical ingestion; after that, the application polls linked accounts every 60 seconds for backfill and Gmail history changes. Gmail push/Pub/Sub is not implemented.
- **Gmail auto-reply is off by default.** Use the Gmail UI toggle, or `GET`/`PUT /api/v1/email/auto-reply/settings`, to read or change the per-account setting. The setting is stored in PostgreSQL.
- **New auto-replied email is indexed before the reply is sent.** After checking whether the account owner already replied, the service embeds and stores the incoming message before invoking the agent's reply tool. If the owner has already replied, it indexes the exchange and sends no second reply. Historical backfill continues to index the selected label's old mail without replying.
- **Messages received while auto-reply is disabled stay pending.** Enabling the setting does not itself replay pending messages; a new Gmail thread event (such as an owner reply) is needed for sync to revisit that thread. New incoming messages received while the setting is enabled are handled automatically.
- The ChromaDB collection is named `knowledge` and is shared by the app. The current vector-store records do not include a user ownership filter, so deployments serving multiple users should not treat the vector store as tenant-isolated.
- Chat sessions, temporary ingestion statuses, per-user Groq API keys, and email send history are held in process memory. They are lost when the process restarts and are not shared between multiple server workers.
- Several ingestion endpoints use FastAPI `BackgroundTasks`; these are in-process tasks, not a durable job queue.

## Architecture and data flow

```text
Platform loader or uploaded PDF
        |
        v
CleanerDispatcher -> LangChainChunker -> SentenceTransformerEmbedder
                                                   |
                                                   v
                                            ChromaVectorStore
                                                   |
                         Agent search tool <-------+
                                |
                                v
                         Groq chat completion
```

The shared ingestion pipeline in `rag/pipeline.py` runs these stages:

1. **Load:** a loader converts source data into the Pydantic models in `models.py`; PDF uploads are extracted into document records.
2. **Clean:** `CleanerDispatcher` selects the source-specific cleaner. For email, this strips HTML where applicable, quoted/forwarded history, signatures, and excess whitespace. Other source cleaners normalize their own content.
3. **Chunk:** `LangChainChunker` splits the text into chunks. Its defaults are 1,000 characters with 200 characters of overlap. Each chunk retains source metadata and an index.
4. **Embed:** `SentenceTransformerEmbedder` uses `all-mpnet-base-v2` locally and selects CUDA when available, otherwise CPU. The model's embedding dimension is obtained from the model at runtime.
5. **Persist:** `ChromaVectorStore` stores chunk text, metadata, IDs, and embeddings in the `knowledge` collection.

The agent controller (`agent/controller.py`) calls the Groq client and executes model-requested tools. The `search` tool queries ChromaDB by semantic similarity and can filter by `source_type` (`email`, `slack`, `discord`, `telegram`, or `pdf`). The `send_email_reply` tool is exposed only while processing an incoming email and hands replies to `services/email_service.py`.

## Requirements

- Python 3.12 or newer is recommended.
- PostgreSQL and a PostgreSQL role allowed to connect to the `postgres` database and create the configured application database.
- A Groq API key to use agent chat or email automation.
- Google OAuth client credentials for application startup and Gmail integration.
- Credentials for any other platform you plan to connect.
- Enough disk space for the local embedding model and ChromaDB data. Inference can use CPU; a CUDA-capable GPU is optional.

## Installation and startup

From the project root, create and activate a virtual environment.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux or macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the repository root using the [configuration](#configuration) section below. Make sure PostgreSQL is running and accessible with those credentials.

The embedding model is initialized while application modules load. The default Hugging Face setting is offline mode, so download/cache the model before starting the server if it is not already available:

```bash
python download_model.py
```

Start the API:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 5000
```

On startup, the application creates the configured PostgreSQL database if it does not exist and creates its ORM tables. The web interface is served at <http://localhost:5000/>. Swagger UI is at <http://localhost:5000/docs>; the OpenAPI schema is at <http://localhost:5000/openapi.json>.

## Configuration

Settings are loaded from environment variables and then `.env` through `pydantic-settings`. The following values have no defaults in `config.py` and must be set for application startup:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Signs application JWT access tokens. Use a long random value. |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID. Required by settings even if Gmail is not used. |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret. |
| `GOOGLE_REDIRECT_URI` | Exact registered Google callback URI; locally, use `http://localhost:5000/auth/google/callback`. |
| `GROQ_API_KEY` | Groq API key used by the agent. |
| `VECTOR_STORE_DIR` | Filesystem directory for persistent ChromaDB data. |
| `POSTGRES_DB` | Application database name; created on startup if absent. |
| `POSTGRES_USER` | PostgreSQL username. |
| `POSTGRES_PASSWORD` | PostgreSQL password. |

Example `.env` for local development:

```env
# Required application and database settings
SECRET_KEY=replace-with-a-long-random-secret
VECTOR_STORE_DIR=./chroma_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=knowledge_assistant
POSTGRES_USER=postgres
POSTGRES_PASSWORD=replace-me

# Google OAuth; register this exact callback in Google Cloud Console
GOOGLE_CLIENT_ID=replace-me.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=replace-me
GOOGLE_REDIRECT_URI=http://localhost:5000/auth/google/callback

# Agent chat (required when using the agent unless a user key is set in the UI/API)
GROQ_API_KEY=replace-me
GROQ_MODEL=qwen/qwen3.6-27b

# Optional JWT settings
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Optional Slack integration
SLACK_CLIENT_ID=
SLACK_CLIENT_SECRET=
SLACK_REDIRECT_URI=http://localhost:5000/slack/oauth/callback
SLACK_BOT_TOKEN=

# Optional Discord integration
DISCORD_CLIENT_ID=
DISCORD_CLIENT_SECRET=
DISCORD_REDIRECT_URI=http://localhost:5000/discord/callback
DISCORD_BOT_TOKEN=

# Optional Telegram integration
TELEGRAM_API_ID=0
TELEGRAM_API_HASH=
```

Additional configuration notes:

- Gmail OAuth requests read and send access. Users who previously linked Google must reconnect and approve the updated permissions before Gmail API sending works.
- Email replies are sent only through the authenticated user's linked Gmail API account. There is no SMTP delivery or fallback. The sender is the linked Gmail address.
- `GROQ_API_KEY` is required at startup and is used by agent chat and email automation.
- Slack, Discord, and Telegram settings are optional unless using those integrations. Their callback URLs must exactly match the provider settings.
- `HF_HUB_OFFLINE` and `HF_HUB_DISABLE_XET` default to `1`. Use `python download_model.py` while network access is available to cache the embedding model before launching the application.
- Do not commit `.env`, OAuth credentials, passwords, or API keys.

## API reference

Most application endpoints require an application JWT. OAuth provider callbacks are called by the providers and do not use the app JWT in the same way as protected application routes. Check `/docs` for request schemas, optional query parameters, and response models.

### Application authentication

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create an application user. |
| `POST` | `/api/v1/auth/login` | Authenticate and receive a JWT access token. |
| `POST` | `/api/v1/auth/logout` | Revoke the presented access token. |

### Platform authentication

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/auth/google/login` | Start Google sign-in; requires app authentication. |
| `GET` | `/auth/google/callback` | Google OAuth callback. |
| `GET` | `/auth/google/status` | Check whether the current user linked Google. |
| `GET` | `/slack/install` | Start Slack OAuth installation; requires app authentication. |
| `GET` | `/slack/oauth/callback` | Slack OAuth callback. |
| `GET` | `/slack/health` | Check Slack OAuth configuration. |
| `DELETE` | `/slack/revoke/{team_id}` | Revoke a stored Slack workspace token. |
| `GET` | `/discord/authorize` | Start Discord authorization; requires app authentication. |
| `GET` | `/discord/callback` | Discord OAuth callback. |
| `GET` | `/discord/health` | Check Discord configuration. |
| `POST` | `/telegram/start` | Request a Telegram login code. |
| `POST` | `/telegram/verify` | Verify the code; requires app authentication. |
| `GET` | `/telegram/health` | Check Telegram API configuration. |
| `DELETE` | `/telegram/revoke/{phone_number}` | Delete the stored Telegram session. |

### Ingestion and source discovery

These endpoints start one-time ingestion jobs. The response generally indicates that a task started; query its status endpoint to see its current in-memory state.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/ingest` | Start historical Gmail backfill for a label (default `INBOX`), up to 60 messages per batch. Old mail is indexed without triggering replies. Requires a linked Google account. |
| `GET` | `/api/v1/ingest/status` | Check Gmail ingestion status for the current user. |
| `GET` | `/api/v1/email/auto-reply/settings` | Get whether Gmail auto-replies are enabled for the linked account. Requires app authentication and linked Google credentials. |
| `PUT` | `/api/v1/email/auto-reply/settings` | Set Gmail auto-replies with JSON body `{"enabled": true}` or `{"enabled": false}`. Disabled by default. |
| `POST` | `/api/v1/email/auto-reply/start` | Disabled shortcut: this route is intentionally blocked to enforce the required Backfill -> Checkpoint Save -> Forward Catch-Up Sync order. |
| `POST` | `/api/v1/ingest/pdf` | Upload PDF files for extraction and indexing. |
| `POST` | `/api/v1/ingest/slack` | Fetch and index Slack messages. |
| `GET` | `/api/v1/ingest/slack/status` | Check Slack ingestion status. |
| `GET` | `/api/v1/slack/channels` | List channels available to the connected Slack integration. |
| `POST` | `/api/v1/ingest/discord` | Fetch and index Discord messages for a guild. |
| `GET` | `/api/v1/ingest/discord/status` | Check Discord ingestion status. |
| `GET` | `/api/v1/discord/guilds` | List connected Discord guilds. |
| `GET` | `/api/v1/discord/channels` | List channels in a guild. |
| `POST` | `/api/v1/ingest/telegram` | Fetch and index messages from a Telegram chat or recent chats. |
| `GET` | `/api/v1/ingest/telegram/status` | Check Telegram ingestion status. |
| `GET` | `/api/v1/telegram/chats` | List recent Telegram chats. |

The platform ingestion routes accept source-specific query parameters such as channel, guild, chat, message limits, and `save_to_db`. See `/docs` for their defaults and constraints.

### Chat and agent settings

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/chat` | Send a message to the authenticated agent. Accepts `message` and optional `session_id`. |
| `POST` | `/api/v1/chat/reset?session_id=...` | Clear the in-memory history for a chat session. |
| `GET` | `/api/v1/settings/groq-api-key` | Get whether a Groq key is configured and its source; never returns the key. |
| `PUT` | `/api/v1/settings/groq-api-key` | Store a Groq key in memory for the authenticated user. |
| `DELETE` | `/api/v1/settings/groq-api-key` | Remove that in-memory user key. |

### Health and docs

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/` | Basic API response. |
| `GET` | `/docs` | Swagger UI. |
| `GET` | `/openapi.json` | OpenAPI schema. |

## Email workflows

### One-time Gmail indexing

1. Register and log in to obtain an application JWT.
2. Link Google using `/auth/google/login` and complete the OAuth callback.
3. Optionally enable auto-replies in the Gmail card. The setting is off by default and can also be managed through `GET` and `PUT /api/v1/email/auto-reply/settings`.
4. Call `POST /api/v1/ingest` with the JWT. The endpoint schedules the initial background fetch of up to 60 messages for the selected label (default `INBOX`).
5. Poll `GET /api/v1/ingest/status` for the in-process job state and stored chunk count.

Historical ingestion is started with `POST /api/v1/ingest` (or the **Start Ingestion** button). The first fetch is limited to 60 messages. If there are older messages, the background scheduler fetches further batches of up to 60, moving the cursor backward until the backfill completes. Historical messages are indexed without being sent to the autoresponder. Forward processing remains disabled until the full historical backfill completes. Gmail captures a `historyId` checkpoint before the initial scan, so messages received during backfill are caught up after it completes.

The system does not allow a bypass shortcut to start forward processing immediately. The required flow is strictly: historical backfill, save the checkpoint, then run the forward catch-up sync. Afterward, linked accounts are checked every 60 seconds. The auto-reply setting is independent: when enabled, the responder only acts after this sequence is complete.

### Process an incoming message and send a reply

1. Gmail history sync passes each newly added message to `GmailAutoResponder`. Incoming email automation is internal to Gmail sync; there is no public endpoint for submitting arbitrary incoming messages.
2. For Gmail messages, `GmailAutoResponder` fetches the current thread and determines whether a later message was sent by the account owner. If so, it does not send an automatic reply and indexes the incoming/reply exchange once.
3. If there is no manual reply, the service checks the per-account auto-reply setting. When enabled, it embeds and stores the incoming email first, then the agent searches email history, composes a grounded reply, and attempts to send it. When disabled, it records the message as pending without replying.
4. After a successful automatic reply, the service indexes the generated reply; the incoming message has already been stored before the send attempt. Gmail message state is persisted to avoid replying to messages already recorded as completed. Gmail history pages are read and deduplicated before the sync checkpoint advances.

The Gmail autoresponder applies to Gmail-history messages and sends only through the linked account's Gmail API credentials. Email sending requires an authenticated app user with linked Gmail credentials; SMTP is not supported. Users with an existing Google connection must reconnect after granting the Gmail send scope. Since the incoming email is indexed before the agent runs, a failed or skipped reply does not undo that indexing.

The email cleaner indexes the cleaned body. Chunk metadata includes fields such as `source_type`, `email_id`, sender, recipients, subject, date, and chunk index. Long messages may produce multiple Chroma records.

## Data storage and privacy

- **PostgreSQL:** application users, connected platform credentials, Gmail auto-reply preferences, and per-message Gmail reply outcomes. The database and ORM tables are created at application startup. Protect the database and backups as sensitive data.
- **ChromaDB:** persistent embeddings and chunk text in the configured `VECTOR_STORE_DIR`, collection `knowledge`. The default `./chroma_db` is relative to the process working directory if used as the configured value.
- **In-memory state:** chat histories, ingestion job status, temporary Groq keys, and the email delivery service's send history are process-local and do not survive restart.
- **Source metadata:** records are tagged by source type so the search tool can filter by platform. The current shared vector store does not enforce per-user isolation.

Do not expose the service with production credentials until authentication, authorization, vector-store tenant isolation, webhook verification, TLS, and secret management are configured for the deployment environment.

## Project layout

```text
agent/                 Agent loop, Groq client, prompts, tools, and state
database/              SQLAlchemy models and PostgreSQL operations
loaders/                Gmail, Slack, Discord, Telegram, and PDF readers
models.py               Pydantic source data models
rag/
  pipeline.py           Clean -> chunk -> embed -> store orchestration
  chunker.py            Text splitting and source metadata
  embedder.py           Local SentenceTransformers embeddings
  preprocessing/        Source-specific content cleaners
  vector_store/          ChromaDB persistence and similarity search
routers/                FastAPI routes for auth, ingestion, chat, and email
schemas/                API request and response models
services/               Authentication, platform integration, Gmail autoresponder, and email delivery
static/                 Browser-based frontend
config.py               Environment-backed settings
initializer.py           Shared database, model, and RAG components
main.py                  FastAPI application setup and router registration
requirements.txt         Python dependencies
download_model.py        Pre-download the configured embedding model
check.py                 Inspect the latest email in local Chroma data
```

## Troubleshooting

- **Settings validation error at startup:** check all required values listed in [Configuration](#configuration), including Google OAuth values even when not using Gmail.
- **PostgreSQL connection or permission error:** verify the server is running and the configured role can connect to the `postgres` database and create databases/tables.
- **Embedding model unavailable:** run `python download_model.py` with network access, then restart the API. The app initializes the model during startup/import.
- **Agent says the Groq key is missing:** set `GROQ_API_KEY` or configure a user key through the settings API. User-provided keys are lost when the server process restarts.
- **Gmail ingestion says the account is not linked:** complete Google OAuth for the logged-in application user and verify `/auth/google/status`.
- **Email reply fails:** verify the authenticated user has linked Google credentials and approved the Gmail send scope. SMTP delivery is not supported.
- **Email was sent but no chunks were stored:** inspect server logs for RAG ingestion errors and check that `VECTOR_STORE_DIR` is writable.
- **No reply happens for a newly received email:** verify Gmail history sync is initialized and the linked account has the Gmail send scope. Check server logs for agent or delivery errors.

## Technology stack

- **API:** FastAPI and Uvicorn
- **Relational storage:** PostgreSQL, SQLAlchemy, and psycopg2
- **Vector database:** ChromaDB
- **Embeddings:** SentenceTransformers (`all-mpnet-base-v2`)
- **LLM and tool calling:** Groq API
- **Text processing:** LangChain text splitters and source-specific cleaners
- **Authentication:** OAuth integrations, JWT, and Argon2 password hashing
- **Platform clients:** Google API client, HTTPX, and Telethon
