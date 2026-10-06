"""
telegram_auth_service.py

Core service layer for Telegram user-account authentication via Telethon.

Telegram has no redirect-based OAuth2 flow. Instead, it uses a two-step
phone-number + code exchange (MTProto). Session strings are persisted
to PostgreSQL via the app's Database singleton.

Flow:
    1. send_code(phone_number)  -> triggers Telegram to send a login code
    2. verify_code(phone_number, code, password=None) -> completes login,
       saves session string to DB

Required env vars: TELEGRAM_API_ID, TELEGRAM_API_HASH
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    SessionPasswordNeededError,
)
from telethon.sessions import StringSession

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telegram app credentials (non-secret values exported for health checks)
# ---------------------------------------------------------------------------

_REQUIRED_VARS = {
    "TELEGRAM_API_ID": str(settings.TELEGRAM_API_ID),
    "TELEGRAM_API_HASH": bool(settings.TELEGRAM_API_HASH.get_secret_value()),
}
_missing = [k for k, v in _REQUIRED_VARS.items() if not v]
if _missing:
    logger.warning("Telegram auth disabled — missing env vars: %s", ", ".join(_missing))


def _get_db():
    from initializer import db
    return db


# ---------------------------------------------------------------------------
# Pending login state (in-memory, single-process only)
# ---------------------------------------------------------------------------

@dataclass
class _PendingLogin:
    client: TelegramClient
    phone_code_hash: str


_pending_clients: dict[str, _PendingLogin] = {}
_pending_lock = asyncio.Lock()


@dataclass
class AuthStartResult:
    phone_number: str


@dataclass
class AuthVerifyResult:
    phone_number: str
    telegram_user_id: int
    already_existed: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def send_code(phone_number: str) -> AuthStartResult:
    """Step 1: ask Telegram to send a login code to the given phone."""
    api_hash = settings.TELEGRAM_API_HASH.get_secret_value()
    if not settings.TELEGRAM_API_ID or not api_hash:
        raise RuntimeError("Telegram API credentials not configured.")

    client = TelegramClient(StringSession(), settings.TELEGRAM_API_ID, api_hash)
    await client.connect()
    try:
        sent = await client.send_code_request(phone_number)
    except FloodWaitError as e:
        await client.disconnect()
        raise RuntimeError(f"Rate limited by Telegram, retry in {e.seconds}s") from e
    except Exception:
        await client.disconnect()
        raise

    async with _pending_lock:
        _pending_clients[phone_number] = _PendingLogin(
            client=client, phone_code_hash=sent.phone_code_hash
        )

    return AuthStartResult(phone_number=phone_number)


async def verify_code(
    phone_number: str,
    code: str,
    password: Optional[str] = None,
    user_id: int | None = None,
) -> AuthVerifyResult:
    """Step 2: complete login with the code (and 2FA password if enabled)."""
    async with _pending_lock:
        pending = _pending_clients.get(phone_number)

    if pending is None:
        raise RuntimeError(
            "No pending login for this phone number. Call send_code first."
        )

    client = pending.client
    phone_code_hash = pending.phone_code_hash
    needs_password = False

    try:
        try:
            await client.sign_in(
                phone=phone_number, code=code, phone_code_hash=phone_code_hash
            )
        except SessionPasswordNeededError:
            if not password:
                needs_password = True
                raise ValueError("This account has 2FA enabled; 'password' is required.")
            await client.sign_in(password=password)
        except (PhoneCodeInvalidError, PhoneCodeExpiredError) as e:
            raise ValueError("Invalid or expired code.") from e

        me = await client.get_me()
        session_string = client.session.save()

        db = _get_db()
        if user_id is None:
            raise RuntimeError("user_id is required to save Telegram credentials.")
        with db.session_context() as session:
            existing = db.find_telegram_credentials(session, phone_number)
            db.save_telegram_credentials(
                session,
                user_id,
                phone_number,
                telegram_user_id=me.id,
                session_string=session_string,
            )

        return AuthVerifyResult(
            phone_number=phone_number,
            telegram_user_id=me.id,
            already_existed=existing is not None,
        )
    finally:
        if not needs_password:
            await client.disconnect()
            async with _pending_lock:
                _pending_clients.pop(phone_number, None)


def get_session_string(phone_number: str) -> str:
    """Retrieve a stored session string for a phone number, or raise."""
    db = _get_db()
    with db.session_context() as session:
        cred = db.find_telegram_credentials(session, phone_number)
        if cred is None:
            raise RuntimeError(
                f"No saved Telegram session for {phone_number}. "
                "Run the auth flow first."
            )
        return cred.session_string
