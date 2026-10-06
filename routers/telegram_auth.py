"""
routers/telegram_auth.py

FastAPI router for Telegram user-account authentication via Telethon.

Endpoints
---------
POST /telegram/start     -- Send a login code to the phone number.
POST /telegram/verify    -- Submit the code to complete login.
GET  /telegram/health    -- Liveness check for Telegram auth config.
DELETE /telegram/revoke/{phone_number} -- Delete stored session.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from services.app_auth_service import get_current_user
from initializer import db
from config import settings
from services.telegram_auth_service import (
    send_code,
    verify_code,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["Telegram Auth"])


class StartAuthRequest(BaseModel):
    phone_number: str = Field(..., examples=["+15551234567"])


class VerifyAuthRequest(BaseModel):
    phone_number: str
    code: str
    password: Optional[str] = None


@router.post("/start", summary="Send Telegram login code")
async def telegram_start(payload: StartAuthRequest):
    """Step 1: request that Telegram send a login code to the phone number."""
    try:
        result = await send_code(payload.phone_number)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to send code: {e}")

    return {"status": "code_sent", "phone_number": result.phone_number}


@router.post("/verify", summary="Verify Telegram login code")
async def telegram_verify(payload: VerifyAuthRequest, user=Depends(get_current_user)):
    """Step 2: submit the received code (and 2FA password, if any) to finish login."""
    try:
        result = await verify_code(
            phone_number=payload.phone_number,
            code=payload.code,
            password=payload.password,
            user_id=user.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Login failed: {e}")

    return {
        "status": "authenticated",
        "telegram_user_id": result.telegram_user_id,
        "already_existed": result.already_existed,
    }


@router.get("/health", summary="Telegram auth health check")
def telegram_health():
    """Liveness / readiness check for the Telegram auth module."""
    issues: list[str] = []

    if not settings.TELEGRAM_API_ID:
        issues.append("TELEGRAM_API_ID is not set.")
    if not settings.TELEGRAM_API_HASH.get_secret_value():
        issues.append("TELEGRAM_API_HASH is not set.")

    health_status = "degraded" if issues else "ok"

    return JSONResponse(
        content={"status": health_status, "issues": issues},
        status_code=status.HTTP_200_OK if health_status == "ok" else status.HTTP_207_MULTI_STATUS,
    )


@router.delete("/revoke/{phone_number}", summary="Revoke stored Telegram session")
def telegram_revoke(phone_number: str, user=Depends(get_current_user), session: Session = Depends(db.get_session)):
    """Delete the stored Telegram session for a phone number."""
    deleted = db.delete_telegram_credentials(session, phone_number)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No session found for this phone number.")
    return {"status": "revoked", "phone_number": phone_number}
