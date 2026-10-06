"""
routers/auth.py

Registration and login endpoints using email/password.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from services.app_auth_service import (
    oauth2_scheme,
    create_access_token,
    hash_password,
    verify_password,
    revoke_token,
)
from initializer import db
from schemas.app_auth import LoginRequest, LoginResponse, RegisterRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Creates a new user account with email and password.",
)
def register(req: RegisterRequest, session: Session = Depends(db.get_session)):
    """Register a new user with email and password."""
    if db.user_exists(session, req.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Email already registered."
        )

    db.create_user(
        session,
        name=req.name,
        email=req.email,
        password_hash=hash_password(req.password)
    )
    return {"message": "User registered successfully."}


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login with email and password",
    description="Authenticates a user and returns a JWT access token.",
)
def login(req: LoginRequest, session: Session = Depends(db.get_session)):
    """Authenticate a user and return a JWT token."""
    user = db.find_user_by_email(session, req.email)
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")

    token = create_access_token(data={"sub": str(user.id)})
    return LoginResponse(access_token=token)


@router.post(
    "/logout",
    summary="Logout and revoke token",
    description="Revokes the current JWT access token.",
)
def logout(token: str = Depends(oauth2_scheme)):
    """Revoke the current access token."""
    revoke_token(token)
    return {"message": "Logged out successfully."}
