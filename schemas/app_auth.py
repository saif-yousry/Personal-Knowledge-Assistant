"""
schemas/auth.py
Schemas for registration and login endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """Incoming registration request."""
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """Incoming login request."""
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
