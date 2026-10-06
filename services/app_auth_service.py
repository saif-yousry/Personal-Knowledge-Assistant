"""
This module provides authentication services for the application, 
including password hashing, password verification, 
and JWT token creation and verification.
"""

import logging
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from sqlalchemy.orm import Session

from config import settings
from initializer import db

logger = logging.getLogger(__name__)

# Create a password hasher instance using Argon 2.
password_hasher = PasswordHash.recommended()

# Extracts the token from the Authorization header.
# Note: This is used as a dependency in FastAPI routes to get the token from the request.
# This enable the authorize button in the Swagger UI.
# The token is then passed to the verify_access_token function to extract the user_id. 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def hash_password(password: str) -> str:
    """
    Hash a password using pwdlib.
    Note that encryption is reversable, 
    so this is not suitable for storing passwords in a database, 
    and that is wht it is not used here.
    """
    return password_hasher.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against a hashed password using pwdlib."""
    return password_hasher.verify(password, hashed_password)


# ------------------------------------------------------------------
# Token functions
# ------------------------------------------------------------------


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token.
    
    Token structure:
    Header:
    {
        Type: "JWT", 
        Algorithm: "HS256"}

    Payload:
    {
        "sub": "123",
        "exp": 1720000000}

    Signature:
    HMACSHA256(
        Base64UrlEncode(Header) + "." + Base64UrlEncode(Payload),
        SecretKey
    )

    The token is signed using the secret key and algorithm specified in settings.
    The token contains the data passed in the `data` argument,
    and an expiration time set to `expires_delta` or the default expiration time.
    The data: dict contain a "sub" key with the user id as its value and a "role" key with the user's role as its value.
    """
    to_encode = data.copy() # create a copy of the data to encode
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    # Encode the token using the secret key and algorithm specified in settings.
    encoded_jwt = jwt.encode(to_encode,
                             settings.SECRET_KEY.get_secret_value(),
                             algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def verify_access_token(token: str) -> int:
    """Verify a JWT access token and return the subject (user id) if valid."""
    _cleanup_revoked()
    if token in _revoked_tokens:
        raise ValueError("Token has been revoked")
    try:
        payload = jwt.decode(token,
                             settings.SECRET_KEY.get_secret_value(),
                             algorithms=[settings.ALGORITHM],
                             options={"require": ["exp", "sub"]}
        )
        sub = payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")
    else:
        return int(sub)  # Return the user id (subject) from the token payload.


def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(db.get_session)):
    """Dependency that verifies the JWT and returns the authenticated user.

    Raises 401 if the token is invalid/expired/revoked.
    Raises 404 if the user no longer exists in the database.
    """
    try:
        user_id = verify_access_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc)) from exc

    user = db.find_user_by_id(session, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )
    return user


def revoke_token(token: str) -> None:
    """Add a token to the in-memory blocklist."""
    _cleanup_revoked()
    try:
        payload = jwt.decode(token,
                             settings.SECRET_KEY.get_secret_value(),
                             algorithms=[settings.ALGORITHM]
        )
        _revoked_tokens[token] = payload.get("exp", datetime.now(UTC).timestamp())
    except jwt.PyJWTError:
        logger.debug("Token revocation skipped — token is already invalid.")


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

# In-memory blocklist of revoked tokens (token_string -> expiry_timestamp).
# Expired entries are cleaned up on every revoke and verify call.
_revoked_tokens: dict[str, float] = {}


def _cleanup_revoked() -> None:
    """Remove expired entries from the revocation blocklist."""
    now = datetime.now(UTC).timestamp()
    expired_keys = [t for t, exp in _revoked_tokens.items() if exp <= now]
    for key in expired_keys:
        del _revoked_tokens[key]
