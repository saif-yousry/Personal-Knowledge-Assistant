"""Health check endpoint for the API."""

from fastapi import APIRouter

base_router = APIRouter(
    prefix="/api/v1",
    tags=["Base Router"]
)

@base_router.get("/")
def welcome():
    """Return a simple health check message."""
    return "The server is Working Right"
