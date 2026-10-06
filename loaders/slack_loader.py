"""
slack_loader.py

Fetches messages from Slack channels via the Slack Web API using httpx.

Data flow:
    Slack API  --(SlackLoader)-->  List[Dict]   (raw message dicts)

This module only fetches and structures raw messages. It performs no
cleaning, chunking, embedding, storage, or retrieval — those belong to
later stages of the RAG pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class SlackLoader:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://slack.com/api"
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

    def list_user_channels(self) -> List[Dict[str, str]]:
        """Discovers all public and private channels the user has access to."""
        channels = []
        cursor = None

        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=15.0) as client:
            while True:
                params = {
                    "types": "public_channel,private_channel",
                    "limit": 100
                }
                if cursor:
                    params["cursor"] = cursor

                response = client.get("/conversations.list", params=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("ok"):
                    raise RuntimeError(f"Slack API Error (list_channels): {data.get('error')}")

                for ch in data.get("channels", []):
                    channels.append({
                        "id": ch["id"],
                        "name": ch.get("name", "unnamed")
                    })

                cursor = data.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

        return channels

    def fetch_channel_messages(self, channel_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches up to limit messages from a specific channel."""
        all_messages: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        remaining = limit

        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=15.0) as client:
            while remaining > 0:
                batch_size = min(remaining, 100)
                params = {
                    "channel": channel_id,
                    "limit": batch_size
                }
                if cursor:
                    params["cursor"] = cursor

                response = client.get("/conversations.history", params=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("ok"):
                    err = data.get("error")
                    if err == "not_in_channel":
                        logger.warning(
                            "Channel %s: Bot is NOT in channel. "
                            "Invite the bot by typing /invite @<BotName> in the channel.",
                            channel_id,
                        )
                    else:
                        logger.warning("Channel %s: %s", channel_id, err)
                    break

                batch = data.get("messages", [])
                if not batch:
                    break

                all_messages.extend(batch)
                remaining -= len(batch)

                cursor = data.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

        return all_messages

    def fetch_all_messages(self, limit_per_channel: int = 20) -> List[Dict[str, Any]]:
        """Discovers all channels automatically and pulls messages from each one."""
        channels = self.list_user_channels()
        all_workspace_messages = []

        for ch in channels:
            logger.info("Fetching channel: #%s (%s)", ch["name"], ch["id"])
            messages = self.fetch_channel_messages(channel_id=ch["id"], limit=limit_per_channel)
            all_workspace_messages.extend(messages)

        return all_workspace_messages


def get_slack_loader_for_team(team_id: Optional[str] = None) -> SlackLoader:
    """
    Instantiate a SlackLoader using the bot token stored in PostgreSQL (or env fallback).
    """
    from services.slack_auth_service import get_bot_token
    token = get_bot_token(team_id=team_id)
    return SlackLoader(access_token=token)
