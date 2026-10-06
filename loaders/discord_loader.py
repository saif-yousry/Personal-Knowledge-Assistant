"""
Discord message loader using bot token and Discord REST API.

Adapted from ai-agent-message-integrations. Uses the app's Database
singleton (via initializer) instead of receiving a SQLAlchemy session.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from services.discord_auth_service import (
    DISCORD_API_BASE_URL,
    DiscordIntegrationError,
    get_bot_token,
)

logger = logging.getLogger(__name__)

MAX_RATE_LIMIT_RETRY_AFTER_SECONDS = 10.0
GUILD_TEXT_CHANNEL_TYPE = 0
GUILD_ANNOUNCEMENT_CHANNEL_TYPE = 5
SUPPORTED_TEXT_CHANNEL_TYPES = frozenset(
    {GUILD_TEXT_CHANNEL_TYPE, GUILD_ANNOUNCEMENT_CHANNEL_TYPE}
)


class AttachmentMetadata(BaseModel):
    id: str
    filename: str
    url: str
    content_type: str | None
    size: int | None


class MentionMetadata(BaseModel):
    user_id: str
    display_name: str


class StickerMetadata(BaseModel):
    sticker_id: str
    name: str
    format_type: int | None = None


class NormalizedMessage(BaseModel):
    platform: Literal["discord"]
    message_id: str
    guild_id: str
    channel_id: str
    channel_name: str | None = None
    author_id: str
    author_name: str | None
    content: str
    timestamp: datetime
    attachments: list[AttachmentMetadata]
    reply_to_message_id: str | None
    mentions: list[MentionMetadata] = Field(default_factory=list)
    stickers: list[StickerMetadata] = Field(default_factory=list)


class ConnectedGuild(BaseModel):
    guild_id: str
    guild_name: str


class DiscoveredChannel(BaseModel):
    channel_id: str
    channel_name: str
    channel_type: int


class GuildMessageLoadResult(BaseModel):
    messages: list[NormalizedMessage]
    supported_channels_discovered: int
    channels_successfully_read: int
    channels_skipped: int


def _bot_headers() -> dict[str, str]:
    bot_token = get_bot_token()
    return {"Authorization": f"Bot {bot_token}"}


def _retry_after(response: httpx.Response) -> float | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    value = payload.get("retry_after") if isinstance(payload, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _get_with_rate_limit_retry(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, str | int] | None,
    sleep_function: Callable[[float], None],
) -> httpx.Response:
    response = client.get(url, headers=headers, params=params)
    if response.status_code != 429:
        return response

    wait_seconds = _retry_after(response)
    if wait_seconds is None:
        raise DiscordIntegrationError("Discord rate limit response was invalid.", 503)
    if wait_seconds > MAX_RATE_LIMIT_RETRY_AFTER_SECONDS:
        raise DiscordIntegrationError("Discord rate limit exceeded.", 503)
    sleep_function(wait_seconds)
    retry_response = client.get(url, headers=headers, params=params)
    if retry_response.status_code == 429:
        raise DiscordIntegrationError("Discord rate limit exceeded.", 503)
    return retry_response


def _require_success(response: httpx.Response, resource: str) -> None:
    if response.status_code == 403:
        raise DiscordIntegrationError(f"Discord access to the {resource} was denied.", 403)
    if response.status_code == 404:
        raise DiscordIntegrationError(f"Discord {resource} was not found.", 404)
    if response.is_error:
        raise DiscordIntegrationError(f"Discord {resource} request failed.")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise DiscordIntegrationError("Discord message data was invalid.")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DiscordIntegrationError("Discord message data was invalid.") from exc
    if timestamp.tzinfo is None:
        raise DiscordIntegrationError("Discord message data was invalid.")
    return timestamp


def _normalize_attachments(value: object) -> list[AttachmentMetadata]:
    if not isinstance(value, list):
        raise DiscordIntegrationError("Discord message data was invalid.")
    attachments: list[AttachmentMetadata] = []
    for attachment in value:
        if not isinstance(attachment, dict):
            raise DiscordIntegrationError("Discord message data was invalid.")
        attachment_id = attachment.get("id")
        filename = attachment.get("filename")
        url = attachment.get("url")
        content_type = attachment.get("content_type")
        size = attachment.get("size")
        if not all(isinstance(item, str) and item for item in (attachment_id, filename, url)):
            raise DiscordIntegrationError("Discord message data was invalid.")
        if content_type is not None and not isinstance(content_type, str):
            raise DiscordIntegrationError("Discord message data was invalid.")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int)):
            raise DiscordIntegrationError("Discord message data was invalid.")
        attachments.append(
            AttachmentMetadata(
                id=attachment_id,
                filename=filename,
                url=url,
                content_type=content_type,
                size=size,
            )
        )
    return attachments


def _normalize_mentions(value: object) -> list[MentionMetadata]:
    if not isinstance(value, list):
        raise DiscordIntegrationError("Discord message data was invalid.")
    mentions: list[MentionMetadata] = []
    for mention in value:
        if not isinstance(mention, dict):
            raise DiscordIntegrationError("Discord message data was invalid.")
        user_id = mention.get("id")
        global_name = mention.get("global_name")
        username = mention.get("username")
        if not isinstance(user_id, str) or not user_id:
            raise DiscordIntegrationError("Discord message data was invalid.")
        if global_name is not None and not isinstance(global_name, str):
            raise DiscordIntegrationError("Discord message data was invalid.")
        if username is not None and not isinstance(username, str):
            raise DiscordIntegrationError("Discord message data was invalid.")
        display_name = (global_name or "").strip() or (username or "").strip()
        if not display_name:
            raise DiscordIntegrationError("Discord message data was invalid.")
        mentions.append(MentionMetadata(user_id=user_id, display_name=display_name))
    return mentions


def _normalize_stickers(value: object) -> list[StickerMetadata]:
    if not isinstance(value, list):
        raise DiscordIntegrationError("Discord message data was invalid.")
    stickers: list[StickerMetadata] = []
    for sticker in value:
        if not isinstance(sticker, dict):
            raise DiscordIntegrationError("Discord message data was invalid.")
        sticker_id = sticker.get("id")
        name = sticker.get("name")
        format_type = sticker.get("format_type")
        if not isinstance(sticker_id, str) or not sticker_id:
            raise DiscordIntegrationError("Discord message data was invalid.")
        if not isinstance(name, str) or not name.strip():
            raise DiscordIntegrationError("Discord message data was invalid.")
        if format_type is not None and (
            isinstance(format_type, bool) or not isinstance(format_type, int)
        ):
            raise DiscordIntegrationError("Discord message data was invalid.")
        stickers.append(
            StickerMetadata(
                sticker_id=sticker_id,
                name=name.strip(),
                format_type=format_type,
            )
        )
    return stickers


def _normalize_message(
    message: object, guild_id: str, channel_id: str, channel_name: str | None = None
) -> NormalizedMessage:
    if not isinstance(message, dict):
        raise DiscordIntegrationError("Discord message data was invalid.")
    message_id = message.get("id")
    content = message.get("content")
    author = message.get("author")
    if not isinstance(message_id, str) or not isinstance(content, str) or not isinstance(author, dict):
        raise DiscordIntegrationError("Discord message data was invalid.")
    author_id = author.get("id")
    if not isinstance(author_id, str) or not author_id:
        raise DiscordIntegrationError("Discord message data was invalid.")
    author_name = author.get("global_name") or author.get("username")
    if author_name is not None and not isinstance(author_name, str):
        raise DiscordIntegrationError("Discord message data was invalid.")

    reference = message.get("message_reference")
    reply_to_message_id: str | None = None
    if reference is not None:
        if not isinstance(reference, dict):
            raise DiscordIntegrationError("Discord message data was invalid.")
        reference_id = reference.get("message_id")
        if reference_id is not None and not isinstance(reference_id, str):
            raise DiscordIntegrationError("Discord message data was invalid.")
        reply_to_message_id = reference_id

    return NormalizedMessage(
        platform="discord",
        message_id=message_id,
        guild_id=guild_id,
        channel_id=channel_id,
        channel_name=channel_name,
        author_id=author_id,
        author_name=author_name,
        content=content,
        timestamp=_parse_timestamp(message.get("timestamp")),
        attachments=_normalize_attachments(message.get("attachments")),
        reply_to_message_id=reply_to_message_id,
        mentions=_normalize_mentions(message.get("mentions", [])),
        stickers=_normalize_stickers(message.get("sticker_items", [])),
    )


def _guild_connection_exists(guild_id: str) -> bool:
    from initializer import db
    with db.session_context() as session:
        return db.discord_guild_connection_exists(session, guild_id)


def load_messages(
    guild_id: str,
    channel_id: str,
    max_messages: int = 500,
    *,
    sleep_function: Callable[[float], None] = time.sleep,
) -> list[NormalizedMessage]:
    if max_messages <= 0:
        raise DiscordIntegrationError("max_messages must be positive.", 400)
    if not _guild_connection_exists(guild_id):
        raise DiscordIntegrationError("Discord connection was not found.", 404)
    headers = _bot_headers()

    with httpx.Client(timeout=10.0) as client:
        channel_response = _get_with_rate_limit_retry(
            client,
            f"{DISCORD_API_BASE_URL}/channels/{channel_id}",
            headers=headers,
            params=None,
            sleep_function=sleep_function,
        )
        _require_success(channel_response, "channel")
        channel = channel_response.json()
        if not isinstance(channel, dict) or channel.get("guild_id") != guild_id:
            raise DiscordIntegrationError("Requested channel is outside the connected guild.", 403)

        messages: list[NormalizedMessage] = []
        before: str | None = None
        while len(messages) < max_messages:
            page_size = min(100, max_messages - len(messages))
            params: dict[str, str | int] = {"limit": page_size}
            if before is not None:
                params["before"] = before
            response = _get_with_rate_limit_retry(
                client,
                f"{DISCORD_API_BASE_URL}/channels/{channel_id}/messages",
                headers=headers,
                params=params,
                sleep_function=sleep_function,
            )
            _require_success(response, "message history")
            page = response.json()
            if not isinstance(page, list):
                raise DiscordIntegrationError("Discord message response was invalid.")
            if not page:
                break

            normalized_page = [
                _normalize_message(message, guild_id, channel_id) for message in page
            ]
            messages.extend(normalized_page)
            if len(messages) >= max_messages or len(page) < page_size:
                break
            before = normalized_page[-1].message_id
        return messages[:max_messages]


def list_connected_guilds(
    *,
    sleep_function: Callable[[float], None] = time.sleep,
) -> list[ConnectedGuild]:
    from initializer import db
    with db.session_context() as session:
        guild_ids = db.list_discord_guild_ids(session)
    if not guild_ids:
        return []
    headers = _bot_headers()

    with httpx.Client(timeout=10.0) as client:
        guilds: list[ConnectedGuild] = []
        for guild_id in guild_ids:
            response = _get_with_rate_limit_retry(
                client,
                f"{DISCORD_API_BASE_URL}/guilds/{guild_id}",
                headers=headers,
                params=None,
                sleep_function=sleep_function,
            )
            if response.status_code in (403, 404):
                continue
            _require_success(response, "guild")
            payload = response.json()
            if (
                not isinstance(payload, dict)
                or payload.get("id") != guild_id
                or not isinstance(payload.get("name"), str)
                or not payload["name"]
            ):
                raise DiscordIntegrationError("Discord guild response was invalid.")
            guilds.append(ConnectedGuild(guild_id=guild_id, guild_name=payload["name"]))
        return guilds


def discover_guild_channels(
    guild_id: str,
    *,
    sleep_function: Callable[[float], None] = time.sleep,
) -> list[DiscoveredChannel]:
    if not _guild_connection_exists(guild_id):
        raise DiscordIntegrationError("Discord connection was not found.", 404)
    headers = _bot_headers()

    with httpx.Client(timeout=10.0) as client:
        response = _get_with_rate_limit_retry(
            client,
            f"{DISCORD_API_BASE_URL}/guilds/{guild_id}/channels",
            headers=headers,
            params=None,
            sleep_function=sleep_function,
        )
        _require_success(response, "guild channel discovery")
        payload = response.json()
        if not isinstance(payload, list):
            raise DiscordIntegrationError("Discord channel discovery response was invalid.")

        channels: list[DiscoveredChannel] = []
        for channel in payload:
            if not isinstance(channel, dict):
                raise DiscordIntegrationError("Discord channel discovery response was invalid.")
            channel_id = channel.get("id")
            channel_name = channel.get("name")
            channel_type = channel.get("type")
            if (
                not isinstance(channel_id, str)
                or not channel_id
                or not isinstance(channel_name, str)
                or not channel_name
                or isinstance(channel_type, bool)
                or not isinstance(channel_type, int)
            ):
                raise DiscordIntegrationError("Discord channel discovery response was invalid.")
            if channel_type in SUPPORTED_TEXT_CHANNEL_TYPES:
                channels.append(
                    DiscoveredChannel(
                        channel_id=channel_id,
                        channel_name=channel_name,
                        channel_type=channel_type,
                    )
                )
        return channels


def load_guild_messages(
    guild_id: str,
    max_messages_per_channel: int = 10,
    *,
    sleep_function: Callable[[float], None] = time.sleep,
) -> GuildMessageLoadResult:
    if max_messages_per_channel <= 0:
        raise DiscordIntegrationError("max_messages_per_channel must be positive.", 400)

    channels = discover_guild_channels(guild_id, sleep_function=sleep_function)
    messages: list[NormalizedMessage] = []
    channels_successfully_read = 0
    channels_skipped = 0
    for channel in channels:
        try:
            channel_messages = load_messages(
                guild_id,
                channel.channel_id,
                max_messages_per_channel,
                sleep_function=sleep_function,
            )
        except DiscordIntegrationError as exc:
            if exc.status_code in (403, 404):
                channels_skipped += 1
                continue
            raise
        channels_successfully_read += 1
        messages.extend(
            message.model_copy(update={"channel_name": channel.channel_name})
            for message in channel_messages
        )
    messages.sort(
        key=lambda message: (message.timestamp, message.channel_id, message.message_id),
        reverse=True,
    )
    return GuildMessageLoadResult(
        messages=messages,
        supported_channels_discovered=len(channels),
        channels_successfully_read=channels_successfully_read,
        channels_skipped=channels_skipped,
    )
