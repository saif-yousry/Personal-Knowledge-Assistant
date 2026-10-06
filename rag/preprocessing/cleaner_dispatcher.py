"""
Selects and applies the correct cleaner based on document type.
"""

from __future__ import annotations

from models import DiscordMessage, Document, Email, SlackMessage, TelegramMessage
from .email_cleaner import EmailCleaner
from .pdf_cleaner import PdfCleaner
from .slack_cleaner import SlackCleaner
from .discord_cleaner import DiscordCleaner
from .telegram_cleaner import TelegramCleaner


class CleanerDispatcher:
    """Routes documents to the correct cleaner based on type."""

    def __init__(self):
        self._cleaners = {
            Email: EmailCleaner(),
            Document: PdfCleaner(),
            SlackMessage: SlackCleaner(),
            DiscordMessage: DiscordCleaner(),
            TelegramMessage: TelegramCleaner(),
        }

    def clean(self, document):
        """Clean a document using the appropriate cleaner."""
        cleaner = self._cleaners.get(type(document))
        if cleaner is None:
            raise TypeError(f"No cleaner registered for {type(document)}")
        return cleaner.clean(document)
