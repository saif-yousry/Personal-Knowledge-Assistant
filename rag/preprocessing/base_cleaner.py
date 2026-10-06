"""
base_cleaner.py

Abstract base class for all content cleaners in the RAG preprocessing
stage.

Single Responsibility Principle
--------------------------------
A Cleaner is responsible for transforming a document's content into
a cleaned form.

This contract is document-type agnostic — cleaners for different document
types (Email, PDF, Slack messages, ...) follow the same
`clean(document) -> document` shape, preserving all metadata and returning
a new object rather than mutating the input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Cleaner(ABC):
    """Abstract base class defining the cleaning contract."""

    @abstractmethod
    def clean(self, source):
        """
        Return a new, cleaned copy of the source object.

        Implementations must not mutate the input object, and must
        preserve all metadata fields untouched — only content fields
        (e.g. body, text) should change.
        """
        raise NotImplementedError
