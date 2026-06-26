"""
Shared utility helpers for the MindScribe application layer.
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """
    Return the current UTC time as a naive ``datetime``.

    ``datetime.utcnow()`` is deprecated from Python 3.12 onwards. We build a
    timezone-aware UTC value and then strip the ``tzinfo`` so the result keeps
    the same naive-UTC semantics the database columns already rely on
    (``DateTime`` columns are declared without ``timezone=True``).

    Returns:
        Current time in UTC, without timezone information.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
