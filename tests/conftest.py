"""
Shared pytest fixtures.

The suite runs against an isolated in-memory SQLite database and avoids the ML
stack entirely, so it needs only sqlalchemy + pandas + pytest installed
(see requirements-dev.txt).
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import JournalEntry, EmotionAnalysis, DistortionAnalysis
from app.repository import JournalRepository


@pytest.fixture
def db_session():
    """A fresh in-memory SQLite session with all tables created (per test)."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def repo(db_session):
    """A JournalRepository bound to the in-memory session."""
    return JournalRepository(db_session)


@pytest.fixture
def make_analysis():
    """
    Factory for an analysis-result dict in the shape repository.create_entry
    expects.

    emotions:    list of (emotion, probability)
    distortions: list of (label, name, description, probability)
    """
    def _make(emotions=None, distortions=None):
        return {
            "emotions": {
                "detected": [
                    {"emotion": e, "probability": p} for e, p in (emotions or [])
                ],
            },
            "distortions": {
                "detected": [
                    {"label": lbl, "name": name, "description": desc, "probability": p}
                    for lbl, name, desc, p in (distortions or [])
                ],
            },
        }

    return _make


@pytest.fixture
def add_entry(db_session):
    """
    Insert a JournalEntry (with optional emotions/distortions and an explicit
    timestamp) directly via the ORM, bypassing the analyzer. Useful for trend
    tests that need control over timestamps.
    """
    def _add(text="entry", user_id="default", timestamp=None,
             emotions=None, distortions=None):
        entry = JournalEntry(
            text=text,
            user_id=user_id,
            timestamp=timestamp or datetime(2026, 1, 1, 12, 0, 0),
        )
        db_session.add(entry)
        db_session.flush()
        for emo, prob in (emotions or []):
            db_session.add(
                EmotionAnalysis(entry_id=entry.id, emotion=emo, probability=prob)
            )
        for lbl, name, desc, prob in (distortions or []):
            db_session.add(
                DistortionAnalysis(
                    entry_id=entry.id,
                    distortion_type=lbl,
                    distortion_name=name,
                    description=desc,
                    probability=prob,
                )
            )
        db_session.commit()
        db_session.refresh(entry)
        return entry

    return _add
