"""SQLite helpers for storing attendee webhook payloads."""

from __future__ import annotations

import json
import os
from typing import Generator

from sqlalchemy import Boolean, Column, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker


# Accept override via env, fall back to local SQLite file.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///attendee_events.db")

# SQLite needs this flag for threaded FastAPI servers.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AttendeeEvent(Base):
    """Store the raw webhook payload for audit + replay."""

    __tablename__ = "attendee_events"

    id = Column(Integer, primary_key=True, index=True)
    idempotency_key = Column(String, nullable=False, unique=True, index=True)
    trigger = Column(String, nullable=False)
    bot_id = Column(String, nullable=True)
    bot_metadata = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    speaker_name = Column(String, nullable=True)
    speaker_uuid = Column(String, nullable=True)
    timestamp_ms = Column(Integer, nullable=True)
    transcript_text = Column(Text, nullable=True)
    speaker_is_host = Column(Boolean, nullable=True)
    speaker_user_uuid = Column(String, nullable=True)


def init_db() -> None:
    """Create tables once on startup."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def serialize_optional_json(value) -> str:
    """Serialize JSON-like payloads; FastAPI expects strings for Text columns."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)

