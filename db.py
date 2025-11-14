"""SQLite helpers for storing attendee webhook payloads."""

from __future__ import annotations

import json
import os
from typing import Generator

from sqlalchemy import Column, Integer, String, Text, create_engine
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
    payload = Column(Text, nullable=False)


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
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)

