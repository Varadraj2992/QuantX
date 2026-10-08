from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, echo=settings.database_echo, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def create_tables() -> None:
    """Create SQLAlchemy tables for the QuantX data model."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()
