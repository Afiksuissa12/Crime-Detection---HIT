import re
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import DATABASE_URL


def _ensure_db_dir(url: str) -> None:
    match = re.match(r"sqlite:///(.*)", url)
    if match:
        path = match.group(1)
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)


_ensure_db_dir(DATABASE_URL)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
