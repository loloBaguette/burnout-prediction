"""
Shared database helpers. Every script in this repo imports from here so there is
exactly one place that knows how to build a Postgres connection.

Reads credentials from the .env file at the repo root (copy .env.example first).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Repo root = the folder this file lives in. Used to resolve every relative path,
# so scripts behave the same no matter which directory you run them from.
REPO_ROOT = Path(__file__).resolve().parent

load_dotenv(REPO_ROOT / ".env")


def get_database_url() -> str:
    """Build the SQLAlchemy connection string from the .env values."""
    user = os.getenv("POSTGRES_USER", "burnout")
    password = os.getenv("POSTGRES_PASSWORD", "burnout_local_dev")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "burnout")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def get_engine():
    """
    Return a SQLAlchemy engine.

    pool_pre_ping=True makes SQLAlchemy check a pooled connection is still alive
    before handing it out -- avoids stale-connection errors during long ETL runs.
    """
    return create_engine(get_database_url(), pool_pre_ping=True, future=True)


def apply_schema() -> None:
    """Run db/schema.sql. Safe to call repeatedly (everything is IF NOT EXISTS)."""
    schema_sql = (REPO_ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(schema_sql))
    print("Schema applied (model_metadata, model_predictions, shap_explanations).")


def resolve_path(path_str: str) -> Path:
    """Turn a possibly-relative path from .env into an absolute path."""
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


if __name__ == "__main__":
    # `python db.py` = a quick connection smoke test.
    with get_engine().connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar_one()
    print("Connected OK ->", version)
