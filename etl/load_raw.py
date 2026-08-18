"""
Step 1 -- Load the raw dataset into Postgres, AS-IS.

The source file is a 1.3 GB JSON array with one employee object per line
(~850,000 of them). That is far too big to `json.load()` into memory, so this
script streams it one line at a time and bulk-inserts with Postgres COPY.

No cleaning, no feature engineering, no renaming happens here -- that is
etl/clean.py's job. The only thing we add is a surrogate primary key.

Run it:  python etl/load_raw.py
"""

import csv
import io
import json
import os
import sys
from pathlib import Path

# Make `import db` work no matter which directory you run this from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db import get_engine, resolve_path, REPO_ROOT

TABLE_NAME = "raw_employee_data"

# How many records to read before deciding each column's SQL type. The dataset
# is machine-generated and uniform, so a few thousand is plenty.
SCHEMA_SAMPLE_SIZE = 20_000

# Rows per COPY batch. Bigger = fewer round trips but more memory.
BATCH_SIZE = 20_000

# The source file has its own `employee_id` ("SYN_00000123"). It is already
# pseudonymous, but per the spec we do NOT use it as the primary key -- we keep
# the anonymisation habit and give every row our own surrogate key instead.
# The original is kept as `source_employee_id` purely so we can trace a row back
# to the file; it is excluded from every model feature and from the dashboard.
SOURCE_ID_FIELD = "employee_id"
SOURCE_ID_COLUMN = "source_employee_id"


def python_type_to_sql(type_counter: dict) -> str:
    """
    Map the Python types we saw in the sample to a Postgres column type.

    Lists become JSONB so the raw table stays a faithful copy of the file --
    clean.py is where they get turned into something model-friendly.
    """
    names = set(type_counter)
    names.discard("NoneType")  # nulls do not tell us anything about the type

    if not names:
        return "TEXT"  # column was null in the whole sample
    if names <= {"bool"}:
        return "BOOLEAN"
    if names <= {"int"}:
        return "BIGINT"
    if names <= {"int", "float"}:
        return "DOUBLE PRECISION"
    if names <= {"list", "dict"}:
        return "JSONB"
    return "TEXT"


def iter_records(path: Path):
    """
    Yield one dict per employee, streaming.

    Fast path: the file is formatted one JSON object per line, so we can strip
    the array punctuation and parse each line on its own.
    Fallback: if that assumption ever breaks, we hand the file to ijson, which
    parses any valid JSON incrementally (slower, but always correct).
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip().rstrip(",")
            # Skip the opening "[" and closing "]" of the outer array.
            if not stripped or stripped in ("[", "]"):
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                print(
                    f"  Line {line_no} is not a standalone JSON object -- "
                    f"switching to the streaming ijson parser."
                )
                yield from _iter_records_ijson(path)
                return


def _iter_records_ijson(path: Path):
    """Fallback parser: handles pretty-printed / arbitrarily formatted JSON."""
    import ijson

    with path.open("rb") as handle:
        yield from ijson.items(handle, "item")


def infer_schema(path: Path):
    """
    Read the first SCHEMA_SAMPLE_SIZE records and work out, per field, which
    Postgres type to use. Returns (ordered field names, {field: sql_type}).
    """
    seen_types: dict[str, dict] = {}
    field_order: list[str] = []

    for i, record in enumerate(iter_records(path)):
        if i >= SCHEMA_SAMPLE_SIZE:
            break
        for key, value in record.items():
            if key not in seen_types:
                seen_types[key] = {}
                field_order.append(key)
            type_name = type(value).__name__
            seen_types[key][type_name] = seen_types[key].get(type_name, 0) + 1

    sql_types = {field: python_type_to_sql(seen_types[field]) for field in field_order}
    return field_order, sql_types


def create_table(engine, field_order, sql_types) -> list[str]:
    """
    Recreate raw_employee_data from scratch and return the column list used for
    COPY (i.e. everything except the auto-generated surrogate key).

    We DROP and recreate so re-running the pipeline is idempotent -- you always
    get exactly one copy of the file's contents, never a doubled-up table.
    """
    column_defs = ["employee_id SERIAL PRIMARY KEY"]
    copy_columns = []

    for field in field_order:
        column = SOURCE_ID_COLUMN if field == SOURCE_ID_FIELD else field
        column_defs.append(f'    "{column}" {sql_types[field]}')
        copy_columns.append(column)

    ddl = (
        f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE;\n"
        f"CREATE TABLE {TABLE_NAME} (\n"
        + ",\n".join(column_defs)
        + "\n);"
    )

    with engine.begin() as conn:
        conn.execute(text(ddl))

    return copy_columns


def to_csv_value(value):
    r"""
    Convert one Python value into the text COPY expects.

    We use \N as the NULL marker (Postgres' default) so that a genuine empty
    string in the data stays an empty string instead of silently becoming NULL.
    """
    if value is None:
        return r"\N"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value)  # JSONB columns take JSON text
    return value


def load(engine, path: Path, field_order, copy_columns) -> int:
    """Stream the file into Postgres in COPY batches. Returns the row count."""
    raw_conn = engine.raw_connection()
    quoted_columns = ", ".join(f'"{c}"' for c in copy_columns)
    copy_sql = (
        f"COPY {TABLE_NAME} ({quoted_columns}) "
        r"FROM STDIN WITH (FORMAT csv, NULL '\N')"
    )

    total = 0
    try:
        cursor = raw_conn.cursor()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        rows_in_batch = 0

        for record in iter_records(path):
            writer.writerow([to_csv_value(record.get(f)) for f in field_order])
            rows_in_batch += 1
            total += 1

            if rows_in_batch >= BATCH_SIZE:
                buffer.seek(0)
                cursor.copy_expert(copy_sql, buffer)
                buffer = io.StringIO()
                writer = csv.writer(buffer)
                rows_in_batch = 0
                if total % 100_000 == 0:
                    print(f"  ... {total:,} rows loaded")

        if rows_in_batch:  # flush whatever is left in the last partial batch
            buffer.seek(0)
            cursor.copy_expert(copy_sql, buffer)

        raw_conn.commit()
    finally:
        raw_conn.close()

    return total


def main() -> None:
    path = resolve_path(os.getenv("RAW_DATA_PATH", "data/raw/synthetic-employee-dataset.json"))
    if not path.exists():
        raise SystemExit(
            f"Dataset not found at {path}\n"
            f"Put the file in data/raw/ (or point RAW_DATA_PATH in .env at it)."
        )

    size_gb = path.stat().st_size / 1024**3
    print(f"Reading {path.name} ({size_gb:.2f} GB)")

    print(f"Inferring column types from the first {SCHEMA_SAMPLE_SIZE:,} records...")
    field_order, sql_types = infer_schema(path)

    engine = get_engine()
    copy_columns = create_table(engine, field_order, sql_types)
    print(f"Created table {TABLE_NAME} with {len(copy_columns) + 1} columns.\n")

    print("Loading (this takes a few minutes for 1.3 GB)...")
    total = load(engine, path, field_order, copy_columns)

    # ---- Report exactly what landed in the database -------------------------
    with engine.connect() as conn:
        row_count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()
        columns = conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = :t ORDER BY ordinal_position"
            ),
            {"t": TABLE_NAME},
        ).all()
        table_size = conn.execute(
            text(f"SELECT pg_size_pretty(pg_total_relation_size('{TABLE_NAME}'))")
        ).scalar_one()

    print(f"\nDone. {total:,} records streamed, {row_count:,} rows in {TABLE_NAME} ({table_size}).")
    print(f"\nReal column list ({len(columns)} columns) -- clean.py is written against THESE:")
    for name, dtype in columns:
        print(f"  {name:35s} {dtype}")


if __name__ == "__main__":
    main()
