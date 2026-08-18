"""
Step 3 -- Clean raw_employee_data into stg_employee_clean.

Everything this script does was decided by the EDA notebook (notebooks/01_eda.ipynb,
section 10). If you want to know *why* a rule exists, that notebook is the answer.

The five real problems in the raw data:
  1. ~782k `role` values and ~13k `department` values carry stray leading whitespace, so
     " Data Analyst" and "Data Analyst" look like two different jobs to an encoder.
  2. `role` is whitespace-only (i.e. genuinely unknown) in ~109k rows, and holds the literal
     string 'nan' in a further 7,481 (`department` in 983). Those pass every NULL and every
     empty-string check while being exactly as useless -- so they are listed explicitly in
     PLACEHOLDER_VALUES and folded into one honest 'Unknown' category.
  3. Five columns are really three: collaboration_score == slack_activity ==
     meeting_participation, and performance_score == goal_achievement_rate. Byte-identical
     in all 849,999 rows.
  4. Two columns hold JSON arrays, which no model can consume directly.
  5. There is no binary label -- only the continuous `burnout_risk`.

The work happens as one INSERT ... SELECT inside Postgres rather than in pandas: 850k rows
never have to travel over the wire, and the whole thing takes a couple of seconds.

The target table is declared with NOT NULL and CHECK constraints on every column, so
"0% NULL / invalid values" (success metric #2) is enforced by the database at write time --
not merely checked afterwards. The assertions at the end are the belt to that pair of braces.

Run it:  python etl/clean.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db import get_engine

SOURCE_TABLE = "raw_employee_data"
TARGET_TABLE = "stg_employee_clean"

# The dataset's own definition of "Severe Burnout Risk" is burnout_risk >= 0.75 -- the EDA
# notebook shows risk_factors_summary is exactly this cut. We reuse it rather than inventing
# a threshold of our own.
BURNOUT_THRESHOLD = 0.75

# Strings that are *present* in the data but carry no information. They survive every
# NULL check and every "is it blank" check -- `role` holds the literal text 'nan' in 7,481
# rows and `department` in 983 -- so they have to be listed explicitly. Compared lowercase
# after trimming.
PLACEHOLDER_VALUES = ["nan", "none", "null", "n/a", "na", "undefined", "-", "?"]

UNKNOWN_LABEL = "Unknown"

# Columns proven identical to another column in the EDA, dropped here.
DUPLICATE_COLUMNS = ["slack_activity", "meeting_participation", "goal_achievement_rate"]

# Every score in the source that is documented as a 0-1 proportion. Checked as such.
UNIT_INTERVAL_COLUMNS = [
    "performance_score",
    "satisfaction_score",
    "workload_score",
    "team_sentiment",
    "project_completion_rate",
    "training_participation",
    "collaboration_score",
    "email_sentiment",
    "role_complexity_score",
    "career_progression_score",
    "burnout_risk",
    "stress_level",
    "turnover_probability_generated",
]


def clean_category_sql(column: str) -> str:
    """
    SQL that turns one raw text column into a trustworthy category.

    Trim -> if what is left is empty or a placeholder, call it 'Unknown' -> otherwise keep it.
    Making the unknowns an explicit category rather than deleting the rows keeps 13% of the
    dataset that would otherwise be thrown away just to satisfy a NULL check.
    """
    placeholders = ", ".join(f"'{v}'" for v in PLACEHOLDER_VALUES)
    return (
        f"CASE WHEN TRIM(r.{column}) = '' "
        f"OR LOWER(TRIM(r.{column})) IN ({placeholders}) "
        f"THEN '{UNKNOWN_LABEL}' ELSE TRIM(r.{column}) END"
    )


def placeholder_check_sql(column: str) -> str:
    """A CHECK constraint body rejecting placeholder text in `column`."""
    placeholders = ", ".join(f"'{v}'" for v in PLACEHOLDER_VALUES)
    return f"LOWER({column}) NOT IN ({placeholders}) AND {column} <> ''"


def build_ddl() -> str:
    """
    The CREATE TABLE for stg_employee_clean.

    Note how much of the data contract is expressed as constraints: if cleaning ever regresses,
    the INSERT fails loudly instead of quietly writing bad rows.
    """
    role_check = placeholder_check_sql("role")
    job_level_check = placeholder_check_sql("job_level")
    department_check = placeholder_check_sql("department")
    unit_checks = ",\n".join(
        f'    {col} DOUBLE PRECISION NOT NULL CHECK ({col} BETWEEN 0 AND 1)'
        for col in UNIT_INTERVAL_COLUMNS
    )
    return f"""
DROP TABLE IF EXISTS {TARGET_TABLE} CASCADE;

CREATE TABLE {TARGET_TABLE} (
    -- Identity. employee_id is our surrogate key, carried over from the raw table.
    employee_id             INTEGER PRIMARY KEY,
    source_employee_id      TEXT    NOT NULL,

    -- Categorical attributes (trimmed; blanks made explicit).
    role                    TEXT    NOT NULL CHECK ({role_check}),
    job_level               TEXT    NOT NULL CHECK ({job_level_check}),
    department              TEXT    NOT NULL CHECK ({department_check}),

    -- Continuous attributes, original units (kept readable for the dashboard).
    tenure_months           INTEGER NOT NULL CHECK (tenure_months >= 0),
    salary                  DOUBLE PRECISION NOT NULL CHECK (salary > 0),
    overtime_hours          DOUBLE PRECISION NOT NULL CHECK (overtime_hours >= 0),

    -- Continuous attributes already expressed as 0-1 proportions.
{unit_checks},

    -- Min-max normalised copies of the three columns that are NOT already 0-1.
    -- These exist so the BI dashboard can put salary, tenure and overtime on a common
    -- scale. The model pipeline does its own scaling, fitted on the training split only,
    -- so nothing here leaks test-set information into training.
    salary_norm             DOUBLE PRECISION NOT NULL CHECK (salary_norm BETWEEN 0 AND 1),
    tenure_months_norm      DOUBLE PRECISION NOT NULL CHECK (tenure_months_norm BETWEEN 0 AND 1),
    overtime_hours_norm     DOUBLE PRECISION NOT NULL CHECK (overtime_hours_norm BETWEEN 0 AND 1),

    -- The two JSON arrays, reduced to something a model can use.
    n_technical_skills      SMALLINT NOT NULL CHECK (n_technical_skills >= 0),
    n_soft_skills           SMALLINT NOT NULL CHECK (n_soft_skills >= 0),

    -- The free-text feedback, reduced to its length. The prose itself is not carried
    -- forward: it is scraped review text with no per-employee meaning, and dropping it
    -- keeps the table small and the anonymisation habit intact.
    feedback_length         INTEGER NOT NULL CHECK (feedback_length >= 0),

    -- Generator by-products. Kept for analysis and for the dashboard, but every one of
    -- these is excluded from the feature set by ml/features.py -- see EDA finding 4.
    communication_patterns  TEXT    NOT NULL,
    persona_name            TEXT    NOT NULL,
    risk_factors_summary    TEXT    NOT NULL,

    -- Outcome columns: known only after the fact, so also excluded from features.
    left_company            BOOLEAN NOT NULL,
    turnover_reason         TEXT    NOT NULL,

    -- The modelling target, derived from burnout_risk at the threshold above.
    is_high_burnout_risk    BOOLEAN NOT NULL
);
"""


def build_insert() -> str:
    """
    The one statement that does all the cleaning.

    Reading order matches the problem list in this file's docstring: trim, fill blanks,
    drop duplicates (by simply not selecting them), unpack JSON, derive the target.
    """
    # Min-max normalisation needs the column extremes, computed once in a CTE.
    role_sql = clean_category_sql("role")
    job_level_sql = clean_category_sql("job_level")
    department_sql = clean_category_sql("department")
    return f"""
INSERT INTO {TARGET_TABLE}
WITH bounds AS (
    SELECT
        MIN(salary) AS min_salary, MAX(salary) AS max_salary,
        MIN(tenure_months) AS min_tenure, MAX(tenure_months) AS max_tenure,
        MIN(overtime_hours) AS min_ot, MAX(overtime_hours) AS max_ot
    FROM {SOURCE_TABLE}
),
deduped AS (
    -- Dedup step. source_employee_id is unique in this file, so this removes nothing today,
    -- but it makes the pipeline safe against a re-delivered file with repeated employees:
    -- we keep the lowest surrogate id per source id.
    SELECT DISTINCT ON (source_employee_id) *
    FROM {SOURCE_TABLE}
    ORDER BY source_employee_id, employee_id
)
SELECT
    r.employee_id,
    TRIM(r.source_employee_id),

    -- Trim, then fold blanks and placeholder text ('nan', 'null', ...) into one explicit
    -- 'Unknown' category. See clean_category_sql() for why they are not deleted.
    {role_sql},
    {job_level_sql},
    {department_sql},

    r.tenure_months::INTEGER,
    r.salary,
    r.overtime_hours,

    r.performance_score,
    r.satisfaction_score,
    r.workload_score,
    r.team_sentiment,
    r.project_completion_rate,
    r.training_participation,
    r.collaboration_score,
    r.email_sentiment,
    r.role_complexity_score,
    r.career_progression_score,
    r.burnout_risk,
    r.stress_level,
    r.turnover_probability_generated,

    -- Min-max scaling. NULLIF guards against a zero-width range (a constant column).
    (r.salary - b.min_salary) / NULLIF(b.max_salary - b.min_salary, 0),
    (r.tenure_months - b.min_tenure)::DOUBLE PRECISION / NULLIF(b.max_tenure - b.min_tenure, 0),
    (r.overtime_hours - b.min_ot) / NULLIF(b.max_ot - b.min_ot, 0),

    jsonb_array_length(r.technical_skills)::SMALLINT,
    jsonb_array_length(r.soft_skills)::SMALLINT,
    LENGTH(r.recent_feedback),

    TRIM(r.communication_patterns),
    TRIM(r.persona_name),
    TRIM(r.risk_factors_summary),

    r.left_company,
    TRIM(r.turnover_reason),

    (r.burnout_risk >= {BURNOUT_THRESHOLD})
FROM deduped r
CROSS JOIN bounds b;
"""


def verify(engine) -> None:
    """
    Success metric #2, verified rather than eyeballed.

    The CHECK/NOT NULL constraints above already make bad rows unwritable, so these
    assertions confirm the table matches what we expect *and* that nothing was silently lost.
    """
    with engine.connect() as conn:
        source_rows = conn.execute(text(f"SELECT COUNT(*) FROM {SOURCE_TABLE}")).scalar_one()
        clean_rows = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE}")).scalar_one()

        columns = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = :t ORDER BY ordinal_position"
            ),
            {"t": TARGET_TABLE},
        ).scalars().all()

        # One scan, every column checked for NULL.
        null_sql = " + ".join(f'(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in columns)
        total_nulls = conn.execute(
            text(f"SELECT SUM({null_sql}) FROM {TARGET_TABLE}")
        ).scalar_one()

        blanks = conn.execute(text(f"""
            SELECT COUNT(*) FROM {TARGET_TABLE}
            WHERE TRIM(role) = '' OR TRIM(job_level) = '' OR TRIM(department) = ''
        """)).scalar_one()

        untrimmed = conn.execute(text(f"""
            SELECT COUNT(*) FROM {TARGET_TABLE}
            WHERE role <> TRIM(role) OR department <> TRIM(department)
               OR job_level <> TRIM(job_level)
        """)).scalar_one()

        placeholders = ", ".join(f"'{v}'" for v in PLACEHOLDER_VALUES)
        placeholder_rows = conn.execute(text(f"""
            SELECT COUNT(*) FROM {TARGET_TABLE}
            WHERE LOWER(role) IN ({placeholders})
               OR LOWER(job_level) IN ({placeholders})
               OR LOWER(department) IN ({placeholders})
        """)).scalar_one()

        unknown_counts = conn.execute(text(f"""
            SELECT
                SUM(CASE WHEN role = '{UNKNOWN_LABEL}' THEN 1 ELSE 0 END),
                SUM(CASE WHEN department = '{UNKNOWN_LABEL}' THEN 1 ELSE 0 END)
            FROM {TARGET_TABLE}
        """)).one()

        duplicate_ids = conn.execute(text(f"""
            SELECT COUNT(*) FROM (
                SELECT source_employee_id FROM {TARGET_TABLE}
                GROUP BY 1 HAVING COUNT(*) > 1
            ) AS d
        """)).scalar_one()

        target_balance = conn.execute(text(f"""
            SELECT is_high_burnout_risk, COUNT(*)
            FROM {TARGET_TABLE} GROUP BY 1 ORDER BY 1
        """)).all()

    print("\n--- Verification (success metric #2) ---")
    print(f"  rows in  {SOURCE_TABLE}: {source_rows:,}")
    print(f"  rows in  {TARGET_TABLE}: {clean_rows:,}  ({source_rows - clean_rows:,} removed as duplicates)")
    print(f"  columns:                 {len(columns)}")
    print(f"  NULL values (all cols):  {total_nulls}")
    print(f"  blank categoricals:      {blanks}")
    print(f"  untrimmed categoricals:  {untrimmed}")
    print(f"  placeholder strings:     {placeholder_rows}   ('nan', 'null', 'n/a', ...)")
    print(f"  duplicate employee ids:  {duplicate_ids}")
    print(f"  folded into '{UNKNOWN_LABEL}':  role {unknown_counts[0]:,}, "
          f"department {unknown_counts[1]:,}")

    # These are the assertions the spec asks for -- the pipeline stops here if cleaning broke.
    assert total_nulls == 0, f"Expected 0 NULLs, found {total_nulls}"
    assert blanks == 0, f"Expected 0 blank categoricals, found {blanks}"
    assert untrimmed == 0, f"Expected 0 untrimmed values, found {untrimmed}"
    assert placeholder_rows == 0, f"Expected 0 placeholder strings, found {placeholder_rows}"
    assert duplicate_ids == 0, f"Expected unique employees, found {duplicate_ids} duplicated ids"
    assert clean_rows > 0, "Clean table is empty"

    print("\n  Target balance (is_high_burnout_risk):")
    for value, count in target_balance:
        print(f"    {str(value):5s}  {count:>9,}  ({100 * count / clean_rows:.2f}%)")

    print("\n  All assertions passed: 0% NULL / invalid values in stg_employee_clean.")


def main() -> None:
    engine = get_engine()

    print(f"Cleaning {SOURCE_TABLE} -> {TARGET_TABLE}")
    print(f"  dropping duplicate columns: {', '.join(DUPLICATE_COLUMNS)}")
    print(f"  target: is_high_burnout_risk = burnout_risk >= {BURNOUT_THRESHOLD}")

    with engine.begin() as conn:
        conn.execute(text(build_ddl()))
        result = conn.execute(text(build_insert()))
        print(f"\n  {result.rowcount:,} rows written.")

    verify(engine)


if __name__ == "__main__":
    main()
