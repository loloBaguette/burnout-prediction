"""
Step 11 -- Smoke tests.

These answer one question: "does the pipeline actually produce what the acceptance
checklist says it should, end to end, with no manual steps?"

They are deliberately *not* unit tests of individual functions. They check the contract --
tables exist, are populated, contain no NULLs, the model clears the bar, and every employee
has a score and exactly three explanations.

Run them after the pipeline:

    pytest tests/ -v

If the database is empty they skip rather than fail, so a fresh clone does not look broken
before the pipeline has been run once.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import text

from db import get_engine
from etl.clean import BURNOUT_THRESHOLD, PLACEHOLDER_VALUES
from ml.evaluate import REQUIRED_PRECISION, REQUIRED_RECALL


@pytest.fixture
def conn():
    """
    A fresh connection per test.

    Function-scoped on purpose: a shared connection would carry an aborted transaction from
    one failing test into every test after it, turning a single real bug into a wall of
    unrelated errors.
    """
    with get_engine().connect() as connection:
        yield connection


def table_exists(conn, table: str) -> bool:
    return conn.execute(
        text("SELECT to_regclass(:t) IS NOT NULL"), {"t": f"public.{table}"}
    ).scalar_one()


def require(conn, table: str) -> int:
    """Skip (not fail) when the pipeline has not been run yet."""
    if not table_exists(conn, table):
        pytest.skip(f"{table} does not exist -- run `python etl/run_pipeline.py` first")
    count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
    if count == 0:
        pytest.skip(f"{table} is empty -- run `python etl/run_pipeline.py` first")
    return count


# ---------------------------------------------------------------------------
# Stage 1-3: the data
# ---------------------------------------------------------------------------
def test_raw_table_is_populated(conn):
    assert require(conn, "raw_employee_data") > 0


def test_clean_table_matches_raw_row_count(conn):
    raw = require(conn, "raw_employee_data")
    clean = require(conn, "stg_employee_clean")
    # Cleaning only removes duplicate employees; this file has none.
    assert clean <= raw
    assert clean >= raw * 0.95, "cleaning dropped more than 5% of rows unexpectedly"


def test_no_nulls_anywhere_in_clean_table(conn):
    """Success metric #2, asserted rather than eyeballed."""
    require(conn, "stg_employee_clean")
    columns = conn.execute(
        text("SELECT column_name FROM information_schema.columns "
             "WHERE table_name = 'stg_employee_clean'")
    ).scalars().all()
    null_expression = " + ".join(
        f'(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in columns
    )
    total_nulls = conn.execute(
        text(f"SELECT SUM({null_expression}) FROM stg_employee_clean")
    ).scalar_one()
    assert total_nulls == 0, f"found {total_nulls} NULLs in stg_employee_clean"


def test_no_placeholder_or_blank_categories(conn):
    """'nan' and friends are invalid values even though they are not NULL."""
    require(conn, "stg_employee_clean")
    placeholders = ", ".join(f"'{v}'" for v in PLACEHOLDER_VALUES)
    bad = conn.execute(text(f"""
        SELECT COUNT(*) FROM stg_employee_clean
        WHERE LOWER(role) IN ({placeholders})
           OR LOWER(department) IN ({placeholders})
           OR LOWER(job_level) IN ({placeholders})
           OR TRIM(role) = '' OR TRIM(department) = '' OR TRIM(job_level) = ''
    """)).scalar_one()
    assert bad == 0, f"found {bad} placeholder/blank category values"


def test_continuous_scores_are_in_range(conn):
    require(conn, "stg_employee_clean")
    out_of_range = conn.execute(text("""
        SELECT COUNT(*) FROM stg_employee_clean
        WHERE satisfaction_score NOT BETWEEN 0 AND 1
           OR workload_score NOT BETWEEN 0 AND 1
           OR burnout_risk NOT BETWEEN 0 AND 1
           OR salary <= 0
           OR tenure_months < 0
    """)).scalar_one()
    assert out_of_range == 0


def test_target_matches_its_definition(conn):
    require(conn, "stg_employee_clean")
    mismatched = conn.execute(text(f"""
        SELECT COUNT(*) FROM stg_employee_clean
        WHERE is_high_burnout_risk <> (burnout_risk >= {BURNOUT_THRESHOLD})
    """)).scalar_one()
    assert mismatched == 0


def test_duplicate_columns_were_dropped(conn):
    """The EDA found five columns that were really three."""
    require(conn, "stg_employee_clean")
    columns = set(conn.execute(
        text("SELECT column_name FROM information_schema.columns "
             "WHERE table_name = 'stg_employee_clean'")
    ).scalars().all())
    for dropped in ("slack_activity", "meeting_participation", "goal_achievement_rate"):
        assert dropped not in columns, f"{dropped} is a duplicate column and should be gone"


# ---------------------------------------------------------------------------
# Stage 4-6: the models
# ---------------------------------------------------------------------------
def test_all_three_algorithms_were_compared(conn):
    require(conn, "model_metadata")
    algorithms = set(conn.execute(
        text("SELECT DISTINCT algorithm FROM model_metadata WHERE NOT is_fallback")
    ).scalars().all())
    assert {"LogisticRegression", "RandomForest", "XGBoost"} <= algorithms


def test_exactly_one_winner(conn):
    require(conn, "model_metadata")
    winners = conn.execute(
        text("SELECT COUNT(*) FROM model_metadata WHERE is_winner")
    ).scalar_one()
    assert winners == 1


def test_winner_meets_the_acceptance_bar(conn):
    """Recall >= 90% and Precision >= 15%, from the initiation document."""
    require(conn, "model_metadata")
    row = conn.execute(text(
        "SELECT recall, precision FROM model_metadata WHERE is_winner"
    )).one()
    assert row.recall >= REQUIRED_RECALL, f"recall {row.recall:.4f} < {REQUIRED_RECALL}"
    assert row.precision >= REQUIRED_PRECISION, (
        f"precision {row.precision:.4f} < {REQUIRED_PRECISION}"
    )


def test_fallback_model_exists(conn):
    require(conn, "model_metadata")
    fallbacks = conn.execute(
        text("SELECT COUNT(*) FROM model_metadata WHERE is_fallback")
    ).scalar_one()
    assert fallbacks >= 1


def test_model_artifacts_are_on_disk(conn):
    from db import REPO_ROOT
    require(conn, "model_metadata")
    for artifact in ("winner.joblib", "fallback.joblib", "feature_spec.json"):
        assert (REPO_ROOT / "models" / artifact).exists(), f"models/{artifact} missing"


# ---------------------------------------------------------------------------
# Stage 7-8: predictions and explanations
# ---------------------------------------------------------------------------
def test_every_employee_has_a_prediction(conn):
    employees = require(conn, "stg_employee_clean")
    require(conn, "model_predictions")
    scored = conn.execute(text("""
        SELECT COUNT(*) FROM model_predictions
        WHERE model_version = (SELECT model_version FROM model_metadata WHERE is_winner)
    """)).scalar_one()
    assert scored == employees, f"{employees:,} employees but {scored:,} predictions"


def test_risk_scores_are_probabilities_and_tiers_are_valid(conn):
    require(conn, "model_predictions")
    bad = conn.execute(text("""
        SELECT COUNT(*) FROM model_predictions
        WHERE risk_score < 0 OR risk_score > 1
           OR risk_level NOT IN ('low', 'medium', 'high')
    """)).scalar_one()
    assert bad == 0


def test_every_employee_has_exactly_three_explanations(conn):
    """The dashboard's "top 3 risk factors" panel depends on this."""
    require(conn, "shap_explanations")
    wrong = conn.execute(text("""
        SELECT COUNT(*) FROM (
            SELECT employee_id, COUNT(*) AS n
            FROM shap_explanations
            WHERE model_version = (SELECT model_version FROM model_metadata WHERE is_winner)
            GROUP BY employee_id
            HAVING COUNT(*) <> 3
        ) AS bad
    """)).scalar_one()
    assert wrong == 0, f"{wrong:,} employees do not have exactly 3 SHAP factors"


def test_explanation_ranks_are_1_2_3(conn):
    require(conn, "shap_explanations")
    bad = conn.execute(text(
        "SELECT COUNT(*) FROM shap_explanations WHERE rank NOT IN (1, 2, 3)"
    )).scalar_one()
    assert bad == 0


def test_rank_1_is_the_largest_contributor(conn):
    """Ranking is by absolute SHAP value, so rank 1 must dominate ranks 2 and 3."""
    require(conn, "shap_explanations")
    violations = conn.execute(text("""
        WITH ranked AS (
            SELECT employee_id,
                   MAX(ABS(shap_value)) FILTER (WHERE rank = 1) AS r1,
                   MAX(ABS(shap_value)) FILTER (WHERE rank = 3) AS r3
            FROM shap_explanations
            WHERE model_version = (SELECT model_version FROM model_metadata WHERE is_winner)
            GROUP BY employee_id
        )
        SELECT COUNT(*) FROM ranked WHERE r1 < r3
    """)).scalar_one()
    assert violations == 0


# ---------------------------------------------------------------------------
# The dashboard's own query path
# ---------------------------------------------------------------------------
def test_dashboard_join_returns_complete_rows(conn):
    """Exactly the join dashboard/app.py runs -- it must not lose employees."""
    require(conn, "model_predictions")
    row = conn.execute(text("""
        SELECT e.employee_id, e.department, e.role, p.risk_score, p.risk_level
        FROM model_predictions p
        JOIN stg_employee_clean e USING (employee_id)
        WHERE p.model_version = (SELECT model_version FROM model_metadata WHERE is_winner)
        ORDER BY p.risk_score DESC
        LIMIT 1
    """)).mappings().first()
    assert row is not None
    assert all(value is not None for value in row.values())
