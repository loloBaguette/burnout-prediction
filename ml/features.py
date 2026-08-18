"""
Step 4 -- Feature engineering and the train/test split.

Every other ML script imports from this file, so there is exactly one definition of
"what the model sees". That matters most for the leakage list below: if a column is
excluded here, it is excluded everywhere.

Two things to understand before changing anything:

1. LEAKAGE_COLUMNS is not a style preference. This is a synthetic dataset generated from
   personas, and the columns listed there are by-products of that generator. Put any of them
   back and you get a model with ~1.00 AUC that would be worthless on real HR data.

2. Encoding is done by hand (explicit category lists saved to models/feature_spec.json)
   rather than with a ColumnTransformer. It is a few more lines, but it means every column
   in the feature matrix has a readable name -- which is what makes the SHAP output in
   ml/explain.py say "satisfaction_score" instead of "feature_37".

Run it on its own to see the resulting shapes:  python ml/features.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from db import get_engine, REPO_ROOT

SOURCE_TABLE = "stg_employee_clean"
TARGET_COLUMN = "is_high_burnout_risk"

# Where the encoding vocabulary is written, so training and scoring agree on column order.
FEATURE_SPEC_PATH = REPO_ROOT / "models" / "feature_spec.json"

TEST_SIZE = 0.20
RANDOM_STATE = 42  # fixed everywhere, so any teammate reproduces the same split

# ---------------------------------------------------------------------------
# Columns that must never become features. See notebooks/01_eda.ipynb, finding 4.
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS = {
    "burnout_risk": "the continuous score the target is derived from",
    "stress_level": "correlates 0.994 with the target and is exactly equal in 46% of rows",
    "risk_factors_summary": "a text restatement of the target, cut at the same 0.75",
    "turnover_probability_generated": "generator internal, correlates 0.59 with the target",
    "persona_name": "the generator's latent class -- every other column is drawn from it",
    "communication_patterns": "a 12-value template keyed directly off persona_name",
    "left_company": "an outcome, not known at prediction time",
    "turnover_reason": "an outcome, not known at prediction time",
}

# Identifiers, and the normalised copies that only exist for the dashboard.
NON_FEATURE_COLUMNS = {
    "employee_id",
    "source_employee_id",
    "salary_norm",           # monotone copy of salary
    "tenure_months_norm",    # monotone copy of tenure_months
    "overtime_hours_norm",   # monotone copy of overtime_hours
    TARGET_COLUMN,
}

NUMERIC_FEATURES = [
    "tenure_months",
    "salary",
    "overtime_hours",
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
    "n_technical_skills",
    "n_soft_skills",
    "feedback_length",
]

CATEGORICAL_FEATURES = ["job_level", "department", "role"]

# `role` has ~150 distinct values with a long tail. Keeping all of them would add 150 mostly
# empty columns, so we keep the most common ones and bucket the rest as "Other".
MAX_ROLE_LEVELS = 30

# Step 6's fallback model: the columns you would expect *any* HR system to have, even a
# spreadsheet. No sentiment scores, no collaboration analytics, no skills inventory.
FALLBACK_FEATURES = [
    "tenure_months",
    "salary",
    "overtime_hours",
    "performance_score",
    "job_level",
    "department",
]


def load_clean_data(limit: int | None = None) -> pd.DataFrame:
    """Read stg_employee_clean into a DataFrame. `limit` is for quick smoke tests."""
    query = f"SELECT * FROM {SOURCE_TABLE} ORDER BY employee_id"
    if limit:
        query += f" LIMIT {limit}"
    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def fit_feature_spec(df: pd.DataFrame, feature_subset: list[str] | None = None) -> dict:
    """
    Work out the encoding vocabulary from the data and return it as a plain dict.

    Saving this makes scoring reproducible: a model trained today can be applied to new
    employees tomorrow and still produce columns in the same order.
    """
    numeric = [c for c in NUMERIC_FEATURES if feature_subset is None or c in feature_subset]
    categorical = [c for c in CATEGORICAL_FEATURES if feature_subset is None or c in feature_subset]

    categories: dict[str, list[str]] = {}
    for col in categorical:
        counts = df[col].value_counts()
        if col == "role":
            keep = counts.head(MAX_ROLE_LEVELS).index.tolist()
            if len(counts) > MAX_ROLE_LEVELS:
                keep.append("Other")
        else:
            keep = counts.index.tolist()
        categories[col] = sorted(keep)

    return {"numeric": numeric, "categorical": categories, "max_role_levels": MAX_ROLE_LEVELS}


def apply_feature_spec(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    Turn the cleaned table into a numeric feature matrix using a saved spec.

    Uses float32 rather than float64 -- with 850k rows that halves the memory the matrix
    occupies (roughly 300 MB instead of 600 MB) at no cost to model quality.
    """
    frames = [df[spec["numeric"]].astype(np.float32)]

    for col, levels in spec["categorical"].items():
        values = df[col]
        # Anything outside the known vocabulary becomes "Other" (only role has an "Other").
        if "Other" in levels:
            values = values.where(values.isin(levels), "Other")
        dummies = pd.get_dummies(values, prefix=col, dtype=np.float32)
        # Reindex so the columns are always the same set, in the same order, even if a
        # particular batch happens not to contain every level.
        expected = [f"{col}_{level}" for level in levels]
        dummies = dummies.reindex(columns=expected, fill_value=np.float32(0))
        frames.append(dummies)

    matrix = pd.concat(frames, axis=1)
    matrix.columns = [str(c) for c in matrix.columns]
    return matrix


def build_features(
    df: pd.DataFrame,
    feature_subset: list[str] | None = None,
    save_spec: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Main entry point: cleaned DataFrame in, (X, y, spec) out.

    Pass `feature_subset` to build the reduced matrix the fallback model uses.
    """
    spec = fit_feature_spec(df, feature_subset)
    X = apply_feature_spec(df, spec)
    y = df[TARGET_COLUMN].astype(int)

    if save_spec:
        FEATURE_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEATURE_SPEC_PATH.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    return X, y, spec


def stratified_split(X: pd.DataFrame, y: pd.Series):
    """
    80/20 split, stratified on the target.

    Stratified because the two classes are not balanced (59/41), so a plain random split
    could hand the test set a noticeably different mix from the training set.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def get_training_data(limit: int | None = None, feature_subset: list[str] | None = None):
    """Convenience wrapper: load, encode, split. Returns everything downstream needs."""
    df = load_clean_data(limit=limit)
    X, y, spec = build_features(df, feature_subset=feature_subset, save_spec=feature_subset is None)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    return df, X, y, X_train, X_test, y_train, y_test, spec


def main() -> None:
    print("Loading stg_employee_clean...")
    df = load_clean_data()
    print(f"  {len(df):,} rows, {len(df.columns)} columns\n")

    print(f"Excluding {len(LEAKAGE_COLUMNS)} leakage columns:")
    for col, reason in LEAKAGE_COLUMNS.items():
        print(f"  - {col:32s} {reason}")

    X, y, spec = build_features(df)

    print(f"\nFeature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns")
    print(f"  {len(spec['numeric'])} numeric")
    for col, levels in spec["categorical"].items():
        print(f"  {len(levels):>3} one-hot columns from {col}")
    print(f"  memory: {X.memory_usage(deep=True).sum() / 1024**2:.0f} MB")

    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"\nStratified 80/20 split:")
    print(f"  train: {len(X_train):,} rows, {100 * y_train.mean():.2f}% positive")
    print(f"  test:  {len(X_test):,} rows, {100 * y_test.mean():.2f}% positive")
    print(f"\nFeature spec saved to {FEATURE_SPEC_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
