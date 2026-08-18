"""
Step 7 -- Per-employee explanations with SHAP.

For each of the 850k employees this writes the three features that pushed their risk score
furthest, with the signed contribution, into shap_explanations. The dashboard turns those
rows into the plain-language "top 3 risk factors" list.

Two implementation notes worth knowing:

1. **We explain the base model, not the calibrated wrapper.** SHAP's TreeExplainer needs the
   actual trees; CalibratedClassifierCV wraps them in an isotonic regression it cannot see
   through. Since calibration is a monotone transform of the score, the *ranking* of feature
   contributions is identical either way -- so explaining the base model gives the right
   top-3 for the calibrated score shown in the dashboard.

2. **Ranking is by absolute value, but the stored value keeps its sign.** A feature can matter
   a great deal *because it lowers* someone's risk, and the dashboard says so ("high
   satisfaction is reducing this person's risk"). Ranking on the absolute value and storing
   the signed number is what lets it do that.

SHAP values are computed in chunks: the full 850k x 90 matrix of contributions would be
~300 MB on its own, and we only ever need three numbers per employee.

Run it:  python ml/explain.py
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import shap
from sqlalchemy import text

from db import get_engine, REPO_ROOT
from ml.features import build_features, load_clean_data

MODELS_DIR = REPO_ROOT / "models"
TOP_N = 3                 # the spec asks for the top 3 factors per employee
SHAP_CHUNK = 50_000       # employees explained per batch


def build_explainer(bundle):
    """
    Pick the right SHAP explainer for the winning model.

    TreeExplainer is the one the spec calls for and the one that runs here, because the
    winning model is XGBoost. The linear branch exists so the pipeline does not break if a
    future run picks Logistic Regression instead -- LinearExplainer is the correct exact
    equivalent for a linear model, and everything downstream is unchanged.
    """
    base_model = bundle["base_model"]
    algorithm = bundle["algorithm"]

    if hasattr(base_model, "get_booster") or hasattr(base_model, "estimators_"):
        print(f"  Using shap.TreeExplainer for {algorithm}")
        return shap.TreeExplainer(base_model)

    print(f"  Using shap.LinearExplainer for {algorithm} (not a tree model)")
    estimator = base_model[-1] if hasattr(base_model, "steps") else base_model
    return shap.LinearExplainer(estimator, np.zeros((1, len(bundle["feature_names"]))))


def top_contributors(shap_values: np.ndarray, feature_names: list[str]) -> tuple:
    """
    Reduce a (rows x features) SHAP matrix to the TOP_N per row.

    argpartition finds the N largest absolute values without fully sorting all 90 columns,
    which matters when you are doing it 850k times. The small N-column slice is then sorted
    properly so rank 1 really is the biggest contributor.
    """
    magnitudes = np.abs(shap_values)
    # Indices of the TOP_N largest magnitudes per row (unordered among themselves).
    partitioned = np.argpartition(-magnitudes, kth=TOP_N - 1, axis=1)[:, :TOP_N]
    # Now order those few properly, largest first.
    rows = np.arange(shap_values.shape[0])[:, None]
    order = np.argsort(-magnitudes[rows, partitioned], axis=1)
    ranked_idx = partitioned[rows, order]
    ranked_val = shap_values[rows, ranked_idx]
    return ranked_idx, ranked_val


def main() -> None:
    engine = get_engine()

    bundle_path = MODELS_DIR / "winner.joblib"
    if not bundle_path.exists():
        raise SystemExit("models/winner.joblib not found -- run `python ml/train.py` first.")
    bundle = joblib.load(bundle_path)
    model_version = bundle["model_version"]
    feature_names = bundle["feature_names"]

    print(f"Explaining {bundle['algorithm']} ({model_version})")

    df = load_clean_data()
    X, _, _ = build_features(df, save_spec=False)
    X = X[feature_names]
    employee_ids = df["employee_id"].to_numpy()
    print(f"  {len(df):,} employees x {len(feature_names)} features")

    explainer = build_explainer(bundle)

    # Clear any previous run for this model version so re-running is idempotent.
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM shap_explanations WHERE model_version = :v"),
            {"v": model_version},
        )

    raw_conn = engine.raw_connection()
    total_rows = 0
    try:
        cursor = raw_conn.cursor()
        copy_sql = (
            "COPY shap_explanations "
            "(employee_id, model_version, feature_name, feature_value, shap_value, rank) "
            "FROM STDIN WITH (FORMAT csv)"
        )

        for start in range(0, len(X), SHAP_CHUNK):
            end = min(start + SHAP_CHUNK, len(X))
            chunk = X.iloc[start:end]

            shap_values = explainer.shap_values(chunk)
            # Some SHAP versions return a list (one array per class) for binary problems.
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            shap_values = np.asarray(shap_values, dtype=np.float64)
            if shap_values.ndim == 3:  # (rows, features, classes)
                shap_values = shap_values[:, :, -1]

            ranked_idx, ranked_val = top_contributors(shap_values, feature_names)
            chunk_values = chunk.to_numpy()
            chunk_ids = employee_ids[start:end]

            buffer = io.StringIO()
            for row in range(len(chunk)):
                for rank in range(TOP_N):
                    feature_idx = ranked_idx[row, rank]
                    name = feature_names[feature_idx]
                    value = chunk_values[row, feature_idx]
                    buffer.write(
                        f"{chunk_ids[row]},{model_version},{name},{value:.6g},"
                        f"{ranked_val[row, rank]:.6f},{rank + 1}\n"
                    )
                    total_rows += 1
            buffer.seek(0)
            cursor.copy_expert(copy_sql, buffer)

            print(f"  ... explained {end:,} / {len(X):,} employees", flush=True)

        raw_conn.commit()
    finally:
        raw_conn.close()

    print(f"\nWrote {total_rows:,} rows into shap_explanations "
          f"({TOP_N} per employee).")

    # ---- What are the drivers across the whole workforce? --------------------
    with engine.connect() as conn:
        drivers = pd.read_sql(
            text("""
                SELECT feature_name,
                       COUNT(*) AS times_in_top3,
                       AVG(shap_value) AS mean_signed_contribution,
                       AVG(ABS(shap_value)) AS mean_magnitude
                FROM shap_explanations
                WHERE model_version = :v
                GROUP BY feature_name
                ORDER BY times_in_top3 DESC
                LIMIT 15
            """),
            conn, params={"v": model_version},
        )
    print("\n  Most common top-3 risk factors across the workforce:")
    print(drivers.to_string(index=False))


if __name__ == "__main__":
    main()
