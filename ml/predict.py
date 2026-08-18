"""
Step 8 -- Score every employee and write the results to model_predictions.

Takes the calibrated winning model from models/winner.joblib, scores all 850k employees,
turns each score into a low/medium/high tier, and bulk-loads the lot into Postgres.

About the tiers: the cut-offs below are deliberately round numbers on a *calibrated*
probability, so they mean something you can say out loud -- "high risk" is an employee the
model gives at least a 70% chance of being in the severe-burnout band. They are not tuned
to the operating threshold from training, because that threshold answers a different
question (how aggressively to screen) and on this dataset it sits at 1.0, which would put
essentially everyone in one tier.

Run it:  python ml/predict.py
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from db import get_engine, REPO_ROOT
from ml.features import build_features, load_clean_data

MODELS_DIR = REPO_ROOT / "models"

# Tier cut-offs on the calibrated probability. Documented in the README and shown in the
# dashboard legend so nobody has to read this file to know what "high" means.
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.30

SCORE_CHUNK = 100_000  # rows scored at a time, to keep memory flat


def assign_tier(scores: np.ndarray) -> np.ndarray:
    """Vectorised score -> tier. Order matters: np.select takes the first match."""
    return np.select(
        [scores >= HIGH_RISK_THRESHOLD, scores >= MEDIUM_RISK_THRESHOLD],
        ["high", "medium"],
        default="low",
    )


def write_predictions(engine, employee_ids, scores, tiers, model_version) -> int:
    """Replace this model version's predictions, then COPY the new ones in."""
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM model_predictions WHERE model_version = :v"),
            {"v": model_version},
        )

    buffer = io.StringIO()
    for employee_id, score, tier in zip(employee_ids, scores, tiers):
        buffer.write(f"{employee_id},{model_version},{score:.6f},{tier}\n")
    buffer.seek(0)

    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        cursor.copy_expert(
            "COPY model_predictions (employee_id, model_version, risk_score, risk_level) "
            "FROM STDIN WITH (FORMAT csv)",
            buffer,
        )
        raw_conn.commit()
    finally:
        raw_conn.close()

    return len(employee_ids)


def main() -> None:
    engine = get_engine()

    bundle_path = MODELS_DIR / "winner.joblib"
    if not bundle_path.exists():
        raise SystemExit("models/winner.joblib not found -- run `python ml/train.py` first.")
    bundle = joblib.load(bundle_path)
    model_version = bundle["model_version"]
    print(f"Scoring with {bundle['algorithm']} ({model_version}), "
          f"{bundle['calibration_method']}-calibrated")

    df = load_clean_data()
    X, _, _ = build_features(df, save_spec=False)
    X = X[bundle["feature_names"]]  # exact training column order
    print(f"  {len(df):,} employees, {X.shape[1]} features")

    # Score in chunks so peak memory stays predictable on a laptop.
    scores = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), SCORE_CHUNK):
        end = min(start + SCORE_CHUNK, len(X))
        scores[start:end] = bundle["model"].predict_proba(X.iloc[start:end])[:, 1]
    tiers = assign_tier(scores)

    written = write_predictions(engine, df["employee_id"].to_numpy(), scores, tiers, model_version)
    print(f"\nWrote {written:,} rows into model_predictions.")

    # ---- Report what the tiers actually look like ---------------------------
    summary = pd.Series(tiers).value_counts().reindex(["high", "medium", "low"], fill_value=0)
    print(f"\n  Risk tiers (high >= {HIGH_RISK_THRESHOLD}, "
          f"medium >= {MEDIUM_RISK_THRESHOLD}, else low):")
    for tier, count in summary.items():
        print(f"    {tier:<7} {count:>9,}  ({100 * count / written:5.2f}%)")

    print(f"\n  Risk score distribution:")
    for label, value in [
        ("min", scores.min()), ("p25", np.percentile(scores, 25)),
        ("median", np.median(scores)), ("p75", np.percentile(scores, 75)),
        ("max", scores.max()), ("mean", scores.mean()),
    ]:
        print(f"    {label:<7} {value:.4f}")

    if summary["medium"] / written < 0.05:
        print(
            "\n  NOTE: the medium tier is nearly empty. That is this dataset's doing, not the\n"
            "  pipeline's -- burnout_risk is an almost deterministic function of two feature\n"
            "  columns, so a well-fitted model is genuinely near-certain about most employees.\n"
            "  See the 'Honest reading of the results' section of the README."
        )


if __name__ == "__main__":
    main()
