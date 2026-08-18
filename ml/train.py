"""
Step 5 -- Train Logistic Regression, Random Forest and XGBoost; compare; save the best.

Flow:
  1. 80/20 stratified split (from ml/features.py).
  2. The 80% training half is split again into 85% fit / 15% calibration. The calibration
     slice is held back so probabilities can be calibrated on data the model has never fitted.
  3. All three algorithms are trained on the fit slice and scored on the untouched test set.
  4. The winner is calibrated (isotonic regression) so that a score of 0.8 really does mean
     "about 80% of employees who score this actually are high-risk" -- required because the
     dashboard presents the number as a probability.
  5. Two files land in models/: the calibrated model (used for scoring) and the bare model
     (used by ml/explain.py, because SHAP's TreeExplainer needs the trees themselves, not a
     calibration wrapper around them).
  6. Metrics for all three models go into the model_metadata table.

Run it:  python ml/train.py
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from xgboost import XGBClassifier

from db import get_engine, REPO_ROOT
from ml import evaluate
from ml.features import RANDOM_STATE, get_training_data

MODELS_DIR = REPO_ROOT / "models"
CALIBRATION_FRACTION = 0.15

# A calibration method has to leave at least this share of employees with a score that is
# neither ~0 nor ~1, otherwise the dashboard's low/medium/high tiers degenerate into two tiers.
MIN_GRADED_SHARE = 5.0


def graded_share(proba) -> float:
    """Percentage of scores strictly inside (0.01, 0.99) -- see MIN_GRADED_SHARE."""
    return float(100 * np.mean((proba > 0.01) & (proba < 0.99)))


def build_models(scale_pos_weight: float) -> dict:
    """
    The three candidates.

    `scale_pos_weight` / `class_weight` tell each algorithm how to trade the two classes off.
    Here the positive class is actually the *majority* (59%), so these weights end up slightly
    favouring the negative class -- the opposite of the usual imbalance case, but the same
    mechanism.
    """
    return {
        # Scaling matters for logistic regression (salary is ~10^5, scores are ~10^0), so it
        # is wrapped in a Pipeline. The tree models below are scale-invariant and need none.
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,          # capped: 680k rows would otherwise grow enormous trees
            min_samples_leaf=20,   # keeps leaves statistically meaningful and training fast
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",    # the fast histogram algorithm; matters at this row count
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def retire_previous_models(engine, keep_fallback: bool = True) -> None:
    """
    Remove the previous run's models before recording this one.

    Two reasons. First, `is_winner` has to identify exactly one model -- the dashboard and
    the tests both assume that, and without this a second training run would leave two rows
    flagged. Second, model_predictions and shap_explanations cascade off model_metadata, so
    without a cleanup every run would leave another 850k predictions and 2.5M explanation
    rows behind for a model nobody is using.

    Fallback rows are left alone by default: ml/fallback_model.py owns those and clears its
    own, so running train.py on its own does not delete a working fallback model.
    """
    with engine.begin() as conn:
        condition = "NOT is_fallback" if keep_fallback else "is_fallback"
        deleted = conn.execute(
            text(f"DELETE FROM model_metadata WHERE {condition}")
        ).rowcount
    if deleted:
        kind = "previous" if keep_fallback else "previous fallback"
        print(f"  Retired {deleted} {kind} model version(s) "
              f"(their predictions and explanations cascaded away).")


def save_metadata(engine, model_version, algorithm, metrics, is_winner, is_fallback, notes):
    """Write one row into model_metadata. Re-runnable: upserts on model_version."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO model_metadata
                    (model_version, algorithm, recall, precision, f1, roc_auc,
                     is_winner, is_fallback, notes, trained_at)
                VALUES
                    (:v, :a, :r, :p, :f, :auc, :w, :fb, :n, NOW())
                ON CONFLICT (model_version) DO UPDATE SET
                    algorithm = EXCLUDED.algorithm, recall = EXCLUDED.recall,
                    precision = EXCLUDED.precision, f1 = EXCLUDED.f1,
                    roc_auc = EXCLUDED.roc_auc, is_winner = EXCLUDED.is_winner,
                    is_fallback = EXCLUDED.is_fallback, notes = EXCLUDED.notes,
                    trained_at = NOW()
            """),
            {
                "v": model_version, "a": algorithm,
                "r": metrics["recall"], "p": metrics["precision"], "f": metrics["f1"],
                "auc": metrics["roc_auc"], "w": is_winner, "fb": is_fallback, "n": notes,
            },
        )


def main() -> None:
    engine = get_engine()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("Loading features...")
    _, _, _, X_train_full, X_test, y_train_full, y_test, spec = get_training_data()
    print(f"  train {X_train_full.shape}, test {X_test.shape}")

    # Hold back a calibration slice from the training half only. The test set stays untouched.
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train_full, y_train_full,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    print(f"  fit {X_fit.shape}, calibration {X_calib.shape}")

    # Ratio of negatives to positives -- what XGBoost's scale_pos_weight expects.
    scale_pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    print(f"  scale_pos_weight = {scale_pos_weight:.4f}\n")

    results, fitted = [], {}
    for name, model in build_models(scale_pos_weight).items():
        print(f"Training {name}...", flush=True)
        started = time.time()
        model.fit(X_fit, y_fit)
        elapsed = time.time() - started

        proba = model.predict_proba(X_test)[:, 1]
        result = evaluate.evaluate_model(name, y_test, proba)
        result["train_seconds"] = elapsed
        results.append(result)
        fitted[name] = model

        print(f"  trained in {elapsed:.1f}s")
        evaluate.print_report(result)

    baseline = evaluate.trivial_baseline(y_test)
    evaluate.print_comparison(results, baseline)

    # ---- Pick and calibrate the winner --------------------------------------
    winner = evaluate.pick_winner(results)
    winner_name = winner["name"]
    winner_model = fitted[winner_name]
    print(f"\nWinner: {winner_name}")

    # Two calibration methods, because they fail in different ways on a target this
    # separable. Isotonic is a step function: it usually wins on Brier score, but on a
    # near-perfectly-separable problem it collapses almost every employee onto exactly 0.0
    # or exactly 1.0, which leaves the dashboard with no "medium" tier to show. Sigmoid
    # (Platt scaling) is a smooth curve, so it keeps the score graded. We fit both, print
    # both, and choose by Brier score *among the methods that keep the score usable*.
    print("Calibrating probabilities on the held-back calibration slice...")
    raw_proba = winner_model.predict_proba(X_test)[:, 1]
    print(f"  uncalibrated       Brier {brier_score_loss(y_test, raw_proba):.5f}  "
          f"graded {graded_share(raw_proba):5.1f}%  distinct {np.unique(raw_proba).size:,}")

    candidates = {}
    for method in ("isotonic", "sigmoid"):
        calibrator = CalibratedClassifierCV(winner_model, method=method, cv="prefit")
        calibrator.fit(X_calib, y_calib)
        proba = calibrator.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, proba)
        graded = graded_share(proba)
        candidates[method] = {"model": calibrator, "proba": proba, "brier": brier, "graded": graded}
        print(f"  {method:<18} Brier {brier:.5f}  graded {graded:5.1f}%  "
              f"distinct {np.unique(proba).size:,}")

    usable = {m: c for m, c in candidates.items() if c["graded"] >= MIN_GRADED_SHARE}
    if usable:
        chosen_method = min(usable, key=lambda m: usable[m]["brier"])
    else:
        # Nothing keeps the score graded -- take the best-calibrated one and say so plainly.
        chosen_method = min(candidates, key=lambda m: candidates[m]["brier"])
        print(f"\n  WARNING: no calibration method keeps more than {MIN_GRADED_SHARE}% of scores "
              f"off the 0/1 endpoints.\n  The dashboard's medium tier will be nearly empty. "
              f"This is a property of the dataset, not a bug -- see the README.")

    calibrated = candidates[chosen_method]["model"]
    cal_proba = candidates[chosen_method]["proba"]
    print(f"\n  Chosen: {chosen_method} calibration "
          f"(Brier {candidates[chosen_method]['brier']:.5f}, "
          f"{candidates[chosen_method]['graded']:.1f}% of scores graded)")
    print("  Calibration is what makes the dashboard's 0-1 number a real probability rather")
    print("  than just a ranking score.")

    calibrated_result = evaluate.evaluate_model(f"{winner_name} (calibrated)", y_test, cal_proba)
    evaluate.print_report(calibrated_result)

    # ---- Persist ------------------------------------------------------------
    model_version = f"{winner_name.lower()}_{run_stamp}"

    joblib.dump(
        {
            "model": calibrated,
            "base_model": winner_model,   # unwrapped, for SHAP
            "algorithm": winner_name,
            "model_version": model_version,
            "feature_names": list(X_test.columns),
            "feature_spec": spec,
            "operating_threshold": calibrated_result["operating"]["threshold"],
            "calibration_method": chosen_method,
        },
        MODELS_DIR / "winner.joblib",
    )
    print(f"\nSaved models/winner.joblib  (version {model_version})")

    retire_previous_models(engine, keep_fallback=True)

    for result in results:
        is_winner = result["name"] == winner_name
        version = model_version if is_winner else f"{result['name'].lower()}_{run_stamp}"
        notes = (
            f"{X_test.shape[1]} features; operating threshold "
            f"{result['operating']['threshold']:.4f} tuned for recall>=90%; "
            f"trained in {result['train_seconds']:.0f}s"
        )
        if is_winner:
            notes += f"; {chosen_method}-calibrated"
        save_metadata(
            engine,
            version,
            result["name"],
            calibrated_result["operating"] if is_winner else result["operating"],
            is_winner=is_winner,
            is_fallback=False,
            notes=notes,
        )
    print(f"Wrote {len(results)} rows into model_metadata (is_winner = {model_version}).")


if __name__ == "__main__":
    main()
