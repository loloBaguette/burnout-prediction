"""
Step 6 -- The fallback model.

Why this exists: the main model uses 90 features, several of which come from systems not
every organisation runs -- sentiment scoring on email, collaboration analytics, a maintained
skills inventory. If any of those feeds is missing or broken, the main model cannot score
anyone at all.

The fallback is trained on the columns you would expect *any* HR system to have, even one
that is really just a spreadsheet:

    tenure_months, salary, overtime_hours, performance_score, job_level, department

It is deliberately weaker. Its job is to keep the pipeline producing a usable risk score
when the richer inputs are unavailable, not to compete with the main model. The comparison
printed at the end shows exactly how much accuracy you give up by falling back -- which is
the number a manager needs in order to decide whether a fallback score is worth acting on.

Run it:  python ml/fallback_model.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from db import get_engine, REPO_ROOT
from ml import evaluate
from ml.features import (
    FALLBACK_FEATURES,
    RANDOM_STATE,
    build_features,
    load_clean_data,
    stratified_split,
)
from ml.train import (
    CALIBRATION_FRACTION,
    MODELS_DIR,
    graded_share,
    retire_previous_models,
    save_metadata,
)


def main() -> None:
    engine = get_engine()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("Fallback model -- reduced 'always available' feature subset")
    print(f"  source columns: {', '.join(FALLBACK_FEATURES)}\n")

    df = load_clean_data()

    # save_spec=False so this does NOT overwrite models/feature_spec.json, which belongs
    # to the main model. The fallback keeps its own spec inside its own .joblib bundle.
    X, y, spec = build_features(df, feature_subset=FALLBACK_FEATURES, save_spec=False)
    print(f"  feature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns "
          f"(main model uses 90)")

    # Same split settings as the main model, so the two are compared on identical test rows.
    X_train_full, X_test, y_train_full, y_test = stratified_split(X, y)
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train_full, y_train_full,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    scale_pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))

    # "Simpler" means both fewer inputs and a smaller model: half the trees of the main
    # XGBoost and shallower ones, which is all a six-column problem warrants.
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining...")
    model.fit(X_fit, y_fit)

    # Calibrated the same way as the main model so the two risk scores mean the same thing
    # and can be shown in the same dashboard column.
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_calib, y_calib)

    proba = calibrated.predict_proba(X_test)[:, 1]
    result = evaluate.evaluate_model("Fallback (6 source columns)", y_test, proba)
    evaluate.print_report(result)
    print(f"\n  Brier score: {brier_score_loss(y_test, proba):.5f}   "
          f"graded scores: {graded_share(proba):.1f}%")

    # ---- How much do you lose by falling back? -------------------------------
    main_bundle_path = MODELS_DIR / "winner.joblib"
    if main_bundle_path.exists():
        bundle = joblib.load(main_bundle_path)
        X_main, _, _ = build_features(df, save_spec=False)
        _, X_main_test, _, y_main_test = stratified_split(X_main, y)
        main_proba = bundle["model"].predict_proba(X_main_test[bundle["feature_names"]])[:, 1]
        main_result = evaluate.evaluate_model(bundle["algorithm"], y_main_test, main_proba)

        print(f"\n{'=' * 72}")
        print("  COST OF FALLING BACK (test set, both at recall >= 90%)")
        print("=" * 72)
        print(f"  {'model':<30} {'features':>9} {'recall':>8} {'precision':>10} {'ROC-AUC':>9}")
        print("  " + "-" * 68)
        for label, res, n_feat in [
            (f"main ({bundle['algorithm']})", main_result, len(bundle["feature_names"])),
            ("fallback", result, X.shape[1]),
        ]:
            op = res["operating"]
            print(f"  {label:<30} {n_feat:>9} {op['recall']:>8.4f} "
                  f"{op['precision']:>10.4f} {op['roc_auc']:>9.4f}")
        delta = main_result["operating"]["roc_auc"] - result["operating"]["roc_auc"]
        print(f"\n  ROC-AUC given up by falling back: {delta:.4f}")

    # ---- Persist -------------------------------------------------------------
    model_version = f"fallback_xgboost_{run_stamp}"
    joblib.dump(
        {
            "model": calibrated,
            "base_model": model,
            "algorithm": "XGBoost (fallback)",
            "model_version": model_version,
            "feature_names": list(X.columns),
            "feature_spec": spec,
            "source_columns": FALLBACK_FEATURES,
            "operating_threshold": result["operating"]["threshold"],
            "calibration_method": "isotonic",
        },
        MODELS_DIR / "fallback.joblib",
    )
    print(f"\nSaved models/fallback.joblib  (version {model_version})")

    # Clears only previous *fallback* rows, leaving the main model untouched.
    retire_previous_models(engine, keep_fallback=False)

    save_metadata(
        engine,
        model_version,
        "XGBoost (fallback)",
        result["operating"],
        is_winner=False,
        is_fallback=True,
        notes=(
            f"reduced subset: {', '.join(FALLBACK_FEATURES)}; "
            f"{X.shape[1]} encoded features; isotonic-calibrated"
        ),
    )
    print("Wrote 1 row into model_metadata (is_fallback = true).")


if __name__ == "__main__":
    main()
