"""
Step 10 -- Run the whole thing with one command.

    python etl/run_pipeline.py

That is the entire manual procedure. It applies the schema, loads the raw file, cleans it,
trains and compares three models, trains the fallback, scores every employee and writes the
SHAP explanations -- in that order, stopping at the first failure.

Useful flags while developing:

    --skip-load     reuse the raw table instead of re-reading the 1.3 GB file (saves ~40s)
    --skip-train    reuse the existing models in models/ and only re-score
    --only clean    run a single stage by name

Each stage prints a banner and its elapsed time, and the run ends with a summary table, so
a failure is easy to locate.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db import apply_schema, get_engine


def stage_schema():
    """Create the model tables if they are not there yet. Safe to repeat."""
    apply_schema()


def stage_load():
    from etl import load_raw
    load_raw.main()


def stage_clean():
    from etl import clean
    clean.main()


def stage_train():
    from ml import train
    train.main()


def stage_fallback():
    from ml import fallback_model
    fallback_model.main()


def stage_predict():
    from ml import predict
    predict.main()


def stage_explain():
    from ml import explain
    explain.main()


# Order matters -- each stage depends on the ones above it.
STAGES = [
    ("schema", stage_schema, "Create model tables"),
    ("load", stage_load, "Load the raw JSON into raw_employee_data"),
    ("clean", stage_clean, "Clean into stg_employee_clean and assert 0 NULLs"),
    ("train", stage_train, "Train and compare LogisticRegression / RandomForest / XGBoost"),
    ("fallback", stage_fallback, "Train the reduced-feature fallback model"),
    ("predict", stage_predict, "Score every employee into model_predictions"),
    ("explain", stage_explain, "Write top-3 SHAP factors into shap_explanations"),
]


def print_banner(index: int, total: int, name: str, description: str) -> None:
    print(f"\n{'#' * 78}")
    print(f"#  [{index}/{total}]  {name.upper()} -- {description}")
    print("#" * 78, flush=True)


def final_report(engine) -> None:
    """What actually ended up in the database, so success is visible not assumed."""
    queries = {
        "raw_employee_data": "SELECT COUNT(*) FROM raw_employee_data",
        "stg_employee_clean": "SELECT COUNT(*) FROM stg_employee_clean",
        "model_metadata": "SELECT COUNT(*) FROM model_metadata",
        "model_predictions": "SELECT COUNT(*) FROM model_predictions",
        "shap_explanations": "SELECT COUNT(*) FROM shap_explanations",
    }
    print(f"\n{'=' * 78}")
    print("  DATABASE CONTENTS")
    print("=" * 78)
    with engine.connect() as conn:
        for table, query in queries.items():
            try:
                count = conn.execute(text(query)).scalar_one()
                print(f"  {table:<22} {count:>12,} rows")
            except Exception:
                print(f"  {table:<22} {'missing':>12}")

        winner = conn.execute(text("""
            SELECT model_version, algorithm, recall, precision, f1, roc_auc
            FROM model_metadata WHERE is_winner ORDER BY trained_at DESC LIMIT 1
        """)).mappings().first()
        if winner:
            print(f"\n  Deployed model: {winner['algorithm']} ({winner['model_version']})")
            print(f"    recall {winner['recall']:.4f} · precision {winner['precision']:.4f} · "
                  f"F1 {winner['f1']:.4f} · ROC-AUC {winner['roc_auc']:.4f}")

    print("\n  Next step:  streamlit run dashboard/app.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the burnout prediction pipeline.")
    parser.add_argument("--skip-load", action="store_true",
                        help="reuse raw_employee_data instead of re-reading the source file")
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse the models in models/ and only re-score")
    parser.add_argument("--only", metavar="STAGE",
                        help=f"run one stage: {', '.join(name for name, _, _ in STAGES)}")
    args = parser.parse_args()

    skipped = set()
    if args.skip_load:
        skipped.add("load")
    if args.skip_train:
        skipped |= {"train", "fallback"}

    if args.only:
        if args.only not in {name for name, _, _ in STAGES}:
            parser.error(f"unknown stage '{args.only}'")
        selected = [s for s in STAGES if s[0] == args.only]
    else:
        selected = [s for s in STAGES if s[0] not in skipped]

    print("Burnout prediction pipeline")
    print(f"  stages: {' -> '.join(name for name, _, _ in selected)}")

    timings = []
    started_all = time.time()

    for index, (name, function, description) in enumerate(selected, start=1):
        print_banner(index, len(selected), name, description)
        started = time.time()
        try:
            function()
        except Exception:
            elapsed = time.time() - started
            print(f"\n  STAGE '{name}' FAILED after {elapsed:.1f}s\n")
            traceback.print_exc()
            print(f"\nPipeline stopped at stage '{name}'. Nothing after it has run.")
            return 1
        elapsed = time.time() - started
        timings.append((name, elapsed))
        print(f"\n  [{name}] finished in {elapsed:.1f}s", flush=True)

    print(f"\n{'=' * 78}")
    print("  PIPELINE COMPLETE")
    print("=" * 78)
    for name, elapsed in timings:
        print(f"  {name:<12} {elapsed:>8.1f}s")
    print(f"  {'TOTAL':<12} {time.time() - started_all:>8.1f}s")

    final_report(get_engine())
    return 0


if __name__ == "__main__":
    sys.exit(main())
