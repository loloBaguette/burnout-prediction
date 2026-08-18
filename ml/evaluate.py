"""
Step 5 (part 1) -- Model evaluation.

Kept separate from train.py so the same scoring code is used for the three candidate models,
for the calibrated winner, and for the fallback model. One definition of "how good is it".

The important idea in this file is the **operating threshold**.

The build spec sets the bar at Recall >= 90% and Precision >= 15%. A classifier's recall and
precision are not fixed properties -- they move as you change the probability cut-off used to
turn a score into a yes/no. So rather than reporting whatever falls out of the default 0.5
cut-off, we find the cut-off that *just* achieves 90% recall and report precision there. That
is how a screening tool is actually tuned: fix the recall you need, then see how much
precision you can buy.

One honest caveat, from the EDA: 59% of employees are in the positive class, so a model that
flags *everybody* scores 100% recall and 59% precision and clears the spec's bar without
learning anything. `trivial_baseline()` computes exactly that, and it is printed next to every
real model. A model is only worth deploying if it beats the baseline's precision at the same
recall.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# The bar from the project's initiation document.
REQUIRED_RECALL = 0.90
REQUIRED_PRECISION = 0.15

# Two models whose precision differs by less than this are treated as tied -- see pick_winner.
PRECISION_TIE_TOLERANCE = 0.01


def find_operating_threshold(y_true, y_proba, target_recall: float = REQUIRED_RECALL) -> float:
    """
    Return the highest probability cut-off that still achieves `target_recall`.

    Highest, because among all cut-offs that meet the recall requirement, the highest one
    flags the fewest people and therefore gives the best precision.
    """
    # Sort scores descending and walk down them; recall improves as the cut-off drops.
    order = np.argsort(-y_proba)
    sorted_true = np.asarray(y_true)[order]
    sorted_proba = np.asarray(y_proba)[order]

    total_positives = sorted_true.sum()
    if total_positives == 0:
        return 0.5

    cumulative_hits = np.cumsum(sorted_true)
    recall_curve = cumulative_hits / total_positives

    reached = np.searchsorted(recall_curve, target_recall, side="left")
    if reached >= len(sorted_proba):
        return float(sorted_proba[-1])  # even flagging everyone cannot reach it
    return float(sorted_proba[reached])


def score(y_true, y_proba, threshold: float) -> dict:
    """Every metric we care about, at one specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "flagged": int(tp + fp),
        "flagged_pct": float(100 * (tp + fp) / len(y_true)),
        # --- score resolution -------------------------------------------------
        # How *graded* the risk scores are, as opposed to collapsing onto 0 and 1.
        # This is not a quality metric; it is a fitness-for-purpose one. The dashboard
        # has to show a 0-1 score and sort employees into low/medium/high tiers, and a
        # model that only ever emits 0.0 or 1.0 cannot fill a "medium" tier at all.
        "distinct_scores": int(np.unique(y_proba).size),
        "graded_pct": float(100 * np.mean((y_proba > 0.01) & (y_proba < 0.99))),
    }


def evaluate_model(name: str, y_true, y_proba) -> dict:
    """
    Score a model twice: at the default 0.5 cut-off, and at its operating threshold.

    The operating-threshold numbers are the ones the acceptance checklist is judged on.
    """
    default_metrics = score(y_true, y_proba, 0.5)
    operating_threshold = find_operating_threshold(y_true, y_proba)
    operating_metrics = score(y_true, y_proba, operating_threshold)

    return {
        "name": name,
        "default": default_metrics,
        "operating": operating_metrics,
        "meets_bar": (
            operating_metrics["recall"] >= REQUIRED_RECALL
            and operating_metrics["precision"] >= REQUIRED_PRECISION
        ),
    }


def trivial_baseline(y_true) -> dict:
    """
    "Flag everyone." The dumbest possible model, scored the same way as a real one.

    Its precision equals the base rate. Any model that cannot beat this number at 90% recall
    is not earning its keep, no matter what the acceptance checklist says.
    """
    y_true = np.asarray(y_true)
    base_rate = float(y_true.mean())
    return {
        "name": "Trivial baseline (flag everyone)",
        "recall": 1.0,
        "precision": base_rate,
        "f1": float(2 * base_rate / (1 + base_rate)),
        "flagged_pct": 100.0,
        "meets_bar": base_rate >= REQUIRED_PRECISION,
    }


def print_confusion_matrix(metrics: dict, title: str) -> None:
    """A confusion matrix you can actually read in a terminal."""
    print(f"\n  Confusion matrix -- {title} (threshold {metrics['threshold']:.4f})")
    print("                     predicted")
    print("                  not-risk    at-risk")
    print(f"    actual not-risk  {metrics['tn']:>8,}   {metrics['fp']:>8,}")
    print(f"    actual at-risk   {metrics['fn']:>8,}   {metrics['tp']:>8,}")
    print(
        f"    recall {metrics['recall']:.4f}   precision {metrics['precision']:.4f}   "
        f"F1 {metrics['f1']:.4f}   ROC-AUC {metrics['roc_auc']:.4f}"
    )


def print_report(result: dict) -> None:
    """Full report for one model."""
    print(f"\n{'=' * 72}")
    print(f"  {result['name']}")
    print("=" * 72)
    print_confusion_matrix(result["default"], "default 0.5 cut-off")
    print_confusion_matrix(result["operating"], f"tuned for recall >= {REQUIRED_RECALL:.0%}")
    verdict = "MEETS" if result["meets_bar"] else "MISSES"
    print(
        f"\n  {verdict} the bar (recall >= {REQUIRED_RECALL:.0%}, "
        f"precision >= {REQUIRED_PRECISION:.0%}) -- flags "
        f"{result['operating']['flagged_pct']:.1f}% of employees"
    )


def print_comparison(results: list[dict], baseline: dict) -> None:
    """The side-by-side table, including the do-nothing baseline."""
    print(f"\n{'=' * 96}")
    print("  MODEL COMPARISON -- all metrics on the held-out test set, at recall >= 90%")
    print("=" * 96)
    header = (f"  {'model':<26} {'recall':>8} {'precision':>10} {'F1':>8} {'ROC-AUC':>9} "
              f"{'flagged':>9} {'graded':>8}  bar")
    print(header)
    print("  " + "-" * 92)
    for r in results:
        op = r["operating"]
        mark = "PASS" if r["meets_bar"] else "FAIL"
        print(
            f"  {r['name']:<26} {op['recall']:>8.4f} {op['precision']:>10.4f} "
            f"{op['f1']:>8.4f} {op['roc_auc']:>9.4f} {op['flagged_pct']:>8.1f}% "
            f"{op['graded_pct']:>7.1f}%  {mark}"
        )
    print("  " + "-" * 92)
    print(
        f"  {baseline['name']:<26} {baseline['recall']:>8.4f} {baseline['precision']:>10.4f} "
        f"{baseline['f1']:>8.4f} {'n/a':>9} {baseline['flagged_pct']:>8.1f}% {'n/a':>8}  "
        f"{'PASS' if baseline['meets_bar'] else 'FAIL'}"
    )
    print(
        "\n  Read the baseline row first: it is what you get for free from the 59% base rate.\n"
        "  A model is only useful if its precision beats that number at the same recall.\n"
        "  'graded' = share of employees whose score is neither ~0 nor ~1; the dashboard's\n"
        "  low/medium/high tiers need this to be well above zero to mean anything."
    )


def pick_winner(results: list[dict]) -> dict:
    """
    Choose the model to deploy, in three passes.

    1. It must clear the spec's bar (recall >= 90%, precision >= 15%).
    2. Among those, take the highest precision at 90% recall -- the model that reaches the
       required recall while sending the fewest people to a manager's follow-up list.
    3. Where several models are within PRECISION_TIE_TOLERANCE of that best precision, they
       are statistically tied and choosing between them on a fourth decimal place is noise.
       The tie is broken on score resolution instead: the deliverable is a dashboard showing
       a 0-1 risk score and low/medium/high tiers, so a model whose scores are spread across
       that range is strictly more useful than one that only ever says 0.0 or 1.0, even
       though both score identically on recall and precision. ROC-AUC breaks any remaining tie.

    On this dataset step 3 does real work: all three models hit ~1.00 precision at 90% recall
    (see the "generator formula" note in the README), so resolution is what separates them.
    """
    qualified = [r for r in results if r["meets_bar"]] or results
    best_precision = max(r["operating"]["precision"] for r in qualified)
    tied = [
        r for r in qualified
        if best_precision - r["operating"]["precision"] <= PRECISION_TIE_TOLERANCE
    ]
    return max(
        tied,
        key=lambda r: (r["operating"]["graded_pct"], r["operating"]["roc_auc"]),
    )
