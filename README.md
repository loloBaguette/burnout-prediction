# Employee Burnout Prediction

An end-to-end pipeline that loads an HR dataset into Postgres, cleans it, trains a
burnout-risk classifier, explains every prediction with SHAP, and serves the results in a
Streamlit dashboard.

One command runs the whole thing:

```bash
python etl/run_pipeline.py
```

---

## Quick start

You need **Docker Desktop** (running) and **Python 3.11+**. Four commands from a fresh clone:

```bash
docker compose up -d --wait
```

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Put the dataset at `data/raw/synthetic-employee-dataset.json` (or point `RAW_DATA_PATH` in
`.env` somewhere else), then:

```bash
.venv/bin/python etl/run_pipeline.py
```

That takes about **six minutes** end to end. When it finishes:

```bash
.venv/bin/streamlit run dashboard/app.py
```

To check everything landed correctly:

```bash
.venv/bin/pytest tests/ -v
```

---

## What each piece does

| File | What it does |
|---|---|
| `docker-compose.yml` | One Postgres 16 service, so all three of us develop against an identical DB |
| `db/schema.sql` | The three model tables. The two data tables are created from the real file, not from guessed column names |
| `db.py` | Shared connection helper — one place that knows how to reach Postgres |
| `etl/load_raw.py` | Streams the 1.3 GB JSON into `raw_employee_data`, as-is |
| `etl/clean.py` | `raw` → `stg_employee_clean`, with 0 NULLs enforced by database constraints |
| `notebooks/01_eda.ipynb` | Profiles all 32 columns. **Read this before changing anything downstream** |
| `ml/features.py` | Encoding, the leakage exclusion list, and the stratified 80/20 split |
| `ml/train.py` | Trains and compares 3 algorithms, calibrates the winner |
| `ml/evaluate.py` | Confusion matrix, recall/precision/F1, and the trivial baseline |
| `ml/fallback_model.py` | A second model on 6 "any HR system has this" columns |
| `ml/explain.py` | SHAP `TreeExplainer` → top-3 factors per employee |
| `ml/predict.py` | Calibrated risk score + low/medium/high tier for all 850k employees |
| `etl/run_pipeline.py` | Chains all of the above |
| `dashboard/app.py` | Streamlit dashboard reading straight from Postgres |
| `tests/test_pipeline.py` | Smoke tests against the acceptance checklist |

Handy flags while developing:

```bash
.venv/bin/python etl/run_pipeline.py --skip-load --skip-train
```

---

## The data

850,000 employees, 31 fields, delivered as a 1.3 GB JSON array (one object per line) rather
than the CSV the spec anticipated. `load_raw.py` streams it, so it never needs more than a
few hundred MB of RAM.

**What the EDA found** — all four of these are documented with evidence in
`notebooks/01_eda.ipynb`:

1. **No NULLs anywhere** — but `role` is whitespace-only in 109,047 rows and holds the
   literal string `'nan'` in 7,481 more. Those pass every `IS NULL` check while being just as
   useless. They are folded into an explicit `Unknown` category rather than deleted, which
   keeps 13% of the dataset that a delete-the-nulls approach would have thrown away.
2. **Five columns are really three.** `collaboration_score`, `slack_activity` and
   `meeting_participation` are byte-identical across all 849,999 rows, as are
   `performance_score` and `goal_achievement_rate`. The copies are dropped.
3. **No ready-made label.** The dataset has a continuous `burnout_risk` (0–1), not a class.
   `risk_factors_summary` turns out to be exactly `burnout_risk` cut at **0.75**, so we adopt
   the dataset's own definition: `is_high_burnout_risk = burnout_risk >= 0.75`.
4. **Eight columns leak the answer** and are excluded from the feature set — most sharply
   `stress_level`, which correlates 0.994 with the target and is *exactly equal* to it in 46%
   of rows. The full list with reasons is at the top of `ml/features.py`.

---

## Honest reading of the results

The pipeline meets every item on the acceptance checklist. Two of those numbers should not
be presented to anyone without the context below.

**The winning model scores ROC-AUC 0.9999. That is not a good result; it is a warning.**

`burnout_risk` in this synthetic dataset is very close to a fixed arithmetic formula of two
columns that are *in* the feature set:

```
burnout_risk  ≈  0.99 + 4.73 × performance_score − 5.93 × project_completion_rate
                 (R² = 0.989, clipped to [0, 1])
```

The model is not discovering a subtle pattern in human behaviour. It is recovering the
formula that generated the data, which any competent model will do almost perfectly. SHAP
agrees: `project_completion_rate` and `performance_score` are top-3 factors for **91.5%** and
**89.9%** of the workforce respectively — the same two columns the formula uses.

We did **not** drop those two features to make the numbers look more realistic. They are
legitimate HR measurements that a real deployment would have, and removing them would be
crippling the model to flatter a metric. But nobody should quote 0.9999 as evidence this
approach works on real employees.

The **fallback model** is the useful reference point here. It is trained on six ordinary HR
columns and excludes `project_completion_rate`, so it cannot recover the formula:

| Model | Features | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| Main (XGBoost) | 90 | 0.9494 | 1.0000 | 0.9999 |
| Fallback | 47 | 0.9170 | 0.6195 | 0.7125 |
| Flag everyone | — | 1.0000 | 0.5898 | — |

ROC-AUC 0.71 and precision barely above the base rate is what this problem actually looks
like without the generator's formula in the inputs. **Expect numbers like the fallback row,
not the main row, on real data.**

**The second caveat: the acceptance bar is nearly free on this dataset.** 59% of employees
are in the positive class, so "flag everyone" scores 100% recall and 59% precision and
clears the spec's Recall ≥ 90% / Precision ≥ 15% bar without learning anything. That is the
bottom row of the table above, and `ml/evaluate.py` prints it next to every model on purpose.
If this bar came from the initiation document, it is worth renegotiating with whoever set it:
it was written for a rare positive class and this target is a majority one.

**A consequence you will see in the dashboard:** because the target is nearly deterministic,
a well-calibrated model is genuinely near-certain about almost everyone. 57.4% of employees
score above 0.99 and 39.7% score below 0.01, leaving the *medium* tier with just **0.28%** of
the workforce (2,392 people out of 850,000). That is the dataset's doing, not a bug — but it does mean the tiering is close to
binary in practice.

---

## How the modelling works

**Split.** Stratified 80/20 (679,999 train / 170,000 test), `random_state=42` everywhere so
we all reproduce the same split. The training half is split again, 85/15, to hold back a
calibration slice the model never fits on.

**Comparison.** Logistic Regression, Random Forest and XGBoost, all scored on the same
untouched test set.

**The operating threshold.** Recall and precision are not fixed properties of a model — they
move with the probability cut-off. Rather than reporting whatever falls out of a default 0.5,
`evaluate.py` finds the highest cut-off that still achieves 90% recall and reports precision
there. That is how a screening tool is actually tuned: fix the recall you need, then see how
much precision you can buy.

**Picking the winner.** Highest precision at 90% recall. On this dataset all three models
land within a rounding error of each other, so a tie-break matters: among models within 1
percentage point of the best precision, we take the one whose scores are most *graded*
(spread across 0–1 rather than piled on the endpoints), because the deliverable is a
dashboard with low/medium/high tiers and a model that only emits 0.0 and 1.0 cannot fill a
medium tier. XGBoost wins on that basis.

**Calibration.** Both isotonic and Platt scaling are fitted and compared on Brier score, so
the dashboard's 0–1 number is a real probability rather than just a ranking. On this dataset
neither keeps the scores meaningfully graded and the pipeline says so out loud when it runs.

**Explanations.** SHAP `TreeExplainer` on the *uncalibrated* model — calibration is a
monotone transform, so it cannot change which factors rank top-3, and TreeExplainer needs the
trees themselves. Ranking is by absolute contribution but the stored value keeps its sign, so
the dashboard can say "high satisfaction is *lowering* this person's risk" rather than
implying every listed factor is bad news.

---

## Database tables

| Table | Rows | Contents |
|---|---|---|
| `raw_employee_data` | 849,999 | The source file, untouched, plus a surrogate `employee_id` |
| `stg_employee_clean` | 849,999 | Cleaned and typed; 0 NULLs, enforced by `NOT NULL`/`CHECK` |
| `model_metadata` | 4 | One row per model: recall, precision, F1, ROC-AUC, `is_winner`, `is_fallback` |
| `model_predictions` | 849,999 | Per employee: calibrated `risk_score` and `risk_level` |
| `shap_explanations` | 2,549,997 | Per employee: top-3 features, signed SHAP value, rank 1–3 |

Re-running the pipeline retires the previous model version, and its predictions and
explanations cascade away with it — so the database does not grow by 3.4M rows every run.

The source file's own `employee_id` (`SYN_00000123`) is kept only as `source_employee_id` for
traceability. It is never a primary key, never a feature, and never shown in the dashboard.

Risk tiers are cut on the calibrated probability: **high ≥ 0.70**, **medium ≥ 0.30**, **low**
below that.

---

## Notes for the team

- `.env` is gitignored; `.env.example` is committed. Never commit the first one.
- The dataset is gitignored too — 1.3 GB does not belong in git.
- **Read `notebooks/01_eda.ipynb` before changing `clean.py` or `features.py`.** Section 10
  lists every decision those two files implement and why.
- `LEAKAGE_COLUMNS` at the top of `ml/features.py` is not a style preference. Putting any of
  those columns back gives you a meaningless model.
- If you add a feature, add a plain-language label for it in `FEATURE_LABELS` in
  `dashboard/app.py`, or the dashboard will show the raw column name to a manager.

## Things we did not do

- **No hyperparameter tuning.** The three models use sensible fixed settings. On a target
  this separable, tuning would move nothing that matters.
- **No text modelling.** `recent_feedback` is scraped review prose with no per-employee
  meaning; it is reduced to its length and dropped. Real feedback text would be worth an NLP
  pass.
- **No fairness audit.** The model uses `department`, `role` and `job_level`, and salary is a
  feature. Before this went anywhere near a real HR decision it would need a bias review
  across protected groups — none of which are in this dataset.
