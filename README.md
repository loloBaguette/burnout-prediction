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

You need **Docker Desktop** (running) and **Python 3.11 or 3.12**.

> **Use 3.11 or 3.12, not 3.13.** `shap` 0.46 ships no pre-built wheel for 3.13, so pip would
> try to compile it from source and fail unless you have a full C++ toolchain installed. Every
> other dependency is fine on 3.13 — shap is the one that pins us.

Pick your platform below; everything after setup is identical.

<details open>
<summary><b>Windows</b> (PowerShell)</summary>

Install the two prerequisites first if you do not have them:

- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) — needs the
  **WSL 2** backend, which its installer offers to set up for you. Launch it and wait for the
  whale icon to stop animating before running anything below.
- [Python 3.11 or 3.12](https://www.python.org/downloads/windows/) — **tick "Add python.exe to
  PATH"** in the installer, or nothing below will be found. Do not use 3.13 (see the note
  above).

Then, from the repo folder:

```powershell
docker compose up -d --wait
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env
```

Put the dataset at `data\raw\synthetic-employee-dataset.json`, then:

```powershell
.venv\Scripts\python etl\run_pipeline.py
.venv\Scripts\streamlit run dashboard\app.py
```

Two things that trip people up on Windows:

- **`py -3.11` not `python3.11`.** The `py` launcher is what the Python installer puts on PATH.
  If `py -3.11` reports no such version, run `py -0` to list what you actually have.
  `py -3.12` works equally well; `py -3.13` does not (see the note above).
- **`.venv\Scripts\` not `.venv/bin/`.** This is the one difference that shows up in every
  command. If you would rather not type it each time, activate the environment once per
  terminal and drop the prefix entirely:

  ```powershell
  .venv\Scripts\Activate.ps1
  python etl\run_pipeline.py
  ```

  If PowerShell refuses with an execution-policy error, this allows local scripts for your
  user only:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```

</details>

<details open>
<summary><b>macOS / Linux</b></summary>

```bash
docker compose up -d --wait
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Put the dataset at `data/raw/synthetic-employee-dataset.json`, then:

```bash
.venv/bin/python etl/run_pipeline.py
.venv/bin/streamlit run dashboard/app.py
```

</details>

The pipeline takes about **six minutes** end to end. The dashboard then opens at
<http://localhost:8501>.

To check everything landed correctly (`.venv\Scripts\pytest` on Windows):

```bash
.venv/bin/pytest tests/ -v
```

### If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `no such file or directory: .venv/bin/...` | You are not in the repo folder, or you are on Windows and need `.venv\Scripts\`. |
| `connection refused` / `could not connect to server` | Postgres is not up. Run `docker compose up -d --wait` and wait for it to report healthy. |
| `docker: unknown command: docker compose` | Docker Desktop is installed but not running. Start it and wait for the whale icon to settle. |
| Dashboard loads but says "No trained model found" | The pipeline has not been run yet. Run `etl/run_pipeline.py`. |
| `port 5432 already in use` | You have another Postgres running. Change `POSTGRES_PORT` in `.env` to e.g. `5433` and re-run `docker compose up -d --wait`. |
| Pipeline fails at the `load` stage | The dataset is not where `RAW_DATA_PATH` in `.env` points. That path is relative to the repo root. |
| `pip install` fails building `shap`, or asks for "Microsoft Visual C++ 14.0" | You are on Python 3.13. Delete `.venv`, recreate it with `py -3.12 -m venv .venv`, and install again. |
| `'.venv\Scripts\Activate.ps1' cannot be loaded because running scripts is disabled` | PowerShell execution policy — see the `Set-ExecutionPolicy` command above. |

Stopping things: `Ctrl-C` stops the dashboard, `docker compose down` stops the database and
keeps the data, `docker compose down -v` throws the data away too.

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

(On Windows: `.venv\Scripts\python etl\run_pipeline.py --skip-load --skip-train`.)

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
- Every path in the code goes through `pathlib`, so scripts work the same on Windows and
  macOS. If you add file handling, use `Path` and pass `encoding="utf-8"` explicitly —
  Windows defaults to cp1252 and will mangle any non-ASCII text without it.
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
