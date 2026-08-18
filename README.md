# Employee Burnout Prediction

This project reads a big file of employee data, cleans it up, trains a machine-learning model
to spot people at risk of burnout, explains *why* it flagged each person, and shows all of it
in a web dashboard you open in your browser.

**New to all this? Start at [Part 1](#part-1--install-the-four-tools-you-need) and work down.
Every step tells you what you should see when it worked.** You do not need to understand the
code to get it running.

---

## Contents

- [What you will end up with](#what-you-will-end-up-with)
- [Part 1 — Install the four tools you need](#part-1--install-the-four-tools-you-need)
- [Part 2 — Get the project onto your computer](#part-2--get-the-project-onto-your-computer)
- [Part 3 — Set it up (once)](#part-3--set-it-up-once)
- [Part 4 — Run it](#part-4--run-it)
- [Part 5 — Using it day to day](#part-5--using-it-day-to-day)
- [When something goes wrong](#when-something-goes-wrong)
- [What the project actually does](#what-the-project-actually-does)
- [What we found in the data](#what-we-found-in-the-data-important-for-the-report)
- [How the modelling works](#how-the-modelling-works)
- [Notes for the team](#notes-for-the-team)

---

## What you will end up with

A dashboard in your browser showing all 850,000 employees, where you can:

- filter by department, role, tenure or risk level
- see each person's **risk score** (a number between 0 and 1) and **risk tier** (low / medium / high)
- see the **top 3 reasons** the model gave that person their score, written in plain English
- see which departments have the most at-risk people

Getting there takes about **20 minutes of installing**, then **6 minutes of the computer doing
the work**. After that, starting it up again takes seconds.

---

## Part 1 — Install the four tools you need

You need four things. Install them in this order. If you already have one, skip it.

### 1.1 A terminal (you already have one)

A terminal is a window where you type commands instead of clicking buttons. Every instruction
in this README is typed into a terminal.

- **Windows:** press the Start key, type `powershell`, and open **Windows PowerShell**.
- **Mac:** press `Cmd + Space`, type `terminal`, and press Enter.

Leave it open — you will use it for everything below.

### 1.2 Git — downloads the project code

Git is the tool that copies the project from GitHub onto your computer.

- **Windows:** download from [git-scm.com/download/win](https://git-scm.com/download/win) and
  run the installer. Click **Next** through every screen — the defaults are correct.
- **Mac:** you probably have it. If not, typing `git` in the terminal will offer to install it.

**Check it worked** — type this and press Enter:

```
git --version
```

✅ You should see something like `git version 2.43.0`. The exact number does not matter.
❌ If you see "not recognized" or "command not found", the install did not finish — close the
terminal, open a new one, and try again. (A new terminal is needed for it to notice new
software.)

### 1.3 Python 3.12 — the language the project is written in

⚠️ **Install Python 3.12, not the newest version.** The newest (3.13) is missing a piece one of
our libraries needs, and the install will fail with a confusing error about "Microsoft Visual
C++". Save yourself the trouble.

- **Windows:** download **Python 3.12** from
  [python.org/downloads/windows](https://www.python.org/downloads/windows/). Scroll to a 3.12
  release and get the **Windows installer (64-bit)**.
  🚨 **On the first screen of the installer, tick the box that says "Add python.exe to PATH"
  before clicking Install.** It is easy to miss and nothing will work without it.
- **Mac:** download **Python 3.12** from
  [python.org/downloads/macos](https://www.python.org/downloads/macos/) and run the installer.

**Check it worked:**

```
py --version
```

(On Mac, use `python3 --version` instead.)

✅ You should see `Python 3.12.something`.
❌ "not recognized"? You missed the "Add python.exe to PATH" tick box. Re-run the installer,
choose **Modify**, and tick it.

### 1.4 Docker Desktop — runs the database

Our project stores its data in a database called PostgreSQL. Rather than each of us installing
and configuring that separately (painful, and we would all end up with slightly different
setups), Docker runs it in a pre-packaged box. You never have to configure the database — you
just start the box.

- Download from
  [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).
- **Windows:** the installer will ask about **WSL 2**. Say yes to everything it offers. It will
  probably ask you to restart your computer — do it.
- **Mac:** drag it to Applications like any other app.

**After installing, open Docker Desktop and leave it running.** You will see a whale icon in
your taskbar (Windows) or menu bar (Mac). Wait until it stops animating — that means the engine
has started. This takes a minute or two the first time.

🚨 **Docker Desktop must be open and running every time you work on this project.** If you
restart your computer, open it again. This is the single most common thing that breaks.

**Check it worked:**

```
docker --version
```

✅ You should see `Docker version 29.x.x` or similar.

---

## Part 2 — Get the project onto your computer

### 2.1 Choose where it should live

Pick a folder. Your Documents folder is fine. In the terminal, go there:

**Windows:**
```powershell
cd $HOME\Documents
```

**Mac:**
```bash
cd ~/Documents
```

> **What `cd` means:** "change directory" — it moves your terminal into a folder, like
> double-clicking a folder in Explorer or Finder. Your terminal is always "inside" one folder,
> and commands run relative to wherever you are. Getting this wrong is the #1 cause of
> "file not found" errors.

### 2.2 Download the project

⚠️ **First: you need access.** This repository is private, so you must be added as a
collaborator before you can download it. If you have not been added, ask the repo owner to go
to **GitHub → the repo → Settings → Collaborators → Add people** and add your GitHub username.

Then:

```
git clone https://github.com/loloBaguette/burnout-prediction.git
```

The first time, Git will ask you to sign in to GitHub — a browser window opens, you log in, and
it remembers you afterwards.

This creates a new folder called `burnout-prediction` containing all the code.

❌ **`repository not found`** almost never means the address is wrong. On a private repo GitHub
says that when you *do not have access* — so it means either you have not been added as a
collaborator, or you are signed in to Git as a different GitHub account.

### 2.3 Go into the project folder

```
cd burnout-prediction
```

✅ **Everything from here on must be typed while you are inside this folder.** If you close the
terminal and come back later, you must `cd` back here first.

To check where you are:

```
pwd
```

✅ It should end in `burnout-prediction`.

### 2.4 Get the dataset

The data file is **1.3 GB** — far too big for GitHub, so it is deliberately not in the repo.
Get it from a teammate (OneDrive, Google Drive, a USB stick — whatever is easiest).

Put the file here, inside the project folder:

```
burnout-prediction\data\raw\synthetic-employee-dataset.json
```

⚠️ **The file name must match exactly.** If yours is called something else, either rename it,
or open the `.env` file you will create in the next part and change the `RAW_DATA_PATH` line to
match your file's name.

---

## Part 3 — Set it up (once)

Four commands. Run them one at a time, in order, and wait for each to finish before starting
the next.

### 3.1 Start the database

```
docker compose up -d --wait
```

The very first time, this downloads PostgreSQL (a few hundred MB), so give it a couple of
minutes. Later runs take about five seconds.

✅ You should see lines ending with `Container burnout_postgres Healthy`.
❌ An error mentioning "daemon" or "pipe" means Docker Desktop is not running. Open it, wait
for the whale to settle, and try again.

### 3.2 Create a virtual environment

**Windows:**
```powershell
py -3.12 -m venv .venv
```

**Mac:**
```bash
python3.12 -m venv .venv
```

> **What this does:** it creates a private copy of Python inside the project, in a hidden folder
> called `.venv`. Libraries we install go in there instead of being dumped system-wide, so this
> project cannot break your other Python work and vice versa. It is standard practice — every
> Python project does this.

✅ Nothing is printed if it worked. A new `.venv` folder appears.

### 3.3 Install the libraries

**Windows:**
```powershell
.venv\Scripts\pip install -r requirements.txt
```

**Mac:**
```bash
.venv/bin/pip install -r requirements.txt
```

This downloads about 15 libraries and takes 2–5 minutes. Lots of text scrolls past — that is
normal.

✅ Ends with `Successfully installed ...` and a long list.
❌ An error about **"Microsoft Visual C++ 14.0"** means you are on Python 3.13. Delete the
`.venv` folder and redo step 3.2 with `py -3.12`.

> **Notice the difference:** Windows uses `.venv\Scripts\`, Mac uses `.venv/bin/`. That prefix
> is how you tell the computer to use the project's private Python instead of the system one.
> It appears in every command from here on.

### 3.4 Create your settings file

**Windows:**
```powershell
Copy-Item .env.example .env
```

**Mac:**
```bash
cp .env.example .env
```

> **What this does:** `.env` holds the database password and the path to the data file. It is
> deliberately not shared on GitHub (passwords never belong in a repo), so each of us makes our
> own from the example. The defaults work as-is — you do not need to edit it.

---

## Part 4 — Run it

### 4.1 Build everything

**Windows:**
```powershell
.venv\Scripts\python etl\run_pipeline.py
```

**Mac:**
```bash
.venv/bin/python etl/run_pipeline.py
```

**This takes about 6 minutes.** It prints a banner for each of its seven stages so you can see
where it is. Leave it alone until it finishes.

✅ It ends with `PIPELINE COMPLETE`, a timing table, and a count of rows in each database table.

What it is doing, in order: creating the database tables → reading the 1.3 GB file into the
database → cleaning the data → training three different models and picking the best → training
a backup model → scoring all 850,000 employees → working out the top 3 reasons for each one.

### 4.2 Open the dashboard

**Windows:**
```powershell
.venv\Scripts\streamlit run dashboard\app.py
```

**Mac:**
```bash
.venv/bin/streamlit run dashboard/app.py
```

✅ Your browser opens automatically at <http://localhost:8501>. If it does not, open that
address yourself.

**To stop the dashboard:** click the terminal window and press `Ctrl + C`.

### 4.3 Check everything is correct (optional)

**Windows:**
```powershell
.venv\Scripts\pytest tests\ -v
```

**Mac:**
```bash
.venv/bin/pytest tests/ -v
```

✅ `18 passed`. These check the data has no gaps, the model meets the required accuracy, and
every employee has a score and exactly three explanations.

---

## Part 5 — Using it day to day

Once set up, you do **not** repeat Parts 1–3. Coming back to the project later:

1. Open **Docker Desktop** and wait for the whale to settle.
2. Open a terminal and go to the project folder:
   ```
   cd $HOME\Documents\burnout-prediction
   ```
   (Mac: `cd ~/Documents/burnout-prediction`)
3. Start the database:
   ```
   docker compose up -d --wait
   ```
4. Open the dashboard:
   ```
   .venv\Scripts\streamlit run dashboard\app.py
   ```

Your data is still in the database — you do **not** need to re-run the 6-minute pipeline unless
you changed the code or the data.

**Getting the latest changes from teammates:**
```
git pull
```

**Shutting down:** `Ctrl + C` stops the dashboard. `docker compose down` stops the database and
keeps your data. `docker compose down -v` deletes the data too (you would need to re-run the
pipeline).

**Re-running only part of the pipeline** while developing — this skips the slow steps:
```
.venv\Scripts\python etl\run_pipeline.py --skip-load --skip-train
```

---

## When something goes wrong

| What you see | What it means | Fix |
|---|---|---|
| `no such file or directory: .venv/bin/...` or `.venv\Scripts\...` | You are not in the project folder | `cd` into `burnout-prediction`, then check with `pwd` |
| `connection refused`, `could not connect to server` | The database is not running | Open Docker Desktop, then `docker compose up -d --wait` |
| `docker: unknown command` or errors about "daemon"/"pipe" | Docker Desktop is not running | Open it, wait for the whale to stop animating |
| `Microsoft Visual C++ 14.0 is required` | You are on Python 3.13 | Delete `.venv`, redo step 3.2 with `py -3.12` |
| `py` or `python` "not recognized" | Python is not on your PATH | Re-run the Python installer, choose **Modify**, tick "Add python.exe to PATH", open a **new** terminal |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell security setting | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Dashboard says **"No trained model found"** | The pipeline has not been run | Do step 4.1 |
| Pipeline fails at the **load** stage | The data file is missing or misnamed | Check `data\raw\` and re-read step 2.4 |
| `port 5432 already in use` | Something else uses that port | Open `.env`, change `POSTGRES_PORT` to `5433`, re-run `docker compose up -d --wait` |
| A command "does nothing" | It is still working | Big steps take minutes. Wait for the prompt to come back |

**Still stuck?** Copy the *whole* error message — the last 20 lines, not just the final one —
and send it to the team. The useful part is usually in the middle.

---

## What the project actually does

### The files

| File | What it does |
|---|---|
| `docker-compose.yml` | Describes the database box so we all get an identical one |
| `db/schema.sql` | The database tables |
| `db.py` | Shared code for connecting to the database |
| `etl/load_raw.py` | Reads the 1.3 GB file into the database, unchanged |
| `etl/clean.py` | Fixes the messy bits and makes a clean copy |
| `notebooks/01_eda.ipynb` | **The analysis notebook — read this one** |
| `ml/features.py` | Turns the data into numbers the model can use |
| `ml/train.py` | Trains three models and picks the best |
| `ml/evaluate.py` | Measures how good each model is |
| `ml/fallback_model.py` | A simpler backup model |
| `ml/explain.py` | Works out the top 3 reasons per employee |
| `ml/predict.py` | Gives everyone a score and a risk tier |
| `etl/run_pipeline.py` | Runs all of the above in order |
| `dashboard/app.py` | The web dashboard |
| `tests/test_pipeline.py` | Checks it all worked |

### Opening the analysis notebook

The notebook shows every chart and finding. To open it:

```
.venv\Scripts\jupyter notebook notebooks\01_eda.ipynb
```

You can also just read it on GitHub — the charts and results are saved inside the file, so it
displays without running anything.

### The database tables

| Table | Rows | What is in it |
|---|---|---|
| `raw_employee_data` | 849,999 | The original file, untouched |
| `stg_employee_clean` | 849,999 | The cleaned version, no gaps |
| `model_metadata` | 4 | How well each model scored |
| `model_predictions` | 849,999 | Everyone's risk score and tier |
| `shap_explanations` | 2,549,997 | Everyone's top 3 reasons (3 rows each) |

Risk tiers: **high** = score 0.70 or above, **medium** = 0.30 to 0.70, **low** = below 0.30.

---

## What we found in the data (important for the report)

### Problems we found in the raw data

All four are documented with evidence in `notebooks/01_eda.ipynb`.

**1. No empty cells — but plenty of useless ones.** Nothing in the file is technically
"missing", so the usual checks all pass. But `role` is just a blank space in 109,047 rows, and
in another 7,481 it literally contains the text `nan` (`department` too, in 983 rows). Those
sail through every "is this empty?" test while being exactly as useless as an empty cell. We
turn them all into a category called `Unknown` rather than deleting those rows — deleting would
have thrown away 13% of the dataset.

**2. Five columns were really three.** `collaboration_score`, `slack_activity` and
`meeting_participation` contain *identical* numbers in all 849,999 rows. Same for
`performance_score` and `goal_achievement_rate`. They are not similar-but-different signals —
they are the same column copied. We keep one of each and drop the copies.

**3. There was no yes/no answer to predict.** The file has `burnout_risk`, a number between 0
and 1 — not a "burned out: yes/no" label, which is what a classifier needs. We discovered that
the file's own `risk_factors_summary` column is exactly `burnout_risk` split at **0.75**, so we
use the dataset's own cut-off rather than inventing one: **high risk = burnout_risk ≥ 0.75**.

**4. Eight columns gave away the answer.** This is the big one, and it is called *leakage*.
`stress_level`, for example, matches `burnout_risk` almost perfectly (and is *exactly equal* in
46% of rows). A model given that column would look brilliant while having learned nothing — it
would just be reading the answer off the page. All eight are excluded, with the reason for each
listed at the top of `ml/features.py`.

> **Leakage in one sentence:** giving the model a column that secretly contains the answer.
> It is the most common way a machine-learning project produces amazing results that fall apart
> the moment it meets real data. Finding it is a real result worth writing up.

### Why our 99.99% accuracy is bad news, not good news

**Read this before putting any number from this project into a slide.**

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


## Notes for the team

- **Never commit `.env`** — it has the database password in it. It is already in `.gitignore`,
  so this happens automatically, but do not go around it.
- **Never commit the dataset.** 1.3 GB does not belong in git. Also already handled.
- **Read `notebooks/01_eda.ipynb` before changing `etl/clean.py` or `ml/features.py`.**
  Section 10 of the notebook lists every decision those two files implement, and why.
- **Do not add anything from `LEAKAGE_COLUMNS`** (top of `ml/features.py`) back into the model.
  It will make the scores look better and the model meaningless.
- **If you add a new feature**, add a plain-English name for it in `FEATURE_LABELS` in
  `dashboard/app.py`, or the dashboard will show a manager a raw column name like
  `n_technical_skills`.
- **If you write code that reads or writes a file**, use `pathlib.Path` and pass
  `encoding="utf-8"`. Windows defaults to a different text encoding and will corrupt anything
  non-English without it.

## Things we deliberately did not do

Worth mentioning in the report as future work:

- **No hyperparameter tuning.** The models use sensible fixed settings. Given how predictable
  this dataset's target is, tuning would not change anything meaningful.
- **No text analysis.** The `recent_feedback` column is scraped review text with no real
  connection to the individual employee, so we reduced it to its length. Genuine feedback text
  would be worth a proper NLP pass.
- **No fairness check.** The model uses `department`, `role`, `job_level`, and salary is an
  input. Before anything like this touched a real HR decision it would need a bias review
  across protected groups — and this dataset does not contain the information needed to do one.
