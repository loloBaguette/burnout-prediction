-- ---------------------------------------------------------------------------
-- Burnout prediction -- database schema
--
-- This file is applied automatically by Docker the first time the Postgres
-- volume is created, AND re-applied (harmlessly) by the pipeline on every run,
-- which is why every statement is IF NOT EXISTS.
--
-- NOTE: `raw_employee_data` and `stg_employee_clean` are NOT defined here.
-- Their columns come from the real dataset, so they are created by
-- etl/load_raw.py and etl/clean.py from the actual observed fields rather than
-- from column names guessed up front.
-- ---------------------------------------------------------------------------

-- One row per trained model. `model_version` is the join key used by the two
-- tables below, so we can keep several model versions side by side.
CREATE TABLE IF NOT EXISTS model_metadata (
    model_version   TEXT PRIMARY KEY,       -- e.g. 'xgboost_20260818_2036'
    algorithm       TEXT        NOT NULL,   -- 'LogisticRegression' | 'RandomForest' | 'XGBoost'
    recall          DOUBLE PRECISION,       -- on the held-out test set
    precision       DOUBLE PRECISION,
    f1              DOUBLE PRECISION,
    roc_auc         DOUBLE PRECISION,
    is_winner       BOOLEAN     NOT NULL DEFAULT FALSE,  -- the model used for predictions
    is_fallback     BOOLEAN     NOT NULL DEFAULT FALSE,  -- the reduced-feature backup model
    notes           TEXT,                                -- e.g. calibration method, feature count
    trained_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per employee per model version: their calibrated risk score and tier.
CREATE TABLE IF NOT EXISTS model_predictions (
    employee_id     INTEGER     NOT NULL,   -- surrogate key from raw_employee_data
    model_version   TEXT        NOT NULL REFERENCES model_metadata(model_version) ON DELETE CASCADE,
    risk_score      DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_level      TEXT        NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (employee_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_version_score
    ON model_predictions (model_version, risk_score DESC);

-- Top-3 SHAP contributors per employee. rank 1 = biggest contributor.
CREATE TABLE IF NOT EXISTS shap_explanations (
    employee_id     INTEGER     NOT NULL,
    model_version   TEXT        NOT NULL REFERENCES model_metadata(model_version) ON DELETE CASCADE,
    feature_name    TEXT        NOT NULL,
    feature_value   TEXT,                   -- the employee's actual value, for plain-language text
    shap_value      DOUBLE PRECISION NOT NULL,  -- signed: + pushes risk up, - pushes it down
    rank            SMALLINT    NOT NULL CHECK (rank BETWEEN 1 AND 3),
    PRIMARY KEY (employee_id, model_version, rank)
);

CREATE INDEX IF NOT EXISTS idx_shap_employee
    ON shap_explanations (model_version, employee_id);
