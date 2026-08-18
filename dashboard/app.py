"""
Step 9 -- The Streamlit dashboard.

Reads straight from Postgres. There is no API layer and no export step: the tables written
by the pipeline are the dashboard's data source.

Run it:  streamlit run dashboard/app.py

Three things it has to show, from the acceptance checklist:
  * a filterable employee list (department / role / tenure),
  * per employee: a 0-1 risk score, a risk tier, and the top-3 contributing factors written
    in plain language,
  * an aggregate view -- risk distribution by department -- for the "BI dashboard" requirement.

Performance note: there are 850k employees, so nothing here ever does `SELECT *`. Every
number on the aggregate tab is computed by Postgres with GROUP BY, and the employee list is
paged with LIMIT. Query results are cached by Streamlit for five minutes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import text

from db import get_engine

st.set_page_config(page_title="Burnout Risk Dashboard", page_icon="", layout="wide")

CACHE_TTL = 300  # seconds
PAGE_SIZE = 200

# ---------------------------------------------------------------------------
# Turning feature names into something a manager can read.
#
# The SHAP table stores raw column names ("satisfaction_score", "department_Sales").
# Nobody outside this repo should have to see those, so each one gets a phrase here.
# Anything missing falls back to a tidied version of the column name, so the dashboard
# never breaks just because a new feature was added.
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "tenure_months": "time at the company",
    "salary": "salary",
    "overtime_hours": "overtime hours",
    "performance_score": "performance rating",
    "satisfaction_score": "job satisfaction",
    "workload_score": "workload",
    "team_sentiment": "team sentiment",
    "project_completion_rate": "project completion rate",
    "training_participation": "training participation",
    "collaboration_score": "collaboration level",
    "email_sentiment": "tone of written communication",
    "role_complexity_score": "role complexity",
    "career_progression_score": "career progression",
    "n_technical_skills": "breadth of technical skills",
    "n_soft_skills": "breadth of soft skills",
    "feedback_length": "length of recent feedback",
}


def humanise_feature(feature_name: str, feature_value: str) -> str:
    """
    "satisfaction_score", "0.21"  ->  "job satisfaction (0.21)"
    "department_Sales",   "1"     ->  "works in Sales"
    """
    if feature_name in FEATURE_LABELS:
        return f"{FEATURE_LABELS[feature_name]} ({feature_value})"

    # One-hot columns are named "<source column>_<level>", and their value is 1 when the
    # employee is in that level and 0 when they are not. Both cases can appear in a top-3
    # list -- "is not a Manager" can genuinely be what moves someone's score -- so each
    # prefix carries a phrasing for either outcome.
    one_hot_phrasing = {
        "department_": ("works in {}", "does not work in {}"),
        "role_": ("job title is {}", "job title is not {}"),
        "job_level_": ("job level is {}", "job level is not {}"),
    }
    for prefix, (when_present, when_absent) in one_hot_phrasing.items():
        if feature_name.startswith(prefix):
            level = feature_name[len(prefix):]
            is_present = str(feature_value).strip() not in ("0", "0.0", "", "nan")
            template = when_present if is_present else when_absent
            return template.format(level)

    # Unknown feature: show a tidied column name rather than crashing or hiding it.
    return f"{feature_name.replace('_', ' ')} ({feature_value})"


def describe_factor(feature_name: str, feature_value: str, shap_value: float) -> str:
    """One plain-language bullet, including which way the factor pushes."""
    direction = "raises" if shap_value > 0 else "lowers"
    return f"{humanise_feature(feature_name, feature_value)} — {direction} risk"


# ---------------------------------------------------------------------------
# Data access. Every query is cached, and every one is parameterised.
# ---------------------------------------------------------------------------
@st.cache_resource
def engine():
    return get_engine()


@st.cache_data(ttl=CACHE_TTL)
def get_active_model() -> dict | None:
    """The model flagged is_winner by ml/train.py -- what the dashboard reports on."""
    with engine().connect() as conn:
        row = conn.execute(text("""
            SELECT model_version, algorithm, recall, precision, f1, roc_auc, trained_at, notes
            FROM model_metadata WHERE is_winner ORDER BY trained_at DESC LIMIT 1
        """)).mappings().first()
    return dict(row) if row else None


@st.cache_data(ttl=CACHE_TTL)
def get_filter_options() -> dict:
    with engine().connect() as conn:
        departments = conn.execute(text(
            "SELECT DISTINCT department FROM stg_employee_clean ORDER BY 1"
        )).scalars().all()
        roles = conn.execute(text(
            "SELECT role FROM stg_employee_clean GROUP BY role ORDER BY COUNT(*) DESC LIMIT 60"
        )).scalars().all()
        tenure = conn.execute(text(
            "SELECT MIN(tenure_months), MAX(tenure_months) FROM stg_employee_clean"
        )).one()
    return {"departments": departments, "roles": sorted(roles),
            "tenure_min": int(tenure[0]), "tenure_max": int(tenure[1])}


def build_where(filters: dict) -> tuple[str, dict]:
    """Shared WHERE clause so the list, the counts and the charts always agree."""
    clauses = ["p.model_version = :version"]
    params: dict = {"version": filters["model_version"]}

    if filters["departments"]:
        clauses.append("e.department = ANY(:departments)")
        params["departments"] = list(filters["departments"])
    if filters["roles"]:
        clauses.append("e.role = ANY(:roles)")
        params["roles"] = list(filters["roles"])
    if filters["tiers"]:
        clauses.append("p.risk_level = ANY(:tiers)")
        params["tiers"] = list(filters["tiers"])

    clauses.append("e.tenure_months BETWEEN :tenure_lo AND :tenure_hi")
    params["tenure_lo"], params["tenure_hi"] = filters["tenure"]

    clauses.append("p.risk_score >= :min_score")
    params["min_score"] = filters["min_score"]

    return " AND ".join(clauses), params


@st.cache_data(ttl=CACHE_TTL)
def get_summary(filters: dict) -> dict:
    where, params = build_where(filters)
    with engine().connect() as conn:
        row = conn.execute(text(f"""
            SELECT COUNT(*) AS n_employees,
                   AVG(p.risk_score) AS mean_score,
                   SUM(CASE WHEN p.risk_level = 'high' THEN 1 ELSE 0 END) AS n_high,
                   SUM(CASE WHEN p.risk_level = 'medium' THEN 1 ELSE 0 END) AS n_medium,
                   SUM(CASE WHEN p.risk_level = 'low' THEN 1 ELSE 0 END) AS n_low
            FROM model_predictions p
            JOIN stg_employee_clean e USING (employee_id)
            WHERE {where}
        """), params).mappings().one()
    return dict(row)


@st.cache_data(ttl=CACHE_TTL)
def get_by_department(filters: dict) -> pd.DataFrame:
    where, params = build_where(filters)
    with engine().connect() as conn:
        return pd.read_sql(text(f"""
            SELECT e.department,
                   COUNT(*) AS employees,
                   AVG(p.risk_score) AS mean_risk,
                   SUM(CASE WHEN p.risk_level = 'high' THEN 1 ELSE 0 END) AS high_risk,
                   100.0 * SUM(CASE WHEN p.risk_level = 'high' THEN 1 ELSE 0 END)
                         / COUNT(*) AS high_risk_pct
            FROM model_predictions p
            JOIN stg_employee_clean e USING (employee_id)
            WHERE {where}
            GROUP BY e.department
            HAVING COUNT(*) >= 20
            ORDER BY high_risk_pct DESC
        """), conn, params=params)


@st.cache_data(ttl=CACHE_TTL)
def get_employees(filters: dict, page: int) -> pd.DataFrame:
    where, params = build_where(filters)
    params["limit"] = PAGE_SIZE
    params["offset"] = page * PAGE_SIZE
    with engine().connect() as conn:
        return pd.read_sql(text(f"""
            SELECT e.employee_id, e.role, e.job_level, e.department,
                   e.tenure_months, e.salary,
                   p.risk_score, p.risk_level
            FROM model_predictions p
            JOIN stg_employee_clean e USING (employee_id)
            WHERE {where}
            ORDER BY p.risk_score DESC, e.employee_id
            LIMIT :limit OFFSET :offset
        """), conn, params=params)


@st.cache_data(ttl=CACHE_TTL)
def get_employee_detail(employee_id: int, model_version: str) -> tuple:
    with engine().connect() as conn:
        profile = conn.execute(text("""
            SELECT e.*, p.risk_score, p.risk_level, p.predicted_at
            FROM stg_employee_clean e
            JOIN model_predictions p USING (employee_id)
            WHERE e.employee_id = :id AND p.model_version = :v
        """), {"id": employee_id, "v": model_version}).mappings().first()

        factors = pd.read_sql(text("""
            SELECT rank, feature_name, feature_value, shap_value
            FROM shap_explanations
            WHERE employee_id = :id AND model_version = :v
            ORDER BY rank
        """), conn, params={"id": employee_id, "v": model_version})
    return (dict(profile) if profile else None), factors


@st.cache_data(ttl=CACHE_TTL)
def get_top_factors_overall(model_version: str) -> pd.DataFrame:
    with engine().connect() as conn:
        return pd.read_sql(text("""
            SELECT feature_name,
                   COUNT(*) AS times_in_top3,
                   AVG(shap_value) AS mean_contribution
            FROM shap_explanations
            WHERE model_version = :v
            GROUP BY feature_name
            ORDER BY times_in_top3 DESC
            LIMIT 12
        """), conn, params={"v": model_version})


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Employee Burnout Risk")

    model = get_active_model()
    if model is None:
        st.error(
            "No trained model found. Run the pipeline first:\n\n"
            "```\npython etl/run_pipeline.py\n```"
        )
        st.stop()

    options = get_filter_options()

    # ---- Sidebar filters ----------------------------------------------------
    with st.sidebar:
        st.header("Filters")
        departments = st.multiselect("Department", options["departments"])
        roles = st.multiselect("Role", options["roles"])
        tenure = st.slider(
            "Tenure (months)",
            options["tenure_min"], options["tenure_max"],
            (options["tenure_min"], options["tenure_max"]),
        )
        tiers = st.multiselect("Risk tier", ["high", "medium", "low"])
        min_score = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)

        st.divider()
        st.caption("**Active model**")
        st.caption(f"{model['algorithm']}  \n`{model['model_version']}`")
        st.caption(
            f"test-set recall {model['recall']:.3f} · precision {model['precision']:.3f} · "
            f"F1 {model['f1']:.3f}"
        )
        st.caption(f"trained {model['trained_at']:%Y-%m-%d %H:%M} UTC")

    filters = {
        "model_version": model["model_version"],
        "departments": departments,
        "roles": roles,
        "tenure": tenure,
        "tiers": tiers,
        "min_score": min_score,
    }

    summary = get_summary(filters)
    if summary["n_employees"] == 0:
        st.warning("No employees match these filters.")
        st.stop()

    # ---- Headline numbers ---------------------------------------------------
    cols = st.columns(5)
    cols[0].metric("Employees", f"{summary['n_employees']:,}")
    cols[1].metric("Mean risk score", f"{summary['mean_score']:.3f}")
    cols[2].metric("High risk", f"{summary['n_high']:,}",
                   f"{100 * summary['n_high'] / summary['n_employees']:.1f}%")
    cols[3].metric("Medium risk", f"{summary['n_medium']:,}")
    cols[4].metric("Low risk", f"{summary['n_low']:,}")

    overview_tab, people_tab, drivers_tab = st.tabs(
        ["Department overview", "Employees", "What drives risk"]
    )

    # ---- Aggregate / BI view ------------------------------------------------
    with overview_tab:
        st.subheader("Risk distribution by department")
        st.caption("Departments with at least 20 matching employees, worst first.")
        by_dept = get_by_department(filters)

        chart = (
            alt.Chart(by_dept.head(25))
            .mark_bar()
            .encode(
                x=alt.X("high_risk_pct:Q", title="% of employees at high risk"),
                y=alt.Y("department:N", sort="-x", title=None),
                color=alt.Color("mean_risk:Q", title="mean risk",
                                scale=alt.Scale(scheme="orangered")),
                tooltip=[
                    alt.Tooltip("department:N"),
                    alt.Tooltip("employees:Q", format=","),
                    alt.Tooltip("high_risk_pct:Q", format=".1f", title="% high risk"),
                    alt.Tooltip("mean_risk:Q", format=".3f", title="mean risk score"),
                ],
            )
            .properties(height=max(300, 22 * min(len(by_dept), 25)))
        )
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(
            by_dept.rename(columns={
                "department": "Department", "employees": "Employees",
                "mean_risk": "Mean risk", "high_risk": "High risk",
                "high_risk_pct": "% high risk",
            }).style.format({"Mean risk": "{:.3f}", "% high risk": "{:.1f}%",
                             "Employees": "{:,}", "High risk": "{:,}"}),
            use_container_width=True, hide_index=True,
        )

    # ---- Employee list + per-person explanation -----------------------------
    with people_tab:
        total_pages = max(1, -(-summary["n_employees"] // PAGE_SIZE))
        page = st.number_input(
            f"Page (showing {PAGE_SIZE} of {summary['n_employees']:,} employees, "
            f"highest risk first)",
            min_value=1, max_value=total_pages, value=1, step=1,
        ) - 1

        employees = get_employees(filters, page)
        st.dataframe(
            employees.rename(columns={
                "employee_id": "ID", "role": "Role", "job_level": "Level",
                "department": "Department", "tenure_months": "Tenure (mo)",
                "salary": "Salary", "risk_score": "Risk score", "risk_level": "Tier",
            }).style.format({"Risk score": "{:.3f}", "Salary": "${:,.0f}"}),
            use_container_width=True, hide_index=True, height=420,
        )

        st.divider()
        st.subheader("Why is this person at risk?")
        chosen = st.selectbox(
            "Employee",
            employees["employee_id"].tolist(),
            format_func=lambda i: f"#{i}",
        )

        if chosen:
            profile, factors = get_employee_detail(chosen, model["model_version"])
            left, right = st.columns([1, 2])

            with left:
                tier_colour = {"high": "🔴", "medium": "🟠", "low": "🟢"}
                st.metric("Risk score", f"{profile['risk_score']:.3f}")
                st.write(f"**Tier:** {tier_colour.get(profile['risk_level'], '')} "
                         f"{profile['risk_level']}")
                st.write(f"**Role:** {profile['role']}")
                st.write(f"**Department:** {profile['department']}")
                st.write(f"**Level:** {profile['job_level']}")
                st.write(f"**Tenure:** {profile['tenure_months']} months")

            with right:
                st.write("**Top 3 contributing factors**")
                if factors.empty:
                    st.info("No SHAP explanation stored for this employee. "
                            "Run `python ml/explain.py`.")
                else:
                    for _, row in factors.iterrows():
                        arrow = "▲" if row["shap_value"] > 0 else "▼"
                        st.write(
                            f"{row['rank']}. {arrow} "
                            f"{describe_factor(row['feature_name'], row['feature_value'], row['shap_value'])}"
                            f"  \n<span style='color:gray;font-size:0.85em'>"
                            f"SHAP contribution {row['shap_value']:+.4f}</span>",
                            unsafe_allow_html=True,
                        )
                    st.caption(
                        "▲ pushes this person's risk up, ▼ pulls it down. "
                        "Contributions come from SHAP and are specific to this employee."
                    )

    # ---- Workforce-level drivers -------------------------------------------
    with drivers_tab:
        st.subheader("Which factors appear most often in someone's top 3?")
        drivers = get_top_factors_overall(model["model_version"])
        if drivers.empty:
            st.info("No SHAP explanations yet. Run `python ml/explain.py`.")
        else:
            drivers["factor"] = drivers["feature_name"].map(
                lambda f: FEATURE_LABELS.get(f, f.replace("_", " "))
            )
            st.altair_chart(
                alt.Chart(drivers).mark_bar().encode(
                    x=alt.X("times_in_top3:Q", title="employees where this is a top-3 factor"),
                    y=alt.Y("factor:N", sort="-x", title=None),
                    color=alt.Color("mean_contribution:Q", title="mean effect",
                                    scale=alt.Scale(scheme="redblue", reverse=True)),
                    tooltip=["factor:N", alt.Tooltip("times_in_top3:Q", format=","),
                             alt.Tooltip("mean_contribution:Q", format=".4f")],
                ).properties(height=380),
                use_container_width=True,
            )
            st.caption(
                "Red = on average pushes risk up, blue = pushes it down. "
                "This is the workforce-level view; individual employees can differ."
            )


if __name__ == "__main__":
    main()
