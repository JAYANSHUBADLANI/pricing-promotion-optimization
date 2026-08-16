"""Scenario tool for the promotional plan.

The recommendation rests on assumptions that nobody can read off this dataset,
chiefly the gross margin and what a display or a circular costs to run. Rather
than pick values and present one number, this lets whoever is deciding move the
assumptions and watch the plan change. If the advice holds across the range they
consider plausible, that is worth more than a point estimate. If it flips inside
that range, they should know before committing a quarter's promotional calendar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pricing.config import load_config  # noqa: E402
from pricing.optimize.policy import OptimizerConstraints  # noqa: E402
from pricing.optimize.promo import (  # noqa: E402
    NO_PROMO,
    MechanicOption,
    ShelfState,
    optimise_promo_plan,
)

st.set_page_config(page_title="Pricing and promotion planner", layout="wide")


@st.cache_data
def load_inputs():
    cfg = load_config()
    tables = cfg.path("reporting", "tables_dir")
    plan_path = tables / "recommended_plan.csv"
    mechanics_path = tables / "mechanic_options.csv"
    if not plan_path.exists() or not mechanics_path.exists():
        return None, None, cfg
    return pd.read_csv(plan_path), pd.read_csv(mechanics_path), cfg


plan_frame, mechanic_frame, cfg = load_inputs()

st.title("Pricing and promotion planner")

if plan_frame is None:
    st.error(
        "No fitted output found. Run `python run_pipeline.py` first, which writes "
        "the tables this tool reads."
    )
    st.stop()

st.caption(
    "Elasticities and mechanic effects are fitted from the dunnhumby Breakfast at "
    "the Frat panel. Gross margin and execution costs are assumptions, not "
    "measurements, and are the controls below."
)

with st.sidebar:
    st.header("Assumptions")
    gross_margin = st.slider(
        "Gross margin at list price", 0.10, 0.60,
        float(cfg["optimizer"]["gross_margin_at_list"]), 0.01,
        help="No cost data ships with this dataset, so this is assumed.",
    )
    margin_floor = st.slider(
        "Minimum contribution margin", 0.0, 0.50,
        float(cfg["optimizer"]["min_margin_floor"]), 0.01,
    )
    max_discount = st.slider(
        "Maximum discount depth", 0.05, 0.60,
        float(cfg["optimizer"]["max_discount"]), 0.01,
    )
    max_promos = st.slider(
        "Shelves that can promote at once", 1,
        len(plan_frame), int(cfg["optimizer"]["max_concurrent_promos"]),
    )
    cost_multiplier = st.slider(
        "Merchandising cost multiplier", 0.0, 8.0, 1.0, 0.25,
        help="Scales the assumed cost of displays and circulars.",
    )
    require_credible = st.checkbox(
        "Exclude shelves with a non credible response", value=True,
        help=(
            "A shelf whose own and cross price elasticities sum to a positive "
            "number is saying that cutting every price on it would reduce volume. "
            "That is not believable and usually means the cross price term has "
            "absorbed correlated promotional timing."
        ),
    )

shelves = [
    ShelfState(
        shelf=row["shelf"],
        category=row["category"],
        own_elasticity=float(row["own_elasticity"]),
        cross_elasticity=float(row["cross_elasticity"]),
        baseline_units=float(row["baseline_units"]),
        reference_price=float(row["reference_price"]),
        baseline_discount=float(row["baseline_discount"]),
        n_products=int(row["n_products"]),
    )
    for _, row in plan_frame.iterrows()
]

mechanics = [
    MechanicOption(
        name=row["mechanic"],
        log_uplift=float(row["log_uplift"]),
        cost_rate=float(row["assumed_cost_rate"]) * cost_multiplier,
    )
    for _, row in mechanic_frame.iterrows()
]

constraints = OptimizerConstraints(
    max_discount=max_discount,
    min_discount=0.0,
    gross_margin_at_list=gross_margin,
    min_margin_floor=min(margin_floor, gross_margin - 0.01),
    max_concurrent_promos=max_promos,
    promo_cap_threshold=float(cfg["optimizer"]["promo_cap_threshold"]),
)

plan = optimise_promo_plan(shelves, mechanics, constraints, require_credible=require_credible)
result = pd.DataFrame(plan.rows)

left, middle, right = st.columns(3)
left.metric("Shelves promoted", f"{plan.n_promoted} of {len(shelves)}")
middle.metric("Contribution change", f"{plan.margin_delta_pct:+.1f}%")
right.metric("Revenue change", f"{plan.revenue_delta_pct:+.1f}%")

st.subheader("Recommended plan")
display = result[[
    "shelf", "category", "n_products", "own_elasticity", "cross_elasticity",
    "net_elasticity", "is_credible", "recommended_discount",
    "recommended_mechanic", "margin_delta",
]].copy()
display["recommended_discount"] = (display["recommended_discount"] * 100).round(1)
display = display.rename(columns={
    "recommended_discount": "discount %",
    "recommended_mechanic": "mechanic",
    "margin_delta": "contribution change",
})
st.dataframe(display.round(3), use_container_width=True, hide_index=True)

promoted = result[result["recommended_mechanic"] != NO_PROMO]
if promoted.empty:
    st.warning(
        "Under these assumptions no shelf is worth promoting. That is a result, "
        "not a failure: discounting only pays where the shelf level response is "
        "steep enough to make up the margin given away."
    )

st.subheader("Where the recommendation is sensitive")
sweep = []
for margin in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    swept = OptimizerConstraints(
        max_discount=max_discount, min_discount=0.0, gross_margin_at_list=margin,
        min_margin_floor=min(margin_floor, margin - 0.01),
        max_concurrent_promos=max_promos,
        promo_cap_threshold=constraints.promo_cap_threshold,
    )
    swept_plan = optimise_promo_plan(
        shelves, mechanics, swept, require_credible=require_credible
    )
    sweep.append({
        "gross margin": margin,
        "shelves promoted": swept_plan.n_promoted,
        "contribution change %": round(swept_plan.margin_delta_pct, 1),
    })
st.dataframe(pd.DataFrame(sweep), use_container_width=True, hide_index=True)

st.caption(
    "Projected changes are model based and hold everything outside the plan "
    "constant. They are not a forecast of what a quarter would deliver."
)
