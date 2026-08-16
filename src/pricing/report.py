"""Turn fitted parameters into a promotional plan and the tables behind it."""

from __future__ import annotations

import json

import arviz as az
import numpy as np
import pandas as pd

from pricing import features as F
from pricing import schema
from pricing.config import Config, load_config
from pricing.models.design import build_design, compute_sufficient_statistics
from pricing.models.elasticity import ElasticityFit
from pricing.optimize.policy import OptimizerConstraints
from pricing.optimize.promo import (
    NO_PROMO,
    MechanicOption,
    ShelfState,
    margin_sensitivity,
    optimise_promo_plan,
)


def load_fit(cfg: Config, frame: pd.DataFrame) -> ElasticityFit:
    design = build_design(frame, cfg)
    stats = compute_sufficient_statistics(design)
    idata = az.from_netcdf(
        str(cfg.path("reporting", "models_dir") / "elasticity_posterior.nc")
    )
    return ElasticityFit(idata=idata, design=design, stats=stats)


def build_shelf_states(frame: pd.DataFrame, fit: ElasticityFit) -> list[ShelfState]:
    """One planning unit per sub category, carrying its net price response.

    Baseline units and price are weekly averages per store across the panel, so
    the plan is expressed at the scale a merchant plans at: a typical store week.
    """
    own = fit.idata.posterior["own_price_elasticity"].mean(("chain", "draw")).values
    cross = fit.idata.posterior["cross_price_elasticity"].mean(("chain", "draw")).values

    n_weeks = frame[schema.WEEK].nunique()
    n_stores = frame[schema.STORE_ID].nunique()

    states: list[ShelfState] = []
    for sub_idx, group in frame.groupby("sub_category_idx"):
        upc_indices = sorted(group["upc_idx"].unique())
        units_by_upc = group.groupby("upc_idx")[schema.UNITS].sum()
        weights = units_by_upc / units_by_upc.sum()
        # Volume weighted, because a shelf's response is dominated by the facings
        # that actually move, not by an unweighted average across facings.
        own_elasticity = float(np.sum([weights[u] * own[u] for u in upc_indices]))

        baseline_units = float(group[schema.UNITS].sum() / (n_weeks * n_stores))
        baseline_revenue = float(group[schema.SPEND].sum() / (n_weeks * n_stores))
        reference_price = (
            baseline_revenue / baseline_units if baseline_units else float(group[schema.PRICE].mean())
        )
        states.append(
            ShelfState(
                shelf=str(group[schema.SUB_CATEGORY].iloc[0]),
                category=str(group[schema.CATEGORY].iloc[0]),
                own_elasticity=own_elasticity,
                cross_elasticity=float(cross[sub_idx]),
                baseline_units=baseline_units,
                reference_price=reference_price,
                baseline_discount=float(group[F.DISCOUNT_DEPTH].mean()),
                n_products=len(upc_indices),
            )
        )
    return sorted(states, key=lambda s: s.shelf)


def build_mechanics(fit: ElasticityFit, cfg: Config) -> list[MechanicOption]:
    """Mechanic options with fitted volume effects and assumed execution costs.

    The volume effect is averaged across categories so that one option list can
    be applied to every shelf. Costs are assumptions, since the data records that
    a display ran but not what it cost, and they are swept separately.
    """
    labels = [b for b in fit.design.blocks if b.name == "mechanic"][0].labels
    values = fit.idata.posterior["mechanic_effect"].mean(("chain", "draw")).values
    by_mechanic: dict[str, list[float]] = {}
    for label, value in zip(labels, values):
        mechanic = label.split("|")[0]
        by_mechanic.setdefault(mechanic, []).append(float(value))

    costs = cfg.get("mechanic_costs", {}) or {}
    options = [MechanicOption(NO_PROMO, 0.0, float(costs.get(NO_PROMO, 0.0)))]
    for mechanic, effects in sorted(by_mechanic.items()):
        options.append(
            MechanicOption(
                name=mechanic,
                log_uplift=float(np.mean(effects)),
                cost_rate=float(costs.get(mechanic, 0.0)),
            )
        )
    return options


def constraints_from_config(cfg: Config) -> OptimizerConstraints:
    settings = cfg["optimizer"]
    return OptimizerConstraints(
        max_discount=float(settings["max_discount"]),
        min_discount=float(settings["min_discount"]),
        gross_margin_at_list=float(settings["gross_margin_at_list"]),
        min_margin_floor=float(settings["min_margin_floor"]),
        max_concurrent_promos=int(settings["max_concurrent_promos"]),
        promo_cap_threshold=float(settings["promo_cap_threshold"]),
    )


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    tables = cfg.path("reporting", "tables_dir")
    frame = pd.read_parquet(cfg.path("data", "processed_dir") / "panel_features.parquet")
    fit = load_fit(cfg, frame)

    shelves = build_shelf_states(frame, fit)
    mechanics = build_mechanics(fit, cfg)
    constraints = constraints_from_config(cfg)

    pd.DataFrame(
        [
            {
                "mechanic": m.name,
                "log_uplift": m.log_uplift,
                "uplift_pct": (m.multiplier - 1) * 100,
                "assumed_cost_rate": m.cost_rate,
            }
            for m in mechanics
        ]
    ).to_csv(tables / "mechanic_options.csv", index=False)

    plan = optimise_promo_plan(shelves, mechanics, constraints)
    plan_frame = pd.DataFrame(plan.rows)
    plan_frame.to_csv(tables / "recommended_plan.csv", index=False)

    sensitivity = margin_sensitivity(
        shelves, mechanics, constraints, cfg["optimizer"]["margin_sensitivity"]
    )
    pd.DataFrame(sensitivity).to_csv(tables / "margin_sensitivity.csv", index=False)

    # Cost sensitivity: the recommendation leans on merchandising, so how much
    # merchandising has to cost before that advice reverses is worth stating.
    cost_rows = []
    for multiplier in cfg["mechanic_costs"].get("sensitivity_multipliers", [1.0]):
        scaled = [
            MechanicOption(m.name, m.log_uplift, m.cost_rate * float(multiplier))
            for m in mechanics
        ]
        scaled_plan = optimise_promo_plan(shelves, scaled, constraints)
        cost_rows.append(
            {
                "cost_multiplier": float(multiplier),
                "n_promoted": scaled_plan.n_promoted,
                "margin_delta_pct": scaled_plan.margin_delta_pct,
                "mechanics_chosen": ";".join(
                    sorted(
                        {
                            r["recommended_mechanic"]
                            for r in scaled_plan.rows
                            if r["recommended_mechanic"] != NO_PROMO
                        }
                    )
                ),
            }
        )
    pd.DataFrame(cost_rows).to_csv(tables / "cost_sensitivity.csv", index=False)

    summary = {
        "n_shelves": len(shelves),
        "n_promoted": plan.n_promoted,
        "max_concurrent_promos": constraints.max_concurrent_promos,
        "gross_margin_assumed": constraints.gross_margin_at_list,
        "baseline_margin_per_store_week": plan.baseline_margin,
        "optimised_margin_per_store_week": plan.optimised_margin,
        "margin_delta_pct": plan.margin_delta_pct,
        "revenue_delta_pct": plan.revenue_delta_pct,
        "promoted_shelves": [
            {
                "shelf": r["shelf"],
                "discount": r["recommended_discount"],
                "mechanic": r["recommended_mechanic"],
                "net_elasticity": r["net_elasticity"],
            }
            for r in plan.rows
            if r["recommended_mechanic"] != NO_PROMO
        ],
        "shelves_left_alone": [
            r["shelf"] for r in plan.rows if r["recommended_mechanic"] == NO_PROMO
        ],
        "n_shelves_excluded_as_not_credible": plan.n_excluded_not_credible,
        "shelves_excluded_as_not_credible": [
            {"shelf": s.shelf, "own": s.own_elasticity, "cross": s.cross_elasticity,
             "net": s.net_elasticity}
            for s in shelves if not s.is_credible
        ],
    }
    (tables / "plan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(plan_frame[[
        "shelf", "own_elasticity", "cross_elasticity", "net_elasticity",
        "recommended_discount", "recommended_mechanic", "is_credible", "margin_delta",
    ]].round(3).to_string(index=False))
    print()
    print(f"promoted {plan.n_promoted} of {len(shelves)} shelves, "
          f"margin {plan.margin_delta_pct:+.2f}%")


if __name__ == "__main__":
    main()
