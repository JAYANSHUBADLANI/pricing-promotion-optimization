"""Promotional lift, split into the part price does and the part merchandising does.

A single discount column would only ever support one question: did volume rise
when price fell. This data supports a sharper one. A promotion here is a price
cut, an in store display, a place in the weekly circular, or some combination,
and the four combinations carry visibly different discount depths. That makes it
possible to ask which part of the lift is bought with margin and which part is
bought with shelf space.

Two views are produced deliberately.

The observational view compares promoted store weeks against non promoted store
weeks for the same product in the same store, so it removes any difference
between products and stores. It is easy to explain and easy to check by hand, and
it is what a merchant would compute. It also overstates the effect, because
promotions are not scheduled at random.

The model view uses the fitted demand model to separate the price contribution,
which is the elasticity applied to the observed depth, from the mechanic
contribution, which is what is left after price is accounted for. The gap between
the two views is itself informative and is reported rather than hidden.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pricing import features as F
from pricing import schema
from pricing.config import Config, load_config
from pricing.models.elasticity import ElasticityFit


def observational_lift(frame: pd.DataFrame) -> pd.DataFrame:
    """Within product and store comparison of promoted against non promoted weeks.

    Each product store pair contributes only if it is observed in both states, so
    the comparison never leans on a product that was always on promotion or never
    on one.
    """
    keys = [schema.STORE_ID, schema.UPC]
    baseline = (
        frame.loc[frame[F.PROMO_STATE] == F.MECHANIC_NONE]
        .groupby(keys)[F.LOG_UNITS]
        .mean()
        .rename("baseline_log_units")
    )
    joined = frame.join(baseline, on=keys)
    joined = joined.loc[joined["baseline_log_units"].notna()]

    rows = []
    for (category, state), group in joined.groupby([schema.CATEGORY, F.PROMO_STATE]):
        if state == F.MECHANIC_NONE:
            continue
        delta = group[F.LOG_UNITS] - group["baseline_log_units"]
        rows.append(
            {
                "category": category,
                "mechanic": state,
                "n_promoted_weeks": int(len(group)),
                "observed_lift_pct": float((np.exp(delta.mean()) - 1) * 100),
                "mean_discount_depth": float(group[F.DISCOUNT_DEPTH].mean()),
                "n_store_product_pairs": int(group.groupby(keys).ngroups),
            }
        )
    return pd.DataFrame(rows).sort_values(["category", "mechanic"]).reset_index(drop=True)


def decompose_lift(frame: pd.DataFrame, fit: ElasticityFit) -> pd.DataFrame:
    """Split modelled lift into a price part and a merchandising part.

    The price part is the elasticity applied to the depth actually run. The
    mechanic part is the fitted effect of the display or circular at that depth.
    Their sum is the modelled lift, and the ratio between them answers the
    question a promotions team actually argues about: are we buying volume with
    margin or with shelf space.
    """
    elasticity = (
        fit.idata.posterior["own_price_elasticity"].mean(("chain", "draw")).values
    )
    mechanic_labels = [
        b for b in fit.design.blocks if b.name == "mechanic"
    ][0].labels
    mechanic_draws = (
        fit.idata.posterior["mechanic_effect"].mean(("chain", "draw")).values
    )
    mechanic_lookup = {
        label: float(value) for label, value in zip(mechanic_labels, mechanic_draws)
    }

    rows = []
    for (category, state), group in frame.groupby([schema.CATEGORY, F.PROMO_STATE]):
        if state == F.MECHANIC_NONE:
            continue
        upc_codes = group["upc_idx"].to_numpy()
        depth = group[F.DISCOUNT_DEPTH].to_numpy()
        # The elasticity acts on the log price ratio, so a depth of d moves log
        # units by beta times log(1 - d).
        price_log_effect = elasticity[upc_codes] * np.log(np.clip(1.0 - depth, 1e-6, None))
        mechanic_log_effect = mechanic_lookup.get(f"{state}|{category}", 0.0)

        total_log = price_log_effect.mean() + mechanic_log_effect
        price_pct = (np.exp(price_log_effect.mean()) - 1) * 100
        mechanic_pct = (np.exp(mechanic_log_effect) - 1) * 100
        total_pct = (np.exp(total_log) - 1) * 100
        rows.append(
            {
                "category": category,
                "mechanic": state,
                "mean_discount_depth": float(depth.mean()),
                "price_lift_pct": float(price_pct),
                "mechanic_lift_pct": float(mechanic_pct),
                "modelled_total_lift_pct": float(total_pct),
                "share_of_lift_from_merchandising": (
                    float(mechanic_log_effect / total_log) if abs(total_log) > 1e-9 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["category", "mechanic"]).reset_index(drop=True)


def cannibalisation_summary(frame: pd.DataFrame, fit: ElasticityFit) -> pd.DataFrame:
    """How much of a product's promotional gain is taken from its own shelf.

    A cut of depth d on one product raises its own volume through the own price
    elasticity. It also lowers the competing price index faced by each of the
    other products in that sub category, and the cross price elasticity turns that
    into a volume loss for them. Comparing the two gives a diversion ratio: the
    fraction of the headline gain that is simply moved off the neighbouring
    facings rather than created.
    """
    own = fit.idata.posterior["own_price_elasticity"].mean(("chain", "draw")).values
    cross = fit.idata.posterior["cross_price_elasticity"].mean(("chain", "draw")).values

    reference_depth = 0.20
    rows = []
    lookup = frame.drop_duplicates("upc_idx").set_index("upc_idx")
    for sub_idx, group in frame.groupby("sub_category_idx"):
        upc_indices = sorted(group["upc_idx"].unique())
        n_products = len(upc_indices)
        if n_products < 2:
            continue
        baseline_units = group.groupby("upc_idx")[schema.UNITS].mean()
        cross_elasticity = float(cross[sub_idx])

        gains, losses = [], []
        for target in upc_indices:
            log_ratio = np.log(1.0 - reference_depth)
            gain = baseline_units[target] * (np.exp(own[target] * log_ratio) - 1.0)
            # Each other product sees its competing index fall by the promoted
            # product's price move, diluted by the number of competitors it has.
            index_shift = log_ratio / (n_products - 1)
            loss = sum(
                baseline_units[other] * (np.exp(cross_elasticity * index_shift) - 1.0)
                for other in upc_indices
                if other != target
            )
            gains.append(gain)
            losses.append(loss)

        total_gain = float(np.sum(gains))
        total_loss = float(np.sum(losses))
        rows.append(
            {
                "sub_category": str(lookup.loc[upc_indices[0], schema.SUB_CATEGORY]),
                "category": str(lookup.loc[upc_indices[0], schema.CATEGORY]),
                "n_products": n_products,
                "cross_elasticity": cross_elasticity,
                "gross_unit_gain": total_gain,
                "units_lost_on_shelf": total_loss,
                "net_unit_gain": total_gain + total_loss,
                "diversion_ratio": float(-total_loss / total_gain) if total_gain else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("diversion_ratio", ascending=False).reset_index(drop=True)


def main() -> None:
    cfg: Config = load_config()
    cfg.ensure_dirs()
    frame = pd.read_parquet(cfg.path("data", "processed_dir") / "panel_features.parquet")

    import arviz as az

    from pricing.models.design import build_design, compute_sufficient_statistics

    design = build_design(frame, cfg)
    stats = compute_sufficient_statistics(design)
    idata = az.from_netcdf(
        str(cfg.path("reporting", "models_dir") / "elasticity_posterior.nc")
    )
    fit = ElasticityFit(idata=idata, design=design, stats=stats)

    tables = cfg.path("reporting", "tables_dir")
    observed = observational_lift(frame)
    modelled = decompose_lift(frame, fit)
    diversion = cannibalisation_summary(frame, fit)

    observed.to_csv(tables / "lift_observational.csv", index=False)
    modelled.to_csv(tables / "lift_decomposition.csv", index=False)
    diversion.to_csv(tables / "cannibalisation.csv", index=False)

    merged = observed.merge(modelled, on=["category", "mechanic"], how="outer")
    merged["observed_minus_modelled_pct"] = (
        merged["observed_lift_pct"] - merged["modelled_total_lift_pct"]
    )
    merged.to_csv(tables / "lift_comparison.csv", index=False)

    summary = {
        "mean_share_of_lift_from_merchandising": float(
            modelled["share_of_lift_from_merchandising"].mean()
        ),
        "mean_diversion_ratio": float(diversion["diversion_ratio"].mean()),
        "worst_diversion_sub_category": str(diversion.iloc[0]["sub_category"]),
        "worst_diversion_ratio": float(diversion.iloc[0]["diversion_ratio"]),
        "mean_observed_minus_modelled_pct": float(
            merged["observed_minus_modelled_pct"].mean()
        ),
    }
    (tables / "lift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(modelled.round(2).to_string(index=False))
    print()
    print(diversion.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
