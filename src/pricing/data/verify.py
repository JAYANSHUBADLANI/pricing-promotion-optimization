"""Profile the raw panel and check it against what the documentation claims.

Documentation and files drift apart, so this stage measures rather than trusts.
It writes a JSON report that later stages and the write up read from, which is
how every number in the README stays tied to the data instead of being typed in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pricing import schema
from pricing.config import Config, load_config
from pricing.data.ingest import build_panel

PROMO_STATES = {
    "no_promo": (0, 0, 0),
    "tpr_only": (0, 0, 1),
    "display_only": (0, 1, 0),
    "feature_only": (1, 0, 0),
    "feature_and_display": (1, 1, 0),
}


def _quantiles(series: pd.Series, qs=(0.0, 0.01, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {}
    return {f"q{int(q * 100):02d}": float(clean.quantile(q)) for q in qs}


def verify(cfg: Config | None = None, panel: pd.DataFrame | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    panel = build_panel(cfg) if panel is None else panel

    weeks = panel[schema.WEEK]
    depth = 1.0 - panel[schema.PRICE] / panel[schema.BASE_PRICE]
    implied = panel[schema.SPEND] / panel[schema.UNITS].replace(0, np.nan)
    price_rel_error = ((implied - panel[schema.PRICE]).abs() / panel[schema.PRICE]).replace(
        [np.inf, -np.inf], np.nan
    )

    n_stores = int(panel[schema.STORE_ID].nunique())
    n_products = int(panel[schema.UPC].nunique())
    n_weeks = int(weeks.nunique())
    grid = n_stores * n_products * n_weeks

    promo_cells: dict[str, dict[str, float]] = {}
    for name, (f, d, t) in PROMO_STATES.items():
        mask = (
            (panel[schema.FEATURE] == f)
            & (panel[schema.DISPLAY] == d)
            & (panel[schema.TPR_ONLY] == t)
        )
        subset_depth = depth[mask]
        promo_cells[name] = {
            "n_rows": int(mask.sum()),
            "share_of_rows": float(mask.mean()),
            "mean_discount_depth": float(subset_depth.mean()) if mask.any() else float("nan"),
            "median_discount_depth": float(subset_depth.median()) if mask.any() else float("nan"),
            "mean_units": float(panel.loc[mask, schema.UNITS].mean()) if mask.any() else float("nan"),
        }
    covered = sum(cell["n_rows"] for cell in promo_cells.values())

    pair = panel.groupby([schema.STORE_ID, schema.UPC])
    weeks_per_pair = pair[schema.WEEK].nunique()
    prices_per_pair = pair[schema.PRICE].nunique()
    price_cv = pair[schema.PRICE].agg(lambda s: s.std() / s.mean() if s.mean() else np.nan)

    expected = cfg["data"]["expected"]
    report: dict[str, Any] = {
        "shape": {
            "n_rows": int(len(panel)),
            "n_columns": int(panel.shape[1]),
            "n_stores": n_stores,
            "n_products": n_products,
            "n_weeks": n_weeks,
            "n_categories": int(panel[schema.CATEGORY].nunique()),
            "n_sub_categories": int(panel[schema.SUB_CATEGORY].nunique()),
            "n_manufacturers": int(panel[schema.MANUFACTURER].nunique()),
        },
        "coverage": {
            "date_min": str(weeks.min().date()),
            "date_max": str(weeks.max().date()),
            "full_grid_rows": int(grid),
            "panel_completeness": float(len(panel) / grid) if grid else float("nan"),
            "weeks_per_store_product_median": float(weeks_per_pair.median()),
            "weeks_per_store_product_min": int(weeks_per_pair.min()),
            "n_store_product_pairs": int(len(weeks_per_pair)),
        },
        "integrity": {
            "duplicate_keys": int(
                panel.duplicated([schema.STORE_ID, schema.UPC, schema.WEEK]).sum()
            ),
            "null_counts": {
                str(k): int(v) for k, v in panel.isna().sum().items() if v > 0
            },
            "n_zero_units": int((panel[schema.UNITS] == 0).sum()),
            "n_negative_units": int((panel[schema.UNITS] < 0).sum()),
            "n_non_positive_price": int((panel[schema.PRICE] <= 0).sum()),
            "n_non_positive_base_price": int((panel[schema.BASE_PRICE] <= 0).sum()),
            # A strong internal check: spend divided by units should reproduce the
            # stated shelf price. Where it does, the price column can be trusted.
            "price_vs_spend_per_unit": {
                "median_relative_error": float(price_rel_error.median()),
                "share_within_1pct": float((price_rel_error < 0.01).mean()),
                "share_within_5pct": float((price_rel_error < 0.05).mean()),
            },
        },
        "identification": {
            "distinct_prices_per_store_product_median": float(prices_per_pair.median()),
            "distinct_prices_per_store_product_q25": float(prices_per_pair.quantile(0.25)),
            "within_pair_price_cv_median": float(price_cv.median()),
            "share_pairs_with_single_price": float((prices_per_pair <= 1).mean()),
        },
        "discount_depth": {
            **_quantiles(depth),
            "mean": float(depth.mean()),
            "share_zero": float((depth.abs() < 1e-9).mean()),
            "share_positive": float((depth > 1e-9).mean()),
            "share_price_above_base": float((depth < -1e-9).mean()),
            "share_above_60pct": float((depth > 0.60).mean()),
        },
        "promo_states": promo_cells,
        "promo_state_coverage": {
            "rows_in_named_states": int(covered),
            "rows_outside_named_states": int(len(panel) - covered),
        },
        "categories": {
            str(k): int(v)
            for k, v in panel.groupby(schema.CATEGORY)[schema.UPC].nunique().items()
        },
        "sub_categories": {
            str(k): int(v)
            for k, v in panel.groupby(schema.SUB_CATEGORY)[schema.UPC].nunique().items()
        },
        "store_segments": {
            str(k): int(v)
            for k, v in panel.drop_duplicates(schema.STORE_ID)[
                schema.STORE_SEGMENT
            ].value_counts(dropna=False).items()
        },
        "states": {
            str(k): int(v)
            for k, v in panel.drop_duplicates(schema.STORE_ID)[schema.STATE]
            .value_counts()
            .items()
        },
        "units": {**_quantiles(panel[schema.UNITS]), "mean": float(panel[schema.UNITS].mean())},
    }

    report["claim_checks"] = {
        "n_rows": {"expected": expected["n_rows"], "actual": report["shape"]["n_rows"],
                   "match": report["shape"]["n_rows"] == expected["n_rows"]},
        "n_stores": {"expected": expected["n_stores"], "actual": n_stores,
                     "match": n_stores == expected["n_stores"]},
        "n_products": {"expected": expected["n_products"], "actual": n_products,
                       "match": n_products == expected["n_products"]},
        "n_weeks": {"expected": expected["n_weeks"], "actual": n_weeks,
                    "match": n_weeks == expected["n_weeks"]},
        "date_min": {"expected": expected["date_min"], "actual": report["coverage"]["date_min"],
                     "match": report["coverage"]["date_min"] == expected["date_min"]},
        "date_max": {"expected": expected["date_max"], "actual": report["coverage"]["date_max"],
                     "match": report["coverage"]["date_max"] == expected["date_max"]},
    }
    report["all_claims_match"] = all(c["match"] for c in report["claim_checks"].values())

    conflicts_path = cfg.path("data", "interim_dir") / "lookup_conflicts.json"
    if conflicts_path.exists():
        report["lookup_conflicts"] = json.loads(conflicts_path.read_text(encoding="utf-8"))

    return report


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    report = verify(cfg)
    out = cfg.path("reporting", "tables_dir") / "data_verification.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    shape, cov = report["shape"], report["coverage"]
    print(f"{shape['n_rows']:,} rows | {shape['n_stores']} stores | "
          f"{shape['n_products']} products | {shape['n_weeks']} weeks")
    print(f"{cov['date_min']} to {cov['date_max']} | "
          f"panel {cov['panel_completeness']:.1%} complete")
    print(f"price vs spend/units within 1pct: "
          f"{report['integrity']['price_vs_spend_per_unit']['share_within_1pct']:.1%}")
    print(f"documented claims all match: {report['all_claims_match']}")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
