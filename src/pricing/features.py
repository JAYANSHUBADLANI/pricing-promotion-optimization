"""Feature construction for the demand model.

Two choices here matter more than the rest and are worth stating plainly.

Price. Both the shelf price and the base price ship with the data, so discount
depth is measured rather than derived. A common shortcut in this kind of work is
to build a price as revenue divided by units, which puts units on both sides of
the regression and manufactures a negative correlation whether or not demand is
actually price sensitive. That trap does not apply here, and avoiding it is a
property of the dataset rather than a clever adjustment.

Competing price. Fifteen pretzel products sit on the same shelf. Cutting the
price on one of them moves volume that partly comes from the others rather than
from new demand. To measure that, each row carries the mean log price of the
other products in the same sub category, store and week. Its coefficient is a
cross price elasticity, and it is what allows the projected gain from a discount
policy to be reported net of what the policy steals from neighbouring products.
The index is unweighted on purpose: weighting by observed volume would pull the
dependent variable into a regressor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricing import schema
from pricing.config import Config, load_config

LOG_UNITS = "log_units"
LOG_PRICE = "log_price"
LOG_BASE_PRICE = "log_base_price"
DISCOUNT_DEPTH = "discount_depth"
PROMO_STATE = "promo_state"
COMPETING_LOG_PRICE = "competing_log_price"
HAS_COMPETITORS = "has_competitors"
WEEK_INDEX = "week_index"
IS_HOLIDAY = "is_holiday"
IS_PROMOTED = "is_promoted"

MECHANIC_NONE = "no_promo"
MECHANIC_TPR = "tpr_only"
MECHANIC_DISPLAY = "display_only"
MECHANIC_FEATURE = "feature_only"
MECHANIC_FEATURE_DISPLAY = "feature_and_display"
MECHANIC_OTHER = "other"

MECHANICS = (
    MECHANIC_TPR,
    MECHANIC_DISPLAY,
    MECHANIC_FEATURE,
    MECHANIC_FEATURE_DISPLAY,
)

GROUP_KEYS = {
    "upc_idx": schema.UPC,
    "sub_category_idx": schema.SUB_CATEGORY,
    "category_idx": schema.CATEGORY,
    "store_idx": schema.STORE_ID,
}


def label_promo_state(frame: pd.DataFrame) -> pd.Series:
    """Map the three promotion flags onto one mutually exclusive state.

    The flags are close to a clean partition already, because a temporary price
    reduction is recorded only when it runs without a circular or a display. Any
    combination outside the five expected ones is labelled explicitly rather than
    folded into the baseline, so it cannot quietly contaminate the comparison.
    """
    feature = frame[schema.FEATURE].astype(int)
    display = frame[schema.DISPLAY].astype(int)
    tpr = frame[schema.TPR_ONLY].astype(int)

    state = pd.Series(MECHANIC_OTHER, index=frame.index, dtype=object)
    state[(feature == 0) & (display == 0) & (tpr == 0)] = MECHANIC_NONE
    state[(feature == 0) & (display == 0) & (tpr == 1)] = MECHANIC_TPR
    state[(feature == 0) & (display == 1) & (tpr == 0)] = MECHANIC_DISPLAY
    state[(feature == 1) & (display == 0) & (tpr == 0)] = MECHANIC_FEATURE
    state[(feature == 1) & (display == 1) & (tpr == 0)] = MECHANIC_FEATURE_DISPLAY
    return state


def competing_price_index(
    frame: pd.DataFrame, scope: str = schema.SUB_CATEGORY
) -> tuple[pd.Series, pd.Series]:
    """Mean log price of the other products on the same shelf, that week.

    Computed as a leave one out mean so a product never competes with itself.
    Where a product is the only one of its scope present in a store week, the
    index falls back to the product's own log price, which makes the competing
    term contribute no variation for that row rather than dropping the row.
    """
    log_price = np.log(frame[schema.PRICE].to_numpy())
    keys = [frame[schema.STORE_ID], frame[schema.WEEK], frame[scope]]
    grouped = pd.Series(log_price, index=frame.index).groupby(keys)
    total = grouped.transform("sum")
    count = grouped.transform("size")

    has_competitors = count > 1
    index = pd.Series(log_price, index=frame.index, dtype=float)
    index[has_competitors] = (
        (total[has_competitors] - log_price[has_competitors.to_numpy()])
        / (count[has_competitors] - 1)
    )
    return index, has_competitors


def fourier_terms(
    week_index: pd.Series, n_terms: int, period: float
) -> dict[str, np.ndarray]:
    """Smooth seasonal basis, which avoids spending 52 parameters on week dummies."""
    out: dict[str, np.ndarray] = {}
    angle = 2.0 * np.pi * week_index.to_numpy() / period
    for k in range(1, n_terms + 1):
        out[f"fourier_sin_{k}"] = np.sin(k * angle)
        out[f"fourier_cos_{k}"] = np.cos(k * angle)
    return out


def build_features(
    frame: pd.DataFrame, cfg: Config | None = None
) -> pd.DataFrame:
    cfg = cfg or load_config()
    settings = cfg["features"]
    out = frame.copy()

    out[LOG_UNITS] = np.log(out[schema.UNITS].to_numpy())
    out[LOG_PRICE] = np.log(out[schema.PRICE].to_numpy())
    out[LOG_BASE_PRICE] = np.log(out[schema.BASE_PRICE].to_numpy())
    out[DISCOUNT_DEPTH] = 1.0 - out[schema.PRICE] / out[schema.BASE_PRICE]

    out[PROMO_STATE] = label_promo_state(out)
    for mechanic in MECHANICS:
        out[f"is_{mechanic}"] = (out[PROMO_STATE] == mechanic).astype(int)
    out[IS_PROMOTED] = (out[PROMO_STATE] != MECHANIC_NONE).astype(int)

    index, has_competitors = competing_price_index(
        out, scope=str(settings.get("competing_price_scope", "sub_category"))
    )
    out[COMPETING_LOG_PRICE] = index
    out[HAS_COMPETITORS] = has_competitors.astype(int)

    first_week = out[schema.WEEK].min()
    out[WEEK_INDEX] = ((out[schema.WEEK] - first_week).dt.days // 7).astype(int)
    for name, values in fourier_terms(
        out[WEEK_INDEX],
        int(settings.get("fourier_terms", 3)),
        float(settings.get("seasonality_period_weeks", 52.18)),
    ).items():
        out[name] = values

    holiday_weeks = set(int(w) for w in settings.get("holiday_weeks_iso", []))
    iso_week = out[schema.WEEK].dt.isocalendar().week.astype(int)
    out[IS_HOLIDAY] = iso_week.isin(holiday_weeks).astype(int)

    for index_name, key in GROUP_KEYS.items():
        codes, _ = pd.factorize(out[key], sort=True)
        out[index_name] = codes

    return out


def fourier_columns(cfg: Config | None = None) -> list[str]:
    cfg = cfg or load_config()
    n = int(cfg["features"].get("fourier_terms", 3))
    return [f"fourier_{fn}_{k}" for k in range(1, n + 1) for fn in ("sin", "cos")]


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    source = cfg.path("data", "processed_dir") / "panel_clean.parquet"
    frame = pd.read_parquet(source)
    features = build_features(frame, cfg)
    out_path = cfg.path("data", "processed_dir") / "panel_features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"{len(features):,} rows, {features.shape[1]} columns")
    print(features[PROMO_STATE].value_counts().to_string())
    print(f"rows without shelf competitors: {(1 - features[HAS_COMPETITORS]).sum():,}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
