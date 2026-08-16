"""Out of sample check on held out weeks.

An in sample fit says how well a model describes weeks it has already seen, which
is a much easier question than the one that matters. The panel is split on time,
the model is refitted on the earlier period only, and it is then asked to predict
volumes in weeks it never saw, under the prices and promotions that were actually
run. Two things come out of that.

First, predictive accuracy under the historical policy. If the model cannot
reproduce what happened when the retailer promoted the way it did, there is no
reason to trust it about a policy nobody has run.

Second, a comparison between the historical policy and the recommended one over
the same weeks. That comparison is a simulation and is reported as one. Its
credibility rests entirely on the first check, which is why both are shown
together rather than the second alone.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pricing import features as F
from pricing import schema
from pricing.config import Config, load_config
from pricing.models.design import build_design, compute_sufficient_statistics
from pricing.models.elasticity import ElasticityFit, fit as fit_elasticity


def split_on_time(frame: pd.DataFrame, holdout_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into an earlier training period and the final weeks."""
    weeks = np.sort(frame[schema.WEEK].unique())
    if holdout_weeks >= len(weeks):
        raise ValueError(
            f"holdout of {holdout_weeks} weeks leaves nothing to train on "
            f"({len(weeks)} weeks available)"
        )
    cutoff = weeks[-holdout_weeks]
    return frame.loc[frame[schema.WEEK] < cutoff], frame.loc[frame[schema.WEEK] >= cutoff]


def predict_log_units(
    frame: pd.DataFrame, fit: ElasticityFit, cfg: Config
) -> np.ndarray:
    """Point predictions from posterior mean coefficients.

    The design for the holdout is built with the training period's centring, so
    the coefficients mean the same thing in both windows. Rebuilding the centring
    on the holdout would silently shift the intercepts and flatter the model.
    """
    design = build_design(frame, cfg, reference=fit.design)
    beta = _posterior_mean_beta(fit)
    return np.asarray(design.matrix @ beta).ravel()


def _posterior_mean_beta(fit: ElasticityFit) -> np.ndarray:
    """Rebuild the full coefficient vector from the posterior."""
    if fit.draws is not None:
        return fit.draws.beta.mean(axis=(0, 1))
    posterior = fit.idata.posterior
    pieces = []
    for name in (
        "intercept", "upc_intercept", "store_intercept", "own_price_elasticity",
        "cross_price_elasticity", "mechanic_effect", "holiday_effect",
        "trend_effect", "seasonality_effect",
    ):
        values = posterior[name].mean(("chain", "draw")).values
        pieces.append(np.atleast_1d(values))
    return np.concatenate(pieces)


def accuracy_metrics(actual_units: np.ndarray, predicted_log: np.ndarray) -> dict:
    """Accuracy on both the log scale the model fits and the unit scale a merchant reads."""
    actual_log = np.log(actual_units)
    predicted_units = np.exp(predicted_log)
    residual = actual_log - predicted_log
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((actual_log - actual_log.mean()) ** 2))
    return {
        "n_observations": int(len(actual_units)),
        "log_scale_rmse": float(np.sqrt(np.mean(residual**2))),
        "log_scale_r_squared": float(1.0 - ss_res / ss_tot) if ss_tot else float("nan"),
        "unit_scale_mape_pct": float(
            np.mean(np.abs(predicted_units - actual_units) / actual_units) * 100
        ),
        "unit_scale_median_ape_pct": float(
            np.median(np.abs(predicted_units - actual_units) / actual_units) * 100
        ),
        "total_units_actual": float(actual_units.sum()),
        "total_units_predicted": float(predicted_units.sum()),
        "total_units_bias_pct": float(
            (predicted_units.sum() - actual_units.sum()) / actual_units.sum() * 100
        ),
    }


def naive_baselines(train: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    """Reference points the model has to beat to be worth anything.

    A model that cannot beat the average of the same product in the same store is
    not doing useful work, however sophisticated its internals.
    """
    keys = [schema.STORE_ID, schema.UPC]
    pair_mean = train.groupby(keys)[F.LOG_UNITS].mean()
    global_mean = float(train[F.LOG_UNITS].mean())

    predicted = holdout.join(pair_mean.rename("pred"), on=keys)["pred"]
    predicted = predicted.fillna(global_mean).to_numpy()
    actual = holdout[schema.UNITS].to_numpy()

    return {
        "store_product_mean": accuracy_metrics(actual, predicted),
        "global_mean": accuracy_metrics(actual, np.full(len(actual), global_mean)),
    }



def accuracy_by_promo_state(
    holdout: pd.DataFrame,
    predicted_log: np.ndarray,
    baseline_log: np.ndarray,
) -> dict:
    """Split accuracy by promotional state.

    Overall accuracy understates what a demand model is for. Most of the variance
    in a store week panel is cross sectional: which product, which store. A per
    pair average captures that and nothing else, which makes it a strong baseline
    on levels and a useless one for the question the model exists to answer. The
    place to look is the promoted weeks, where the baseline has no way to respond
    to a price cut or a display and the model does. If the model does not win
    clearly there, its elasticities should not be trusted to price anything.
    """
    out: dict[str, dict] = {}
    actual = holdout[schema.UNITS].to_numpy()
    states = holdout[F.PROMO_STATE].to_numpy()
    for state in np.unique(states):
        mask = states == state
        model = accuracy_metrics(actual[mask], predicted_log[mask])
        base = accuracy_metrics(actual[mask], baseline_log[mask])
        out[str(state)] = {
            "n_observations": int(mask.sum()),
            "model_log_rmse": model["log_scale_rmse"],
            "baseline_log_rmse": base["log_scale_rmse"],
            "rmse_improvement_pct": float(
                (base["log_scale_rmse"] - model["log_scale_rmse"])
                / base["log_scale_rmse"] * 100
            ),
            "model_mape_pct": model["unit_scale_mape_pct"],
            "baseline_mape_pct": base["unit_scale_mape_pct"],
            "model_total_units_bias_pct": model["total_units_bias_pct"],
            "baseline_total_units_bias_pct": base["total_units_bias_pct"],
        }
    return out


def baseline_predictions(train: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """Per store product average from the training window."""
    keys = [schema.STORE_ID, schema.UPC]
    pair_mean = train.groupby(keys)[F.LOG_UNITS].mean()
    global_mean = float(train[F.LOG_UNITS].mean())
    predicted = holdout.join(pair_mean.rename("pred"), on=keys)["pred"]
    return predicted.fillna(global_mean).to_numpy()


def run(cfg: Config | None = None, frame: pd.DataFrame | None = None) -> dict:
    cfg = cfg or load_config()
    if frame is None:
        frame = pd.read_parquet(
            cfg.path("data", "processed_dir") / "panel_features.parquet"
        )
    holdout_weeks = int(cfg["backtest"]["holdout_weeks"])
    train, holdout = split_on_time(frame, holdout_weeks)

    # Products or stores absent from training cannot be predicted, so they are
    # excluded and counted rather than quietly given a default.
    known_upc = set(train[schema.UPC].unique())
    known_store = set(train[schema.STORE_ID].unique())
    usable = holdout[
        holdout[schema.UPC].isin(known_upc) & holdout[schema.STORE_ID].isin(known_store)
    ]
    n_dropped = len(holdout) - len(usable)

    fit = fit_elasticity(cfg, train)

    predicted_log = predict_log_units(usable, fit, cfg)
    actual = usable[schema.UNITS].to_numpy()

    report = {
        "holdout_weeks": holdout_weeks,
        "train_weeks": int(train[schema.WEEK].nunique()),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(usable)),
        "holdout_rows_dropped_unseen": int(n_dropped),
        "train_date_max": str(train[schema.WEEK].max().date()),
        "holdout_date_min": str(usable[schema.WEEK].min().date()),
        "holdout_date_max": str(usable[schema.WEEK].max().date()),
        "model": accuracy_metrics(actual, predicted_log),
        "baselines": naive_baselines(train, usable),
        "by_promo_state": accuracy_by_promo_state(
            usable, predicted_log, baseline_predictions(train, usable)
        ),
    }
    model_rmse = report["model"]["log_scale_rmse"]
    baseline_rmse = report["baselines"]["store_product_mean"]["log_scale_rmse"]
    report["rmse_improvement_vs_store_product_mean_pct"] = float(
        (baseline_rmse - model_rmse) / baseline_rmse * 100
    )
    return report


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    report = run(cfg)
    out = cfg.path("reporting", "tables_dir") / "backtest.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    model = report["model"]
    base = report["baselines"]["store_product_mean"]
    print(f"train {report['train_rows']:,} rows to {report['train_date_max']}")
    print(f"holdout {report['holdout_rows']:,} rows "
          f"{report['holdout_date_min']} to {report['holdout_date_max']}")
    print(f"model    log RMSE {model['log_scale_rmse']:.4f}  "
          f"R2 {model['log_scale_r_squared']:.4f}  MAPE {model['unit_scale_mape_pct']:.1f}%")
    print(f"baseline log RMSE {base['log_scale_rmse']:.4f}  "
          f"R2 {base['log_scale_r_squared']:.4f}  MAPE {base['unit_scale_mape_pct']:.1f}%")
    print(f"improvement overall {report['rmse_improvement_vs_store_product_mean_pct']:.1f}%")
    print()
    print(f"{'promo state':<22}{'n':>8}{'model':>9}{'baseline':>10}{'gain':>8}")
    for state, values in sorted(report["by_promo_state"].items()):
        print(f"{state:<22}{values['n_observations']:>8,}"
              f"{values['model_log_rmse']:>9.4f}{values['baseline_log_rmse']:>10.4f}"
              f"{values['rmse_improvement_pct']:>7.1f}%")


if __name__ == "__main__":
    main()
