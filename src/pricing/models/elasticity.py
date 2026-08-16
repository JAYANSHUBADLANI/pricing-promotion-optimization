"""Hierarchical Bayesian demand model.

The specification is the constant elasticity form in logs:

    log(units) = a_upc + a_store + b_upc * log(price)
                 + c_subcat * log(competing price)
                 + mechanic and seasonal terms
                 + error

with the own price elasticity partially pooled through three levels:

    b_upc     ~ Normal(mu_subcat, sigma_upc)
    mu_subcat ~ Normal(mu_category, sigma_subcat)
    mu_category ~ Normal(mu_global, sigma_category)

Partial pooling earns its place here rather than being decoration. Fitting each
product separately produces at least one positive own price elasticity, which
says demand rises when price rises. That is a small sample artefact, and pooling
towards the shelf and category means is what pulls those estimates back to
something a merchant could act on, while leaving genuinely distinctive products
free to differ.

The likelihood is evaluated through precomputed sufficient statistics, so the
sampler never touches the half million rows. See models/design.py for why that
is exact rather than an approximation.

Residual variance is estimated separately per category, since a frozen pizza
store week and a mouthwash store week are not equally noisy and forcing them to
share a variance would distort the elasticity credible intervals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc.blocking import DictToArrayBijection, RaveledVars
from scipy.optimize import minimize

from pricing.config import Config, load_config
from pricing.models.gibbs import GibbsResult, PriorSpec
from pricing.models.gibbs import sample as gibbs_sample
from pricing.models.design import (
    DesignMatrix,
    SufficientStatistics,
    build_design,
    compute_sufficient_statistics,
    ols_reference,
)


@dataclass
class ElasticityFit:
    idata: az.InferenceData
    design: DesignMatrix
    stats: SufficientStatistics
    draws: GibbsResult | None = None

    def elasticity_table(self) -> pd.DataFrame:
        """Posterior own price elasticity per product, with intervals."""
        maps = self.design.index_maps
        draws = self.idata.posterior["own_price_elasticity"].stack(
            sample=("chain", "draw")
        ).values
        summary = az.summary(
            self.idata, var_names=["own_price_elasticity"], hdi_prob=0.94
        ).reset_index(drop=True)
        table = pd.DataFrame(
            {
                "upc": maps["upc_labels"],
                "sub_category": [maps["sub_labels"][i] for i in maps["upc_to_sub"]],
                "category": [maps["cat_labels"][i] for i in maps["upc_to_cat"]],
                "elasticity_mean": draws.mean(axis=1),
                "elasticity_sd": draws.std(axis=1),
                "hdi_low": summary["hdi_3%"].to_numpy(),
                "hdi_high": summary["hdi_97%"].to_numpy(),
                "prob_elastic": (draws < -1.0).mean(axis=1),
                "r_hat": summary["r_hat"].to_numpy(),
                "ess_bulk": summary["ess_bulk"].to_numpy(),
            }
        )
        return table.sort_values("elasticity_mean").reset_index(drop=True)

    def cross_price_table(self) -> pd.DataFrame:
        maps = self.design.index_maps
        draws = self.idata.posterior["cross_price_elasticity"].stack(
            sample=("chain", "draw")
        ).values
        return pd.DataFrame(
            {
                "sub_category": maps["sub_labels"],
                "cross_elasticity_mean": draws.mean(axis=1),
                "cross_elasticity_sd": draws.std(axis=1),
                "prob_positive": (draws > 0).mean(axis=1),
            }
        ).sort_values("cross_elasticity_mean", ascending=False).reset_index(drop=True)

    def mechanic_table(self) -> pd.DataFrame:
        """Promotional mechanic effects, expressed as a multiplier on units."""
        maps = self.design.index_maps
        draws = self.idata.posterior["mechanic_effect"].stack(
            sample=("chain", "draw")
        ).values
        labels = [b for b in self.design.blocks if b.name == "mechanic"][0].labels
        rows = []
        for i, label in enumerate(labels):
            mechanic, category = label.split("|")
            values = draws[i]
            rows.append(
                {
                    "mechanic": mechanic,
                    "category": category,
                    "log_effect_mean": float(values.mean()),
                    "uplift_multiplier": float(np.exp(values).mean()),
                    "uplift_pct": float((np.exp(values) - 1).mean() * 100),
                    "hdi_low_pct": float((np.exp(np.quantile(values, 0.03)) - 1) * 100),
                    "hdi_high_pct": float((np.exp(np.quantile(values, 0.97)) - 1) * 100),
                    "prob_positive": float((values > 0).mean()),
                }
            )
        del maps
        return pd.DataFrame(rows)


def laplace_mass_matrix(
    model: pm.Model, maxiter: int = 1000, epsilon: float = 1e-5, floor: float = 1e-8
) -> tuple[dict, np.ndarray]:
    """Posterior mode and a full mass matrix from a Laplace approximation.

    The sampler has to cope with a posterior whose directions differ in scale by
    roughly three orders of magnitude. A global intercept informed by half a
    million rows is pinned down far more tightly than a hierarchical scale
    parameter informed by four categories. Left to adapt a diagonal mass matrix
    on its own, the sampler spends its whole budget on adaptation and reports a
    tree depth stuck at the maximum, which means around a thousand gradient
    evaluations for every single draw.

    Because the likelihood runs through sufficient statistics, a gradient costs
    well under a millisecond. That makes it cheap to find the mode with L-BFGS
    and then build the Hessian by finite differences of the gradient. Its inverse
    is used as the mass matrix, which leaves the sampler working in coordinates
    where the posterior is close to isotropic.

    Returns the mode as a point dictionary and the mass matrix as a dense array.
    """
    initial = model.initial_point()
    raveled = DictToArrayBijection.map(initial)
    info = raveled.point_map_info
    logp_fn = model.compile_logp()
    dlogp_fn = model.compile_dlogp()

    def to_point(x: np.ndarray):
        return DictToArrayBijection.rmap(RaveledVars(x, info))

    def negative_logp(x: np.ndarray) -> float:
        return -float(logp_fn(to_point(x)))

    def negative_grad(x: np.ndarray) -> np.ndarray:
        return -np.asarray(dlogp_fn(to_point(x)), dtype=float)

    result = minimize(
        negative_logp,
        raveled.data.astype(float),
        jac=negative_grad,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    mode = result.x

    n = len(mode)
    hessian = np.zeros((n, n))
    for i in range(n):
        forward, backward = mode.copy(), mode.copy()
        forward[i] += epsilon
        backward[i] -= epsilon
        hessian[i] = (negative_grad(forward) - negative_grad(backward)) / (2.0 * epsilon)
    hessian = 0.5 * (hessian + hessian.T)

    # Invert through the eigendecomposition with a floor, so a direction the
    # Hessian reports as flat becomes a wide but finite step rather than an error.
    values, vectors = np.linalg.eigh(hessian)
    values = np.clip(values, floor, None)
    covariance = (vectors / values) @ vectors.T
    covariance = 0.5 * (covariance + covariance.T)

    return to_point(mode), covariance


def build_model(design: DesignMatrix, stats: SufficientStatistics, cfg: Config) -> pm.Model:
    settings = cfg["elasticity"]
    maps = design.index_maps
    n_upc = int(maps["n_upc"])
    n_store = int(maps["n_store"])
    n_sub = int(maps["n_sub"])
    n_cat = int(maps["n_cat"])
    upc_to_sub = np.asarray(maps["upc_to_sub"], dtype=int)
    sub_to_cat = np.asarray(maps["sub_to_cat"], dtype=int)

    n_mechanic = [b for b in design.blocks if b.name == "mechanic"][0].size
    n_season = [b for b in design.blocks if b.name == "seasonality"][0].size

    with pm.Model() as model:
        xtx = pm.Data("xtx", stats.xtx)
        xty = pm.Data("xty", stats.xty)
        yty = pm.Data("yty", stats.yty)
        n_obs = pm.Data("n_obs", stats.n_obs.astype(float))

        intercept = pm.Normal("intercept", mu=0.0, sigma=5.0)

        # Product and store effects are constrained to sum to zero. Without that
        # constraint they are only softly identified: adding a constant to every
        # product effect and subtracting it from the global intercept leaves the
        # likelihood unchanged, which leaves the sampler exploring a ridge it can
        # never resolve. Pinning the sum makes the decomposition identified, and
        # as a side effect removes the geometry that was making sampling slow.
        # Product and store intercepts get a fixed weakly informative scale rather
        # than a hierarchical one. Every product is observed in the order of ten
        # thousand store weeks and every store across every product, so these
        # intercepts are pinned down by the data and pooling them changes nothing
        # in the answer. What it does change is the geometry: a scale parameter
        # sitting above a large block of tightly determined coefficients creates a
        # funnel, and the sampler pays for it in divergences. Pooling is kept where
        # it earns its place, on the elasticities, and dropped where it does not.
        upc_intercept = pm.ZeroSumNormal(
            "upc_intercept", sigma=float(settings.get("prior_upc_intercept_sd", 3.0)),
            shape=n_upc,
        )
        store_intercept = pm.ZeroSumNormal(
            "store_intercept", sigma=float(settings.get("prior_store_intercept_sd", 1.5)),
            shape=n_store,
        )

        # Three level partial pooling on own price elasticity, non centred so the
        # sampler does not have to fight the funnel geometry at each level.
        mu_global = pm.Normal(
            "mu_global_elasticity",
            mu=float(settings["prior_elasticity_mean"]),
            sigma=float(settings["prior_elasticity_sd"]),
        )
        sigma_category = pm.HalfNormal(
            "sigma_category_elasticity", sigma=float(settings["prior_group_sd"])
        )
        z_category = pm.Normal("z_category_elasticity", mu=0.0, sigma=1.0, shape=n_cat)
        mu_category = pm.Deterministic(
            "mu_category_elasticity", mu_global + sigma_category * z_category
        )

        sigma_sub = pm.HalfNormal(
            "sigma_sub_elasticity", sigma=float(settings["prior_group_sd"])
        )
        z_sub = pm.Normal("z_sub_elasticity", mu=0.0, sigma=1.0, shape=n_sub)
        mu_sub = pm.Deterministic(
            "mu_sub_elasticity", mu_category[sub_to_cat] + sigma_sub * z_sub
        )

        sigma_upc = pm.HalfNormal(
            "sigma_upc_elasticity", sigma=float(settings["prior_group_sd"])
        )
        z_upc = pm.Normal("z_upc_elasticity", mu=0.0, sigma=1.0, shape=n_upc)
        own_price = pm.Deterministic(
            "own_price_elasticity", mu_sub[upc_to_sub] + sigma_upc * z_upc
        )

        # Cross price sits at sub category level. Estimating it per product would
        # spend parameters on a term the data identifies far less sharply than
        # own price, and the shelf is the level a merchant reasons about anyway.
        mu_cross = pm.Normal(
            "mu_cross_elasticity",
            mu=float(settings["prior_cross_mean"]),
            sigma=float(settings["prior_cross_sd"]),
        )
        sigma_cross = pm.HalfNormal("sigma_cross_elasticity", sigma=0.5)
        cross_price = pm.Normal(
            "cross_price_elasticity", mu=mu_cross, sigma=sigma_cross, shape=n_sub
        )

        mechanic = pm.Normal("mechanic_effect", mu=0.0, sigma=0.5, shape=n_mechanic)
        holiday = pm.Normal("holiday_effect", mu=0.0, sigma=0.5, shape=n_cat)
        trend = pm.Normal("trend_effect", mu=0.0, sigma=1.0, shape=n_cat)
        seasonality = pm.Normal("seasonality_effect", mu=0.0, sigma=0.5, shape=n_season)

        beta = pt.concatenate(
            [
                pt.reshape(intercept, (1,)),
                upc_intercept,
                store_intercept,
                own_price,
                cross_price,
                mechanic,
                holiday,
                trend,
                seasonality,
            ]
        )

        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0, shape=len(stats.group_labels))

        # Gaussian log likelihood written through sufficient statistics. Identical
        # in value and gradient to summing over every row, but the cost no longer
        # scales with the number of rows.
        quadratic = (
            yty
            - 2.0 * pt.dot(xty, beta)
            + pt.sum(pt.tensordot(xtx, beta, axes=[[2], [0]]) * beta, axis=1)
        )
        log_likelihood = pt.sum(
            -0.5 * n_obs * pt.log(2.0 * np.pi * sigma_y**2)
            - quadratic / (2.0 * sigma_y**2)
        )
        pm.Potential("sufficient_statistic_likelihood", log_likelihood)

    return model




def to_inference_data(result: GibbsResult, design: DesignMatrix) -> az.InferenceData:
    """Wrap Gibbs draws as an ArviZ object so the usual diagnostics apply.

    Being exact and free of tuning parameters does not exempt a sampler from
    having its mixing checked, so r hat and effective sample size are reported
    for the Gibbs output exactly as they would be for a gradient sampler.
    """
    maps = design.index_maps
    posterior = {
        "own_price_elasticity": result.block("own_price"),
        "cross_price_elasticity": result.block("cross_price"),
        "mechanic_effect": result.block("mechanic"),
        "holiday_effect": result.block("holiday"),
        "trend_effect": result.block("trend"),
        "upc_intercept": result.block("upc_intercept"),
        "store_intercept": result.block("store_intercept"),
        "sigma_y": result.sigma_y,
        "mu_global_elasticity": result.mu_global,
        "mu_category_elasticity": result.mu_category,
        "mu_sub_elasticity": result.mu_sub,
        "sigma_category_elasticity": result.sigma_category,
        "sigma_sub_elasticity": result.sigma_sub,
        "sigma_upc_elasticity": result.sigma_upc,
        "mu_cross_elasticity": result.mu_cross,
        "sigma_cross_elasticity": result.sigma_cross,
    }
    coords = {
        "upc": maps["upc_labels"],
        "store": maps["store_labels"],
        "sub_category": maps["sub_labels"],
        "category": maps["cat_labels"],
        "mechanic_category": list(
            [b for b in design.blocks if b.name == "mechanic"][0].labels
        ),
    }
    dims = {
        "own_price_elasticity": ["upc"],
        "upc_intercept": ["upc"],
        "store_intercept": ["store"],
        "cross_price_elasticity": ["sub_category"],
        "mechanic_effect": ["mechanic_category"],
        "holiday_effect": ["category"],
        "trend_effect": ["category"],
        "sigma_y": ["category"],
        "mu_category_elasticity": ["category"],
        "mu_sub_elasticity": ["sub_category"],
    }
    return az.from_dict(posterior=posterior, coords=coords, dims=dims)


def fit(
    cfg: Config | None = None,
    frame: pd.DataFrame | None = None,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
) -> ElasticityFit:
    """Fit the hierarchical demand model with the blocked Gibbs sampler."""
    cfg = cfg or load_config()
    settings = cfg["elasticity"]
    if frame is None:
        frame = pd.read_parquet(
            cfg.path("data", "processed_dir") / "panel_features.parquet"
        )

    design = build_design(frame, cfg)
    stats = compute_sufficient_statistics(design)
    prior = PriorSpec(
        upc_intercept_sd=float(settings.get("prior_upc_intercept_sd", 3.0)),
        store_intercept_sd=float(settings.get("prior_store_intercept_sd", 1.5)),
        elasticity_mean=float(settings["prior_elasticity_mean"]),
        elasticity_mean_sd=float(settings["prior_elasticity_sd"]),
        cross_mean=float(settings["prior_cross_mean"]),
        cross_mean_sd=float(settings["prior_cross_sd"]),
        group_sd_prior_scale=float(settings["prior_group_sd"]),
    )
    result = gibbs_sample(
        design,
        stats,
        draws=int(draws if draws is not None else settings["draws"]),
        tune=int(tune if tune is not None else settings["tune"]),
        chains=int(chains if chains is not None else settings["chains"]),
        seed=cfg.seed,
        prior=prior,
    )
    return ElasticityFit(
        idata=to_inference_data(result, design),
        design=design,
        stats=stats,
        draws=result,
    )


def diagnostics(fit_result: ElasticityFit) -> dict:
    """Convergence summary, reported rather than assumed."""
    variables = [
        "own_price_elasticity",
        "cross_price_elasticity",
        "mechanic_effect",
        "mu_global_elasticity",
        "sigma_y",
    ]
    summary = az.summary(fit_result.idata, var_names=variables, hdi_prob=0.94)
    posterior = fit_result.idata.posterior

    beta_hat = ols_reference(fit_result.stats)
    own_slice = fit_result.design.slice_for("own_price")
    posterior_mean = (
        fit_result.idata.posterior["own_price_elasticity"].mean(("chain", "draw")).values
    )
    return {
        "sampler": "blocked Gibbs with slice steps on group scales",
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "n_draws_total": int(posterior.sizes["chain"] * posterior.sizes["draw"]),
        "n_chains": int(posterior.sizes["chain"]),
        "n_parameters": int(fit_result.stats.n_columns),
        "n_observations": int(fit_result.stats.n_obs.sum()),
        "residual_sigma_by_category": {
            label: float(v)
            for label, v in zip(
                fit_result.stats.group_labels,
                fit_result.idata.posterior["sigma_y"].mean(("chain", "draw")).values,
            )
        },
        "unpooled_ols_own_price": {
            "mean": float(beta_hat[own_slice].mean()),
            "min": float(beta_hat[own_slice].min()),
            "max": float(beta_hat[own_slice].max()),
            "n_positive": int((beta_hat[own_slice] > 0).sum()),
        },
        "pooled_posterior_own_price": {
            "mean": float(posterior_mean.mean()),
            "min": float(posterior_mean.min()),
            "max": float(posterior_mean.max()),
            "n_positive": int((posterior_mean > 0).sum()),
        },
    }


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    result = fit(cfg)

    tables_dir = cfg.path("reporting", "tables_dir")
    models_dir = cfg.path("reporting", "models_dir")
    result.elasticity_table().to_csv(tables_dir / "elasticity_by_product.csv", index=False)
    result.cross_price_table().to_csv(tables_dir / "cross_price_elasticity.csv", index=False)
    result.mechanic_table().to_csv(tables_dir / "mechanic_effects.csv", index=False)

    diag = diagnostics(result)
    (tables_dir / "elasticity_diagnostics.json").write_text(
        json.dumps(diag, indent=2), encoding="utf-8"
    )
    result.idata.to_netcdf(str(Path(models_dir) / "elasticity_posterior.nc"))
    if result.draws is not None:
        np.save(Path(models_dir) / "elasticity_beta_draws.npy", result.draws.beta)

    print(f"parameters {diag['n_parameters']}, observations {diag['n_observations']:,}")
    print(f"max r_hat {diag['max_r_hat']:.4f}, min ess {diag['min_ess_bulk']:.0f}, "
          f"draws {diag['n_draws_total']:,}")
    print(f"unpooled OLS positive elasticities: "
          f"{diag['unpooled_ols_own_price']['n_positive']}")
    print(f"pooled posterior positive elasticities: "
          f"{diag['pooled_posterior_own_price']['n_positive']}")


if __name__ == "__main__":
    main()
