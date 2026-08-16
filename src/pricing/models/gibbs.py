"""Blocked Gibbs sampler for the hierarchical demand model.

Why this rather than a general purpose gradient sampler.

The demand model is a Gaussian linear model with Gaussian priors arranged in a
hierarchy. Every conditional distribution in it is available in closed form. That
makes gradient based sampling the wrong tool: it has to discover numerically a
geometry that can simply be written down. In practice that showed up as a tree
depth pinned at its maximum, hundreds of divergent transitions, and a fit that
would not finish, all for a model whose conditionals are textbook.

The sampler below exploits three things.

1. The whole coefficient vector is drawn jointly from its exact multivariate
   normal conditional. Drawing it as one block rather than parameter by parameter
   means the strong correlations between a product's intercept and its price
   slope, or between the global intercept and the product effects, are handled
   exactly instead of being something the sampler has to crawl along. This is
   also why no identifying constraint is needed on the intercepts: the joint
   draw handles the collinearity, and proper priors keep the posterior proper.

2. The conditional depends on the data only through the sufficient statistics
   computed in models/design.py, so each iteration costs a single Cholesky
   factorisation of a 243 by 243 matrix rather than a pass over 522,154 rows.

3. Variance and mean hyperparameters that are conjugate are drawn directly. The
   group level standard deviations are given half normal priors, which are not
   conjugate but are one dimensional, so they are drawn with slice sampling. That
   keeps the recommended prior rather than switching to an inverse gamma for
   convenience, which is known to behave badly when there are only four groups.

The result is an exact sampler with no step size, no acceptance rate and no
divergences, running in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pricing.models.design import DesignMatrix, SufficientStatistics


@dataclass
class PriorSpec:
    """Fixed prior scales for the coefficient blocks that are not pooled."""

    intercept_sd: float = 10.0
    upc_intercept_sd: float = 3.0
    store_intercept_sd: float = 1.5
    mechanic_sd: float = 0.5
    holiday_sd: float = 0.5
    trend_sd: float = 1.0
    seasonality_sd: float = 0.5
    elasticity_mean: float = -2.0
    elasticity_mean_sd: float = 1.5
    cross_mean: float = 0.3
    cross_mean_sd: float = 0.5
    group_sd_prior_scale: float = 0.5
    sigma_y_prior_shape: float = 2.0
    sigma_y_prior_scale: float = 1.0


@dataclass
class GibbsState:
    beta: np.ndarray
    sigma_y: np.ndarray
    mu_global: float
    mu_category: np.ndarray
    mu_sub: np.ndarray
    sigma_category: float
    sigma_sub: float
    sigma_upc: float
    mu_cross: float
    sigma_cross: float


@dataclass
class GibbsResult:
    """Posterior draws, shaped chain by draw by parameter."""

    beta: np.ndarray
    sigma_y: np.ndarray
    mu_global: np.ndarray
    mu_category: np.ndarray
    mu_sub: np.ndarray
    sigma_category: np.ndarray
    sigma_sub: np.ndarray
    sigma_upc: np.ndarray
    mu_cross: np.ndarray
    sigma_cross: np.ndarray
    slices: dict[str, slice] = field(default_factory=dict)

    def block(self, name: str) -> np.ndarray:
        """Draws for one named block of the coefficient vector."""
        return self.beta[..., self.slices[name]]


def _slice_sample_positive(
    log_density, current: float, rng: np.random.Generator, width: float = 0.5
) -> float:
    """Univariate slice sampler on a positive parameter.

    Stepping out and shrinking as in Neal's algorithm. Used for the group level
    standard deviations, where a half normal prior is preferred to a conjugate
    inverse gamma and the cost of not being conjugate is negligible in one
    dimension.
    """
    current = float(max(current, 1e-6))
    log_y = log_density(current) + np.log(rng.uniform())

    lower = max(current - width * rng.uniform(), 1e-8)
    upper = lower + width
    while lower > 1e-8 and log_density(lower) > log_y:
        lower = max(lower - width, 1e-8)
    while log_density(upper) > log_y:
        upper += width

    for _ in range(100):
        candidate = rng.uniform(lower, upper)
        if log_density(candidate) > log_y:
            return float(candidate)
        if candidate < current:
            lower = candidate
        else:
            upper = candidate
    return current


def _log_half_normal(value: float, scale: float) -> float:
    if value <= 0:
        return -np.inf
    return -0.5 * (value / scale) ** 2


def _build_prior(
    state: GibbsState,
    slices: dict[str, slice],
    n_columns: int,
    upc_to_sub: np.ndarray,
    prior: PriorSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal prior precision and prior mean for the coefficient vector."""
    precision = np.zeros(n_columns)
    mean = np.zeros(n_columns)

    precision[slices["intercept"]] = 1.0 / prior.intercept_sd**2
    precision[slices["upc_intercept"]] = 1.0 / prior.upc_intercept_sd**2
    precision[slices["store_intercept"]] = 1.0 / prior.store_intercept_sd**2
    precision[slices["mechanic"]] = 1.0 / prior.mechanic_sd**2
    precision[slices["holiday"]] = 1.0 / prior.holiday_sd**2
    precision[slices["trend"]] = 1.0 / prior.trend_sd**2
    precision[slices["seasonality"]] = 1.0 / prior.seasonality_sd**2

    precision[slices["own_price"]] = 1.0 / state.sigma_upc**2
    mean[slices["own_price"]] = state.mu_sub[upc_to_sub]

    precision[slices["cross_price"]] = 1.0 / state.sigma_cross**2
    mean[slices["cross_price"]] = state.mu_cross

    return precision, mean


def _draw_beta(
    stats: SufficientStatistics,
    precision: np.ndarray,
    mean: np.ndarray,
    sigma_y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Joint draw of every coefficient from its exact normal conditional.

    Posterior precision is the sum over variance groups of X'X divided by the
    group variance, plus the diagonal prior precision. One Cholesky factorisation
    gives both the mean and a correlated normal draw.
    """
    weights = 1.0 / sigma_y**2
    posterior_precision = np.einsum("g,gij->ij", weights, stats.xtx)
    posterior_precision[np.diag_indices_from(posterior_precision)] += precision
    rhs = weights @ stats.xty + precision * mean

    chol = np.linalg.cholesky(posterior_precision)
    posterior_mean = np.linalg.solve(
        chol.T, np.linalg.solve(chol, rhs)
    )
    noise = np.linalg.solve(chol.T, rng.standard_normal(len(rhs)))
    return posterior_mean + noise


def _draw_sigma_y(
    stats: SufficientStatistics,
    beta: np.ndarray,
    prior: PriorSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Conjugate inverse gamma draw for the residual scale of each category."""
    out = np.empty(len(stats.n_obs))
    for g in range(len(stats.n_obs)):
        rss = max(stats.residual_sum_of_squares(beta, g), 1e-12)
        shape = prior.sigma_y_prior_shape + 0.5 * stats.n_obs[g]
        scale = prior.sigma_y_prior_scale + 0.5 * rss
        out[g] = np.sqrt(scale / rng.gamma(shape))
    return out


def _draw_normal_mean(
    values: np.ndarray, sigma_within: float, prior_mean: float, prior_sd: float,
    rng: np.random.Generator,
) -> float:
    """Conjugate normal draw for the mean of a group of exchangeable values."""
    n = len(values)
    precision = n / sigma_within**2 + 1.0 / prior_sd**2
    centre = (values.sum() / sigma_within**2 + prior_mean / prior_sd**2) / precision
    return float(centre + rng.standard_normal() / np.sqrt(precision))


def sample(
    design: DesignMatrix,
    stats: SufficientStatistics,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 42,
    prior: PriorSpec | None = None,
    thin: int = 1,
) -> GibbsResult:
    prior = prior or PriorSpec()
    maps = design.index_maps
    upc_to_sub = np.asarray(maps["upc_to_sub"], dtype=int)
    sub_to_cat = np.asarray(maps["sub_to_cat"], dtype=int)
    n_upc, n_sub, n_cat = int(maps["n_upc"]), int(maps["n_sub"]), int(maps["n_cat"])
    p = stats.n_columns

    slices = {block.name: design.slice_for(block.name) for block in design.blocks}
    kept = len(range(0, draws, thin))

    out = GibbsResult(
        beta=np.empty((chains, kept, p)),
        sigma_y=np.empty((chains, kept, len(stats.n_obs))),
        mu_global=np.empty((chains, kept)),
        mu_category=np.empty((chains, kept, n_cat)),
        mu_sub=np.empty((chains, kept, n_sub)),
        sigma_category=np.empty((chains, kept)),
        sigma_sub=np.empty((chains, kept)),
        sigma_upc=np.empty((chains, kept)),
        mu_cross=np.empty((chains, kept)),
        sigma_cross=np.empty((chains, kept)),
        slices=slices,
    )

    for chain in range(chains):
        rng = np.random.default_rng(seed + 1000 * chain)
        state = GibbsState(
            beta=np.zeros(p),
            sigma_y=np.full(len(stats.n_obs), 0.6),
            mu_global=prior.elasticity_mean,
            mu_category=np.full(n_cat, prior.elasticity_mean),
            mu_sub=np.full(n_sub, prior.elasticity_mean),
            sigma_category=0.3,
            sigma_sub=0.3,
            sigma_upc=0.5,
            mu_cross=prior.cross_mean,
            sigma_cross=0.3,
        )

        stored = 0
        for iteration in range(tune + draws):
            precision, mean = _build_prior(state, slices, p, upc_to_sub, prior)
            state.beta = _draw_beta(stats, precision, mean, state.sigma_y, rng)
            state.sigma_y = _draw_sigma_y(stats, state.beta, prior, rng)

            elasticities = state.beta[slices["own_price"]]
            cross = state.beta[slices["cross_price"]]

            # Sub category means, pooled between their products and their category.
            for j in range(n_sub):
                members = elasticities[upc_to_sub == j]
                n_members = len(members)
                precision_j = n_members / state.sigma_upc**2 + 1.0 / state.sigma_sub**2
                centre = (
                    members.sum() / state.sigma_upc**2
                    + state.mu_category[sub_to_cat[j]] / state.sigma_sub**2
                ) / precision_j
                state.mu_sub[j] = centre + rng.standard_normal() / np.sqrt(precision_j)

            for k in range(n_cat):
                members = state.mu_sub[sub_to_cat == k]
                n_members = len(members)
                precision_k = n_members / state.sigma_sub**2 + 1.0 / state.sigma_category**2
                centre = (
                    members.sum() / state.sigma_sub**2
                    + state.mu_global / state.sigma_category**2
                ) / precision_k
                state.mu_category[k] = centre + rng.standard_normal() / np.sqrt(precision_k)

            state.mu_global = _draw_normal_mean(
                state.mu_category, state.sigma_category,
                prior.elasticity_mean, prior.elasticity_mean_sd, rng,
            )
            state.mu_cross = _draw_normal_mean(
                cross, state.sigma_cross, prior.cross_mean, prior.cross_mean_sd, rng
            )

            scale = prior.group_sd_prior_scale
            state.sigma_upc = _slice_sample_positive(
                lambda s: _log_half_normal(s, scale)
                - len(elasticities) * np.log(s)
                - 0.5 * np.sum((elasticities - state.mu_sub[upc_to_sub]) ** 2) / s**2,
                state.sigma_upc, rng,
            )
            state.sigma_sub = _slice_sample_positive(
                lambda s: _log_half_normal(s, scale)
                - n_sub * np.log(s)
                - 0.5 * np.sum((state.mu_sub - state.mu_category[sub_to_cat]) ** 2) / s**2,
                state.sigma_sub, rng,
            )
            state.sigma_category = _slice_sample_positive(
                lambda s: _log_half_normal(s, scale)
                - n_cat * np.log(s)
                - 0.5 * np.sum((state.mu_category - state.mu_global) ** 2) / s**2,
                state.sigma_category, rng,
            )
            state.sigma_cross = _slice_sample_positive(
                lambda s: _log_half_normal(s, scale)
                - n_sub * np.log(s)
                - 0.5 * np.sum((cross - state.mu_cross) ** 2) / s**2,
                state.sigma_cross, rng,
            )

            if iteration >= tune and (iteration - tune) % thin == 0:
                out.beta[chain, stored] = state.beta
                out.sigma_y[chain, stored] = state.sigma_y
                out.mu_global[chain, stored] = state.mu_global
                out.mu_category[chain, stored] = state.mu_category
                out.mu_sub[chain, stored] = state.mu_sub
                out.sigma_category[chain, stored] = state.sigma_category
                out.sigma_sub[chain, stored] = state.sigma_sub
                out.sigma_upc[chain, stored] = state.sigma_upc
                out.mu_cross[chain, stored] = state.mu_cross
                out.sigma_cross[chain, stored] = state.sigma_cross
                stored += 1

    return out
