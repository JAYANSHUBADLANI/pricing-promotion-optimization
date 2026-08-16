"""Design matrix and sufficient statistics for the demand model.

Sampling a hierarchical model over half a million rows is normally the slow part
of a project like this. It does not have to be. The demand model is linear and
Gaussian in logs, so the likelihood depends on the data only through three
quantities per residual variance group:

    y'y        a scalar
    X'y        a vector of length p
    X'X        a p by p matrix

Once those are precomputed, every log density and gradient evaluation costs
O(p squared) regardless of how many rows produced them. With p around 240 that
turns a model that would otherwise sample for hours into one that samples in
seconds, and it is exact rather than an approximation or a subsample.

The design matrix itself is built sparse. Almost every column is a one hot
interaction, so a row touches roughly fourteen columns out of two hundred and
forty. Materialising it densely would cost about a gigabyte for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from pricing import features as F
from pricing import schema
from pricing.config import Config, load_config


@dataclass(frozen=True)
class BlockSpec:
    """One contiguous group of columns in the design matrix."""

    name: str
    size: int
    labels: tuple[str, ...]


@dataclass
class DesignMatrix:
    matrix: sparse.csr_matrix
    target: np.ndarray
    blocks: list[BlockSpec]
    variance_group: np.ndarray
    variance_group_labels: tuple[str, ...]
    centering: dict[str, float] = field(default_factory=dict)
    index_maps: dict[str, dict] = field(default_factory=dict)

    @property
    def n_columns(self) -> int:
        return self.matrix.shape[1]

    @property
    def n_rows(self) -> int:
        return self.matrix.shape[0]

    def slice_for(self, name: str) -> slice:
        start = 0
        for block in self.blocks:
            if block.name == name:
                return slice(start, start + block.size)
            start += block.size
        raise KeyError(f"No block named {name!r}")

    def column_labels(self) -> list[str]:
        labels: list[str] = []
        for block in self.blocks:
            labels.extend(f"{block.name}[{label}]" for label in block.labels)
        return labels


@dataclass
class SufficientStatistics:
    """Per group y'y, X'y, X'X and row counts."""

    yty: np.ndarray
    xty: np.ndarray
    xtx: np.ndarray
    n_obs: np.ndarray
    group_labels: tuple[str, ...]
    n_columns: int

    def residual_sum_of_squares(self, beta: np.ndarray, group: int) -> float:
        """Reconstruct the residual sum of squares without touching the raw rows."""
        return float(
            self.yty[group]
            - 2.0 * beta @ self.xty[group]
            + beta @ self.xtx[group] @ beta
        )


def _group_means(values: np.ndarray, codes: np.ndarray, n_levels: int) -> np.ndarray:
    """Mean of `values` within each level of `codes`, in level order."""
    totals = np.bincount(codes, weights=values, minlength=n_levels)
    counts = np.bincount(codes, minlength=n_levels).astype(float)
    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)


def _one_hot(codes: np.ndarray, n_levels: int, values: np.ndarray | None = None):
    n = len(codes)
    data = np.ones(n, dtype=float) if values is None else np.asarray(values, dtype=float)
    return sparse.csr_matrix(
        (data, (np.arange(n), codes)), shape=(n, n_levels)
    )


def build_design(
    frame: pd.DataFrame,
    cfg: Config | None = None,
    reference: "DesignMatrix | None" = None,
) -> DesignMatrix:
    """Assemble the sparse design matrix and the response vector.

    Continuous columns are centred so that the intercept stays interpretable and
    the normal equations stay well conditioned. Centring a log price shifts the
    intercept but leaves the elasticity coefficient untouched, which is the whole
    point of the log log form.

    Pass `reference` when building a design for data the model was not fitted on,
    such as a holdout window. It reuses the original centring constants and level
    counts, so a coefficient means the same thing in both windows. Recomputing
    them on the new data would shift every intercept and quietly flatter the
    out of sample numbers, which is the sort of error that does not announce
    itself: the code runs, the metrics look plausible, and they are wrong.
    """
    cfg = cfg or load_config()
    n = len(frame)

    upc_codes = frame["upc_idx"].to_numpy()
    store_codes = frame["store_idx"].to_numpy()
    sub_codes = frame["sub_category_idx"].to_numpy()
    cat_codes = frame["category_idx"].to_numpy()

    if reference is None:
        n_upc = int(upc_codes.max()) + 1
        n_store = int(store_codes.max()) + 1
        n_sub = int(sub_codes.max()) + 1
        n_cat = int(cat_codes.max()) + 1
    else:
        maps = reference.index_maps
        n_upc, n_store = int(maps["n_upc"]), int(maps["n_store"])
        n_sub, n_cat = int(maps["n_sub"]), int(maps["n_cat"])
        for codes, size, label in (
            (upc_codes, n_upc, "product"), (store_codes, n_store, "store"),
            (sub_codes, n_sub, "sub category"), (cat_codes, n_cat, "category"),
        ):
            if len(codes) and int(codes.max()) >= size:
                raise ValueError(
                    f"frame contains a {label} level absent from the reference design"
                )

    log_price = frame[F.LOG_PRICE].to_numpy(dtype=float)
    competing = frame[F.COMPETING_LOG_PRICE].to_numpy(dtype=float)

    # Centre each continuous predictor within the group whose slope it feeds.
    # Shifting a regressor cannot change its slope, so the elasticities are
    # untouched, but it makes each slope orthogonal to its own group intercept.
    # Without this the sampler spends every iteration walking a long diagonal
    # ridge between a product's intercept and its price slope, which shows up as
    # a tree depth pinned at the maximum and a fit that takes minutes instead of
    # seconds. The group means are kept so predictions can be reconstructed.
    if reference is None:
        price_centre = _group_means(log_price, upc_codes, n_upc)
        competing_centre = _group_means(competing, sub_codes, n_sub)
    else:
        price_centre = np.asarray(reference.centering["log_price_by_upc"], dtype=float)
        competing_centre = np.asarray(
            reference.centering["competing_log_price_by_sub_category"], dtype=float
        )
    log_price_c = log_price - price_centre[upc_codes]
    competing_c = competing - competing_centre[sub_codes]

    week = frame[F.WEEK_INDEX].to_numpy(dtype=float)
    if reference is None:
        trend_scale = float(week.max()) if week.max() > 0 else 1.0
        trend_centre = float((week / trend_scale).mean())
    else:
        trend_scale = float(reference.centering["trend_scale"])
        trend_centre = float(reference.centering["trend_centre"])
    trend = week / trend_scale - trend_centre

    fourier_cols = F.fourier_columns(cfg)
    fourier = frame[fourier_cols].to_numpy(dtype=float)

    parts: list[sparse.csr_matrix] = []
    blocks: list[BlockSpec] = []

    def add(name: str, block: sparse.csr_matrix, labels: Sequence[str]) -> None:
        parts.append(block)
        blocks.append(BlockSpec(name=name, size=block.shape[1], labels=tuple(labels)))

    if reference is None:
        upc_labels = [str(u) for u in sorted(frame[schema.UPC].unique())]
        store_labels = [str(s) for s in sorted(frame[schema.STORE_ID].unique())]
        sub_labels = [str(s) for s in sorted(frame[schema.SUB_CATEGORY].unique())]
        cat_labels = [str(c) for c in sorted(frame[schema.CATEGORY].unique())]
    else:
        maps = reference.index_maps
        upc_labels = list(maps["upc_labels"])
        store_labels = list(maps["store_labels"])
        sub_labels = list(maps["sub_labels"])
        cat_labels = list(maps["cat_labels"])

    add("intercept", sparse.csr_matrix(np.ones((n, 1))), ["global"])
    add("upc_intercept", _one_hot(upc_codes, n_upc), upc_labels)
    add("store_intercept", _one_hot(store_codes, n_store), store_labels)
    add("own_price", _one_hot(upc_codes, n_upc, log_price_c), upc_labels)
    add("cross_price", _one_hot(sub_codes, n_sub, competing_c), sub_labels)

    mechanic_labels: list[str] = []
    mechanic_blocks: list[sparse.csr_matrix] = []
    for mechanic in F.MECHANICS:
        flag = frame[f"is_{mechanic}"].to_numpy(dtype=float)
        mechanic_blocks.append(_one_hot(cat_codes, n_cat, flag))
        mechanic_labels.extend(f"{mechanic}|{c}" for c in cat_labels)
    add("mechanic", sparse.hstack(mechanic_blocks, format="csr"), mechanic_labels)

    holiday = frame[F.IS_HOLIDAY].to_numpy(dtype=float)
    add("holiday", _one_hot(cat_codes, n_cat, holiday), cat_labels)
    add("trend", _one_hot(cat_codes, n_cat, trend), cat_labels)

    seasonal_blocks = []
    seasonal_labels: list[str] = []
    for j, column in enumerate(fourier_cols):
        seasonal_blocks.append(_one_hot(cat_codes, n_cat, fourier[:, j]))
        seasonal_labels.extend(f"{column}|{c}" for c in cat_labels)
    add("seasonality", sparse.hstack(seasonal_blocks, format="csr"), seasonal_labels)

    matrix = sparse.hstack(parts, format="csr")
    # Interaction columns are built by scaling a one hot block, which leaves an
    # explicit stored zero wherever the interacting flag is off. Dropping them
    # cuts the stored entries per row by roughly a third and costs nothing.
    matrix.eliminate_zeros()
    target = frame[F.LOG_UNITS].to_numpy(dtype=float)

    return DesignMatrix(
        matrix=matrix,
        target=target,
        blocks=blocks,
        variance_group=cat_codes,
        variance_group_labels=tuple(cat_labels),
        centering={
            "log_price_by_upc": price_centre.tolist(),
            "competing_log_price_by_sub_category": competing_centre.tolist(),
            "trend_scale": trend_scale,
            "trend_centre": trend_centre,
        },
        index_maps={
            "upc_to_sub": (
                reference.index_maps["upc_to_sub"] if reference is not None
                else _child_to_parent(frame, "upc_idx", "sub_category_idx")
            ),
            "sub_to_cat": (
                reference.index_maps["sub_to_cat"] if reference is not None
                else _child_to_parent(frame, "sub_category_idx", "category_idx")
            ),
            "upc_to_cat": (
                reference.index_maps["upc_to_cat"] if reference is not None
                else _child_to_parent(frame, "upc_idx", "category_idx")
            ),
            "n_upc": n_upc,
            "n_store": n_store,
            "n_sub": n_sub,
            "n_cat": n_cat,
            "upc_labels": upc_labels,
            "store_labels": store_labels,
            "sub_labels": sub_labels,
            "cat_labels": cat_labels,
        },
    )


def _child_to_parent(frame: pd.DataFrame, child: str, parent: str) -> np.ndarray:
    """Map each child level onto its parent level, checking the nesting holds."""
    pairs = frame[[child, parent]].drop_duplicates()
    if pairs[child].duplicated().any():
        offenders = pairs.loc[pairs[child].duplicated(keep=False), child].unique()
        raise ValueError(
            f"{child} does not nest cleanly inside {parent}; offending levels: {offenders}"
        )
    return pairs.sort_values(child)[parent].to_numpy(dtype=int)


def compute_sufficient_statistics(design: DesignMatrix) -> SufficientStatistics:
    """Reduce the data to per group y'y, X'y and X'X.

    This is the step that decouples sampling cost from row count. It runs once.
    """
    p = design.n_columns
    labels = design.variance_group_labels
    n_groups = len(labels)

    yty = np.zeros(n_groups)
    xty = np.zeros((n_groups, p))
    xtx = np.zeros((n_groups, p, p))
    n_obs = np.zeros(n_groups, dtype=int)

    for g in range(n_groups):
        mask = design.variance_group == g
        rows = design.matrix[mask]
        y = design.target[mask]
        n_obs[g] = int(mask.sum())
        yty[g] = float(y @ y)
        xty[g] = np.asarray(rows.T @ y).ravel()
        xtx[g] = np.asarray((rows.T @ rows).todense())

    return SufficientStatistics(
        yty=yty, xty=xty, xtx=xtx, n_obs=n_obs,
        group_labels=labels, n_columns=p,
    )


def ols_reference(stats: SufficientStatistics, ridge: float = 1e-8) -> np.ndarray:
    """Pooled least squares solution from the sufficient statistics alone.

    Used to sanity check the Bayesian fit and to supply sensible starting values.
    A ridge term keeps the normal equations solvable despite the intercept being
    collinear with the group effects by construction.
    """
    xtx = stats.xtx.sum(axis=0)
    xty = stats.xty.sum(axis=0)
    return np.linalg.solve(xtx + ridge * np.eye(stats.n_columns), xty)
