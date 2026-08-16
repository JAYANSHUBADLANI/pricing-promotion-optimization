"""Does the sampler recover parameters it was never told?

A custom sampler is only worth having if it is right. The strongest available
check is to simulate data from the model with known coefficients, fit it, and ask
whether the truth lands inside the posterior. That catches sign errors, wrong
conditionals and mis-indexed hierarchies in a way that inspecting the code does
not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricing import features as F
from pricing import schema
from pricing.config import load_config
from pricing.models import gibbs
from pricing.models.design import build_design, compute_sufficient_statistics, ols_reference

TRUE_ELASTICITY = {0: -3.0, 1: -2.4, 2: -1.8, 3: -1.2, 4: -0.8, 5: -2.0}
NOISE_SD = 0.15


def simulate(seed: int = 11, n_weeks: int = 60, n_stores: int = 6) -> pd.DataFrame:
    """Generate a panel from a known log log demand curve."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2010-01-03", periods=n_weeks, freq="7D")
    store_effect = rng.normal(0, 0.3, size=n_stores)
    rows = []
    for store in range(n_stores):
        for upc, elasticity in TRUE_ELASTICITY.items():
            base_price = 2.0 + 0.4 * upc
            for week in weeks:
                depth = rng.choice([0.0, 0.10, 0.20, 0.30], p=[0.6, 0.15, 0.15, 0.10])
                price = base_price * (1 - depth)
                log_units = (
                    4.0
                    + store_effect[store]
                    + 0.2 * upc
                    + elasticity * np.log(price)
                    + rng.normal(0, NOISE_SD)
                )
                units = max(1, int(np.exp(log_units)))
                rows.append(
                    {
                        schema.WEEK: week,
                        schema.STORE_ID: store,
                        schema.UPC: upc,
                        schema.UNITS: units,
                        schema.SPEND: units * price,
                        schema.PRICE: price,
                        schema.BASE_PRICE: base_price,
                        schema.FEATURE: 0,
                        schema.DISPLAY: 0,
                        schema.TPR_ONLY: int(depth > 0),
                        schema.CATEGORY: "CAT_A" if upc < 3 else "CAT_B",
                        schema.SUB_CATEGORY: f"SUB_{upc // 3}",
                    }
                )
    return F.build_features(pd.DataFrame(rows), load_config())


@pytest.fixture(scope="module")
def fitted():
    frame = simulate()
    design = build_design(frame)
    stats = compute_sufficient_statistics(design)
    result = gibbs.sample(design, stats, draws=600, tune=300, chains=2, seed=5)
    return frame, design, stats, result


class TestParameterRecovery:
    def test_elasticities_land_near_the_truth(self, fitted) -> None:
        _, design, _, result = fitted
        draws = result.block("own_price")
        posterior_mean = draws.mean(axis=(0, 1))
        truth = np.array([TRUE_ELASTICITY[i] for i in range(len(posterior_mean))])
        np.testing.assert_allclose(posterior_mean, truth, atol=0.25)

    def test_truth_sits_inside_the_credible_interval(self, fitted) -> None:
        _, _, _, result = fitted
        draws = result.block("own_price").reshape(-1, len(TRUE_ELASTICITY))
        low = np.quantile(draws, 0.005, axis=0)
        high = np.quantile(draws, 0.995, axis=0)
        truth = np.array([TRUE_ELASTICITY[i] for i in range(draws.shape[1])])
        inside = (truth >= low) & (truth <= high)
        assert inside.all(), f"outside interval: {np.where(~inside)[0]}"

    def test_residual_scale_is_recovered(self, fitted) -> None:
        _, _, _, result = fitted
        assert result.sigma_y.mean() == pytest.approx(NOISE_SD, abs=0.05)

    def test_ordering_of_elasticities_is_preserved(self, fitted) -> None:
        _, _, _, result = fitted
        posterior_mean = result.block("own_price").mean(axis=(0, 1))
        truth = np.array([TRUE_ELASTICITY[i] for i in range(len(posterior_mean))])
        assert np.corrcoef(posterior_mean, truth)[0, 1] > 0.98


class TestSamplerBehaviour:
    def test_chains_agree_with_each_other(self, fitted) -> None:
        _, _, _, result = fitted
        per_chain = result.block("own_price").mean(axis=1)
        spread = per_chain.max(axis=0) - per_chain.min(axis=0)
        assert spread.max() < 0.15

    def test_posterior_is_not_a_point_mass(self, fitted) -> None:
        _, _, _, result = fitted
        assert result.block("own_price").std(axis=(0, 1)).min() > 0.0

    def test_group_scales_stay_positive(self, fitted) -> None:
        _, _, _, result = fitted
        for values in (result.sigma_upc, result.sigma_sub, result.sigma_category,
                       result.sigma_cross, result.sigma_y):
            assert np.all(values > 0)

    def test_agrees_with_least_squares_when_pooling_is_weak(self, fitted) -> None:
        # With plenty of data per product the hierarchy should barely shrink, so
        # the posterior means should sit close to the unpooled solution. A large
        # gap would mean the prior is doing work the data should be doing.
        _, design, stats, result = fitted
        ols = ols_reference(stats)[design.slice_for("own_price")]
        posterior_mean = result.block("own_price").mean(axis=(0, 1))
        np.testing.assert_allclose(posterior_mean, ols, atol=0.2)

    def test_draw_shapes_are_as_requested(self, fitted) -> None:
        _, design, stats, result = fitted
        assert result.beta.shape == (2, 600, stats.n_columns)
        assert result.sigma_y.shape[2] == len(stats.group_labels)


class TestShrinkage:
    def test_a_sparsely_observed_product_is_pulled_towards_its_shelf(self) -> None:
        # Partial pooling should matter most where the data is thinnest. Cutting
        # one product down to a handful of weeks should move its estimate towards
        # its neighbours relative to the unpooled fit.
        frame = simulate(seed=3)
        sparse_upc = 4
        keep = frame[schema.UPC] != sparse_upc
        thin = frame[(frame[schema.UPC] == sparse_upc)].head(8)
        reduced = pd.concat([frame[keep], thin], ignore_index=True)
        reduced = F.build_features(
            reduced.drop(columns=[c for c in reduced.columns if c.startswith("fourier_")]),
            load_config(),
        )

        design = build_design(reduced)
        stats = compute_sufficient_statistics(design)
        result = gibbs.sample(design, stats, draws=400, tune=200, chains=2, seed=9)

        pooled = result.block("own_price").mean(axis=(0, 1))
        unpooled = ols_reference(stats)[design.slice_for("own_price")]
        shelf_mates = [i for i in range(len(pooled)) if i // 3 == sparse_upc // 3 and i != sparse_upc]
        shelf_mean = pooled[shelf_mates].mean()

        assert abs(pooled[sparse_upc] - shelf_mean) < abs(unpooled[sparse_upc] - shelf_mean)
