"""The sufficient statistics shortcut has to be exact, not approximately right.

The whole model rests on the claim that reducing half a million rows to y'y, X'y
and X'X loses nothing. These tests check that claim numerically rather than
trusting the algebra, and check the holdout path that quietly broke once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricing import features as F
from pricing import schema
from pricing.config import load_config
from pricing.models.design import (
    build_design,
    compute_sufficient_statistics,
    ols_reference,
)


@pytest.fixture(scope="module")
def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    weeks = pd.date_range("2010-01-03", periods=40, freq="7D")
    rows = []
    for store in range(4):
        for upc in range(6):
            for week_index, week in enumerate(weeks):
                base = 3.0 + 0.5 * upc
                depth = rng.choice([0.0, 0.1, 0.25], p=[0.7, 0.2, 0.1])
                price = base * (1 - depth)
                units = max(
                    1, int(np.exp(3.0 - 2.0 * np.log(price) + rng.normal(0, 0.2)))
                )
                rows.append(
                    {
                        schema.WEEK: week,
                        schema.STORE_ID: store,
                        schema.UPC: upc,
                        schema.UNITS: units,
                        schema.SPEND: units * price,
                        schema.PRICE: price,
                        schema.BASE_PRICE: base,
                        schema.FEATURE: int(depth > 0.2),
                        schema.DISPLAY: 0,
                        schema.TPR_ONLY: int(0 < depth <= 0.2),
                        schema.CATEGORY: "CAT_A" if upc < 3 else "CAT_B",
                        schema.SUB_CATEGORY: f"SUB_{upc // 3}",
                    }
                )
                del week_index
    frame = pd.DataFrame(rows)
    return F.build_features(frame, load_config())


class TestSufficientStatistics:
    def test_reproduce_residual_sum_of_squares_exactly(self, synthetic_panel) -> None:
        design = build_design(synthetic_panel)
        stats = compute_sufficient_statistics(design)
        beta = ols_reference(stats)

        residual = design.target - design.matrix @ beta
        for group in range(len(stats.group_labels)):
            mask = design.variance_group == group
            direct = float(residual[mask] @ residual[mask])
            via_stats = stats.residual_sum_of_squares(beta, group)
            assert via_stats == pytest.approx(direct, rel=1e-9)

    def test_cross_products_match_a_direct_computation(self, synthetic_panel) -> None:
        design = build_design(synthetic_panel)
        stats = compute_sufficient_statistics(design)
        dense = design.matrix.toarray()
        for group in range(len(stats.group_labels)):
            mask = design.variance_group == group
            np.testing.assert_allclose(
                stats.xtx[group], dense[mask].T @ dense[mask], rtol=1e-9, atol=1e-8
            )
            np.testing.assert_allclose(
                stats.xty[group], dense[mask].T @ design.target[mask], rtol=1e-9, atol=1e-8
            )

    def test_row_counts_add_up(self, synthetic_panel) -> None:
        design = build_design(synthetic_panel)
        stats = compute_sufficient_statistics(design)
        assert int(stats.n_obs.sum()) == len(synthetic_panel)


class TestDesignLayout:
    def test_blocks_cover_every_column_once(self, synthetic_panel) -> None:
        design = build_design(synthetic_panel)
        assert sum(b.size for b in design.blocks) == design.n_columns
        assert len(design.column_labels()) == design.n_columns

    def test_row_touches_only_a_handful_of_columns(self, synthetic_panel) -> None:
        # Sparsity here is structural rather than incidental: a row carries one
        # product, one store, one category and a fixed seasonal basis, so the
        # number of non zeros per row stays constant as levels are added while
        # the column count grows. That is what makes the sparse representation
        # worth having on the real panel.
        design = build_design(synthetic_panel)
        per_row = design.matrix.nnz / design.n_rows
        n_fourier = len(F.fourier_columns(load_config()))
        assert per_row <= 7 + n_fourier

    def test_price_is_centred_within_product(self, synthetic_panel) -> None:
        design = build_design(synthetic_panel)
        column = design.slice_for("own_price")
        block = design.matrix[:, column].toarray()
        # Each product's own price column must average to zero, which is what
        # makes its slope orthogonal to its intercept.
        for j in range(block.shape[1]):
            values = block[:, j]
            nonzero = values[values != 0]
            if len(nonzero):
                assert nonzero.mean() == pytest.approx(0.0, abs=1e-9)

    def test_nesting_violation_is_caught(self, synthetic_panel) -> None:
        broken = synthetic_panel.copy()
        broken.loc[broken.index[0], "sub_category_idx"] = 99
        with pytest.raises(ValueError, match="does not nest"):
            build_design(broken)


class TestHoldoutDesign:
    def test_reference_reuses_training_centring(self, synthetic_panel) -> None:
        # Rebuilding centring on a holdout window shifts every intercept and
        # silently corrupts out of sample metrics. The reference path exists to
        # prevent that, so it has to be checked.
        weeks = np.sort(synthetic_panel[schema.WEEK].unique())
        train = synthetic_panel[synthetic_panel[schema.WEEK] < weeks[-8]]
        holdout = synthetic_panel[synthetic_panel[schema.WEEK] >= weeks[-8]]

        train_design = build_design(train)
        holdout_design = build_design(holdout, reference=train_design)

        assert holdout_design.n_columns == train_design.n_columns
        np.testing.assert_allclose(
            holdout_design.centering["log_price_by_upc"],
            train_design.centering["log_price_by_upc"],
        )
        assert (
            holdout_design.centering["trend_scale"]
            == train_design.centering["trend_scale"]
        )

    def test_holdout_without_reference_would_differ(self, synthetic_panel) -> None:
        weeks = np.sort(synthetic_panel[schema.WEEK].unique())
        train = synthetic_panel[synthetic_panel[schema.WEEK] < weeks[-8]]
        holdout = synthetic_panel[synthetic_panel[schema.WEEK] >= weeks[-8]]
        train_design = build_design(train)
        naive = build_design(holdout)
        assert naive.centering["trend_scale"] != train_design.centering["trend_scale"]

    def test_unseen_level_is_rejected(self, synthetic_panel) -> None:
        train = synthetic_panel[synthetic_panel[schema.UPC] < 5]
        train_design = build_design(train)
        with pytest.raises(ValueError, match="absent from the reference"):
            build_design(synthetic_panel, reference=train_design)
