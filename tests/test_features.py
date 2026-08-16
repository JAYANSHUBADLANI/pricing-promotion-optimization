"""Feature construction, especially the pieces the conclusions rest on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricing import features as F
from pricing import schema


def toy_frame() -> pd.DataFrame:
    """Two products on one shelf in one store across two weeks."""
    return pd.DataFrame(
        {
            schema.WEEK: pd.to_datetime(
                ["2010-01-02", "2010-01-02", "2010-01-09", "2010-01-09"]
            ),
            schema.STORE_ID: [1, 1, 1, 1],
            schema.UPC: [10, 20, 10, 20],
            schema.UNITS: [10, 20, 30, 40],
            schema.SPEND: [20.0, 40.0, 45.0, 80.0],
            schema.PRICE: [2.0, 2.0, 1.5, 2.0],
            schema.BASE_PRICE: [2.0, 2.0, 2.0, 2.0],
            schema.FEATURE: [0, 0, 0, 1],
            schema.DISPLAY: [0, 0, 1, 1],
            schema.TPR_ONLY: [0, 0, 0, 0],
            schema.CATEGORY: ["C", "C", "C", "C"],
            schema.SUB_CATEGORY: ["S", "S", "S", "S"],
        }
    )


class TestPromoState:
    def test_all_flags_off_is_the_baseline(self) -> None:
        state = F.label_promo_state(toy_frame())
        assert state.iloc[0] == F.MECHANIC_NONE

    def test_display_alone_is_labelled(self) -> None:
        assert F.label_promo_state(toy_frame()).iloc[2] == F.MECHANIC_DISPLAY

    def test_feature_and_display_together_is_its_own_state(self) -> None:
        assert F.label_promo_state(toy_frame()).iloc[3] == F.MECHANIC_FEATURE_DISPLAY

    def test_unexpected_combination_is_flagged_not_folded_into_baseline(self) -> None:
        # A row marked as both a temporary price reduction and a display is not
        # one of the documented states. It must not silently become no_promo,
        # which would contaminate the comparison group.
        frame = toy_frame()
        frame.loc[0, schema.TPR_ONLY] = 1
        frame.loc[0, schema.DISPLAY] = 1
        assert F.label_promo_state(frame).iloc[0] == F.MECHANIC_OTHER


class TestCompetingPriceIndex:
    def test_leaves_the_product_itself_out(self) -> None:
        frame = toy_frame()
        index, has_competitors = F.competing_price_index(frame)
        # In week two product 10 costs 1.5 and product 20 costs 2.0, so each
        # should see only the other's price.
        assert index.iloc[2] == pytest.approx(np.log(2.0))
        assert index.iloc[3] == pytest.approx(np.log(1.5))
        assert has_competitors.all()

    def test_falls_back_to_own_price_when_alone_on_the_shelf(self) -> None:
        frame = toy_frame().iloc[[0]].copy()
        index, has_competitors = F.competing_price_index(frame)
        assert not has_competitors.iloc[0]
        assert index.iloc[0] == pytest.approx(np.log(frame[schema.PRICE].iloc[0]))

    def test_index_is_not_weighted_by_units(self) -> None:
        # Weighting by observed volume would pull the dependent variable into a
        # regressor. Changing units must leave the index untouched.
        frame = toy_frame()
        before, _ = F.competing_price_index(frame)
        frame[schema.UNITS] = frame[schema.UNITS] * 17
        after, _ = F.competing_price_index(frame)
        pd.testing.assert_series_equal(before, after)


class TestDerivedColumns:
    def test_discount_depth_matches_price_over_base(self) -> None:
        frame = toy_frame()
        depth = 1.0 - frame[schema.PRICE] / frame[schema.BASE_PRICE]
        assert depth.iloc[2] == pytest.approx(0.25)

    def test_fourier_terms_are_bounded_and_paired(self) -> None:
        week = pd.Series(np.arange(200))
        terms = F.fourier_terms(week, n_terms=3, period=52.18)
        assert len(terms) == 6
        for values in terms.values():
            assert np.all(np.abs(values) <= 1.0 + 1e-12)

    def test_fourier_repeats_after_one_period(self) -> None:
        week = pd.Series([0.0, 52.18])
        terms = F.fourier_terms(week, n_terms=1, period=52.18)
        assert terms["fourier_sin_1"][0] == pytest.approx(terms["fourier_sin_1"][1], abs=1e-9)
