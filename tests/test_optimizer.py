"""Tests for the constrained discount optimiser.

The most useful property of this model is that parts of it have a closed form
solution, so the numerical optimiser can be checked against an exact answer
rather than against a previously recorded output.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricing.optimize import (
    CategoryState,
    OptimizerConstraints,
    closed_form_margin_optimum,
    margin,
    margin_accretive_elasticity_threshold,
    max_discount_from_margin_floor,
    optimize_policy,
    revenue,
    units,
)
from pricing.optimize.policy import policy_under_posterior


def make_states() -> list[CategoryState]:
    return [
        CategoryState("Elastic", elasticity=-5.0, baseline_units=1000.0, reference_price=10.0),
        CategoryState("Moderate", elasticity=-3.5, baseline_units=500.0, reference_price=15.0),
        CategoryState("Inelastic", elasticity=-1.2, baseline_units=800.0, reference_price=20.0),
    ]


class TestDemandCurve:
    def test_no_discount_returns_baseline_units(self) -> None:
        state = CategoryState("A", -2.0, 1000.0, 10.0, baseline_discount=0.0)
        assert units(state, 0.0) == pytest.approx(1000.0)

    def test_discount_increases_units_for_negative_elasticity(self) -> None:
        state = CategoryState("A", -2.0, 1000.0, 10.0)
        assert units(state, 0.2) > units(state, 0.0)

    def test_baseline_discount_is_the_reference_point(self) -> None:
        state = CategoryState("A", -2.0, 1000.0, 10.0, baseline_discount=0.1)
        assert units(state, 0.1) == pytest.approx(1000.0)

    def test_elasticity_definition_holds(self) -> None:
        # A one percent price cut should move volume by beta percent.
        state = CategoryState("A", -2.5, 1000.0, 10.0)
        base = units(state, 0.0)
        moved = units(state, 0.01)
        pct_price = np.log(1.0 - 0.01)
        pct_units = np.log(moved / base)
        assert pct_units / pct_price == pytest.approx(-2.5, rel=1e-9)

    def test_revenue_is_price_times_units(self) -> None:
        state = CategoryState("A", -2.0, 1000.0, 10.0)
        d = 0.15
        assert revenue(state, d) == pytest.approx(units(state, d) * 10.0 * (1 - d))

    def test_margin_is_zero_when_discount_equals_list_margin(self) -> None:
        state = CategoryState("A", -2.0, 1000.0, 10.0)
        assert margin(state, 0.35, gross_margin_at_list=0.35) == pytest.approx(0.0)


class TestClosedForms:
    @pytest.mark.parametrize("m0", [0.20, 0.35, 0.50])
    def test_threshold_is_minus_one_over_margin(self, m0: float) -> None:
        assert margin_accretive_elasticity_threshold(m0) == pytest.approx(-1.0 / m0)

    @pytest.mark.parametrize("beta", [-0.5, -1.0, -2.0, -2.8])
    def test_flat_categories_should_not_be_discounted(self, beta: float) -> None:
        # At a 35 percent list margin the break even elasticity is about -2.86.
        assert closed_form_margin_optimum(beta, 0.35) == 0.0

    @pytest.mark.parametrize("beta", [-3.0, -4.0, -6.0, -10.0])
    def test_steep_categories_have_a_positive_optimum(self, beta: float) -> None:
        assert closed_form_margin_optimum(beta, 0.35) > 0.0

    @pytest.mark.parametrize("beta", [-3.0, -4.0, -6.0, -10.0])
    def test_closed_form_beats_neighbours_on_a_grid(self, beta: float) -> None:
        m0 = 0.35
        state = CategoryState("A", beta, 1000.0, 10.0)
        best = closed_form_margin_optimum(beta, m0)
        grid = np.linspace(0.0, m0 - 1e-6, 4001)
        values = margin(state, grid, m0)
        assert best == pytest.approx(float(grid[int(np.argmax(values))]), abs=2e-3)

    def test_margin_floor_maps_to_a_discount_cap(self) -> None:
        cap = max_discount_from_margin_floor(0.35, 0.15)
        assert cap == pytest.approx((0.35 - 0.15) / (1 - 0.15))
        # At that discount the realised margin rate sits exactly on the floor.
        realised = (0.35 - cap) / (1 - cap)
        assert realised == pytest.approx(0.15)

    def test_floor_above_list_margin_forbids_discounting(self) -> None:
        assert max_discount_from_margin_floor(0.20, 0.30) == 0.0


class TestOptimiser:
    def test_matches_closed_form_when_constraints_do_not_bind(self) -> None:
        constraints = OptimizerConstraints(
            max_discount=0.40,
            gross_margin_at_list=0.35,
            min_margin_floor=0.0,
            max_concurrent_promos=3,
        )
        states = make_states()
        result = optimize_policy(states, constraints, objective="margin", n_multistart=8)
        for state, chosen in zip(states, result.discounts):
            expected = closed_form_margin_optimum(state.elasticity, 0.35)
            assert chosen == pytest.approx(expected, abs=1e-4)

    def test_respects_the_margin_floor(self) -> None:
        constraints = OptimizerConstraints(
            max_discount=0.40,
            gross_margin_at_list=0.35,
            min_margin_floor=0.30,
            max_concurrent_promos=3,
        )
        states = [CategoryState("VeryElastic", -20.0, 1000.0, 10.0)]
        result = optimize_policy(states, constraints, objective="margin", n_multistart=8)
        cap = max_discount_from_margin_floor(0.35, 0.30)
        assert result.discounts.max() <= cap + 1e-9
        for row in result.per_category:
            assert row["realised_margin_rate"] >= 0.30 - 1e-9

    def test_respects_the_maximum_discount(self) -> None:
        constraints = OptimizerConstraints(
            max_discount=0.05,
            gross_margin_at_list=0.60,
            min_margin_floor=0.0,
            max_concurrent_promos=3,
            promo_cap_threshold=0.01,
        )
        states = [CategoryState("VeryElastic", -20.0, 1000.0, 10.0)]
        result = optimize_policy(states, constraints, objective="margin", n_multistart=8)
        assert result.discounts.max() <= 0.05 + 1e-9

    def test_respects_the_concurrent_promotion_cap(self) -> None:
        constraints = OptimizerConstraints(
            max_discount=0.30,
            gross_margin_at_list=0.50,
            min_margin_floor=0.0,
            max_concurrent_promos=1,
            promo_cap_threshold=0.05,
        )
        states = [
            CategoryState("A", -8.0, 1000.0, 10.0),
            CategoryState("B", -7.5, 1000.0, 10.0),
            CategoryState("C", -7.0, 1000.0, 10.0),
        ]
        result = optimize_policy(states, constraints, objective="margin", n_multistart=8)
        promoted = int(np.sum(result.discounts > constraints.promo_cap_threshold + 1e-9))
        assert promoted <= 1

    def test_cap_picks_the_most_valuable_category(self) -> None:
        constraints = OptimizerConstraints(
            max_discount=0.30,
            gross_margin_at_list=0.50,
            min_margin_floor=0.0,
            max_concurrent_promos=1,
            promo_cap_threshold=0.05,
        )
        states = [
            CategoryState("Small", -8.0, 10.0, 10.0),
            CategoryState("Large", -8.0, 10000.0, 10.0),
        ]
        result = optimize_policy(states, constraints, objective="margin", n_multistart=8)
        chosen = dict(zip(result.categories, result.discounts))
        assert chosen["Large"] > constraints.promo_cap_threshold
        assert chosen["Small"] <= constraints.promo_cap_threshold + 1e-9

    def test_optimised_margin_is_never_worse_than_baseline(self) -> None:
        constraints = OptimizerConstraints(min_margin_floor=0.10, max_concurrent_promos=2)
        result = optimize_policy(make_states(), constraints, objective="margin", n_multistart=8)
        assert result.optimised_margin >= result.baseline_margin - 1e-6

    def test_revenue_objective_goes_to_a_corner(self) -> None:
        # Revenue is monotone in discount, so an elastic category should sit at
        # the cap and an inelastic one at zero. This is a property of the demand
        # form, not a solver artefact, and it is why margin is the default.
        constraints = OptimizerConstraints(
            max_discount=0.25,
            gross_margin_at_list=0.35,
            min_margin_floor=0.0,
            max_concurrent_promos=3,
            promo_cap_threshold=0.05,
        )
        states = [
            CategoryState("Elastic", -4.0, 1000.0, 10.0),
            CategoryState("Inelastic", -0.6, 1000.0, 10.0),
        ]
        result = optimize_policy(states, constraints, objective="revenue", n_multistart=8)
        chosen = dict(zip(result.categories, result.discounts))
        assert chosen["Elastic"] == pytest.approx(0.25, abs=1e-4)
        assert chosen["Inelastic"] == pytest.approx(0.0, abs=1e-4)

    def test_subproblem_count_matches_the_admissible_subsets(self) -> None:
        # Three categories with a cap of two gives 1 + 3 + 3 = 7 subsets.
        constraints = OptimizerConstraints(max_concurrent_promos=2)
        result = optimize_policy(make_states(), constraints, n_multistart=3)
        assert result.n_subproblems == 7

    def test_rejects_an_empty_category_list(self) -> None:
        with pytest.raises(ValueError):
            optimize_policy([], OptimizerConstraints())


class TestValidation:
    def test_negative_reference_price_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            CategoryState("A", -2.0, 100.0, -1.0)

    def test_baseline_discount_of_one_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            CategoryState("A", -2.0, 100.0, 10.0, baseline_discount=1.0)

    def test_max_below_min_discount_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            OptimizerConstraints(min_discount=0.3, max_discount=0.1)


class TestUncertaintyPropagation:
    def test_posterior_spread_reflects_elasticity_spread(self) -> None:
        states = make_states()
        constraints = OptimizerConstraints(min_margin_floor=0.10, max_concurrent_promos=3)
        result = optimize_policy(states, constraints, n_multistart=8)
        rng = np.random.default_rng(0)
        draws = np.array([s.elasticity for s in states]) + rng.normal(
            0.0, 0.5, size=(200, len(states))
        )
        out = policy_under_posterior(states, draws, result.discounts, constraints)
        assert out["margin_delta"].shape == (200,)
        assert np.isfinite(out["margin_delta"]).all()
        assert out["margin_delta"].std() > 0.0

    def test_wrong_draw_shape_is_rejected(self) -> None:
        states = make_states()
        with pytest.raises(ValueError):
            policy_under_posterior(
                states, np.zeros((10, 2)), np.zeros(3), OptimizerConstraints()
            )
