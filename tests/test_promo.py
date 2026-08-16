"""Shelf level planning: net elasticity, mechanic choice and the credibility gate."""

from __future__ import annotations

import numpy as np
import pytest

from pricing.optimize.policy import OptimizerConstraints
from pricing.optimize.promo import (
    NO_PROMO,
    MechanicOption,
    ShelfState,
    margin_sensitivity,
    optimise_promo_plan,
)


def shelf(name="SHELF", own=-2.5, cross=0.1, units=1000.0, price=4.0, **kwargs):
    return ShelfState(
        shelf=name, category="CAT", own_elasticity=own, cross_elasticity=cross,
        baseline_units=units, reference_price=price, **kwargs,
    )


def mechanics():
    return [
        MechanicOption(NO_PROMO, 0.0, 0.0),
        MechanicOption("display_only", 0.40, 0.02),
        MechanicOption("feature_and_display", 0.70, 0.045),
    ]


class TestNetElasticity:
    def test_net_is_own_plus_cross(self) -> None:
        assert shelf(own=-2.0, cross=0.5).net_elasticity == pytest.approx(-1.5)

    def test_substitution_makes_a_shelf_less_responsive_than_its_products(self) -> None:
        # This is the whole point of planning at shelf level. A product that looks
        # responsive on its own is less so once the gain taken from its
        # neighbours is netted off.
        product_level = shelf(own=-2.0, cross=0.0)
        shelf_level = shelf(own=-2.0, cross=1.2)
        assert abs(shelf_level.net_elasticity) < abs(product_level.net_elasticity)

    def test_cannibalisation_can_overwhelm_own_response(self) -> None:
        heavily_substituting = shelf(own=-0.5, cross=1.2)
        assert heavily_substituting.net_elasticity > 0
        assert not heavily_substituting.is_credible

    def test_a_normal_shelf_is_credible(self) -> None:
        assert shelf(own=-2.5, cross=0.1).is_credible


class TestDemand:
    def test_no_discount_and_no_mechanic_returns_baseline(self) -> None:
        state = shelf()
        assert state.units(0.0, MechanicOption(NO_PROMO)) == pytest.approx(1000.0)

    def test_mechanic_multiplies_volume(self) -> None:
        state = shelf()
        bare = state.units(0.1, MechanicOption(NO_PROMO))
        supported = state.units(0.1, MechanicOption("display_only", np.log(2.0)))
        assert supported == pytest.approx(2.0 * bare)

    def test_execution_cost_reduces_margin(self) -> None:
        state = shelf()
        free = MechanicOption("display_only", 0.4, 0.0)
        costly = MechanicOption("display_only", 0.4, 0.05)
        assert state.margin(0.1, costly, 0.35) < state.margin(0.1, free, 0.35)


class TestPlanning:
    def test_non_credible_shelves_are_never_promoted(self) -> None:
        shelves = [
            shelf("GOOD", own=-3.0, cross=0.05),
            shelf("BROKEN", own=-0.4, cross=1.3),
        ]
        plan = optimise_promo_plan(shelves, mechanics(), OptimizerConstraints())
        chosen = dict(zip(plan.shelves, plan.mechanics))
        assert chosen["BROKEN"] == NO_PROMO
        assert plan.n_excluded_not_credible == 1

    def test_credibility_gate_can_be_lifted_deliberately(self) -> None:
        shelves = [shelf("BROKEN", own=-0.4, cross=1.3)]
        gated = optimise_promo_plan(shelves, mechanics(), OptimizerConstraints())
        ungated = optimise_promo_plan(
            shelves, mechanics(), OptimizerConstraints(), require_credible=False
        )
        assert gated.n_promoted == 0
        assert ungated.n_promoted >= gated.n_promoted

    def test_respects_the_concurrent_promotion_cap(self) -> None:
        shelves = [shelf(f"S{i}", own=-3.0 - i * 0.1) for i in range(5)]
        constraints = OptimizerConstraints(max_concurrent_promos=2)
        plan = optimise_promo_plan(shelves, mechanics(), constraints)
        assert plan.n_promoted <= 2

    def test_respects_the_margin_floor(self) -> None:
        shelves = [shelf(own=-8.0)]
        constraints = OptimizerConstraints(
            gross_margin_at_list=0.35, min_margin_floor=0.30, max_concurrent_promos=1
        )
        plan = optimise_promo_plan(shelves, mechanics(), constraints)
        depth = plan.discounts[0]
        realised = (0.35 - depth) / (1 - depth)
        assert realised >= 0.30 - 1e-9

    def test_margin_never_falls_below_leaving_everything_alone(self) -> None:
        shelves = [shelf(f"S{i}", own=-1.0 - i * 0.5) for i in range(4)]
        plan = optimise_promo_plan(shelves, mechanics(), OptimizerConstraints())
        assert plan.optimised_margin >= plan.baseline_margin - 1e-6

    def test_prefers_the_larger_shelf_when_the_cap_binds(self) -> None:
        shelves = [
            shelf("SMALL", own=-3.0, units=10.0),
            shelf("LARGE", own=-3.0, units=10000.0),
        ]
        constraints = OptimizerConstraints(max_concurrent_promos=1)
        plan = optimise_promo_plan(shelves, mechanics(), constraints)
        chosen = dict(zip(plan.shelves, plan.mechanics))
        assert chosen["LARGE"] != NO_PROMO
        assert chosen["SMALL"] == NO_PROMO

    def test_expensive_mechanics_stop_being_worth_it(self) -> None:
        shelves = [shelf(own=-3.0)]
        cheap = optimise_promo_plan(shelves, mechanics(), OptimizerConstraints())
        dear = optimise_promo_plan(
            shelves,
            [MechanicOption(NO_PROMO, 0.0, 0.0),
             MechanicOption("display_only", 0.40, 5.0)],
            OptimizerConstraints(),
        )
        assert cheap.margin_delta >= dear.margin_delta

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            optimise_promo_plan([], mechanics(), OptimizerConstraints())


class TestSensitivity:
    def test_sweep_covers_every_requested_margin(self) -> None:
        shelves = [shelf(f"S{i}", own=-2.0 - i * 0.4) for i in range(3)]
        margins = [0.20, 0.30, 0.40]
        rows = margin_sensitivity(shelves, mechanics(), OptimizerConstraints(), margins)
        assert [r["gross_margin_at_list"] for r in rows] == margins

    def test_thinner_margins_do_not_recommend_deeper_cuts(self) -> None:
        # Discounting costs more when there is less margin to give away, so the
        # recommended depth should not increase as assumed margin falls.
        shelves = [shelf(own=-4.0)]
        rows = margin_sensitivity(
            shelves, [MechanicOption(NO_PROMO)], OptimizerConstraints(min_margin_floor=0.05),
            [0.20, 0.30, 0.40],
        )
        depths = [r["mean_recommended_discount"] for r in rows]
        assert depths == sorted(depths)
