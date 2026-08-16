"""Promotional planning: which shelves to promote, how deep, and with what support.

This sits on top of optimize/policy.py and changes the decision in two ways that
the data supports and that materially change the recommendation.

Net rather than own elasticity. A promotion is planned for a shelf, not for a
single facing, so when a sub category goes on promotion every product on it moves
together. When they all move together the competing price index moves with the
own price, and the volume response is governed by the sum of the own price and
cross price elasticities rather than by the own price elasticity alone. Ignoring
that is the single easiest way to overstate the case for discounting: pretzels
look responsive product by product and turn out to be close to inert as a shelf,
because most of what one facing gains is taken from the facing beside it.

Mechanic as a decision, not a given. A price cut can be run bare, backed by an
in store display, placed in the circular, or both. Those carry different volume
effects and different costs, and the fitted model estimates them separately, so
the plan chooses among them instead of assuming a depth alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from pricing.optimize.policy import (
    OptimizerConstraints,
    max_discount_from_margin_floor,
)

NO_PROMO = "none"


@dataclass(frozen=True)
class MechanicOption:
    """A way of supporting a price cut, and what it does to volume and cost."""

    name: str
    log_uplift: float = 0.0
    # Execution cost expressed as a fraction of the shelf's baseline revenue.
    # Display space and circular pages are not free, and a plan that treats them
    # as free will always recommend the most heavily supported option.
    cost_rate: float = 0.0

    @property
    def multiplier(self) -> float:
        return float(np.exp(self.log_uplift))


@dataclass(frozen=True)
class ShelfState:
    """Baseline trading position for one sub category, the unit of planning."""

    shelf: str
    category: str
    own_elasticity: float
    cross_elasticity: float
    baseline_units: float
    reference_price: float
    baseline_discount: float = 0.0
    n_products: int = 1

    @property
    def is_credible(self) -> bool:
        """Whether the estimated shelf response can be acted on.

        A shelf whose net elasticity is not negative is saying that cutting the
        price of everything on it would reduce the volume sold, which no shelf
        does. When that happens the cross price term has absorbed something other
        than substitution, most likely the fact that neighbouring products tend
        to be promoted in the same weeks, so the competing price index moves with
        unobserved demand rather than against it. The estimate is reported, and
        it is refused as a basis for a recommendation.
        """
        return self.net_elasticity < 0.0

    @property
    def net_elasticity(self) -> float:
        """Response of the whole shelf when every facing on it moves together.

        Own price elasticity measures what happens when one product moves and its
        neighbours hold still. A shelf promotion is not that. Adding the cross
        price elasticity converts a product level response into the shelf level
        response the plan actually buys.
        """
        return self.own_elasticity + self.cross_elasticity

    def units(self, discount: float, mechanic: MechanicOption) -> float:
        ratio = (1.0 - discount) / (1.0 - self.baseline_discount)
        return self.baseline_units * ratio**self.net_elasticity * mechanic.multiplier

    def revenue(self, discount: float, mechanic: MechanicOption) -> float:
        return self.units(discount, mechanic) * self.reference_price * (1.0 - discount)

    def margin(
        self, discount: float, mechanic: MechanicOption, gross_margin_at_list: float
    ) -> float:
        contribution = (
            self.units(discount, mechanic)
            * self.reference_price
            * (gross_margin_at_list - discount)
        )
        execution_cost = (
            mechanic.cost_rate * self.baseline_units * self.reference_price
        )
        return contribution - execution_cost


@dataclass
class PromoPlan:
    shelves: list[str]
    discounts: np.ndarray
    mechanics: list[str]
    baseline_margin: float
    optimised_margin: float
    baseline_revenue: float
    optimised_revenue: float
    rows: list[dict] = field(default_factory=list)
    n_promoted: int = 0
    n_evaluations: int = 0
    n_excluded_not_credible: int = 0

    @property
    def margin_delta(self) -> float:
        return self.optimised_margin - self.baseline_margin

    @property
    def margin_delta_pct(self) -> float:
        if self.baseline_margin == 0:
            return float("nan")
        return 100.0 * self.margin_delta / self.baseline_margin

    @property
    def revenue_delta_pct(self) -> float:
        if self.baseline_revenue == 0:
            return float("nan")
        return 100.0 * (self.optimised_revenue - self.baseline_revenue) / self.baseline_revenue


def default_mechanics() -> list[MechanicOption]:
    return [MechanicOption(NO_PROMO, 0.0, 0.0)]


def _best_depth_for(
    shelf: ShelfState,
    mechanic: MechanicOption,
    constraints: OptimizerConstraints,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Best depth for one shelf under one mechanic, by bounded scalar search.

    With the mechanic fixed the objective is one dimensional and smooth, so a
    bounded search is both faster and more reliable than a multistart gradient
    solve. The endpoints are checked explicitly because the optimum is frequently
    a corner rather than an interior point.
    """
    if upper <= lower:
        value = shelf.margin(lower, mechanic, constraints.gross_margin_at_list)
        return lower, value

    def negative(d: float) -> float:
        return -shelf.margin(float(d), mechanic, constraints.gross_margin_at_list)

    result = minimize_scalar(negative, bounds=(lower, upper), method="bounded")
    candidates = [lower, upper]
    if result.success:
        candidates.append(float(result.x))
    values = [
        shelf.margin(c, mechanic, constraints.gross_margin_at_list) for c in candidates
    ]
    best = int(np.argmax(values))
    return float(candidates[best]), float(values[best])


def optimise_promo_plan(
    shelves: Sequence[ShelfState],
    mechanics: Sequence[MechanicOption] | None = None,
    constraints: OptimizerConstraints | None = None,
    require_credible: bool = True,
) -> PromoPlan:
    """Choose depth and mechanic per shelf under the planning constraints.

    The cap on how many shelves can run at once is combinatorial, so admissible
    promoted sets are enumerated exactly rather than approximated with a penalty.
    Within a set, each shelf's depth and mechanic are chosen independently,
    which is valid because the objective is additive across shelves once the set
    is fixed and cannibalisation has already been absorbed into net elasticity.
    """
    constraints = constraints or OptimizerConstraints()
    mechanics = list(mechanics) if mechanics else default_mechanics()
    shelves = list(shelves)
    if not shelves:
        raise ValueError("optimise_promo_plan needs at least one shelf")

    no_promo = next(
        (m for m in mechanics if m.name == NO_PROMO), MechanicOption(NO_PROMO, 0.0, 0.0)
    )
    promo_mechanics = [m for m in mechanics if m.name != NO_PROMO] or [no_promo]

    floor_cap = max_discount_from_margin_floor(
        constraints.gross_margin_at_list, constraints.min_margin_floor
    )
    upper = max(min(constraints.max_discount, floor_cap), constraints.min_discount)
    promo_floor = max(constraints.min_discount, constraints.promo_cap_threshold)

    n = len(shelves)
    cap = min(constraints.max_concurrent_promos, n)
    evaluations = 0

    # Value of leaving a shelf alone, and the best it can do if promoted.
    idle_value: list[float] = []
    promoted_best: list[tuple[float, str, float]] = []
    for shelf in shelves:
        idle_value.append(
            shelf.margin(constraints.min_discount, no_promo, constraints.gross_margin_at_list)
        )
        best_value = -np.inf
        best_depth, best_mechanic = constraints.min_discount, no_promo.name
        if require_credible and not shelf.is_credible:
            # Left at baseline on purpose. Its estimated response is not usable,
            # and a plan that promotes on the strength of it would be reporting
            # confidence the model has not earned.
            promoted_best.append((constraints.min_discount, no_promo.name, idle_value[-1]))
            continue
        for mechanic in promo_mechanics:
            evaluations += 1
            depth, value = _best_depth_for(
                shelf, mechanic, constraints, promo_floor, upper
            )
            if value > best_value:
                best_value, best_depth, best_mechanic = value, depth, mechanic.name
        promoted_best.append((best_depth, best_mechanic, best_value))

    # With the set fixed the objective is separable, so the best set of a given
    # size is the one with the largest gains from promoting. Enumerating sizes up
    # to the cap and taking the top gains is exact and avoids a full subset sweep.
    gains = np.array([promoted_best[i][2] - idle_value[i] for i in range(n)])
    order = np.argsort(-gains)

    base_total = float(np.sum(idle_value))
    best_total = base_total
    best_set: tuple[int, ...] = ()
    for size in range(1, cap + 1):
        chosen = tuple(int(i) for i in order[:size])
        total = base_total + float(gains[list(chosen)].sum())
        if total > best_total + 1e-12:
            best_total = total
            best_set = chosen

    discounts = np.full(n, constraints.min_discount, dtype=float)
    chosen_mechanics = [no_promo.name] * n
    for i in best_set:
        discounts[i] = promoted_best[i][0]
        chosen_mechanics[i] = promoted_best[i][1]

    mechanic_lookup = {m.name: m for m in mechanics}
    if no_promo.name not in mechanic_lookup:
        mechanic_lookup[no_promo.name] = no_promo

    baseline_margin = baseline_revenue = 0.0
    optimised_margin = optimised_revenue = 0.0
    rows: list[dict] = []
    for i, shelf in enumerate(shelves):
        mechanic = mechanic_lookup[chosen_mechanics[i]]
        b_margin = shelf.margin(
            shelf.baseline_discount, no_promo, constraints.gross_margin_at_list
        )
        b_revenue = shelf.revenue(shelf.baseline_discount, no_promo)
        o_margin = shelf.margin(discounts[i], mechanic, constraints.gross_margin_at_list)
        o_revenue = shelf.revenue(discounts[i], mechanic)
        baseline_margin += b_margin
        baseline_revenue += b_revenue
        optimised_margin += o_margin
        optimised_revenue += o_revenue
        rows.append(
            {
                "shelf": shelf.shelf,
                "category": shelf.category,
                "n_products": shelf.n_products,
                "own_elasticity": shelf.own_elasticity,
                "cross_elasticity": shelf.cross_elasticity,
                "net_elasticity": shelf.net_elasticity,
                "is_credible": shelf.is_credible,
                "recommended_discount": float(discounts[i]),
                "recommended_mechanic": chosen_mechanics[i],
                "baseline_units": shelf.baseline_units,
                "reference_price": shelf.reference_price,
                "baseline_discount": shelf.baseline_discount,
                "projected_units": shelf.units(discounts[i], mechanic),
                "baseline_margin": b_margin,
                "projected_margin": o_margin,
                "margin_delta": o_margin - b_margin,
            }
        )

    return PromoPlan(
        shelves=[s.shelf for s in shelves],
        discounts=discounts,
        mechanics=chosen_mechanics,
        baseline_margin=baseline_margin,
        optimised_margin=optimised_margin,
        baseline_revenue=baseline_revenue,
        optimised_revenue=optimised_revenue,
        rows=rows,
        n_promoted=len(best_set),
        n_evaluations=evaluations,
        n_excluded_not_credible=int(sum(1 for s in shelves if not s.is_credible)),
    )


def margin_sensitivity(
    shelves: Sequence[ShelfState],
    mechanics: Sequence[MechanicOption],
    constraints: OptimizerConstraints,
    margins: Iterable[float],
) -> list[dict]:
    """Re-run the plan across assumed gross margins.

    No cost data ships with this dataset, so the gross margin is an assumption
    rather than a measurement, and it is the assumption the recommendation is
    most sensitive to. Sweeping it is the honest way to present the result: the
    reader sees where the advice changes rather than one number resting on a
    figure nobody verified.
    """
    out: list[dict] = []
    for margin in margins:
        adjusted = OptimizerConstraints(
            max_discount=constraints.max_discount,
            min_discount=constraints.min_discount,
            gross_margin_at_list=float(margin),
            min_margin_floor=min(constraints.min_margin_floor, float(margin) - 0.01),
            max_concurrent_promos=constraints.max_concurrent_promos,
            promo_cap_threshold=constraints.promo_cap_threshold,
        )
        plan = optimise_promo_plan(shelves, mechanics, adjusted)
        out.append(
            {
                "gross_margin_at_list": float(margin),
                "n_promoted": plan.n_promoted,
                "promoted_shelves": [
                    r["shelf"] for r in plan.rows if r["recommended_mechanic"] != NO_PROMO
                ],
                "mean_recommended_discount": float(
                    np.mean([r["recommended_discount"] for r in plan.rows])
                ),
                "margin_delta_pct": plan.margin_delta_pct,
            }
        )
    return out
