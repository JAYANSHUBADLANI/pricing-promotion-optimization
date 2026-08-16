"""Constrained discount policy optimisation.

The demand side is the constant elasticity form fitted upstream:

    units(p) = A * p ** beta

with beta the price elasticity of demand for a category, normally negative. If a
category currently sells `baseline_units` at a discount of `baseline_discount`
off a reference price P, then moving to discount d gives

    units(d) = baseline_units * ((1 - d) / (1 - baseline_discount)) ** beta

Revenue and contribution follow directly:

    revenue(d) = units(d) * P * (1 - d)
    margin(d)  = units(d) * P * (m0 - d)

where m0 is the gross margin rate on a unit sold at list price, so unit cost is
P * (1 - m0) and the contribution per discounted unit is P * (1 - d) - P * (1 - m0).

Two properties of this model shape everything below and are worth stating up
front, because they are the substance of the recommendation rather than an
artefact of the solver.

1. Under a revenue objective the optimum is always at a corner. Revenue is
   proportional to (1 - d) ** (1 + beta), which is monotone in d, so a category
   either goes to the deepest allowed discount (when beta < -1) or to no
   discount at all. Revenue maximisation alone is not a useful pricing
   objective, and the interesting problem is margin.

2. Under a margin objective there is an interior optimum, and it exists only for
   sufficiently elastic categories. Setting the derivative to zero gives

       d* = (1 + beta * m0) / (1 + beta)

   which is positive only when beta < -1 / m0. At a 35 percent list margin that
   threshold is an elasticity steeper than about -2.86. Any category flatter
   than that should not be discounted at all, whatever the promotional calendar
   says.

The optimiser below solves the constrained problem numerically with
scipy.optimize, and the closed forms above are used in the test suite to check
that the numerical answer is right where an analytic answer exists.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import minimize

Objective = Literal["margin", "revenue"]


@dataclass(frozen=True)
class CategoryState:
    """Baseline trading position for one product category."""

    category: str
    elasticity: float
    baseline_units: float
    reference_price: float
    baseline_discount: float = 0.0

    def __post_init__(self) -> None:
        if self.baseline_units < 0:
            raise ValueError(f"{self.category}: baseline_units must be non negative")
        if self.reference_price <= 0:
            raise ValueError(f"{self.category}: reference_price must be positive")
        if not 0.0 <= self.baseline_discount < 1.0:
            raise ValueError(f"{self.category}: baseline_discount must be in [0, 1)")


@dataclass(frozen=True)
class OptimizerConstraints:
    """Operating limits the recommended policy has to respect."""

    max_discount: float = 0.40
    min_discount: float = 0.0
    gross_margin_at_list: float = 0.35
    min_margin_floor: float = 0.15
    max_concurrent_promos: int = 3
    promo_cap_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_discount <= self.max_discount < 1.0:
            raise ValueError("Require 0 <= min_discount <= max_discount < 1")
        if not 0.0 < self.gross_margin_at_list < 1.0:
            raise ValueError("gross_margin_at_list must be in (0, 1)")
        if not 0.0 <= self.min_margin_floor < 1.0:
            raise ValueError("min_margin_floor must be in [0, 1)")
        if self.max_concurrent_promos < 0:
            raise ValueError("max_concurrent_promos must be non negative")


@dataclass
class PolicyResult:
    """The recommended discount per category and what it is expected to produce."""

    categories: list[str]
    discounts: np.ndarray
    baseline_discounts: np.ndarray
    objective: Objective
    baseline_revenue: float
    baseline_margin: float
    optimised_revenue: float
    optimised_margin: float
    per_category: list[dict[str, float]] = field(default_factory=list)
    n_promoted: int = 0
    binding_constraints: list[str] = field(default_factory=list)
    solver_success: bool = True
    n_subproblems: int = 0

    @property
    def revenue_delta(self) -> float:
        return self.optimised_revenue - self.baseline_revenue

    @property
    def margin_delta(self) -> float:
        return self.optimised_margin - self.baseline_margin

    @property
    def revenue_delta_pct(self) -> float:
        if self.baseline_revenue == 0:
            return float("nan")
        return 100.0 * self.revenue_delta / self.baseline_revenue

    @property
    def margin_delta_pct(self) -> float:
        if self.baseline_margin == 0:
            return float("nan")
        return 100.0 * self.margin_delta / self.baseline_margin


def units(state: CategoryState, discount: float | np.ndarray) -> float | np.ndarray:
    """Expected units at a given discount under constant elasticity demand."""
    price_ratio = (1.0 - np.asarray(discount, dtype=float)) / (1.0 - state.baseline_discount)
    return state.baseline_units * price_ratio**state.elasticity


def revenue(state: CategoryState, discount: float | np.ndarray) -> float | np.ndarray:
    d = np.asarray(discount, dtype=float)
    return units(state, d) * state.reference_price * (1.0 - d)


def margin(
    state: CategoryState,
    discount: float | np.ndarray,
    gross_margin_at_list: float,
) -> float | np.ndarray:
    d = np.asarray(discount, dtype=float)
    return units(state, d) * state.reference_price * (gross_margin_at_list - d)


def max_discount_from_margin_floor(
    gross_margin_at_list: float, min_margin_floor: float
) -> float:
    """Deepest discount that still clears a contribution margin floor.

    The realised margin rate at discount d is (m0 - d) / (1 - d). Requiring that
    to be at least the floor f rearranges to d <= (m0 - f) / (1 - f).
    """
    if min_margin_floor >= gross_margin_at_list:
        return 0.0
    return (gross_margin_at_list - min_margin_floor) / (1.0 - min_margin_floor)


def margin_accretive_elasticity_threshold(gross_margin_at_list: float) -> float:
    """Elasticity below which discounting starts to add contribution.

    Discounting is worth doing only when beta < -1 / m0. A category with an
    elasticity flatter than this threshold loses more on price than it recovers
    on volume, at any depth.
    """
    return -1.0 / gross_margin_at_list


def closed_form_margin_optimum(
    elasticity: float, gross_margin_at_list: float
) -> float:
    """Unconstrained margin maximising discount for one category.

    Contribution is proportional to (1 - d) ** beta * (m0 - d). Its derivative
    changes sign only where d * (1 + beta) = 1 + beta * m0, but that stationary
    point is a maximum inside the feasible range only when the derivative at
    d = 0 is positive, which happens exactly when beta < -1 / m0. For any
    category flatter than that threshold the contribution curve is decreasing
    from the start and the answer is no discount at all.

    Returns 0 when the category is not elastic enough for discounting to pay.
    """
    if elasticity >= margin_accretive_elasticity_threshold(gross_margin_at_list):
        return 0.0
    denominator = 1.0 + elasticity
    if abs(denominator) < 1e-12:
        return 0.0
    optimum = (1.0 + elasticity * gross_margin_at_list) / denominator
    if not np.isfinite(optimum) or optimum <= 0.0:
        return 0.0
    # The stationary point approaches m0 from below as elasticity steepens and
    # never reaches it, so this clip is a guard rather than an active bound.
    return float(min(optimum, gross_margin_at_list))


def _objective_fn(
    states: Sequence[CategoryState],
    constraints: OptimizerConstraints,
    objective: Objective,
) -> Callable[[np.ndarray], float]:
    """Total objective across categories, negated for a minimiser."""

    def total(discounts: np.ndarray) -> float:
        value = 0.0
        for state, d in zip(states, discounts):
            if objective == "revenue":
                value += float(revenue(state, float(d)))
            else:
                value += float(margin(state, float(d), constraints.gross_margin_at_list))
        return -value

    return total


def _evaluate(
    states: Sequence[CategoryState],
    discounts: np.ndarray,
    constraints: OptimizerConstraints,
) -> tuple[float, float, list[dict[str, float]]]:
    """Total revenue, total margin and a per category breakdown."""
    total_revenue = 0.0
    total_margin = 0.0
    rows: list[dict[str, float]] = []
    for state, d in zip(states, discounts):
        d = float(d)
        u = float(units(state, d))
        r = float(revenue(state, d))
        m = float(margin(state, d, constraints.gross_margin_at_list))
        total_revenue += r
        total_margin += m
        rows.append(
            {
                "units": u,
                "revenue": r,
                "margin": m,
                "discount": d,
                "realised_margin_rate": (
                    (constraints.gross_margin_at_list - d) / (1.0 - d)
                    if d < 1.0
                    else float("nan")
                ),
            }
        )
    return total_revenue, total_margin, rows


def _solve_subset(
    states: Sequence[CategoryState],
    constraints: OptimizerConstraints,
    objective: Objective,
    promoted: frozenset[int],
    upper: float,
    rng: np.random.Generator,
    n_multistart: int,
) -> tuple[np.ndarray, float, bool]:
    """Optimise depth for a fixed set of promoted categories.

    Categories outside `promoted` are held at or below the promotion threshold,
    which is how the cap on simultaneous promotions is enforced exactly rather
    than through a penalty term.
    """
    n = len(states)
    bounds: list[tuple[float, float]] = []
    for index in range(n):
        if index in promoted:
            low = max(constraints.min_discount, constraints.promo_cap_threshold)
            high = upper
            if low > high:
                # The margin floor forbids a promotion deep enough to count as
                # one, so this subset is infeasible.
                return np.zeros(n), -np.inf, False
            bounds.append((low, high))
        else:
            bounds.append(
                (
                    constraints.min_discount,
                    min(constraints.promo_cap_threshold, upper),
                )
            )

    fn = _objective_fn(states, constraints, objective)
    best_x = np.array([b[0] for b in bounds], dtype=float)
    best_value = -fn(best_x)
    success = False

    starts = [np.array([b[0] for b in bounds], dtype=float),
              np.array([b[1] for b in bounds], dtype=float),
              np.array([0.5 * (b[0] + b[1]) for b in bounds], dtype=float)]
    for _ in range(max(0, n_multistart - len(starts))):
        starts.append(
            np.array([rng.uniform(b[0], b[1]) for b in bounds], dtype=float)
        )

    for start in starts:
        with warnings.catch_warnings():
            # SLSQP evaluates trial points marginally outside the box and warns
            # each time. The bounds are still honoured in the returned solution,
            # so the warning is noise rather than a signal here.
            warnings.filterwarnings(
                "ignore", message="Values in x were outside bounds"
            )
            result = minimize(fn, start, method="SLSQP", bounds=bounds,
                              options={"maxiter": 300, "ftol": 1e-12})
        if result.success:
            success = True
            value = -float(result.fun)
            if value > best_value:
                best_value = value
                best_x = np.clip(result.x, [b[0] for b in bounds], [b[1] for b in bounds])
    return best_x, best_value, success


def optimize_policy(
    states: Sequence[CategoryState],
    constraints: OptimizerConstraints | None = None,
    objective: Objective = "margin",
    n_multistart: int = 20,
    seed: int = 42,
) -> PolicyResult:
    """Find the discount per category that maximises the chosen objective.

    The promotion cap is a combinatorial constraint, so the solver enumerates
    every admissible set of promoted categories and runs a bounded continuous
    optimisation inside each one. That is exact for the category counts a
    retailer actually plans against, at the cost of scaling poorly if the number
    of categories grows into the dozens. The number of subproblems solved is
    reported so that cost is visible rather than hidden.
    """
    constraints = constraints or OptimizerConstraints()
    states = list(states)
    if not states:
        raise ValueError("optimize_policy needs at least one category")

    rng = np.random.default_rng(seed)
    floor_cap = max_discount_from_margin_floor(
        constraints.gross_margin_at_list, constraints.min_margin_floor
    )
    upper = min(constraints.max_discount, floor_cap)
    upper = max(upper, constraints.min_discount)

    n = len(states)
    cap = min(constraints.max_concurrent_promos, n)

    best_x: np.ndarray | None = None
    best_value = -np.inf
    any_success = False
    n_subproblems = 0

    for size in range(cap + 1):
        for subset in itertools.combinations(range(n), size):
            n_subproblems += 1
            x, value, success = _solve_subset(
                states, constraints, objective, frozenset(subset), upper, rng, n_multistart
            )
            any_success = any_success or success
            if value > best_value:
                best_value = value
                best_x = x

    if best_x is None:
        raise RuntimeError("Optimiser found no feasible policy")

    baseline_discounts = np.array([s.baseline_discount for s in states], dtype=float)
    base_revenue, base_margin, _ = _evaluate(states, baseline_discounts, constraints)
    opt_revenue, opt_margin, rows = _evaluate(states, best_x, constraints)

    binding: list[str] = []
    if np.any(best_x >= upper - 1e-6) and upper < constraints.max_discount - 1e-9:
        binding.append("margin_floor")
    if np.any(best_x >= constraints.max_discount - 1e-6):
        binding.append("max_discount")
    n_promoted = int(np.sum(best_x > constraints.promo_cap_threshold + 1e-9))
    if n_promoted >= constraints.max_concurrent_promos and constraints.max_concurrent_promos < n:
        binding.append("max_concurrent_promos")

    for row, state in zip(rows, states):
        row["category"] = state.category  # type: ignore[assignment]
        row["elasticity"] = state.elasticity
        row["baseline_units"] = state.baseline_units
        row["reference_price"] = state.reference_price
        row["baseline_discount"] = state.baseline_discount

    return PolicyResult(
        categories=[s.category for s in states],
        discounts=best_x,
        baseline_discounts=baseline_discounts,
        objective=objective,
        baseline_revenue=base_revenue,
        baseline_margin=base_margin,
        optimised_revenue=opt_revenue,
        optimised_margin=opt_margin,
        per_category=rows,
        n_promoted=n_promoted,
        binding_constraints=sorted(set(binding)),
        solver_success=any_success,
        n_subproblems=n_subproblems,
    )


def policy_under_posterior(
    states: Sequence[CategoryState],
    elasticity_draws: np.ndarray,
    discounts: np.ndarray,
    constraints: OptimizerConstraints,
) -> dict[str, np.ndarray]:
    """Propagate elasticity uncertainty through a fixed policy.

    `elasticity_draws` has shape (n_draws, n_categories) and comes from the
    posterior of the hierarchical model. Holding the recommended discounts fixed
    and re-evaluating under each draw shows how much of the headline uplift
    survives the uncertainty in the elasticities themselves. A recommendation
    whose credible interval straddles zero is worth flagging rather than
    presenting as a point estimate.
    """
    elasticity_draws = np.atleast_2d(np.asarray(elasticity_draws, dtype=float))
    if elasticity_draws.shape[1] != len(states):
        raise ValueError(
            "elasticity_draws must have one column per category, got "
            f"{elasticity_draws.shape[1]} for {len(states)} categories"
        )

    n_draws = elasticity_draws.shape[0]
    revenue_delta = np.zeros(n_draws)
    margin_delta = np.zeros(n_draws)

    for i in range(n_draws):
        drawn = [
            CategoryState(
                category=s.category,
                elasticity=float(elasticity_draws[i, j]),
                baseline_units=s.baseline_units,
                reference_price=s.reference_price,
                baseline_discount=s.baseline_discount,
            )
            for j, s in enumerate(states)
        ]
        base_r, base_m, _ = _evaluate(
            drawn, np.array([s.baseline_discount for s in drawn]), constraints
        )
        opt_r, opt_m, _ = _evaluate(drawn, discounts, constraints)
        revenue_delta[i] = opt_r - base_r
        margin_delta[i] = opt_m - base_m

    return {"revenue_delta": revenue_delta, "margin_delta": margin_delta}


def states_from_frame(
    frame: Iterable[dict[str, float]],
) -> list[CategoryState]:
    """Build optimiser inputs from an iterable of record dicts."""
    return [
        CategoryState(
            category=str(row["category"]),
            elasticity=float(row["elasticity"]),
            baseline_units=float(row["baseline_units"]),
            reference_price=float(row["reference_price"]),
            baseline_discount=float(row.get("baseline_discount", 0.0)),
        )
        for row in frame
    ]
