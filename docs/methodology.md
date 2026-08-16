# Methodology

## Specification

Demand is modelled in the constant elasticity form. For product `i` in store `s`
in week `t`:

```
log(units) = a_i + a_s + b_i * log(price)
             + c_g * log(competing price index)
             + m[mechanic, category]
             + h[category] * holiday + r[category] * trend
             + seasonal[category]
             + error
```

`b_i` is the own price elasticity of product `i`. `c_g` is the cross price
elasticity of sub category `g`. Errors are normal with a separate variance per
category, because a frozen pizza store week and a mouthwash store week are not
equally noisy and forcing a shared variance distorts the credible intervals.

## Why the price variable is not constructed

A frequent shortcut in elasticity work is to build a price as revenue divided by
units, because a unit price is often not recorded. Doing that puts `units` on
both sides of the regression. Any measurement error in units then pushes the
constructed price the other way, producing a negative coefficient whether or not
demand responds to price at all. The bias is toward finding elasticity, which is
exactly the direction that makes the result look good.

That problem does not arise here. This dataset ships both the shelf price and the
base price, so discount depth is measured directly. As a check that the price
column is trustworthy, spend divided by units is compared against the stated
price on every row; the agreement is within one percent throughout.

## Partial pooling, and where it is not used

The own price elasticity is pooled through three levels: product, sub category,
category. Fitting each product independently produces estimates that are
implausible for the products with the least price variation, and pooling pulls
those toward the shelf they sit on while leaving genuinely distinctive products
free to differ. `tests/test_gibbs.py` checks this behaviour directly by thinning
one product's history and confirming its estimate moves toward its shelf mates.

Product and store intercepts are deliberately *not* pooled hierarchically. Every
product is observed across thousands of store weeks, so those intercepts are
determined by the data and pooling changes nothing in the answer. What it does
change is the posterior geometry: a scale parameter above a large block of tightly
determined coefficients creates a funnel, and the sampler pays for it. Pooling is
used where it earns its place and dropped where it does not.

## Cross price and cannibalisation

Each row carries the mean log price of the other products in the same sub
category, store and week, computed leave one out so nothing competes with itself.
The index is unweighted on purpose: weighting by observed volume would pull the
dependent variable into a regressor.

The coefficient on that index is a cross price elasticity. It converts into a
diversion ratio, the share of a promotion's apparent gain that came off
neighbouring facings rather than from new demand.

For planning, the relevant quantity is the **net elasticity**, own plus cross.
When an entire shelf is promoted together, the competing price moves with the own
price, so the shelf level response is the sum. A shelf whose net elasticity is not
negative is claiming that cutting every price on it reduces units sold. Nothing
does that, so such estimates are reported and refused as a basis for planning.

## Computation

### Sufficient statistics

The model is Gaussian and linear in logs, so the likelihood depends on the data
only through `y'y`, `X'y` and `X'X`, computed once per variance group. Every
density and gradient evaluation then costs `O(p squared)` in the coefficients
rather than `O(n)` in the rows. This is exact, not an approximation or a
subsample. `tests/test_design.py` verifies that the reduction reproduces a direct
computation to nine significant figures.

The design matrix is built sparse. A row touches roughly a dozen of the 243
columns, so a dense representation would cost about a gigabyte for no benefit.

### Blocked Gibbs

Every conditional in this model is available in closed form, which makes a
general purpose gradient sampler the wrong tool: it has to discover numerically a
geometry that can be written down. Attempting it produced a tree depth pinned at
the maximum and hundreds of divergent transitions.

The sampler used instead draws all 243 coefficients jointly from their exact
multivariate normal conditional, via one Cholesky factorisation per iteration.
Drawing them as a block rather than one at a time handles the strong correlations
between a product's intercept and its price slope exactly, and it is why no
identifying constraint is needed on the intercepts: the joint draw absorbs the
collinearity and proper priors keep the posterior proper.

Residual variances and hierarchical means are conjugate and drawn directly. Group
level standard deviations keep half normal priors, which are not conjugate, and
are drawn by slice sampling. That is a deliberate choice over an inverse gamma,
which is known to behave badly when there are only four groups.

The result has no step size, no acceptance rate and no divergences. Mixing is
still checked with `r_hat` and effective sample size, since being exact does not
exempt a sampler from being slow to mix.

## Optimisation

Contribution for a shelf at discount `d` under mechanic `m`:

```
margin(d, m) = units(d, m) * price * (m0 - d) - execution_cost(m)
```

where `m0` is the gross margin at list price. Two closed form results shape the
recommendation and are used in `tests/test_optimizer.py` to check the numerical
solver against an exact answer:

- Under a **revenue** objective the optimum is always a corner, because revenue
  is proportional to `(1 - d)^(1 + b)` and therefore monotone in depth. Revenue
  maximisation is not a usable pricing objective on its own.
- Under a **margin** objective the optimum is interior and positive only when the
  elasticity is steeper than `-1 / m0`. At a 35 percent list margin that is about
  -2.86. Anything flatter should not be discounted at any depth.

The cap on simultaneous promotions is combinatorial and is enforced exactly.
Because the objective is separable across shelves once the promoted set is fixed,
and cannibalisation is already absorbed into net elasticity, the best set of a
given size is the one with the largest gains from promoting, so no full subset
sweep is needed.

## Validation

The panel is split on time, the model refit on the earlier period only, and asked
to predict weeks it never saw under the prices and promotions actually run. The
holdout design inherits the training design's centring constants and level
counts; rebuilding them on the holdout shifts every intercept and quietly
flatters the metrics, which is checked in `tests/test_design.py`.

The benchmark is the average of the same product in the same store. That is a
strong baseline for levels, because most variance in a store week panel is cross
sectional, and it cannot respond to price at all. Reporting overall accuracy
against it understates what the model is for, so accuracy is broken out by
promotional state.

## Known limitations

**Promotions are not randomly assigned.** Displays go on products the buying team
expects to sell. Whatever part of that anticipation the model does not control for
lands in the mechanic effect and inflates it. This is the assumption the
recommendation is most exposed to, and no amount of additional history fixes it.

**No cost data.** Gross margin and the cost of running a display are assumptions.
The elasticity threshold at which discounting pays is exactly `-1 / m0`, so the
margin assumption sets the recommendation. Both are swept rather than asserted.

**Three products retain a positive own price elasticity.** They are all private
label pretzels, on the shelf with the highest cross price elasticity, which points
at own and competing price being too collinear there to separate. Pooling does not
rescue them.

**Errors are assumed homoscedastic within category.** Variance almost certainly
also varies by store size and by promotional state.

**Narrow scope.** Four categories, 55 products, one retailer, 2009 to 2012. The
method carries over; these particular elasticities do not.
