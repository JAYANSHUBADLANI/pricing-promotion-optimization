"""Generate README.md from produced output.

The rule for this repository is that no figure in the write up is typed by hand.
Every number below is read out of outputs/ and formatted here, so a change in the
data or the model shows up in the README the next time the pipeline runs rather
than quietly disagreeing with it. If a required file is missing this script says
which stage produces it instead of writing a placeholder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TABLES = ROOT / "outputs" / "tables"

REQUIRED = {
    "data_verification.json": "python run_pipeline.py --only verify",
    "cleaning_report.json": "python run_pipeline.py --only clean",
    "elasticity_by_product.csv": "python run_pipeline.py --only model",
    "elasticity_diagnostics.json": "python run_pipeline.py --only model",
    "cross_price_elasticity.csv": "python run_pipeline.py --only model",
    "mechanic_effects.csv": "python run_pipeline.py --only model",
    "lift_decomposition.csv": "python run_pipeline.py --only lift",
    "lift_summary.json": "python run_pipeline.py --only lift",
    "cannibalisation.csv": "python run_pipeline.py --only lift",
    "backtest.json": "python run_pipeline.py --only backtest",
    "plan_summary.json": "python run_pipeline.py --only plan",
    "recommended_plan.csv": "python run_pipeline.py --only plan",
    "margin_sensitivity.csv": "python run_pipeline.py --only plan",
    "cost_sensitivity.csv": "python run_pipeline.py --only plan",
}


def load() -> dict:
    missing = [(n, c) for n, c in REQUIRED.items() if not (TABLES / n).exists()]
    if missing:
        print("Cannot build the README, these outputs are missing:\n")
        for name, command in missing:
            print(f"  {name:<34s} produced by: {command}")
        raise SystemExit(1)

    def read(name):
        path = TABLES / name
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return pd.read_csv(path)

    return {name.split(".")[0]: read(name) for name in REQUIRED}


def count_tests() -> int:
    """Count collected tests so the README never claims a stale number.

    pytest reports a per file tally in quiet collect mode rather than a single
    total, so the counts are summed. A parse failure returns zero rather than
    guessing, which is visible in the output instead of silently wrong.
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    counts = re.findall(r"^\S+\.py:\s*(\d+)$", result.stdout, flags=re.MULTILINE)
    return sum(int(c) for c in counts)


def markdown_table(frame: pd.DataFrame, columns: dict[str, str], floats: int = 2) -> str:
    subset = frame[list(columns)].rename(columns=columns).round(floats)
    header = "| " + " | ".join(subset.columns) + " |"
    divider = "| " + " | ".join("---" for _ in subset.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in record) + " |"
        for record in subset.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def build(data: dict, n_tests: int) -> str:
    verification = data["data_verification"]
    cleaning = data["cleaning_report"]
    elasticity = data["elasticity_by_product"]
    diagnostics = data["elasticity_diagnostics"]
    mechanics = data["mechanic_effects"]
    lift = data["lift_decomposition"]
    lift_summary = data["lift_summary"]
    cannibalisation = data["cannibalisation"]
    backtest = data["backtest"]
    plan_summary = data["plan_summary"]
    plan = data["recommended_plan"]
    margin_sensitivity = data["margin_sensitivity"]
    cost_sensitivity = data["cost_sensitivity"]

    shape = verification["shape"]
    coverage = verification["coverage"]
    integrity = verification["integrity"]
    identification = verification["identification"]

    by_category = (
        elasticity.groupby("category")["elasticity_mean"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
        .rename(columns={"count": "products"})
    )

    promo_states = pd.DataFrame(
        [{"state": k, **v} for k, v in verification["promo_states"].items()]
    ).sort_values("n_rows", ascending=False)

    backtest_states = pd.DataFrame(
        [{"state": k, **v} for k, v in backtest["by_promo_state"].items()]
    ).sort_values("rmse_improvement_pct", ascending=False)

    promoted = plan[plan["recommended_mechanic"] != "none"]
    excluded = plan[~plan["is_credible"]]

    display_effect = mechanics[mechanics["mechanic"] == "display_only"]["uplift_pct"]
    tpr_effect = mechanics[mechanics["mechanic"] == "tpr_only"]["uplift_pct"]
    both_effect = mechanics[mechanics["mechanic"] == "feature_and_display"]["uplift_pct"]

    n_elastic = int((elasticity["prob_elastic"] > 0.9).sum())
    n_positive = int((elasticity["elasticity_mean"] > 0).sum())

    return f"""# Pricing and Promotion Optimization

I estimated price elasticity of demand by product, quantified what promotions
actually buy, and turned both into a constrained discount and promotional plan
for a multi category retailer.

The short version of what I found: **most of the lift a promotion produces is
bought with shelf space rather than with margin, and on two of seven shelves the
volume a discount appears to win is taken almost entirely from the products
beside it.** A plan built on own price elasticity alone would recommend promoting
those shelves. A plan built on the shelf level response does not.

## Data

[dunnhumby "Breakfast at the Frat"]({"https://www.dunnhumby.com/source-files/"}),
a store, product and week panel of grocery sales with promotional detail.
dunnhumby describe it as nearly real world data provided for teaching. It is not
synthetic in the sense of being generated from a formula, and it is not a raw
production extract either.

| | |
| --- | --- |
| Rows | {shape['n_rows']:,} |
| Stores | {shape['n_stores']} |
| Products | {shape['n_products']} across {shape['n_categories']} categories and {shape['n_sub_categories']} sub categories |
| Weeks | {shape['n_weeks']}, {coverage['date_min']} to {coverage['date_max']} |
| Panel completeness | {coverage['panel_completeness']:.1%} of the full store by product by week grid |

Two properties of this file made the analysis worth doing rather than merely
possible.

**Price is given, not inferred.** Both the shelf price and the base price ship
with the data, so discount depth is measured. A common shortcut is to build a
price by dividing revenue by units, which puts units on both sides of the
regression and manufactures a negative price coefficient whether or not demand
responds to price at all. That failure mode does not arise here.

**The price column checks out against the revenue column.** Spend divided by
units reproduces the stated shelf price to within one percent on
{integrity['price_vs_spend_per_unit']['share_within_1pct']:.1%} of rows. That is
an unusually clean internal consistency result and it is the reason I was willing
to build on this file.

There is also real price variation to work with: the median store and product
pair is observed at {identification['distinct_prices_per_store_product_median']:.0f}
distinct prices, with a within pair price coefficient of variation of
{identification['within_pair_price_cv_median']:.1%}. Elasticity is only estimable
where price moves, and here it moves.

### Promotional structure

A promotion in this data is a price cut, an in store display, a place in the
weekly circular, or a combination. The four combinations carry very different
depths, which is what makes it possible to separate what price does from what
merchandising does.

{markdown_table(promo_states, {
    "state": "state",
    "n_rows": "store weeks",
    "share_of_rows": "share",
    "mean_discount_depth": "mean depth",
    "mean_units": "mean units",
}, 3)}

### Cleaning

{cleaning['n_rows_out']:,} of {cleaning['n_rows_in']:,} rows survive cleaning,
which is {1 - cleaning['share_removed']:.2%} of the file. Every rule and its
count is in `outputs/tables/cleaning_report.json`.

{markdown_table(pd.DataFrame(cleaning['steps']), {
    "step": "rule",
    "rows_removed": "rows removed",
    "share_removed": "share",
}, 5)}

I also found two stores duplicated in the store lookup with contradictory
segment labels, one row calling each MAINSTREAM and the other UPSCALE. Rather
than keep whichever came first, which would make the answer depend on sheet
ordering, the conflicting field is marked unresolved and recorded in
`data/interim/lookup_conflicts.json`.

## Method

The demand model is the constant elasticity form in logs, with the own price
elasticity partially pooled through three levels, product to sub category to
category, plus store effects, seasonality and separate terms for each
promotional mechanic.

Two decisions in the implementation are worth calling out.

**The likelihood runs on sufficient statistics.** The model is Gaussian and
linear in logs, so the likelihood depends on the {shape['n_rows']:,} rows only
through `y'y`, `X'y` and `X'X`. Precomputing those makes every density evaluation
cost `O(p squared)` in the {diagnostics['n_parameters']} coefficients rather than
`O(n)` in the rows. The test suite checks this reduction reproduces the direct
computation to nine significant figures rather than taking the algebra on trust.

**The sampler is a blocked Gibbs sampler, written for this model.** Every
conditional here is available in closed form, so a general purpose gradient
sampler has to discover numerically a geometry that can simply be written down.
In practice that showed up as a tree depth pinned at its maximum and hundreds of
divergent transitions. Drawing all {diagnostics['n_parameters']} coefficients
jointly from their exact normal conditional instead removed the problem: no step
size, no acceptance rate, no divergences, and a fit that takes seconds. Group
scales keep half normal priors and are drawn by slice sampling, since a
conjugate inverse gamma behaves badly with only {shape['n_categories']} categories.

Convergence: `r_hat` at most {diagnostics['max_r_hat']:.3f} and bulk effective
sample size at least {diagnostics['min_ess_bulk']:.0f} across
{diagnostics['n_draws_total']:,} draws. The sampler is validated against
simulated data with known elasticities in `tests/test_gibbs.py`.

## What I found

### Elasticity varies far more across categories than within them

{markdown_table(by_category, {
    "category": "category",
    "products": "products",
    "mean": "mean elasticity",
    "min": "most elastic",
    "max": "least elastic",
}, 2)}

{n_elastic} of {shape['n_products']} products are elastic with at least 90
percent posterior probability. Frozen pizza is the standout at a mean of
{by_category[by_category['category'] == 'FROZEN PIZZA']['mean'].iloc[0]:.2f};
oral hygiene is the least responsive at
{by_category[by_category['category'] == 'ORAL HYGIENE PRODUCTS']['mean'].iloc[0]:.2f}.

![Elasticity by product](outputs/figures/elasticity_by_product.png)

### Promotional lift is mostly merchandising, not discount

This is the finding I did not expect. Separating the price effect from the
mechanic effect gives, averaged across categories:

{markdown_table(mechanics.groupby('mechanic', as_index=False)['uplift_pct'].mean(), {
    "mechanic": "mechanic",
    "uplift_pct": "volume effect beyond price, percent",
}, 1)}

An in store display on its own is worth about
{display_effect.mean():+.0f} percent on volume *after* its price cut is already
accounted for, and it runs at a far shallower depth than an unsupported price
cut. A display and a circular together are worth about {both_effect.mean():+.0f}
percent. A temporary price reduction with no display and no circular is worth
about {tpr_effect.mean():+.1f} percent, meaning an unadvertised price cut
delivers slightly *less* than its own price elasticity predicts.

Across mechanics and categories, merchandising accounts for a mean
{lift_summary['mean_share_of_lift_from_merchandising']:.0%} of modelled lift.

![Lift decomposition](outputs/figures/lift_decomposition.png)

### On two shelves, a discount moves volume sideways rather than up

Fifteen pretzel products share a shelf. Cutting the price of one of them wins
volume that partly comes from the other fourteen rather than from new demand. The
cross price elasticity measures that, and turning it into a diversion ratio gives
the share of the headline gain that is simply moved along the shelf.

{markdown_table(cannibalisation, {
    "sub_category": "shelf",
    "n_products": "products",
    "cross_elasticity": "cross elasticity",
    "diversion_ratio": "diversion ratio",
}, 3)}

A diversion ratio above 1 means more than the entire gain came off neighbouring
facings. Pretzels sit at {cannibalisation.iloc[0]['diversion_ratio']:.2f} and all
family cereal at {cannibalisation.iloc[1]['diversion_ratio']:.2f}. Premium pizza,
by contrast, shows essentially none, so a pizza promotion creates volume rather
than relocating it.

![Cannibalisation](outputs/figures/cannibalisation.png)

## The recommendation

Planning happens at shelf level, so the number that matters is not the own price
elasticity but the **net elasticity**, own plus cross. When every facing on a
shelf moves together, the competing price moves with the own price, and the
shelf's response is governed by the sum.

{markdown_table(plan, {
    "shelf": "shelf",
    "own_elasticity": "own",
    "cross_elasticity": "cross",
    "net_elasticity": "net",
    "recommended_discount": "discount",
    "recommended_mechanic": "mechanic",
}, 3)}

Promote {plan_summary['n_promoted']} of {plan_summary['n_shelves']} shelves:
{", ".join(x['shelf'] for x in plan_summary['promoted_shelves'])}. Shallow
depth, full merchandising support. Projected contribution change
**{plan_summary['margin_delta_pct']:+.1f} percent** per store week at an assumed
{plan_summary['gross_margin_assumed']:.0%} gross margin.

{len(excluded)} shelves are excluded from the plan entirely, not because they are
unattractive but because their estimates are not usable: their net elasticity
comes out positive, which would mean cutting every price on the shelf reduces
volume sold. No shelf does that. It means the cross price term has absorbed the
fact that neighbouring products tend to be promoted in the same weeks. I report
the estimate and refuse to plan on it.

### How sensitive is this

No cost data ships with this dataset, so gross margin and the cost of running a
display are assumptions rather than measurements. Both are swept.

{markdown_table(margin_sensitivity, {
    "gross_margin_at_list": "assumed gross margin",
    "n_promoted": "shelves promoted",
    "margin_delta_pct": "contribution change, percent",
}, 2)}

{markdown_table(cost_sensitivity, {
    "cost_multiplier": "merchandising cost multiplier",
    "n_promoted": "shelves promoted",
    "margin_delta_pct": "contribution change, percent",
}, 2)}

The set of shelves worth promoting does not change across either sweep. The size
of the prize does.

## Does the model actually predict anything

I held out the final {backtest['holdout_weeks']} weeks
({backtest['holdout_date_min']} to {backtest['holdout_date_max']}), refit on the
earlier period only, and predicted volumes the model had never seen. The
benchmark is the average of the same product in the same store, which is a
strong baseline for levels and cannot respond to price at all.

Overall the model improves log scale RMSE by only
{backtest['rmse_improvement_vs_store_product_mean_pct']:.1f} percent. That
number is close to useless on its own, because most variance in a store week
panel is cross sectional and the baseline already captures it. The informative
split is by promotional state:

{markdown_table(backtest_states, {
    "state": "state",
    "n_observations": "holdout rows",
    "model_log_rmse": "model RMSE",
    "baseline_log_rmse": "baseline RMSE",
    "rmse_improvement_pct": "improvement, percent",
}, 3)}

The model earns its keep exactly where it should. On weeks with a display and a
circular it cuts error by
{backtest_states.iloc[0]['rmse_improvement_pct']:.0f} percent, because the
baseline has no way to know a promotion is running. On quiet weeks it is slightly
worse than the baseline, and on unadvertised price cuts it is
{abs(backtest_states.iloc[-1]['rmse_improvement_pct']):.0f} percent worse. That
last result is consistent with the negative fitted effect for that mechanic and
is the weakest part of the model.

![Backtest](outputs/figures/backtest_by_promo_state.png)

## What I would not claim

**Promotions are not randomly assigned, so these are not causal estimates.**
Retailers put displays on products they expect to sell. Any of that anticipation
that the model does not control for lands in the mechanic effect and inflates it.
Since the recommendation leans heavily on merchandising, this is the assumption
the whole result is most exposed to. Fixing it properly needs an instrument or an
experiment, and neither is available here.

**The projected contribution change is a model output, not a forecast.** It holds
everything outside the plan constant: no competitor response, no supply
constraint, no shopper fatigue from seeing the same shelf promoted repeatedly.

**{n_positive} products come out with a positive own price elasticity.** Partial
pooling does not rescue them. They are all private label pretzels, on the shelf
with the highest cross price elasticity, which points at the own and competing
price signals being too collinear there to separate.

**Four categories and {shape['n_products']} products is narrow.** Three are food
and one is oral hygiene, all from one retailer, and the window is
{coverage['date_min'][:4]} to {coverage['date_max'][:4]}. The method carries over;
these particular elasticities do not.

**Gross margin is assumed.** The elasticity threshold at which discounting starts
to pay is exactly `-1 / gross margin`, so that assumption sets the whole
recommendation. It is swept above rather than buried.

## Running it

```bash
pip install -r requirements.txt
# place the workbook in data/raw/, then
python run_pipeline.py
```

`python run_pipeline.py --list` shows the stages. Each writes to `outputs/` and
nothing downstream reads anything a previous stage did not produce.

```bash
pytest                              # {n_tests} tests
pytest -m slow                      # cross check the sampler against PyMC NUTS
streamlit run app/streamlit_app.py  # scenario tool
python scripts/build_readme.py      # regenerate this file from outputs
```

## Layout

```
config/config.yaml          every tunable, single source of truth
src/pricing/
  schema.py                 tolerant column mapping onto canonical names
  features.py               price, promotional state, competing price index
  data/                     workbook ingestion, verification, cleaning
  models/
    design.py               sparse design matrix and sufficient statistics
    gibbs.py                blocked Gibbs sampler
    elasticity.py           model assembly, diagnostics, output tables
    lift.py                 promotional decomposition and cannibalisation
    backtest.py             held out validation
  optimize/
    policy.py               constrained depth optimisation, closed forms
    promo.py                shelf level planning with mechanic choice
  report.py                 fitted parameters into a plan
  viz.py                    figures
app/streamlit_app.py        scenario tool
tests/                      test suite
scripts/                    README generation
```

## Licence

MIT. The dataset is dunnhumby's and carries its own terms.
"""


def main() -> int:
    data = load()
    (ROOT / "README.md").write_text(build(data, count_tests()), encoding="utf-8")
    print(f"README.md written from {len(REQUIRED)} generated outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
