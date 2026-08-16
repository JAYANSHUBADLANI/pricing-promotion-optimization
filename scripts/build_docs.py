"""Generate the executive summary and business case from produced output.

Same rule as the README: the prose is written here, the numbers are read from
outputs/. A document that quotes a figure the pipeline no longer produces is
worse than no document, because it looks authoritative.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TABLES = ROOT / "outputs" / "tables"
DOCS = ROOT / "docs"


def read(name: str):
    path = TABLES / name
    if not path.exists():
        raise SystemExit(
            f"Missing {path.relative_to(ROOT)}. Run `python run_pipeline.py` first."
        )
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def executive_summary() -> str:
    verification = read("data_verification.json")
    plan_summary = read("plan_summary.json")
    plan = read("recommended_plan.csv")
    cannibalisation = read("cannibalisation.csv")
    mechanics = read("mechanic_effects.csv")
    backtest = read("backtest.json")
    lift_summary = read("lift_summary.json")
    margin_sensitivity = read("margin_sensitivity.csv")

    shape = verification["shape"]
    promoted = ", ".join(x["shelf"] for x in plan_summary["promoted_shelves"])
    excluded = plan[~plan["is_credible"]]["shelf"].tolist()
    by_mechanic = mechanics.groupby("mechanic")["uplift_pct"].mean()
    heavy = cannibalisation[cannibalisation["diversion_ratio"] > 1.0]
    best_state = max(
        backtest["by_promo_state"].items(), key=lambda kv: kv[1]["rmse_improvement_pct"]
    )
    worst_state = min(
        backtest["by_promo_state"].items(), key=lambda kv: kv[1]["rmse_improvement_pct"]
    )

    return f"""# Executive summary

## The question

Which product shelves should carry a promotion next quarter, how deep should the
discount go, and what promotional support should sit behind it.

## What I did

I fitted a hierarchical demand model to {shape['n_rows']:,} store weeks covering
{shape['n_products']} products across {shape['n_stores']} stores and
{shape['n_weeks']} weeks, estimating a separate price elasticity for every
product while pooling information across shelves and categories. I then measured
what each promotional mechanic adds beyond its price cut, measured how much of a
promotion's gain is taken from neighbouring products rather than created, and
solved for the plan that maximises contribution under a margin floor, a maximum
depth and a cap on how many shelves can promote at once.

## The recommendation

Promote {plan_summary['n_promoted']} of {plan_summary['n_shelves']} shelves:
**{promoted}**. Run them at shallow depth with full merchandising support rather
than deep discounts with none. Projected contribution change of
**{plan_summary['margin_delta_pct']:+.0f} percent** per store week at an assumed
{plan_summary['gross_margin_assumed']:.0%} gross margin, holding between
{margin_sensitivity['margin_delta_pct'].min():.0f} and
{margin_sensitivity['margin_delta_pct'].max():.0f} percent across every gross
margin assumption tested.

## The three findings behind it

**1. Merchandising does most of the work, not discount depth.** An in store
display is worth roughly {by_mechanic.get('display_only', 0):+.0f} percent on
volume after its price cut is already accounted for. A display and a circular
together are worth about {by_mechanic.get('feature_and_display', 0):+.0f}
percent. A price cut with no display and no circular is worth about
{by_mechanic.get('tpr_only', 0):+.1f} percent, which is to say it slightly
underdelivers against its own price elasticity. Across mechanics and categories
merchandising accounts for a mean
{lift_summary['mean_share_of_lift_from_merchandising']:.0%} of modelled lift.
The practical reading is that promotional budget is better spent on space and
visibility than on depth.

**2. On some shelves a discount relocates volume instead of creating it.** For
{len(heavy)} of {len(cannibalisation)} shelves the diversion ratio exceeds 1,
meaning more than the whole apparent gain comes off neighbouring facings:
{", ".join(f"{r.sub_category} at {r.diversion_ratio:.2f}" for r in heavy.itertuples())}.
Premium pizza shows essentially none. Planning on own price elasticity alone
would recommend promoting the shelves that cannibalise hardest.

**3. Two shelves cannot be planned on at all.** {", ".join(excluded)} produce a
net price response that is positive, which would mean cutting every price on the
shelf sells fewer units. That is not believable, and it indicates the model has
picked up the tendency for neighbouring products to be promoted in the same weeks
rather than genuine substitution. They are reported and excluded rather than
quietly included.

## How much to trust it

The model was refit on data ending {backtest['train_date_max']} and asked to
predict {backtest['holdout_weeks']} weeks it had never seen. Against a per store
per product average it cuts error by
{best_state[1]['rmse_improvement_pct']:.0f} percent on weeks running a display
and a circular, which is where a demand model should earn its keep. On
{worst_state[0].replace('_', ' ')} weeks it is
{abs(worst_state[1]['rmse_improvement_pct']):.0f} percent worse than that
baseline, which is the weakest part of the result.

The largest caveat is that promotions are not randomly assigned. Retailers put
displays on products they expect to sell, and any of that anticipation the model
does not control for inflates the merchandising effect. Since the recommendation
leans on merchandising, that is the assumption the conclusion is most exposed to.
Separating the two properly requires a deliberate test rather than more history.

## What I would do next

Run the recommended plan against a holdout set of stores for a quarter, with
mechanic assignment randomised within matched store pairs. That converts the
largest assumption in this work into a measurement, and it is cheap relative to
the promotional spend it would inform.
"""


def business_case() -> str:
    verification = read("data_verification.json")
    plan = read("recommended_plan.csv")
    plan_summary = read("plan_summary.json")
    cannibalisation = read("cannibalisation.csv")
    elasticity = read("elasticity_by_product.csv")
    cost_sensitivity = read("cost_sensitivity.csv")

    shape = verification["shape"]
    promoted = plan[plan["recommended_mechanic"] != "none"]
    left_alone = plan[
        (plan["recommended_mechanic"] == "none") & (plan["is_credible"])
    ]
    pizza = elasticity[elasticity["category"] == "FROZEN PIZZA"]["elasticity_mean"].mean()
    oral = elasticity[
        elasticity["category"] == "ORAL HYGIENE PRODUCTS"
    ]["elasticity_mean"].mean()
    pretzels = cannibalisation[cannibalisation["sub_category"] == "PRETZELS"]

    return f"""# Business case

## Who this is for

A category manager at a regional grocery chain, building next quarter's
promotional calendar. She has {shape['n_sub_categories']} shelves she could
promote across {shape['n_categories']} categories, room to run
{plan_summary['max_concurrent_promos']} at a time without the store looking like
a permanent sale, and a finance partner who will ask what the margin cost buys.

Her current process is the one most teams use: last year's calendar, adjusted for
what the buyers negotiated and what looked like it worked. The weakness is not
effort, it is that "looked like it worked" compares a promoted week against a
normal week and calls the difference the effect of the promotion. That
comparison cannot tell her three things she needs.

## The three questions the usual approach cannot answer

**Was the lift worth the margin?** A shelf that sells 60 percent more units at 25
percent off has not necessarily made money. Whether it has depends on the
elasticity, and elasticity varies enormously here: frozen pizza averages
{pizza:.1f} while oral hygiene averages {oral:.1f}. Treating them the same is
the expensive mistake.

**Did the promotion create volume or move it?** When she promotes one pretzel
brand, the units it gains partly come from the other fourteen pretzel products on
the same shelf. The category total barely moves while the margin cost is real.
This analysis puts a number on that: pretzels divert
{pretzels['diversion_ratio'].iloc[0]:.2f} of the apparent gain from neighbouring
facings, meaning the shelf as a whole loses.

**Was it the price or the display?** Promotions bundle a price cut with a display
and a circular, so the two are almost never separated. They have very different
costs. Giving away margin is expensive; putting a product at the end of an aisle
is comparatively cheap. If most of the lift is coming from the placement, she is
paying for the wrong half.

## What the analysis changes

The recommendation is to promote {plan_summary['n_promoted']} shelves rather
than the {plan_summary['n_shelves']} available, at shallow depth with full
merchandising support:

{chr(10).join(f"- **{r.shelf}** at {r.recommended_discount:.0%} with {r.recommended_mechanic.replace('_', ' ')}, net elasticity {r.net_elasticity:.2f}" for r in promoted.itertuples())}

And to leave alone:

{chr(10).join(f"- **{r.shelf}**, net elasticity {r.net_elasticity:.2f}, not responsive enough to pay for the margin given away" for r in left_alone.itertuples())}

The shift in thinking is from depth to support. The plan spends its promotional
budget on placement and visibility, and keeps discounts shallow enough that the
margin floor is never close to binding.

## What it is worth

Projected contribution change of {plan_summary['margin_delta_pct']:+.0f} percent
per store week. That number moves with what merchandising actually costs, which
this dataset does not record. Scaling the assumed execution cost from half to
{cost_sensitivity['cost_multiplier'].max():.0f} times the baseline gives a range
of {cost_sensitivity['margin_delta_pct'].min():.0f} to
{cost_sensitivity['margin_delta_pct'].max():.0f} percent, and the set of shelves
worth promoting does not change anywhere in that range.

The honest framing for the finance conversation is not the point estimate. It is
that the ranking is robust and the magnitude is not.

## What she should ask for before committing

The single largest assumption is that display and circular placement causes the
lift attributed to it, rather than reflecting which products the buying team
already expected to move. Randomising mechanic assignment across matched store
pairs for one quarter would settle it. Until that runs, the ranking of shelves
is trustworthy and the size of the prize is an estimate.
"""


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "executive_summary.md").write_text(executive_summary(), encoding="utf-8")
    (DOCS / "business_case.md").write_text(business_case(), encoding="utf-8")
    print("docs/executive_summary.md and docs/business_case.md written from outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
