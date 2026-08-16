# Progress log

A record of what was built, what was verified, and what is still open. Every
number in the README and in `docs/` is written by a script from generated output
and read back from `outputs/`, never typed by hand.

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repo scaffold and config | Complete |
| 1 | Ingestion, verification, cleaning, features | Complete |
| 2 | Demand model, lift decomposition, cannibalisation, backtest | Complete |
| 3 | Constrained optimiser, scenario tool | Complete |
| 4 | Business case, executive summary, README | Complete |

## Dataset change during the build

The project was specified against a Kaggle retail dataset with a single discount
column. Two files were checked before settling.

The first candidate turned out to be a different dataset entirely, an alcohol
distribution report with no price column, no discount column, four non
consecutive months and monthly granularity. Price elasticity is not estimable
without a price, so it was rejected rather than worked around.

The dataset used instead is dunnhumby's "Breakfast at the Frat", which is better
suited than the original specification in two ways that changed the analysis:
both the shelf price and the base price are given, so discount depth is measured
rather than constructed, and promotional support is recorded as three separate
mechanics rather than one flag, which is what makes the lift decomposition
possible.

## What was verified rather than assumed

- Row count, store count, product count, week count and date range all match what
  the file actually contains, checked in `outputs/tables/data_verification.json`.
- Spend divided by units reproduces the stated shelf price to within one percent
  on every row.
- Two stores appear twice in the store lookup with contradictory segment labels.
  The conflicting field is marked unresolved rather than resolved by sheet order,
  and recorded in `data/interim/lookup_conflicts.json`.
- 99.47 percent of rows survive cleaning, with every rule and count recorded.

## Notable decisions

**The sampler was written for this model.** A general purpose gradient sampler
could not fit it in reasonable time: tree depth pinned at the maximum and
hundreds of divergent transitions, because the posterior geometry it was trying
to learn numerically is available in closed form. A blocked Gibbs sampler drawing
all 243 coefficients jointly removed the problem. It is validated two ways:
against simulated data with known elasticities, and against an independent PyMC
NUTS fit of the same model on a small panel, which has to agree with it.

**The likelihood runs on sufficient statistics.** Sampling cost is independent of
the 522,154 rows. Verified exact against direct computation in the test suite.

**Planning happens at shelf level using net elasticity.** Own plus cross, because
promoting a whole shelf moves the competing price along with the own price. This
changed the recommendation: two shelves that look attractive on own price
elasticity turn out to divert more than their entire gain from neighbouring
products.

**A credibility gate refuses non credible estimates.** Two shelves produce a
positive net elasticity, which is not economically possible. They are reported
and excluded from the plan rather than silently included.

## Open items

- Add a name to the copyright line in `LICENSE`. It currently reads
  `Copyright (c) 2026` with no holder, which is valid but impersonal.
- `data/raw/` holds the source workbook at roughly 30 MB. It is excluded by
  `.gitignore`, so anyone cloning the repository has to download it themselves.
  The path and the instructions are in the README and in
  `src/pricing/data/ingest.py`.
- The `outputs/` directory is committed so the README renders its figures and
  tables on a fresh clone without a pipeline run. Remove it from version control
  if that is not wanted.
- The weakest empirical result is that the model is worse than a per pair average
  on unadvertised price cut weeks. Worth investigating before presenting this as
  finished work.
