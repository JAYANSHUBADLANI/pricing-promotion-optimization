# Executive summary

## The question

Which product shelves should carry a promotion next quarter, how deep should the
discount go, and what promotional support should sit behind it.

## What I did

I fitted a hierarchical demand model to 524,950 store weeks covering
55 products across 77 stores and
156 weeks, estimating a separate price elasticity for every
product while pooling information across shelves and categories. I then measured
what each promotional mechanic adds beyond its price cut, measured how much of a
promotion's gain is taken from neighbouring products rather than created, and
solved for the plan that maximises contribution under a margin floor, a maximum
depth and a cap on how many shelves can promote at once.

## The recommendation

Promote 3 of 7 shelves:
**ADULT CEREAL, KIDS CEREAL, PIZZA/PREMIUM**. Run them at shallow depth with full merchandising support rather
than deep discounts with none. Projected contribution change of
**+57 percent** per store week at an assumed
35% gross margin, holding between
56 and
61 percent across every gross
margin assumption tested.

## The three findings behind it

**1. Merchandising does most of the work, not discount depth.** An in store
display is worth roughly +55 percent on
volume after its price cut is already accounted for. A display and a circular
together are worth about +114
percent. A price cut with no display and no circular is worth about
-4.9 percent, which is to say it slightly
underdelivers against its own price elasticity. Across mechanics and categories
merchandising accounts for a mean
34% of modelled lift.
The practical reading is that promotional budget is better spent on space and
visibility than on depth.

**2. On some shelves a discount relocates volume instead of creating it.** For
2 of 7 shelves the diversion ratio exceeds 1,
meaning more than the whole apparent gain comes off neighbouring facings:
PRETZELS at 1.37, ALL FAMILY CEREAL at 1.19.
Premium pizza shows essentially none. Planning on own price elasticity alone
would recommend promoting the shelves that cannibalise hardest.

**3. Two shelves cannot be planned on at all.** ALL FAMILY CEREAL, PRETZELS produce a
net price response that is positive, which would mean cutting every price on the
shelf sells fewer units. That is not believable, and it indicates the model has
picked up the tendency for neighbouring products to be promoted in the same weeks
rather than genuine substitution. They are reported and excluded rather than
quietly included.

## How much to trust it

The model was refit on data ending 2011-10-05 and asked to
predict 13 weeks it had never seen. Against a per store
per product average it cuts error by
53 percent on weeks running a display
and a circular, which is where a demand model should earn its keep. On
tpr only weeks it is
14 percent worse than that
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
