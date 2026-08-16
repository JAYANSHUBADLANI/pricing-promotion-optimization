"""Cross check the custom sampler against an independent implementation.

The Gibbs sampler in models/gibbs.py is written specifically for this model, so
it carries the risk that any bespoke numerical code carries: it could be
confidently wrong in a way that parameter recovery on simulated data does not
catch, because both the simulation and the sampler come from the same
understanding of the model.

The defence is a second implementation that shares none of that code. The same
model is expressed in PyMC and sampled with NUTS, and the two posteriors are
required to agree. NUTS is slow enough on the full panel to be impractical, which
is why the Gibbs sampler exists, but on a small panel it is perfectly usable as a
reference.

Marked slow. Run with `pytest -m slow`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricing import features as F
from pricing import schema
from pricing.config import load_config
from pricing.models import gibbs
from pricing.models.design import build_design, compute_sufficient_statistics

pytestmark = pytest.mark.slow

TRUE = {0: -2.6, 1: -1.9, 2: -1.3, 3: -2.2}


def small_panel(seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2010-01-03", periods=30, freq="7D")
    rows = []
    for store in range(2):
        for upc, elasticity in TRUE.items():
            base = 2.5 + 0.3 * upc
            for week in weeks:
                depth = rng.choice([0.0, 0.15, 0.30], p=[0.6, 0.25, 0.15])
                price = base * (1 - depth)
                units = max(
                    1,
                    int(np.exp(4.0 + 0.15 * store + elasticity * np.log(price)
                               + rng.normal(0, 0.18))),
                )
                rows.append({
                    schema.WEEK: week, schema.STORE_ID: store, schema.UPC: upc,
                    schema.UNITS: units, schema.SPEND: units * price,
                    schema.PRICE: price, schema.BASE_PRICE: base,
                    schema.FEATURE: 0, schema.DISPLAY: 0,
                    schema.TPR_ONLY: int(depth > 0),
                    schema.CATEGORY: "CAT_A" if upc < 2 else "CAT_B",
                    schema.SUB_CATEGORY: f"SUB_{upc // 2}",
                })
    return F.build_features(pd.DataFrame(rows), load_config())


@pytest.fixture(scope="module")
def both_posteriors():
    pymc = pytest.importorskip("pymc")
    from pricing.models.elasticity import build_model

    cfg = load_config()
    frame = small_panel()
    design = build_design(frame, cfg)
    stats = compute_sufficient_statistics(design)

    gibbs_result = gibbs.sample(design, stats, draws=1500, tune=500, chains=2, seed=3)
    gibbs_mean = gibbs_result.block("own_price").mean(axis=(0, 1))

    model = build_model(design, stats, cfg)
    with model:
        idata = pymc.sample(
            draws=400, tune=400, chains=1, cores=1, target_accept=0.85,
            random_seed=3, progressbar=False, compute_convergence_checks=False,
        )
    nuts_mean = idata.posterior["own_price_elasticity"].mean(("chain", "draw")).values
    return gibbs_mean, nuts_mean


def test_two_independent_samplers_agree(both_posteriors) -> None:
    gibbs_mean, nuts_mean = both_posteriors
    np.testing.assert_allclose(gibbs_mean, nuts_mean, atol=0.20)


def test_both_orderings_match_the_truth(both_posteriors) -> None:
    """Both samplers should rank the products the same way the truth does.

    The panel here is deliberately small, because its job is to make an
    independent NUTS fit cheap enough to run as a check. At that size neither
    sampler pins an elasticity down tightly, so demanding a close absolute match
    to the simulated values would be testing the sample size rather than the
    code. Recovery to a tight tolerance is tested on a larger panel in
    tests/test_gibbs.py. What matters here is that two independent
    implementations agree with each other and both order the products correctly.
    """
    gibbs_mean, nuts_mean = both_posteriors
    truth = np.array([TRUE[i] for i in sorted(TRUE)])
    assert np.corrcoef(gibbs_mean, truth)[0, 1] > 0.9
    assert np.corrcoef(nuts_mean, truth)[0, 1] > 0.9
