# causal-lift

> **Incrementality measurement for retail media — and any other channel where geo or SKU holdouts are the only credible answer.**
> Open-source Python library. MIT licensed. Pre- and post-experiment workflow in one tool.

[![PyPI](https://img.shields.io/pypi/v/causal-lift.svg)](https://pypi.org/project/causal-lift/)
[![Python](https://img.shields.io/pypi/pyversions/causal-lift.svg)](https://pypi.org/project/causal-lift/)
[![CI](https://github.com/rt23-dev/causal-lift/actions/workflows/test.yml/badge.svg)](https://github.com/rt23-dev/causal-lift/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The problem `causal-lift` was built to solve

**Retail media ad networks** — Amazon Sponsored Products, Walmart Connect, Target Roundel, Instacart, Kroger Precision — are the fastest-growing $50B+ advertising category. The brand managers spending money on them have **no credible way to measure whether the ads actually drove incremental sales**, or whether the network's 7-day attribution window is crediting paid for sales the customer would have made organically.

Existing tools (Pacvue, Skai, Perpetua) optimise bids. They don't measure causal lift. The brand-side measurement teams at CPG companies $20–500M in revenue run SKU-level holdouts manually in spreadsheets, or pay $50K+/year to Haus or Tatari for managed services.

**`causal-lift` is the open-source workflow that sits in that gap.** Design a SKU-level or geo-level holdout, run it, get a credible verdict with confidence intervals — for any channel, any retailer.

It works equally well for:

- **Retail media** (Amazon Ads, Walmart Connect, Target Roundel) — SKU-level holdouts
- **Out-of-home / TV / podcast** (the classic "physical media" use case) — DMA-level geo holdouts
- **Digital prospecting** — region-level holdouts where you can selectively pause Meta/Google in test markets
- **MMM as a sanity check** — when a real experiment isn't feasible, the regression-based MMM with safety gates gives you a directional answer

## Install

```bash
pip install causal-lift
```

## 60-second example: SKU-level Amazon Ads holdout

```python
import causal_lift as cl
import pandas as pd

# 16+ weeks of revenue history per ASIN
baseline = pd.read_csv("amazon_sales_by_asin.csv", parse_dates=["date"])

# STEP 1 — design the experiment
design = cl.design_geo_holdout(
    baseline,
    geo_column="asin",       # works equally well with "dma", "region", "store_id"
    n_treated=3,
    duration_weeks=4,
    expected_lift_pct=0.08,  # we expect ~8% revenue change from pausing ads
)
print(design.summary())
# → Selected 3 ASINs to pause for 4 weeks. MDE = 2.1% at 80% power.

# STEP 2 — pause Sponsored Products on the treated ASINs for 4 weeks

# STEP 3 — analyse what actually happened
result = cl.analyze_geo_holdout(
    experiment_revenue,
    geo_column="asin",
    treated_geos=design.treated_geos,
    pre_period_end="2025-04-21",
    post_period_start="2025-04-28",
    spend_change=-120_000,   # the $120K of Amazon spend you paused
)
print(result.summary())
```

Sample output:

```
Verdict:             NEGATIVE_LIFT
Measured lift:       -9.0%
95% CI:              [-10.5%, -7.5%]
p-value:             0.000

Treated ASINs:       B0C2PROTEIN-CHC, B0C1PROTEIN-VAN, B0E2BCAA-BBRY
Spend change:        $-120,000
Implied iROAS:       0.75x

Rationale: Pausing ads on the treated ASINs caused revenue to drop 9%
relative to the 9 control ASINs. Amazon Sponsored Products IS causally
incremental on these SKUs, but at 0.75x iROAS — below break-even.
```

The same workflow drives TV/OOH/podcast holdouts — pass `geo_column="dma"` instead of `"asin"` and you have a billboard incrementality test. See [`examples/billboard_holdout.py`](examples/billboard_holdout.py) and [`examples/amazon_sku_holdout.py`](examples/amazon_sku_holdout.py).

## What's in the library

| Module | What it does |
|---|---|
| `cl.design_geo_holdout` | Pre-experiment: pick treated/control units, compute statistical power, report minimum detectable effect at 80% power |
| `cl.analyze_geo_holdout` | Post-experiment: diff-in-diff with stationary block bootstrap CI, verdict (`LIFT_DETECTED` / `NEGATIVE_LIFT` / `NO_EFFECT` / `INCONCLUSIVE`), implied iROAS |
| `cl.RegressionMMM` | Multi-channel MMM with auto-tuned adstock, HAC standard errors, four safety gates that refuse confident-wrong recommendations |
| `cl.GeoMMM` | Multi-geo aggregation when you have regional disaggregation |
| `cl.recommend_reallocation` | Budget shift recommender with conservation accounting |
| `cl.loaders.*` | CSV converters for **Amazon Ads, Walmart Connect**, Shopify, Meta Ads, Google Ads |

## Honest safety gates

The library refuses to give you a confident answer when the data can't support one. Four gates demote `SCALE` recommendations to `INCONCLUSIVE`:

1. **High VIF** (> 10) — collinear channels can't be individually identified
2. **Always-on channels** (> 85% non-zero) — can't separate from baseline trend
3. **Aggregate implausibility** (Σ iROAS·spend > 50% of revenue) — model is over-attributing
4. **CI wider than point estimate** — not enough precision to commit budget

No other open-source MMM ships an explicit refusal label. The rationale and competitive position are in [`docs/case-studies.md`](docs/case-studies.md) and [`docs/product-brief.md`](docs/product-brief.md).

## Benchmarked

Regression tests pin the library against Meta Robyn's standard MMM benchmark dataset (`dt_simulated_weekly.csv`). Earlier versions of `causal-lift` failed this benchmark with confident-wrong SCALE recommendations on always-on channels; the current safety gates catch the failure. See [`tests/test_benchmark.py`](tests/test_benchmark.py).

| Metric                                | v0.1.0      | v0.4.0      |
|---|---|---|
| Model R²                              | 0.48        | 0.83        |
| Aggregate implied incremental share   | 44%         | 10–20%      |
| Confident `SCALE` on 20x+ iROAS       | yes         | **no**      |

## What this library does NOT do

- **Direct platform API ingestion.** OAuth-based pulls from Amazon Ads, Meta, Google, and the retail media networks land in v0.5+. For now, use CSV exports — every major platform supports them.
- **Saturation curves.** No Hill-style diminishing-returns modelling. The linear coefficient is the *average* iROAS over your observed spend range.
- **Bayesian priors.** Pure frequentist OLS + bootstrap. Use PyMC-Marketing if you need priors.
- **Channel-shift correction.** TV → branded search mediation is a separate problem. Run a branded-search holdout to size it.

## Roadmap

- [x] Cadence-aware seasonality (v0.1.1)
- [x] Plausibility safety gates (v0.1.1)
- [x] Geometric adstock with auto-tuned decay (v0.2.0)
- [x] Multi-geo analyzer + bootstrap CIs + budget reallocator (v0.3.0)
- [x] Geo holdout design + analysis — the product layer (v0.4.0)
- [x] **Retail media loaders (Amazon Ads, Walmart Connect)** (v0.5.0)
- [ ] Direct API integrations with retail media networks (v0.6)
- [ ] Cross-retailer aggregation (Amazon + Walmart + Target in one pane)
- [ ] Hosted experiment-tracking dashboard

## Reading guide

- **First time?** Start with [`examples/amazon_sku_holdout.py`](examples/amazon_sku_holdout.py) — runnable end-to-end, recovers a known 8% lift exactly.
- **Coming from physical media?** See [`examples/billboard_holdout.py`](examples/billboard_holdout.py) — same workflow, geo unit instead of SKU.
- **Want to understand the safety gates?** See [`docs/case-studies.md`](docs/case-studies.md) — Acme Skincare walkthrough where all four gates fire and the library refuses to give bad answers.
- **Strategic / commercial?** See [`docs/product-brief.md`](docs/product-brief.md) — pricing, ICP, competitive position.

## Contributing

PRs welcome — especially retail media network loaders, SKU-level analysis patterns, and case studies with real anonymised brand data. Issues for `INCONCLUSIVE` outputs you think should be actionable are highly valued; they're how we improve the safety-gate thresholds.

Run the tests with `pytest`. Lint with `ruff check`.

## License

MIT.
