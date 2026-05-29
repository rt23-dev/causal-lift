# causal-lift

> A lightweight, transparent marketing mix model for DTC brands.
> Estimate the causal lift of each ad channel from spend + sales data — and find out when you can't trust the estimate.

[![PyPI](https://img.shields.io/pypi/v/causal-lift.svg)](https://pypi.org/project/causal-lift/)
[![Python](https://img.shields.io/pypi/pyversions/causal-lift.svg)](https://pypi.org/project/causal-lift/)
[![CI](https://github.com/rt23-dev/causal-lift/actions/workflows/test.yml/badge.svg)](https://github.com/rt23-dev/causal-lift/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why

Platform-reported ROAS (the number Meta and Google show in their dashboards) is inflated. It credits revenue that would have happened anyway and double-counts across platforms. **`causal-lift` estimates how much revenue each channel actually drove**, using a regression-based marketing mix model with proper time-series controls.

It is opinionated about being honest with you:

- ⚠️ Warns when your spend variation looks endogenous (algorithmic bidding)
- 📊 Reports VIF per channel — flags collinear estimates as unreliable
- 📈 Uses HAC (Newey-West) standard errors to correct for autocorrelation
- 🎯 Anchors `SCALE` / `HOLD` / `CUT` recommendations to your actual contribution margin, not an assumed 1x breakeven
- 🚫 Refuses to recommend `SCALE` on always-on channels or when aggregate implied incremental share is implausibly high — returns `INCONCLUSIVE` instead

If the model can't make a defensible claim about a channel, it says so instead of fabricating confidence.

## Install

```bash
pip install causal-lift
```

## 30-second quickstart

```python
import causal_lift as cl

# 1. Synthetic data with known ground-truth iROAS for testing
data = cl.generate_synthetic_data(n_days=90, seed=42)

# 2. Run the analysis at your actual contribution margin
result = cl.analyze(
    data.spend_df,        # columns: date, channel, spend
    data.sales_df,        # columns: date, revenue
    contribution_margin=0.30,
)

print(result.summary())
```

Output:

```
causal-lift analysis  |  method: RegressionMMM (multivariate OLS, trend + cadence-aware seasonality, HAC/Newey-West SEs, plausibility gates)
  observations: 90 (daily)  |  R-squared: 0.849  |  DW: 2.22
  contribution margin: 30%  ->  breakeven iROAS = 3.33x
  aggregate implied incremental share: 44% of revenue

  channel          iROAS                CI95     VIF   on%           rec
  ----------------------------------------------------------------------
  facebook         1.92x        [1.69, 2.14]     1.1  100%           CUT
  google           3.83x        [3.37, 4.29]     1.1  100%  INCONCLUSIVE
  tiktok           1.20x       [-0.18, 2.58]     1.1  100%           CUT
```

Notice `google` returns **INCONCLUSIVE** instead of `SCALE` despite an iROAS of 3.83x — because the channel runs in 100% of periods, the model can't separate its effect from baseline. The library tells you to run a budget holdout instead of giving you a confident-but-wrong answer.

## Benchmarked on Meta's Robyn dataset

`causal-lift` is regression-tested against [Robyn's `dt_simulated_weekly.csv`](https://github.com/facebookexperimental/Robyn) — the de-facto industry MMM benchmark (208 weekly observations × 5 paid-media channels × 4-year span). The benchmark catches the exact failure mode that broke earlier versions: confident `SCALE` recommendations on channels with implausibly inflated iROAS estimates.

| Metric                                | v0.1.0 | v0.1.1 |
|---|---|---|
| Model R²                               | 0.48  | **0.82** |
| Durbin-Watson                          | 1.23  | **1.79** |
| Aggregate implied incremental share    | 44%   | **10%** |
| `search_S` recommendation              | SCALE @ 85x | **HOLD @ 11x** |
| `facebook_S` recommendation            | SCALE @ 62x | **HOLD @ 14x** |
| Any `SCALE` on iROAS > 20x             | yes   | **no**  |

The benchmark CSV and reproducibility script are vendored at [`examples/benchmark_results/`](examples/benchmark_results/) and [`examples/benchmark.py`](examples/benchmark.py). Six regression tests in [`tests/test_benchmark.py`](tests/test_benchmark.py) keep the library honest as it evolves — any future change that re-introduces a confident-but-wrong `SCALE` will fail CI.

## With your own data

```python
from causal_lift.data import load_spend_csv, load_sales_csv
import causal_lift as cl

spend_df = load_spend_csv("spend.csv")    # date, channel, spend
sales_df = load_sales_csv("sales.csv")    # date, revenue (orders optional)

result = cl.analyze(spend_df, sales_df, contribution_margin=0.30)

result.to_dataframe().to_csv("results.csv", index=False)
```

## CLI

```bash
# Generate synthetic sample data
causal-lift sample --days 90 --out-dir ./data

# Run analysis
causal-lift analyze ./data/spend.csv ./data/sales.csv --margin 0.30

# JSON output
causal-lift analyze ./data/spend.csv ./data/sales.csv --json -o results.json
```

## Method

Multivariate OLS with cadence-aware time-series controls:

```
revenue_t  =  α
           +  β_trend · t
           +  (DOW dummies, daily data only)
           +  (annual Fourier sin/cos, when span ≥ 365 days)
           +  Σ_i β_i · spend_{it}
           +  ε_t
```

`β_i` is the causal incremental ROAS for channel `i` — $ revenue per $ spent, after controlling for the time trend and seasonality. Standard errors use HAC (Newey-West) with auto-selected bandwidth, robust to both heteroskedasticity and autocorrelation.

Three safety gates layer over the regression output:

1. **VIF > 10** → recommendation forced to `HOLD` (estimate not identified due to collinearity).
2. **Channel active in >85% of periods AND would-be SCALE** → `INCONCLUSIVE` (same-day OLS can't separate its effect from baseline).
3. **Σ(iROAS · spend) / revenue > 50%** → all `SCALE` labels demoted to `INCONCLUSIVE` with a banner warning (model is collectively over-attributing).

See [`examples/interpreting_results.md`](examples/interpreting_results.md) for a practical guide to reading the outputs.

## What this library does NOT do (yet)

- **Adstock / carryover.** Brand-awareness channels with delayed conversions (TikTok, YouTube) will look weaker than they are. On the roadmap.
- **Geo holdouts / synthetic control.** No regional disaggregation. Single time-series only.
- **Saturation curves.** No diminishing-returns modelling. If you're far up an S-curve, the linear coefficient under-extrapolates.
- **Bayesian priors.** Pure frequentist OLS. If you need to inject business knowledge as priors, this isn't your tool.
- **Causal identification under endogenous spend.** If your campaigns use Meta Advantage+ or Google Smart Bidding, your spend co-moves with demand and OLS biases iROAS upward. The library surfaces a warning; it does not correct for it. Budget holdout experiments or geo randomisation are the proper fix.

The library is intentionally simple. It is a starting point and a diagnostic — not a replacement for a properly-designed lift study.

## Roadmap

- [ ] Geo holdout / synthetic control method
- [ ] Adstock + saturation curves (full MMM)
- [ ] Budget allocation optimiser
- [ ] Bootstrap CIs as alternative to HAC
- [ ] Bayesian prior support
- [ ] More data loaders (Shopify, Meta Ads, Google Ads APIs)

## Hosted playground

If you want to try the library without installing it, run the React + FastAPI demo in [`playground/`](playground/):

```bash
cd playground/backend && python -m uvicorn main:app --port 8000 &
cd playground/frontend && npm install && npm run dev
```

Open <http://localhost:5173> and upload your CSVs (or click "Use Sample Data").

## Contributing

This is alpha software. Issues and PRs welcome — especially:

- Bug reports with reproducible synthetic-data examples
- Documentation improvements
- Additional analysis methods (geo holdout, adstock)

Run the tests with `pytest`.

## License

MIT
