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

- ⏳ **Adstock / carryover** — geometric decay per channel, auto-tuned by adjusted R². TV and OOH no longer look dead because their effect spreads across weeks.
- ⚠️ Warns when your spend variation looks endogenous (algorithmic bidding)
- 📊 Reports VIF per channel — flags collinear estimates as unreliable
- 📈 Uses HAC (Newey-West) standard errors to correct for autocorrelation
- 🎯 Anchors `SCALE` / `HOLD` / `CUT` recommendations to your actual contribution margin, not an assumed 1x breakeven
- 🚫 Refuses to recommend `SCALE` when (a) the channel runs in >85% of periods, (b) aggregate implied incremental share exceeds 50% of revenue, or (c) the 95% CI is wider than the point estimate — returns `INCONCLUSIVE` instead

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
causal-lift analysis  |  method: RegressionMMM (multivariate OLS, trend + cadence-aware seasonality, geometric adstock, HAC/Newey-West SEs, plausibility gates)
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

(Adstock auto-selected `θ=0` for all three channels in this example because the synthetic data has no carryover baked in. On real DTC data with TV / OOH / YouTube, you'll typically see `θ > 0` for those channels.)

## Benchmarked on Meta's Robyn dataset

`causal-lift` is regression-tested against [Robyn's `dt_simulated_weekly.csv`](https://github.com/facebookexperimental/Robyn) — the de-facto industry MMM benchmark (208 weekly observations × 5 paid-media channels × 4-year span). The benchmark catches the exact failure mode that broke earlier versions: confident `SCALE` recommendations on channels with implausibly inflated iROAS estimates.

| Metric                              | v0.1.0      | v0.1.1     | v0.2.0       |
|---|---|---|---|
| Model R²                             | 0.48        | 0.82       | **0.83**     |
| Durbin-Watson                        | 1.23        | 1.79       | **1.84**     |
| Aggregate implied incremental share  | 44%         | 10%        | **20%**      |
| `tv_S` recommendation                | SCALE @ 7x  | HOLD @ 3.8x | **SCALE @ 5.8x** (CI [3.2, 8.4]) |
| `ooh_S` iROAS                        | 0.63x       | 0.17x      | **1.05x**    (closer to true) |
| `search_S` recommendation            | SCALE @ 85x | HOLD @ 11x | HOLD @ 25x   (CI crosses 0) |
| `facebook_S` recommendation          | SCALE @ 62x | HOLD @ 14x | **INCONCLUSIVE @ 30x** (CI wider than point) |
| Any `SCALE` on iROAS > 20x           | yes         | no         | **no**       |

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

Multivariate OLS with cadence-aware time-series controls and geometric adstock:

```
adstocked_spend_{it}  =  (1-θ_i) · Σ_{k=0}^{t} θ_i^k · spend_{i,t-k}      ← per-channel carryover

revenue_t            =  α
                     +  β_trend · t
                     +  (DOW dummies, daily data only)
                     +  (annual Fourier sin/cos, when span ≥ 365 days)
                     +  Σ_i β_i · adstocked_spend_{it}                     ← causal effects
                     +  ε_t
```

`β_i` is the causal incremental ROAS for channel `i` — $ revenue per $ spent, after controlling for time trend, seasonality, and carryover. The adstock impulse response is normalised to sum to 1, so the iROAS interpretation is preserved: $1 of raw spend produces β of revenue, spread across periods according to the geometric decay.

The decay parameter `θ_i` is auto-selected per channel by greedy grid search over `{0.0, 0.3, 0.5, 0.7}` against adjusted R². Override with `RegressionMMM(adstock={"facebook": 0.0, "tv": 0.6})` for explicit control, or disable with `adstock=None`.

Standard errors use HAC (Newey-West) with auto-selected bandwidth, robust to heteroskedasticity and autocorrelation.

Four safety gates layer over the regression output:

1. **VIF > 10** → recommendation forced to `HOLD` (estimate not identified due to collinearity).
2. **Channel active in >85% of periods AND would-be SCALE** → `INCONCLUSIVE` (same-day OLS can't separate its effect from baseline).
3. **Σ(iROAS · adstocked_spend) / revenue > 50%** → all `SCALE` labels demoted to `INCONCLUSIVE` (model is collectively over-attributing).
4. **95% CI wider than the point estimate AND would-be SCALE** → `INCONCLUSIVE` (estimate too imprecise to commit budget).

See [`examples/interpreting_results.md`](examples/interpreting_results.md) for a practical guide to reading the outputs, and [`docs/case-studies.md`](docs/case-studies.md) for a deep dive on a fictional $15M-ARR brand plus a catalog of 6 use-case scenarios and 5 anti-cases.

## What this library does NOT do

These are deliberate limitations. Each is documented so you can decide whether `causal-lift` is the right tool for your situation.

- **Geo holdouts / synthetic control.** Single time-series only — no regional disaggregation. If you have geo-level data and a treated/control split, run a proper synthetic-control study (Abadie-Diamond-Hainmueller is the canonical reference). The Magic of an SC study is identification *with no parametric assumptions about response curves*; that's a different product. Coming as a separate analyzer class.
- **Saturation curves.** No Hill or log-transform of spend. The linear coefficient is the *average* iROAS over your observed spend range; if you're far up an S-curve, marginal iROAS is lower than the average. Robyn handles this with explicit average-vs-marginal reporting; we don't, deliberately, because it adds two more hyperparameters per channel and a corresponding overfitting surface. If you're considering scaling spend > 50% from current levels, the linear extrapolation will overstate the gain.
- **Bayesian priors.** Pure frequentist OLS with HAC SEs. If you need to inject business knowledge as priors (e.g., "we know facebook iROAS is between 2 and 5x"), use [PyMC-Marketing](https://www.pymc-marketing.io/) or the [Robyn](https://github.com/facebookexperimental/Robyn) Bayesian fork. Adding PyMC as a dependency would 10x the install footprint and isn't worth it for the default user.
- **Causal identification under endogenous spend.** If your campaigns use Meta Advantage+ or Google Smart Bidding, your spend co-moves with demand and OLS biases iROAS upward. This is a structural limitation of any regression-based MMM. The library surfaces a warning; it cannot correct for it without an instrumental variable, a budget holdout, or geo randomisation. **You should run periodic budget holdouts regardless of which tool you use.**

The library is intentionally simple. It is a starting point and a diagnostic — not a replacement for a properly-designed lift study.

## Roadmap

- [x] Cadence-aware seasonality (v0.1.1)
- [x] Plausibility safety gates (v0.1.1)
- [x] Geometric adstock with auto-tuned decay (v0.2.0)
- [x] CI precision gate (v0.2.0)
- [ ] Geo holdout / synthetic control method (separate analyzer)
- [ ] Budget allocation optimiser (given iROAS estimates, recommend dollar reallocations)
- [ ] Bootstrap CIs as an alternative to HAC
- [ ] Data loaders for Shopify, Meta Ads, Google Ads APIs

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
