# Changelog

All notable changes to `causal-lift` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-05-27

Critical safety fixes triggered by running the library on Meta Robyn's
`dt_simulated_weekly.csv` benchmark dataset.  The 0.1.0 release produced
confident `SCALE` recommendations on channels with implausibly inflated
iROAS estimates (search 85x, facebook 62x) because the trend control alone
could not separate channel effects from annual seasonality.  This release
addresses the three failure modes the benchmark exposed.

### Added
- `INCONCLUSIVE` recommendation label for channels whose estimates are
  flagged as untrustworthy by the safety gates below.
- **Cadence detection** — daily / weekly / irregular inferred from the
  date index.  DOW dummies are only added for daily data (previously they
  were always added and silently dropped on weekly data).
- **Annual Fourier seasonality** — sin/cos terms at period 365.25 (daily)
  or 52 (weekly) are added automatically when the data spans ≥ 1 year.
  Substantially reduces the variance that was previously absorbed by
  spend coefficients.
- **Always-on / baseline-confound gate** — channels active in >85% of
  periods that would otherwise be `SCALE` are downgraded to
  `INCONCLUSIVE`, with a recommendation to run a budget holdout.
- **Aggregate plausibility gate** — if Σ(iROAS·spend) / revenue exceeds
  50%, all `SCALE` labels are demoted to `INCONCLUSIVE` and a warning
  banner is emitted.  Catches the case where the model collectively
  over-attributes revenue to paid media.
- `AnalysisResult.cadence` and `.implied_incremental_share` fields
  exposed on the public API.
- `ChannelResult.nonzero_share` field exposed on the public API.
- `tests/test_benchmark.py` — six regression tests against the Robyn
  dataset that guard against the failures above.
- Benchmark artefacts checked into `examples/benchmark_results/`
  (Robyn CSV, baseline estimates, run log) for reproducibility.

### Changed
- `recommend()` signature accepts `nonzero_share` to drive the always-on
  gate.
- `R²` warning threshold raised from 0.30 to 0.50 — the benchmark showed
  0.48 was not "fine" but was previously below the warning floor.
- Method label updated to reflect the new feature set: "RegressionMMM
  (multivariate OLS, trend + cadence-aware seasonality, HAC/Newey-West
  SEs, plausibility gates)".

### Benchmark
On the Robyn `dt_simulated_weekly.csv` dataset (208 weekly observations,
5 paid-media channels, 4-year span):

|                                  | 0.1.0  | 0.1.1   |
|----------------------------------|--------|---------|
| Model R²                          | 0.48  | **0.82** |
| Durbin-Watson                     | 1.23  | **1.79** |
| Aggregate implied incremental %   | 44%   | **10%** |
| `search_S` recommendation         | SCALE | HOLD    |
| `facebook_S` recommendation       | SCALE | HOLD    |
| Any `SCALE` on 20x+ iROAS estimate | yes   | **no**  |

## [0.1.0] — 2026-05-25

Initial release.

### Added
- `RegressionMMM`: multivariate OLS with linear time trend, day-of-week
  controls, and HAC (Newey-West) heteroskedasticity-and-autocorrelation-robust
  standard errors.
- VIF (Variance Inflation Factor) computed per channel; channels with VIF > 10
  are flagged as unreliable and forced to a `HOLD` recommendation regardless of
  point estimate.
- Margin-aware `SCALE` / `HOLD` / `CUT` recommendations driven by user-provided
  contribution margin (defaults to 0.30).
- Durbin-Watson diagnostic surfaced in results.
- `generate_synthetic_data()` for ground-truth-known sample data with
  per-channel budget experiments.
- CLI: `causal-lift analyze` and `causal-lift sample`.
- Module-level convenience function `causal_lift.analyze(spend_df, sales_df)`.

### Known limitations
- No adstock / saturation modelling. Brand-awareness channels (TikTok, YouTube)
  will appear weaker than their true contribution.
- Identification assumes spend is budget-driven (exogenous). Algorithmic
  bidding (Meta Advantage+, Google Smart Bidding) violates this; the library
  surfaces a warning but does not correct for it.
- No geo holdout / synthetic control. Single-population time-series only.
