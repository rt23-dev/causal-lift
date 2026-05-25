# Changelog

All notable changes to `causal-lift` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
