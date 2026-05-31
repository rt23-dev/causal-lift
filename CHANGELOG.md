# Changelog

All notable changes to `causal-lift` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-05-30

**Retail-media pivot.** Repositions ``causal-lift`` from a generic
"DTC MMM library" into the open-source incrementality workflow for
brands spending on retail media networks (Amazon Sponsored Products,
Walmart Connect, Target Roundel, Instacart, Kroger Precision).  The
underlying methodology is unchanged — the geo-holdout machinery in
v0.4 already generalises to SKU-level holdouts via the
``geo_column`` parameter — but the loaders, examples, README, and
product brief are now retail-media-first.

### Added
- ``load_amazon_ads_csv`` — Amazon Advertising Console exports
  (Sponsored Products / Sponsored Brands / Sponsored Display).
  Supports day-level, campaign-level, or ASIN-level grouping.
- ``load_amazon_sales_csv`` — Amazon Business Reports (Detail Page
  Sales and Traffic by ASIN) → daily per-SKU organic + paid revenue.
- ``load_walmart_ads_csv`` — Walmart Connect Ad Center exports,
  day / campaign / item-level grouping.
- ``examples/amazon_sku_holdout.py`` — RidgeBev fictional CPG brand
  runs a 4-week SKU-level holdout on Amazon Sponsored Products.
  The library recovers a known true 8% revenue drop as -9.0%
  (95% CI [-10.5%, -7.5%]) with p < 0.001 and computes the correct
  implied iROAS of 0.75x. Verdict: NEGATIVE_LIFT — ad spend was below
  break-even on the treated SKUs.

### Changed
- README rewritten to lead with retail media as the primary use case;
  physical media (TV/OOH/podcast) and DTC MMM repositioned as
  secondary use cases.
- ``docs/product-brief.md`` rewritten: new ICP (CPG brand ops directors
  at $50-500M revenue spending $1-25M/year on retail media), new
  competitive analysis (Pacvue, Skai, Perpetua, Haus, Tatari, Recast),
  new pricing tiers (with $15K/month multi-retailer tier added).

### Test additions
- 7 new tests covering Amazon Ads CSV (day/campaign/ASIN grouping),
  Amazon Sales CSV, Walmart Connect CSV (day/item grouping).
- Total test count: 82 → 89.

## [0.4.0] — 2026-05-29

**The product layer.** Adds the geo-holdout experimentation loop that
turns ``causal-lift`` from a passive measurement library into the
foundation for a managed-services product targeting brands running
physical media (TV, OOH, podcast, radio, direct mail).

### Added
- ``design_geo_holdout`` — pre-experiment design.  Given baseline
  revenue by DMA, auto-selects treated/control geos by maximising
  pre-period log-revenue correlation, runs a diff-in-diff power
  calculation, and returns minimum detectable effect at 80% power.
- ``analyze_geo_holdout`` — post-experiment analysis.  Computes the
  treated-vs-control diff-in-diff with a stationary block bootstrap
  CI (robust to autocorrelation).  Returns a categorical verdict
  (``LIFT_DETECTED`` / ``NO_EFFECT`` / ``NEGATIVE_LIFT`` /
  ``INCONCLUSIVE``) and the implied iROAS on the spend change.
- ``GeoHoldoutDesign`` and ``GeoHoldoutResult`` public dataclasses
  with ``.summary()`` and ``.to_dict()`` methods.
- ``examples/billboard_holdout.py`` — end-to-end OOH workflow.  On
  fully synthetic data the analysis recovers a 7% true lift as
  +7.6% (95% CI [+5.6%, +9.7%]) and computes the correct implied
  iROAS.
- ``docs/product-brief.md`` — strategic positioning document.

### Changed
- New dependency: ``scipy >= 1.10`` (for the normal-distribution CDF
  used in the power calculation).

### Test additions
- ``tests/test_experiments.py`` — 19 tests covering design selection,
  power monotonicity, post-experiment verdict logic, bootstrap
  determinism, and false-positive control under no-true-lift.
- Total test count: 63 → 82.

## [0.3.0] — 2026-05-29

Clears four of the open roadmap items in one shot: multi-geo analyzer,
bootstrap CIs, budget optimiser, and data loaders for the three most-asked-for
sources.

### Added
- **``GeoMMM``** analyzer for multi-geo data.  When ``spend_df`` and
  ``sales_df`` contain a recognised geo column (``geo``, ``region``, ``dma``,
  ``state``, ``country``), ``cl.analyze`` auto-routes to ``GeoMMM`` which:
    - Fits an independent ``RegressionMMM`` per geo.
    - Aggregates per-channel iROAS via spend-weighted (or median) cross-geo
      pooling.
    - Reports cross-geo 95% percentile CI alongside per-geo HAC/bootstrap CIs.
    - Exposes ``result.per_geo`` for drill-down.
- **Stationary block bootstrap CIs** in ``RegressionMMM`` via
  ``inference="bootstrap"``.  Block length auto-sized to ``n^(1/3)`` (Politis &
  Romano 1994).  Useful when residuals are heavily non-normal or DW < 1.0.
- **``recommend_reallocation``** — budget reallocation recommender that takes
  an ``AnalysisResult`` and a current allocation and proposes per-channel
  dollar shifts.  SCALE channels get +20% (capped at +30%), CUT channels get
  −50% (capped at −30%), HOLD / INCONCLUSIVE channels stay put.  Returns a
  ``ReallocationPlan`` with conservation accounting and an expected-lift
  estimate.
- **``causal_lift.loaders``** module:
    - ``load_shopify_orders_csv`` — Shopify "Orders export" CSV → daily revenue.
    - ``load_meta_ads_insights_csv`` — Meta Ads Manager export → daily or
      per-campaign spend.
    - ``load_google_ads_report_csv`` — Google Ads campaign report → daily or
      per-campaign spend.
    - Direct API variants (``fetch_*_api``) ship as ``NotImplementedError``
      stubs with clear messages; full OAuth flows land in v0.4.

### Changed
- Public API additions (no breaking changes): ``GeoMMM``,
  ``GeoAnalysisResult``, ``detect_geo_column``, ``recommend_reallocation``,
  ``ReallocationPlan``, ``ChannelRecommendation``.

### Test additions
- ``tests/test_geo.py``      — 8 tests
- ``tests/test_bootstrap.py`` — 5 tests
- ``tests/test_optimiser.py`` — 9 tests
- ``tests/test_loaders.py``   — 8 tests
- Total test count: 33 → 63

## [0.2.0] — 2026-05-29

Adds geometric adstock and a CI-precision safety gate.  The headline change:
the library now models carryover effects (TV and OOH no longer look dead
because their spend disperses across weeks), and refuses to issue `SCALE`
on estimates whose 95% CI is wider than the point estimate itself.

### Added
- **Geometric adstock** in `RegressionMMM`.
  - `adstock="auto"` (default): greedy per-channel grid search over
    `{0.0, 0.3, 0.5, 0.7}` by adjusted R².
  - `adstock=dict[str, float]`: explicit per-channel decay.
  - `adstock=None`: legacy v0.1 behaviour (no adstock).
  - Custom grid via `adstock_grid=(0.0, 0.2, 0.4, 0.6, 0.8)`.
  - Impulse response is normalised so β preserves its iROAS interpretation.
- **Precision gate**: `SCALE` recommendations are demoted to
  `INCONCLUSIVE` when the 95% CI width exceeds the point estimate. Catches
  the case where the model has high model fit but is genuinely uncertain
  about a specific channel's contribution.
- `AnalysisResult.adstock_thetas: dict[str, float]` exposes the selected
  decay per channel on the public API.

### Changed
- Method label updated: "RegressionMMM (multivariate OLS, trend +
  cadence-aware seasonality, geometric adstock, HAC/Newey-West SEs,
  plausibility gates)".
- `incremental_revenue` per channel now uses `β · Σ(adstocked_spend)`
  instead of `β · Σ(raw_spend)`.  With normalised adstock these differ only
  by small edge effects, but the former is exactly the model-implied
  contribution.
- `test_high_vif_forces_hold` updated to accept either `HOLD` or
  `INCONCLUSIVE` (adstock auto-search can break perfect collinearity in
  the design matrix, dropping VIF below the 10 threshold while the
  precision gate correctly catches the imprecision).

### Benchmark
On the Robyn `dt_simulated_weekly.csv` dataset:

|                                  | 0.1.0      | 0.1.1     | 0.2.0      |
|----------------------------------|-----------|-----------|-----------|
| Model R²                          | 0.48     | 0.82      | **0.83**  |
| Durbin-Watson                     | 1.23     | 1.79      | **1.84**  |
| `tv_S` recommendation             | SCALE @ 7x | HOLD @ 3.8x | **SCALE @ 5.8x** |
| `ooh_S` iROAS                     | 0.63x     | 0.17x     | **1.05x** |
| `facebook_S` recommendation       | SCALE @ 62x | HOLD @ 14x | **INCONCLUSIVE @ 30x** |

The `tv_S` recovery is the headline: adstock revealed that TV has a real,
identifiable lift that same-day regression was missing.

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
