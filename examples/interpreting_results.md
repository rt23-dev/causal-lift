# Interpreting `causal-lift` results

A practical guide to what each output field means and how to use it without
fooling yourself.

## The core fields

### `incremental_roas`
The model's causal estimate of $ revenue per $ spent for this channel, after
controlling for organic time trend and day-of-week effects.

- **What it is:** an OLS coefficient β_i from `revenue ~ trend + DOW + Σ spend_i`.
- **What it is not:** a guaranteed return. The point estimate is the *centre*
  of a distribution. Always read it together with the CI.

### `confidence_interval`
A 95% confidence interval on `incremental_roas`, computed using HAC
(Newey-West) standard errors to correct for autocorrelation in daily revenue
residuals.

- **Narrow CI** (e.g. [1.7, 2.1]) → high confidence in the point estimate.
- **Wide CI** (e.g. [0.2, 4.5]) → genuinely don't know if the channel is
  making or losing money.
- **CI straddles breakeven** → cannot rule out either profitability or value
  destruction. Run an experiment before acting.

### `vif_score`
Variance Inflation Factor for this channel's spend after partialling out
trend and DOW. Measures multicollinearity with other channels.

- **VIF < 5** → estimate is well-identified.
- **5 ≤ VIF ≤ 10** → moderate collinearity; treat point estimate with caution.
- **VIF > 10** → estimate is not credible. The library forces a `HOLD`
  recommendation regardless of the point estimate. Run a budget experiment
  with deliberate, independent variation in this channel to fix it.

### `breakeven_roas`
`1 / contribution_margin`. The iROAS your channel must exceed to be net-positive
on contribution profit. **All recommendations are anchored to this number.**

Examples:
- 30% margin → 3.33x breakeven
- 40% margin → 2.50x breakeven
- 50% margin → 2.00x breakeven

### `recommendation`
- `SCALE` — `iROAS ≥ breakeven` AND CI lower bound ≥ 75% of breakeven.
- `CUT` — `iROAS < 85% of breakeven` AND CI upper bound `< breakeven`.
- `HOLD` — everything else (uncertainty too wide to act, or VIF > 10).

Treat these as **hypotheses to test**, not decisions to execute blindly.

## Diagnostics

### `r_squared`
Fraction of revenue variance the model explains. Below 0.3 → something
material is missing from your data (promotions, seasonality, organic press).

### `durbin_watson`
Autocorrelation diagnostic on residuals. ~2 = no autocorrelation. < 1.5 = positive autocorrelation; HAC SEs partially correct but a richer model would be better.

### `attribution_proxy_roas`
A *naive* proportional attribution baseline: on each day, total revenue is
allocated to channels by their share of total daily spend. This is **not**
what your ad platforms report. It exists only to show the gap between naive
and causal attribution. Compare against your real platform numbers manually.

## Common mistakes

1. **Acting on a point estimate inside a wide CI.** If the CI straddles your
   breakeven, the model is telling you it doesn't know. Don't pretend it does.

2. **Trusting a high iROAS estimate when VIF > 5.** Multicollinearity makes
   per-channel coefficients unstable. Adding or removing one channel can flip
   another's sign. Budget experiments fix this.

3. **Ignoring the endogeneity warning.** If your campaigns use Meta
   Advantage+, Google Smart Bidding, or any rules-based scaling that responds
   to demand, your spend is endogenous and the model overestimates iROAS.
   Magnitude unknown without experiments.

4. **Forgetting that this model has no adstock.** Brand-awareness channels
   (TikTok, YouTube, Pinterest) drive delayed conversions that this model
   can't capture. Their iROAS will look artificially low.

5. **Comparing the attribution proxy to Meta's reported ROAS.** They will
   never match. Different mechanism.

## What to do with the results

- **SCALE channels:** increase spend in 20-30% increments and re-measure. If
  iROAS holds, keep scaling. If it drops, you've hit saturation.
- **CUT channels:** before fully cutting, run a 2-week budget holdout to
  confirm. Sometimes a low iROAS estimate is a multicollinearity artefact.
- **HOLD channels:** the highest-value action is usually a budget experiment.
  Pick one channel, swing its budget by ±30%, hold others constant, re-run.
