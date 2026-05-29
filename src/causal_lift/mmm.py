"""
Regression-based Marketing Mix Model.

Multivariate OLS of revenue on per-channel spend, with linear time trend
plus cadence-appropriate seasonality controls.  Heteroskedasticity-and-
autocorrelation-consistent (HAC / Newey-West) standard errors.

Safety gates layered over the regression:
  1. VIF > 10                    -> recommendation forced to HOLD
  2. nonzero_share > 85%         -> SCALE recommendations demoted to INCONCLUSIVE
  3. Σ(iROAS·spend) / revenue
     exceeds plausibility ceiling -> all SCALE recs demoted, banner warning

What this model is honest about
-------------------------------
1. **Endogeneity.** Algorithmic bidding (Meta Advantage+, Google Smart
   Bidding) increases spend on high-demand days.  Same-day OLS will bias
   spend coefficients upward.  Surfaced as a warning; not corrected.
2. **Multicollinearity.** Correlated spend series cannot be individually
   identified.  VIF > 10 -> HOLD recommendation regardless of point estimate.
3. **Adstock / carryover.** Same-day regression ignores delayed effects.
   Flighted brand channels (TV, OOH) will appear weaker than they are.
4. **Always-on / baseline confound.** Channels with near-100% activation
   co-vary with trend and unmodelled baseline.  These produce confidently
   wrong "high iROAS" estimates -> auto-demoted to INCONCLUSIVE.
5. **Autocorrelation.** HAC (Newey-West) SEs correct residual serial
   correlation; Durbin-Watson is reported as a diagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from causal_lift.analyzer import (
    AGGREGATE_SHARE_CEILING,
    ALWAYS_ON_THRESHOLD,
    AnalysisResult,
    BaseModel,
    ChannelResult,
)


class RegressionMMM(BaseModel):
    """
    Lightweight regression-based marketing mix model.

    Model
    -----
    revenue_t = α
              + β_trend · t
              + (DOW dummies, daily only)
              + (annual Fourier sin/cos, when span ≥ 365 days)
              + Σ_i β_i · spend_{it}
              + ε_t

    β_i = incremental ROAS for channel i (causal $ revenue per $ spent).
    """

    def __init__(self, min_obs: int = 21):
        self.min_obs = min_obs

    def fit(
        self,
        spend_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        contribution_margin: float = 0.30,
    ) -> AnalysisResult:
        warnings: list[str] = []
        contribution_margin = max(0.05, min(0.95, contribution_margin))
        breakeven_roas = 1.0 / contribution_margin

        daily, channels = self._build_daily_panel(spend_df, sales_df)
        if not channels:
            raise ValueError("No channel spend columns found after merging datasets.")

        if len(daily) < self.min_obs:
            warnings.append(
                f"Only {len(daily)} periods of overlapping data — estimates are unreliable. "
                f"Recommend ≥ 60 periods for stable inference."
            )

        # Cadence detection — affects DOW handling and Fourier period
        cadence = self._detect_cadence(daily.index)
        span_days = (daily.index.max() - daily.index.min()).days
        if cadence == "irregular":
            warnings.append(
                "Date index has irregular spacing. Trend and seasonality controls assume "
                "evenly-spaced observations; estimates may be biased."
            )

        warnings.append(
            "⚠️ Identification assumption: this model assumes spend variation is "
            "budget-driven (exogenous to same-day demand). Algorithmic bidding "
            "(Meta Advantage+, Google Smart Bidding) violates this — spend will be "
            "correlated with unobserved demand shocks, biasing iROAS upward. "
            "Budget holdout experiments or geo randomisation provide stronger causal "
            "identification. Use these estimates as directional signals, not exact truths."
        )

        # Per-channel always-on share — feeds into the recommendation gates
        nonzero_share: dict[str, float] = {
            ch: float((daily[ch] > 0).mean()) for ch in channels
        }
        always_on = [ch for ch, s in nonzero_share.items() if s > ALWAYS_ON_THRESHOLD]
        if always_on:
            warnings.append(
                f"Always-on channels detected ({', '.join(always_on)} — active in "
                f">85% of periods). Same-day OLS cannot cleanly separate their effect "
                f"from baseline/trend. Their estimates are kept but SCALE recommendations "
                f"are downgraded to INCONCLUSIVE. Run a budget holdout to validate."
            )

        attr_proxy = self._compute_attribution_proxy(daily, channels)

        X = self._build_features(daily, channels, cadence=cadence, span_days=span_days)  # noqa: N806
        y = daily["revenue"]
        n = len(y)
        max_lags = max(1, int(4 * (n / 100) ** (2 / 9)))
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

        dw = float(durbin_watson(model.resid))
        if dw < 1.5:
            warnings.append(
                f"Durbin-Watson={dw:.2f} suggests positive autocorrelation in residuals. "
                f"HAC SEs partially correct for this."
            )

        if model.rsquared < 0.5:
            warnings.append(
                f"Model R²={model.rsquared:.2f} is below 0.50 — revenue may be driven by "
                f"factors outside the uploaded spend data (promotions, organic press, "
                f"competitor activity, seasonality beyond annual). Interpret estimates "
                f"conservatively."
            )

        vif_scores = self._compute_vifs(X, channels)
        high_vif = [ch for ch, v in vif_scores.items() if v > 10]
        if high_vif:
            warnings.append(
                f"High multicollinearity for: {', '.join(high_vif)} (VIF > 10). "
                f"Individual iROAS estimates are not reliably identified."
            )

        # Build per-channel results with first-pass recommendations
        channel_results: list[ChannelResult] = []
        for ch in channels:
            raw_coef = float(model.params[ch])
            ci_low, ci_high = model.conf_int(alpha=0.05).loc[ch]
            total_spend = float(daily[ch].sum())
            vif = vif_scores.get(ch)
            share = nonzero_share[ch]

            display_roas = max(0.0, raw_coef)
            incremental_rev = display_roas * total_spend

            rec, reason = self.recommend(
                display_roas,
                [float(ci_low), float(ci_high)],
                breakeven_roas=breakeven_roas,
                vif=vif,
                nonzero_share=share,
            )

            channel_results.append(
                ChannelResult(
                    channel=ch,
                    total_spend=total_spend,
                    attribution_proxy_roas=attr_proxy.get(ch, 0.0),
                    incremental_roas=display_roas,
                    incremental_revenue=incremental_rev,
                    confidence_interval=[float(ci_low), float(ci_high)],
                    recommendation=rec,
                    recommendation_reason=reason,
                    model_fit=float(model.rsquared),
                    vif_score=vif,
                    raw_coef=raw_coef,
                    nonzero_share=share,
                )
            )

        # Aggregate plausibility gate — if implied incremental share is silly, demote SCALEs
        total_revenue = float(daily["revenue"].sum())
        total_incremental = sum(c.incremental_revenue for c in channel_results)
        implied_share = total_incremental / total_revenue if total_revenue > 0 else 0.0

        if implied_share > AGGREGATE_SHARE_CEILING:
            warnings.append(
                f"Aggregate implied incremental share is {implied_share:.0%} of revenue "
                f"(threshold {AGGREGATE_SHARE_CEILING:.0%}). The model is over-attributing "
                f"revenue to paid media — likely confounded with baseline / trend / "
                f"seasonality. All SCALE recommendations are downgraded to INCONCLUSIVE."
            )
            for c in channel_results:
                if c.recommendation == "SCALE":
                    c.recommendation = "INCONCLUSIVE"
                    c.recommendation_reason = (
                        f"Demoted from SCALE: aggregate implied incremental share is "
                        f"{implied_share:.0%}, which is implausibly high. The model is "
                        f"over-attributing revenue to paid media; this channel's "
                        f"{c.incremental_roas:.1f}x estimate cannot be trusted."
                    )

        return AnalysisResult(
            channels=channel_results,
            method_used=(
                "RegressionMMM (multivariate OLS, trend + cadence-aware seasonality, "
                "HAC/Newey-West SEs, plausibility gates)"
            ),
            total_revenue=total_revenue,
            total_spend=float(daily[channels].sum().sum()),
            r_squared=float(model.rsquared),
            observations=n,
            contribution_margin=contribution_margin,
            breakeven_roas=round(breakeven_roas, 2),
            durbin_watson=dw,
            cadence=cadence,
            implied_incremental_share=implied_share,
            warnings=warnings,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_cadence(idx: pd.DatetimeIndex) -> str:
        """
        Inspect the modal gap between consecutive timestamps.

        Returns one of: "daily", "weekly", "irregular".
        """
        if len(idx) < 2:
            return "daily"
        diffs = pd.Series(idx).diff().dropna()
        mode = diffs.mode().iloc[0]
        days = mode.days
        if days == 1:
            return "daily"
        if 5 <= days <= 8:
            return "weekly"
        # Anything else is treated as irregular (monthly, mixed, etc.)
        return "irregular"

    @staticmethod
    def _build_daily_panel(
        spend_df: pd.DataFrame, sales_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        spend_wide = spend_df.pivot_table(
            index="date", columns="channel", values="spend", aggfunc="sum", fill_value=0
        )
        spend_wide.columns.name = None
        sales_idx = sales_df.set_index("date")[["revenue"]]
        daily = sales_idx.join(spend_wide, how="inner").sort_index().fillna(0)
        channels = [c for c in daily.columns if c != "revenue" and daily[c].sum() > 0]
        return daily, channels

    @staticmethod
    def _build_features(
        daily: pd.DataFrame,
        channels: list[str],
        cadence: str,
        span_days: int,
    ) -> pd.DataFrame:
        """
        Construct the OLS design matrix.

        Always includes:    const, linear trend, channel spend columns.
        Daily-only:         day-of-week dummies (Sun = reference).
        Span >= 365 days:   annual Fourier sin/cos (handles seasonal patterns).
        """
        n = len(daily)
        feat: dict[str, np.ndarray] = {}
        feat["trend"] = np.arange(n, dtype=float)

        # Day-of-week dummies — only meaningful for daily data
        if cadence == "daily":
            dow = daily.index.dayofweek
            for d in range(6):  # Sunday = reference
                feat[f"dow_{d}"] = (dow == d).astype(float)

        # Annual Fourier seasonality — needs ≥ 1 year of data
        if span_days >= 365:
            t = np.arange(n, dtype=float)
            if cadence == "weekly":
                period = 52.0
            elif cadence == "daily":
                period = 365.25
            else:
                # irregular cadence — fall back to a generic annual cycle on row index
                period = float(n) * (365.0 / max(span_days, 1))
            feat["sin_annual"] = np.sin(2 * np.pi * t / period)
            feat["cos_annual"] = np.cos(2 * np.pi * t / period)

        for ch in channels:
            feat[ch] = daily[ch].values.astype(float)

        return sm.add_constant(pd.DataFrame(feat, index=daily.index))

    @staticmethod
    def _compute_vifs(X: pd.DataFrame, channels: list[str]) -> dict[str, float]:  # noqa: N803
        X_arr = X.values  # noqa: N806
        col_names = list(X.columns)
        vifs: dict[str, float] = {}
        for ch in channels:
            idx = col_names.index(ch)
            try:
                vifs[ch] = float(variance_inflation_factor(X_arr, idx))
            except Exception:
                vifs[ch] = float("inf")
        return vifs

    @staticmethod
    def _compute_attribution_proxy(
        daily: pd.DataFrame, channels: list[str]
    ) -> dict[str, float]:
        """Naive proportional daily attribution — NOT what platforms report."""
        total_daily_spend = daily[channels].sum(axis=1)
        result: dict[str, float] = {}
        for ch in channels:
            ch_total = daily[ch].sum()
            if ch_total == 0:
                result[ch] = 0.0
                continue
            valid = total_daily_spend > 0
            share = daily.loc[valid, ch] / total_daily_spend[valid]
            attributed = (share * daily.loc[valid, "revenue"]).sum()
            result[ch] = float(attributed / ch_total)
        return result
