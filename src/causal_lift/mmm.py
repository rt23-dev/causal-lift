"""
Regression-based Marketing Mix Model.

Multivariate OLS of daily revenue on per-channel spend, with linear time
trend and day-of-week controls.  Heteroskedasticity-and-autocorrelation-
consistent (HAC / Newey-West) standard errors.  Variance Inflation Factors
are reported per channel and used to gate confidence in individual estimates.

What this model is honest about
-------------------------------
1. **Endogeneity.** Algorithmic bidding (Meta Advantage+, Google Smart
   Bidding) increases spend on high-demand days. This biases the spend
   coefficients upward. Surfaced as a warning; not corrected.
2. **Multicollinearity.** Channels that scale budgets together have
   correlated spend series. OLS cannot separate their individual effects.
   VIF > 10 means the channel's estimate is not credible.
3. **Adstock / carryover.** Same-day regression ignores delayed effects.
   Brand channels (TikTok, YouTube) will appear weaker than they are.
4. **Autocorrelation.** HAC (Newey-West) SEs correct for residual serial
   correlation; Durbin-Watson is reported as a diagnostic.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from causal_lift.analyzer import AnalysisResult, BaseModel, ChannelResult


class RegressionMMM(BaseModel):
    """
    Lightweight regression-based marketing mix model.

    Model
    -----
    revenue_t = α + β_trend·t + Σ_{k=0}^{5} β_dow_k·DOW_{kt} + Σ_i β_i·spend_{it} + ε_t

    β_i = incremental ROAS for channel i (causal $ revenue per $ spent).
    """

    MIN_OBS = 21

    def __init__(self, min_obs: int = 21):
        self.min_obs = min_obs

    def fit(
        self,
        spend_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        contribution_margin: float = 0.30,
    ) -> AnalysisResult:
        warnings: List[str] = []
        contribution_margin = max(0.05, min(0.95, contribution_margin))
        breakeven_roas = 1.0 / contribution_margin

        daily, channels = self._build_daily_panel(spend_df, sales_df)
        if not channels:
            raise ValueError("No channel spend columns found after merging datasets.")

        if len(daily) < self.min_obs:
            warnings.append(
                f"Only {len(daily)} days of overlapping data — estimates are unreliable. "
                f"Recommend ≥ 60 days for stable inference."
            )

        warnings.append(
            "⚠️ Identification assumption: this model assumes spend variation is "
            "budget-driven (exogenous to same-day demand). Algorithmic bidding "
            "(Meta Advantage+, Google Smart Bidding) violates this — spend will be "
            "correlated with unobserved demand shocks, biasing iROAS upward. "
            "Budget holdout experiments or geo randomisation provide stronger causal "
            "identification. Use these estimates as directional signals, not exact truths."
        )

        attr_proxy = self._compute_attribution_proxy(daily, channels)

        X = self._build_features(daily, channels)
        y = daily["revenue"]
        n = len(y)
        max_lags = max(1, int(4 * (n / 100) ** (2 / 9)))
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

        dw = float(durbin_watson(model.resid))
        if dw < 1.5:
            warnings.append(
                f"Durbin-Watson={dw:.2f} suggests positive autocorrelation in residuals. "
                f"HAC SEs partially correct for this, but consider longer differencing "
                f"windows or a lag-augmented model."
            )

        if model.rsquared < 0.3:
            warnings.append(
                f"Model R²={model.rsquared:.2f} is low — revenue may be driven by factors "
                f"outside the uploaded spend data (promotions, organic press, seasonality). "
                f"Interpret all estimates conservatively."
            )

        vif_scores = self._compute_vifs(X, channels)
        high_vif = [ch for ch, v in vif_scores.items() if v > 10]
        if high_vif:
            warnings.append(
                f"High multicollinearity detected for: {', '.join(high_vif)} (VIF > 10). "
                f"These channels have correlated spend series — their individual iROAS "
                f"estimates are not reliably identified. Run budget experiments with "
                f"deliberate, independent variation."
            )

        channel_results: List[ChannelResult] = []
        for ch in channels:
            raw_coef = float(model.params[ch])
            ci_low, ci_high = model.conf_int(alpha=0.05).loc[ch]
            total_spend = float(daily[ch].sum())
            vif = vif_scores.get(ch)

            display_roas = max(0.0, raw_coef)
            incremental_rev = display_roas * total_spend

            rec, reason = self.recommend(
                display_roas,
                [float(ci_low), float(ci_high)],
                breakeven_roas=breakeven_roas,
                vif=vif,
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
                )
            )

        return AnalysisResult(
            channels=channel_results,
            method_used="RegressionMMM (multivariate OLS, trend + DOW controls, HAC/Newey-West SEs)",
            total_revenue=float(daily["revenue"].sum()),
            total_spend=float(daily[channels].sum().sum()),
            r_squared=float(model.rsquared),
            observations=n,
            contribution_margin=contribution_margin,
            breakeven_roas=round(breakeven_roas, 2),
            durbin_watson=dw,
            warnings=warnings,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_daily_panel(
        spend_df: pd.DataFrame, sales_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        spend_wide = spend_df.pivot_table(
            index="date", columns="channel", values="spend", aggfunc="sum", fill_value=0
        )
        spend_wide.columns.name = None
        sales_idx = sales_df.set_index("date")[["revenue"]]
        daily = sales_idx.join(spend_wide, how="inner").sort_index().fillna(0)
        channels = [c for c in daily.columns if c != "revenue" and daily[c].sum() > 0]
        return daily, channels

    @staticmethod
    def _build_features(daily: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
        n = len(daily)
        feat: Dict[str, np.ndarray] = {}
        feat["trend"] = np.arange(n, dtype=float)
        dow = daily.index.dayofweek
        for d in range(6):  # Sunday = reference category
            feat[f"dow_{d}"] = (dow == d).astype(float)
        for ch in channels:
            feat[ch] = daily[ch].values.astype(float)
        return sm.add_constant(pd.DataFrame(feat, index=daily.index))

    @staticmethod
    def _compute_vifs(X: pd.DataFrame, channels: List[str]) -> Dict[str, float]:
        X_arr = X.values
        col_names = list(X.columns)
        vifs: Dict[str, float] = {}
        for ch in channels:
            idx = col_names.index(ch)
            try:
                vifs[ch] = float(variance_inflation_factor(X_arr, idx))
            except Exception:
                vifs[ch] = float("inf")
        return vifs

    @staticmethod
    def _compute_attribution_proxy(
        daily: pd.DataFrame, channels: List[str]
    ) -> Dict[str, float]:
        """
        Proportional daily attribution proxy.

        On each day, allocate total revenue to channels by their spend share.
        This is a *naive baseline* that overcounts because it credits organic
        revenue to whoever spent money that day. It is NOT what advertising
        platforms actually report — platforms use click/view attribution windows
        on tracked conversions. The proxy exists only to illustrate the gap
        between naive and causal attribution. Compare to your actual platform
        dashboards manually.
        """
        total_daily_spend = daily[channels].sum(axis=1)
        result: Dict[str, float] = {}
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
