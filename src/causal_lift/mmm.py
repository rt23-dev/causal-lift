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

    # Default geometric adstock decay grid for the "auto" search.
    # 0.0 = no carryover; 0.7 = ~2-period half-life (≈ 2 weeks for weekly, 2 days for daily).
    DEFAULT_ADSTOCK_GRID: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7)

    def __init__(
        self,
        min_obs: int = 21,
        adstock: str | dict[str, float] | None = "auto",
        adstock_grid: tuple[float, ...] | None = None,
        inference: str = "hac",
        n_bootstrap: int = 1000,
        random_state: int = 0,
    ):
        """
        Parameters
        ----------
        min_obs : int
            Minimum observations before a warning fires.
        adstock : "auto" | dict[str, float] | None
            Geometric adstock decay applied to each channel's spend before
            regression.  Models carryover effects (TV/OOH/YouTube linger;
            search/display do not).
            - "auto" (default): greedy per-channel grid search by adjusted R².
            - dict mapping channel name to theta ∈ [0, 1): explicit override.
            - None: no adstock (legacy v0.1 behaviour).
        adstock_grid : tuple of floats, optional
            Values searched when `adstock="auto"`.  Defaults to
            (0.0, 0.3, 0.5, 0.7).
        inference : "hac" | "bootstrap", default "hac"
            How to compute standard errors and confidence intervals.
            - "hac": Newey-West heteroskedasticity-and-autocorrelation-
              consistent SEs (closed-form, fast).
            - "bootstrap": stationary block bootstrap (Politis-Romano).  Slower
              but more robust to non-normal residuals and heavy autocorrelation
              (DW < 1.0).  Block length set automatically.
        n_bootstrap : int, default 1000
            Number of bootstrap replicates when `inference="bootstrap"`.
        random_state : int, default 0
            Seed for bootstrap resampling.  Pure cosmetic if `inference="hac"`.
        """
        self.min_obs = min_obs
        self.adstock = adstock
        self.adstock_grid = (
            tuple(adstock_grid) if adstock_grid is not None else self.DEFAULT_ADSTOCK_GRID
        )
        if inference not in {"hac", "bootstrap"}:
            raise ValueError(f"inference must be 'hac' or 'bootstrap', got {inference!r}")
        self.inference = inference
        self.n_bootstrap = int(n_bootstrap)
        self.random_state = int(random_state)

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

        # Adstock: select per-channel decay (auto / explicit / off)
        adstock_thetas = self._resolve_adstock_thetas(daily, channels, cadence, span_days)
        if any(t > 0 for t in adstock_thetas.values()):
            nz_adstock = ", ".join(
                f"{ch}={t:.1f}" for ch, t in adstock_thetas.items() if t > 0
            )
            warnings.append(
                f"Adstock applied (geometric, per-channel auto-selected by adjusted R²): "
                f"{nz_adstock}. theta=0 means no carryover; theta=0.7 means a long tail."
            )

        X = self._build_features(  # noqa: N806
            daily, channels, cadence=cadence, span_days=span_days,
            adstock_thetas=adstock_thetas,
        )
        y = daily["revenue"]
        n = len(y)
        max_lags = max(1, int(4 * (n / 100) ** (2 / 9)))
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

        # Optionally replace HAC CIs with stationary bootstrap CIs.
        bootstrap_cis: dict[str, tuple[float, float]] | None = None
        if self.inference == "bootstrap":
            bootstrap_cis = self._stationary_bootstrap_cis(X, y, channels)
            warnings.append(
                f"Inference method: stationary block bootstrap "
                f"({self.n_bootstrap} replicates). CIs replace the HAC closed-form "
                f"intervals; point estimates are unchanged."
            )

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
            if bootstrap_cis is not None and ch in bootstrap_cis:
                ci_low, ci_high = bootstrap_cis[ch]
            else:
                ci_low, ci_high = model.conf_int(alpha=0.05).loc[ch]
            total_spend = float(daily[ch].sum())
            vif = vif_scores.get(ch)
            share = nonzero_share[ch]

            display_roas = max(0.0, raw_coef)
            # Use adstocked spend sum for incremental_revenue (β·Σadstock(spend) is
            # the true model-implied contribution; with normalised adstock this is
            # very close to β·Σspend with only small edge effects).
            adstocked_total = float(
                self._apply_adstock(daily[ch].values.astype(float), adstock_thetas[ch]).sum()
            )
            incremental_rev = display_roas * adstocked_total

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
                "geometric adstock, HAC/Newey-West SEs, plausibility gates)"
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
            adstock_thetas=adstock_thetas,
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

    @classmethod
    def _build_features(
        cls,
        daily: pd.DataFrame,
        channels: list[str],
        cadence: str,
        span_days: int,
        adstock_thetas: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Construct the OLS design matrix.

        Always includes:    const, linear trend, channel spend columns
                            (geometrically adstocked if `adstock_thetas` given).
        Daily-only:         day-of-week dummies (Sun = reference).
        Span >= 365 days:   annual Fourier sin/cos (handles seasonal patterns).
        """
        if adstock_thetas is None:
            adstock_thetas = {ch: 0.0 for ch in channels}

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
            raw = daily[ch].values.astype(float)
            theta = adstock_thetas.get(ch, 0.0)
            feat[ch] = cls._apply_adstock(raw, theta) if theta > 0 else raw

        return sm.add_constant(pd.DataFrame(feat, index=daily.index))

    @staticmethod
    def _apply_adstock(spend: np.ndarray, theta: float) -> np.ndarray:
        """
        Geometric adstock with normalised impulse response.

        Impulse response: (1-θ), (1-θ)θ, (1-θ)θ², ...   →   sums to 1.
        Normalisation preserves the iROAS interpretation of the spend
        coefficient: $1 of raw spend still produces β of (time-shifted) revenue.

        Recurrence: raw_out[t] = spend[t] + θ · raw_out[t-1],
                    adstocked[t] = (1-θ) · raw_out[t]
        """
        if theta <= 0:
            return spend
        if theta >= 1:
            raise ValueError(f"adstock theta must be < 1, got {theta}")
        n = len(spend)
        out = np.zeros(n, dtype=float)
        out[0] = spend[0]
        for t in range(1, n):
            out[t] = spend[t] + theta * out[t - 1]
        return out * (1.0 - theta)

    def _resolve_adstock_thetas(
        self,
        daily: pd.DataFrame,
        channels: list[str],
        cadence: str,
        span_days: int,
    ) -> dict[str, float]:
        """Dispatch to whichever adstock-selection mode is configured."""
        if self.adstock is None:
            return {ch: 0.0 for ch in channels}
        if isinstance(self.adstock, dict):
            return {ch: float(self.adstock.get(ch, 0.0)) for ch in channels}
        if self.adstock == "auto":
            return self._select_adstock_auto(daily, channels, cadence, span_days)
        raise ValueError(
            f"adstock must be 'auto', a dict, or None — got {self.adstock!r}"
        )

    def _select_adstock_auto(
        self,
        daily: pd.DataFrame,
        channels: list[str],
        cadence: str,
        span_days: int,
    ) -> dict[str, float]:
        """
        Greedy per-channel grid search over `adstock_grid`.

        For each channel in turn, holds all other thetas at their current best
        and picks the value of this channel's theta that maximises **adjusted**
        R² (which penalises model complexity, mitigating overfit risk).

        Uses plain OLS (not HAC) during the search — fits are ~1ms each so the
        whole search is sub-second on typical data.  The chosen thetas are
        then used in the final HAC-SE fit done by `fit()`.
        """
        thetas: dict[str, float] = {ch: 0.0 for ch in channels}
        y = daily["revenue"]
        for target in channels:
            best_r2_adj = -np.inf
            best_theta = 0.0
            for theta in self.adstock_grid:
                trial = {**thetas, target: theta}
                X_trial = self._build_features(  # noqa: N806
                    daily, channels, cadence=cadence, span_days=span_days,
                    adstock_thetas=trial,
                )
                try:
                    m = sm.OLS(y, X_trial).fit()
                    r2_adj = float(m.rsquared_adj)
                except Exception:
                    continue
                if r2_adj > best_r2_adj:
                    best_r2_adj = r2_adj
                    best_theta = theta
            thetas[target] = best_theta
        return thetas

    def _stationary_bootstrap_cis(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        channels: list[str],
        alpha: float = 0.05,
    ) -> dict[str, tuple[float, float]]:
        """
        Stationary block bootstrap (Politis & Romano, 1994) for time-series CIs.

        Each replicate resamples observations in geometrically-distributed
        blocks (expected block length n^(1/3)), preserving short-range
        autocorrelation structure in the residuals.

        Returns a dict mapping channel name → (ci_lower, ci_upper) at
        ``1 - alpha`` coverage.  Channels whose bootstrap distribution
        degenerates (e.g. perfect collinearity in some resamples) fall back to
        ``(nan, nan)``.
        """
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        avg_block_length = max(2.0, n ** (1.0 / 3.0))
        p_restart = 1.0 / avg_block_length

        X_arr = X.values  # noqa: N806
        y_arr = y.values
        col_names = list(X.columns)
        ch_indices = {ch: col_names.index(ch) for ch in channels}

        boot_coefs: dict[str, list[float]] = {ch: [] for ch in channels}

        for _ in range(self.n_bootstrap):
            # Build stationary-bootstrap indices
            idx = np.empty(n, dtype=np.int64)
            i = int(rng.integers(0, n))
            for t in range(n):
                idx[t] = i
                if rng.random() < p_restart:
                    i = int(rng.integers(0, n))
                else:
                    i = (i + 1) % n
            X_b = X_arr[idx]  # noqa: N806
            y_b = y_arr[idx]
            try:
                m = sm.OLS(y_b, X_b).fit()
                for ch, j in ch_indices.items():
                    boot_coefs[ch].append(float(m.params[j]))
            except Exception:
                continue

        cis: dict[str, tuple[float, float]] = {}
        lo_q, hi_q = 100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)
        for ch in channels:
            coefs = np.asarray(boot_coefs[ch], dtype=float)
            if coefs.size == 0:
                cis[ch] = (float("nan"), float("nan"))
                continue
            cis[ch] = (
                float(np.percentile(coefs, lo_q)),
                float(np.percentile(coefs, hi_q)),
            )
        return cis

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
