"""
Geo holdout experiment design and analysis — the product layer.

This is the piece that turns ``causal-lift`` from a passive measurement
library into an active experimentation tool.  It answers two questions
brands actually pay for:

1.  **Pre-experiment.**  *Given my baseline revenue across DMAs, which
    geos should I treat, which should be controls, how long should the
    test run, and what lift can I credibly detect?*

2.  **Post-experiment.**  *Given the actual treated/control results,
    what was the causal lift?  Does it cross statistical significance?
    What's the implied iROAS on the spend change?*

Designed for physical-media use cases (TV, OOH, podcast, radio, DM)
where geo-based experiments are the only credible identification
strategy — but the same logic works for digital channels you can
selectively pause by region.

Method
------

**Geo matching.**  Treated DMAs are chosen by maximising similarity to
the remaining control pool.  Similarity is the Pearson correlation of
log-revenue between candidate DMAs, weighted by the revenue magnitude
match.  This is a lightweight alternative to full synthetic control —
fast, robust, and good enough for most practical experiments.

**Power calculation.**  Standard diff-in-diff power formula applied to
the pooled treated-vs-control revenue series, using historical baseline
variance.  Reports both:

- ``power_at_expected_lift`` — probability the test rejects the null
  given the expected effect.
- ``minimum_detectable_effect`` — smallest lift the test can reliably
  detect at 80 % power.

**Post-experiment analysis.**  Computes the per-period diff-in-diff:

  ``lift_t = (revenue_treated_t / pre_period_treated_mean) -
             (revenue_control_t / pre_period_control_mean)``

then aggregates across the post period with a stationary block
bootstrap for the confidence interval (robust to autocorrelation in
the daily / weekly series).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# ── Pre-experiment design ─────────────────────────────────────────────────────


@dataclass
class GeoHoldoutDesign:
    """Output of :func:`design_geo_holdout`."""

    treated_geos: list[str]
    control_geos: list[str]
    duration_weeks: int
    expected_lift_pct: float
    significance_level: float
    power_at_expected_lift: float
    minimum_detectable_effect: float    # smallest lift detectable at 80% power
    baseline_treated_weekly: float       # avg weekly revenue across treated geos
    baseline_control_weekly: float
    baseline_variance: float
    similarity_score: float              # 0-1, treated-vs-control fit quality
    rationale: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "treated_geos": self.treated_geos,
            "control_geos": self.control_geos,
            "duration_weeks": self.duration_weeks,
            "expected_lift_pct": round(self.expected_lift_pct, 4),
            "significance_level": self.significance_level,
            "power_at_expected_lift": round(self.power_at_expected_lift, 3),
            "minimum_detectable_effect": round(self.minimum_detectable_effect, 4),
            "baseline_treated_weekly": round(self.baseline_treated_weekly, 2),
            "baseline_control_weekly": round(self.baseline_control_weekly, 2),
            "baseline_variance": round(self.baseline_variance, 2),
            "similarity_score": round(self.similarity_score, 3),
            "rationale": self.rationale,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        lines = [
            "Geo Holdout Experiment Design",
            "=" * 50,
            f"Treated DMAs:        {', '.join(self.treated_geos)}",
            f"Control DMAs:        {len(self.control_geos)} geos "
            f"({', '.join(self.control_geos[:4])}"
            f"{'...' if len(self.control_geos) > 4 else ''})",
            f"Duration:            {self.duration_weeks} weeks",
            "",
            f"Expected lift:       {self.expected_lift_pct:.1%}",
            f"Power at expected:   {self.power_at_expected_lift:.0%}",
            f"Min detectable:      {self.minimum_detectable_effect:.1%} (at 80% power)",
            f"Significance level:  {self.significance_level:.0%}",
            "",
            f"Baseline (treated):  ${self.baseline_treated_weekly:,.0f}/wk avg",
            f"Baseline (control):  ${self.baseline_control_weekly:,.0f}/wk avg",
            f"Treated/control fit: {self.similarity_score:.2f} "
            f"({'good' if self.similarity_score > 0.7 else 'acceptable' if self.similarity_score > 0.5 else 'POOR — pick different geos'})",
            "",
            f"Rationale: {self.rationale}",
        ]
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


def design_geo_holdout(
    baseline: pd.DataFrame,
    *,
    n_treated: int = 3,
    duration_weeks: int = 4,
    expected_lift_pct: float = 0.05,
    significance_level: float = 0.05,
    treated_geos: list[str] | None = None,
    geo_column: str = "geo",
    date_column: str = "date",
    revenue_column: str = "revenue",
) -> GeoHoldoutDesign:
    """
    Design a geo holdout experiment from historical baseline revenue.

    The function picks treated geos that best match the remainder of the
    pool (high pre-period correlation + similar magnitudes), then runs a
    diff-in-diff power calculation against the proposed lift.

    Parameters
    ----------
    baseline : DataFrame
        Long-format historical data with columns ``geo``, ``date``,
        ``revenue``.  Recommended: 12-26 weeks of weekly observations
        across at least 5 geos.
    n_treated : int, default 3
        How many geos to treat (rest become controls).
    duration_weeks : int, default 4
        Planned experiment duration.
    expected_lift_pct : float, default 0.05
        Lift you expect the spend change to produce (e.g. 5%).  Drives
        power calculation.
    significance_level : float, default 0.05
        Statistical significance threshold (alpha).
    treated_geos : list[str], optional
        If provided, use these as the treated set instead of
        auto-selecting.  Useful when the brand has constraints
        (e.g. "must include NYC, must exclude LA").
    geo_column, date_column, revenue_column : str
        Override column names if your DataFrame uses different ones.

    Returns
    -------
    GeoHoldoutDesign
    """
    warnings: list[str] = []
    df = baseline.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    geos = sorted(df[geo_column].unique())
    if len(geos) < 4:
        warnings.append(
            f"Only {len(geos)} geos in baseline data. Recommend ≥5 for a "
            "credible treated/control split."
        )
    if n_treated >= len(geos):
        raise ValueError(
            f"n_treated ({n_treated}) must be less than total geos ({len(geos)})."
        )

    # Pivot to wide format (one column per geo) for similarity calc
    wide = df.pivot_table(
        index=date_column, columns=geo_column, values=revenue_column, aggfunc="sum"
    ).fillna(0)

    if treated_geos is None:
        treated_geos, similarity = _select_treated_geos(wide, n_treated)
    else:
        missing = set(treated_geos) - set(geos)
        if missing:
            raise ValueError(f"Treated geos not found in baseline: {missing}")
        similarity = _similarity_score(wide, treated_geos)

    control_geos = [g for g in geos if g not in treated_geos]

    # Baseline weekly revenue + variance
    weekly_treated = wide[treated_geos].sum(axis=1)
    weekly_control = wide[control_geos].sum(axis=1)
    baseline_treated = float(weekly_treated.mean())
    baseline_control = float(weekly_control.mean())
    # Pooled variance of the (treated - control)/control ratio, which is
    # roughly the noise floor of our diff-in-diff estimator
    ratio_series = (weekly_treated / baseline_treated) - (
        weekly_control / baseline_control
    )
    baseline_variance = float(ratio_series.var()) if len(ratio_series) > 1 else 0.0

    # Power calc: standard z-test on the diff-in-diff at alpha level
    if baseline_variance <= 0:
        power = 0.0
        mde = float("inf")
        warnings.append(
            "Baseline variance is zero — power calculation degenerate. "
            "Check that the baseline has enough periods of real variation."
        )
    else:
        se = np.sqrt(baseline_variance / max(duration_weeks, 1))
        z_alpha = stats.norm.ppf(1 - significance_level / 2)
        z_lift = expected_lift_pct / se - z_alpha
        power = float(stats.norm.cdf(z_lift))
        # Minimum detectable effect at 80% power
        z_power = stats.norm.ppf(0.8)
        mde = float(se * (z_alpha + z_power))

    if power < 0.5:
        warnings.append(
            f"Power at expected lift is only {power:.0%}. Consider running "
            f"the test longer ({int(duration_weeks * (0.8 / max(power, 0.1)))}+ weeks) "
            "or expecting a larger lift."
        )

    if similarity < 0.5:
        warnings.append(
            f"Treated/control similarity is {similarity:.2f} — below the 0.5 "
            "rough threshold. The diff-in-diff estimator assumes treated and "
            "control geos move together pre-treatment. Consider hand-picking "
            "more similar geos."
        )

    rationale = (
        f"Selected {n_treated} treated geos to maximise pre-period correlation "
        f"with the remaining {len(control_geos)} control geos. At {duration_weeks} "
        f"weeks of duration, the experiment can detect a {mde:.1%} lift at 80% power "
        f"(alpha={significance_level:.0%})."
    )

    return GeoHoldoutDesign(
        treated_geos=treated_geos,
        control_geos=control_geos,
        duration_weeks=duration_weeks,
        expected_lift_pct=expected_lift_pct,
        significance_level=significance_level,
        power_at_expected_lift=power,
        minimum_detectable_effect=mde,
        baseline_treated_weekly=baseline_treated,
        baseline_control_weekly=baseline_control,
        baseline_variance=baseline_variance,
        similarity_score=similarity,
        rationale=rationale,
        warnings=warnings,
    )


def _select_treated_geos(wide: pd.DataFrame, n_treated: int) -> tuple[list[str], float]:
    """
    Greedy selection: pick the set of n_treated geos that maximises
    similarity to the remaining control pool.

    Similarity = mean Pearson correlation of log-revenue series.
    """
    geos = list(wide.columns)
    log_rev = np.log1p(wide)

    # Greedy: at each step add the candidate geo that yields the highest
    # treated-vs-control similarity. Always return n_treated geos (callers
    # explicitly asked for that many); the final similarity score reflects
    # the final selection, not the intermediate maximum.
    remaining = set(geos)
    selected: list[str] = []

    for _ in range(n_treated):
        best_candidate = None
        best_candidate_score = -np.inf
        for cand in remaining:
            trial = selected + [cand]
            score = _similarity_score(wide, trial, log_rev=log_rev)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = cand
        if best_candidate is None:
            break
        selected.append(best_candidate)
        remaining.discard(best_candidate)

    final_score = _similarity_score(wide, selected, log_rev=log_rev) if selected else 0.0
    return selected, float(final_score)


def _similarity_score(
    wide: pd.DataFrame, treated: list[str], log_rev: pd.DataFrame | None = None
) -> float:
    """Mean Pearson correlation of log-revenue between treated set and rest."""
    if log_rev is None:
        log_rev = np.log1p(wide)
    treated_mean = log_rev[treated].mean(axis=1)
    controls = [g for g in wide.columns if g not in treated]
    if not controls:
        return 0.0
    control_mean = log_rev[controls].mean(axis=1)
    if treated_mean.std() == 0 or control_mean.std() == 0:
        return 0.0
    return float(treated_mean.corr(control_mean))


# ── Post-experiment analysis ──────────────────────────────────────────────────


@dataclass
class GeoHoldoutResult:
    """Output of :func:`analyze_geo_holdout`."""

    measured_lift_pct: float                  # observed lift in treated vs control
    confidence_interval: list[float]          # 95% CI on the lift
    is_significant: bool
    verdict: str                              # "LIFT_DETECTED" | "INCONCLUSIVE" | "NO_EFFECT"
    p_value: float
    implied_iroas: float | None            # if spend_change is provided
    spend_change: float | None
    treated_geos: list[str]
    control_geos: list[str]
    pre_period_treated_baseline: float
    post_period_treated_actual: float
    counterfactual_estimate: float            # what treated would have been without treatment
    rationale: str

    def to_dict(self) -> dict:
        return {
            "measured_lift_pct": round(self.measured_lift_pct, 4),
            "confidence_interval": [round(x, 4) for x in self.confidence_interval],
            "is_significant": self.is_significant,
            "verdict": self.verdict,
            "p_value": round(self.p_value, 4),
            "implied_iroas": (
                round(self.implied_iroas, 2) if self.implied_iroas is not None else None
            ),
            "spend_change": (
                round(self.spend_change, 2) if self.spend_change is not None else None
            ),
            "treated_geos": self.treated_geos,
            "control_geos": self.control_geos,
            "pre_period_treated_baseline": round(self.pre_period_treated_baseline, 2),
            "post_period_treated_actual": round(self.post_period_treated_actual, 2),
            "counterfactual_estimate": round(self.counterfactual_estimate, 2),
            "rationale": self.rationale,
        }

    def summary(self) -> str:
        ci_lo, ci_hi = self.confidence_interval
        lines = [
            "Geo Holdout Experiment — Results",
            "=" * 50,
            f"Verdict:             {self.verdict}",
            f"Measured lift:       {self.measured_lift_pct:+.1%}",
            f"95% CI:              [{ci_lo:+.1%}, {ci_hi:+.1%}]",
            f"p-value:             {self.p_value:.3f}",
            "",
            f"Treated DMAs:        {', '.join(self.treated_geos)}",
            f"Control DMAs:        {len(self.control_geos)} geos",
            "",
            f"Pre-period baseline: ${self.pre_period_treated_baseline:,.0f}/wk (treated)",
            f"Post-period actual:  ${self.post_period_treated_actual:,.0f}/wk (treated)",
            f"Counterfactual est:  ${self.counterfactual_estimate:,.0f}/wk "
            f"(what treated would have been)",
        ]
        if self.implied_iroas is not None:
            lines.append("")
            lines.append(f"Spend change:        ${self.spend_change:,.0f}")
            lines.append(f"Implied iROAS:       {self.implied_iroas:.2f}x")
        lines.append("")
        lines.append(f"Rationale: {self.rationale}")
        return "\n".join(lines)


def analyze_geo_holdout(
    revenue: pd.DataFrame,
    *,
    treated_geos: list[str],
    pre_period_end: str,
    post_period_start: str,
    spend_change: float | None = None,
    geo_column: str = "geo",
    date_column: str = "date",
    revenue_column: str = "revenue",
    significance_level: float = 0.05,
    n_bootstrap: int = 1000,
) -> GeoHoldoutResult:
    """
    Analyse a completed geo holdout experiment.

    Computes the diff-in-diff between treated and control DMAs across the
    pre and post periods.  Uses a stationary block bootstrap for the CI to
    handle autocorrelation in the weekly revenue series.

    Parameters
    ----------
    revenue : DataFrame
        Long-format ``geo, date, revenue`` covering BOTH pre-period and
        post-period.
    treated_geos : list[str]
        DMAs that received the treatment (paused spend, scaled spend, etc.).
    pre_period_end : str
        Date string (inclusive) marking the end of the baseline period.
    post_period_start : str
        Date string (inclusive) marking the start of the experiment period.
    spend_change : float, optional
        Net spend change in treated geos during the post period.  When
        provided, the result includes ``implied_iroas``.
    significance_level : float, default 0.05
        alpha for the verdict.
    n_bootstrap : int, default 1000
        Bootstrap replicates for the CI.

    Returns
    -------
    GeoHoldoutResult
    """
    df = revenue.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    pre_end = pd.to_datetime(pre_period_end)
    post_start = pd.to_datetime(post_period_start)

    if post_start <= pre_end:
        raise ValueError("post_period_start must be after pre_period_end.")

    all_geos = sorted(df[geo_column].unique())
    missing = set(treated_geos) - set(all_geos)
    if missing:
        raise ValueError(f"Treated geos not present in revenue data: {missing}")
    control_geos = [g for g in all_geos if g not in treated_geos]
    if not control_geos:
        raise ValueError("No control geos. At least one non-treated geo is required.")

    pre_mask = df[date_column] <= pre_end
    post_mask = df[date_column] >= post_start
    treated_mask = df[geo_column].isin(treated_geos)

    pre_treated = df.loc[pre_mask & treated_mask, revenue_column].sum()
    pre_control = df.loc[pre_mask & ~treated_mask, revenue_column].sum()
    post_treated = df.loc[post_mask & treated_mask, revenue_column].sum()
    post_control = df.loc[post_mask & ~treated_mask, revenue_column].sum()

    # Diff-in-diff ratio: how much did treated change vs control?
    treated_ratio = post_treated / pre_treated if pre_treated > 0 else np.nan
    control_ratio = post_control / pre_control if pre_control > 0 else np.nan
    lift = treated_ratio / control_ratio - 1 if control_ratio else np.nan

    # Counterfactual: what would treated have looked like if it grew like control?
    counterfactual = pre_treated * control_ratio if not np.isnan(control_ratio) else np.nan

    # Per-period series for bootstrap CI
    pivot = df.pivot_table(
        index=date_column,
        columns=geo_column,
        values=revenue_column,
        aggfunc="sum",
    ).fillna(0)
    pre_pivot = pivot.loc[pivot.index <= pre_end]
    post_pivot = pivot.loc[pivot.index >= post_start]

    rng = np.random.default_rng(0)
    boot_lifts: list[float] = []
    n_post = len(post_pivot)
    if n_post >= 2:
        avg_block = max(2, int(n_post ** (1 / 3)))
        p_restart = 1.0 / avg_block
        for _ in range(n_bootstrap):
            # Resample post-period rows with stationary block bootstrap
            idx = np.empty(n_post, dtype=np.int64)
            i = int(rng.integers(0, n_post))
            for t in range(n_post):
                idx[t] = i
                if rng.random() < p_restart:
                    i = int(rng.integers(0, n_post))
                else:
                    i = (i + 1) % n_post
            post_b = post_pivot.iloc[idx]
            pt_b = post_b[treated_geos].sum().sum()
            pc_b = post_b[control_geos].sum().sum()
            tr_b = pt_b / pre_treated if pre_treated > 0 else np.nan
            cr_b = pc_b / pre_control if pre_control > 0 else np.nan
            if not (np.isnan(tr_b) or np.isnan(cr_b)) and cr_b > 0:
                boot_lifts.append(tr_b / cr_b - 1)

    if len(boot_lifts) >= 20:
        lo, hi = float(np.percentile(boot_lifts, 2.5)), float(np.percentile(boot_lifts, 97.5))
        # Two-sided p-value: fraction of bootstrap distribution on the wrong side of 0
        boot_arr = np.array(boot_lifts)
        p_value = 2.0 * min(float(np.mean(boot_arr <= 0)), float(np.mean(boot_arr >= 0)))
    else:
        lo, hi = float("nan"), float("nan")
        p_value = float("nan")

    is_significant = (not np.isnan(p_value)) and p_value < significance_level
    if is_significant and lift > 0:
        verdict = "LIFT_DETECTED"
    elif is_significant and lift < 0:
        verdict = "NEGATIVE_LIFT"
    elif not np.isnan(lift) and abs(lift) < 0.005:
        verdict = "NO_EFFECT"
    else:
        verdict = "INCONCLUSIVE"

    implied_iroas = None
    if spend_change is not None and abs(spend_change) > 0:
        # Use the counterfactual gap as the incremental revenue
        incremental_rev = post_treated - counterfactual if not np.isnan(counterfactual) else np.nan
        if not np.isnan(incremental_rev):
            implied_iroas = float(incremental_rev / spend_change)

    rationale = (
        f"Compared post-period revenue ratio in {len(treated_geos)} treated DMAs "
        f"({treated_ratio:.2f}) to {len(control_geos)} control DMAs ({control_ratio:.2f}). "
        f"The {lift:+.1%} relative lift is "
        f"{'statistically significant' if is_significant else 'not statistically significant'} "
        f"at alpha={significance_level:.0%} "
        f"(p={p_value:.3f}, bootstrap 95% CI [{lo:+.1%}, {hi:+.1%}])."
    )

    return GeoHoldoutResult(
        measured_lift_pct=float(lift) if not np.isnan(lift) else 0.0,
        confidence_interval=[lo, hi],
        is_significant=bool(is_significant),
        verdict=verdict,
        p_value=float(p_value) if not np.isnan(p_value) else 1.0,
        implied_iroas=implied_iroas,
        spend_change=spend_change,
        treated_geos=list(treated_geos),
        control_geos=control_geos,
        pre_period_treated_baseline=float(pre_treated / max(len(pre_pivot), 1)),
        post_period_treated_actual=float(post_treated / max(n_post, 1)),
        counterfactual_estimate=(
            float(counterfactual / max(n_post, 1)) if not np.isnan(counterfactual) else 0.0
        ),
        rationale=rationale,
    )
