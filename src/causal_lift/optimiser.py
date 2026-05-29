"""
Budget reallocation recommender.

Takes an :class:`AnalysisResult` and a current weekly (or daily) budget
allocation and suggests dollar reallocations that move spend from underperforming
channels to overperforming ones — subject to plausibility caps.

**Important: this is not an optimiser in the mathematical sense.**  Without
saturation curves we have no marginal-return information, so the "optimal"
allocation under a pure linear model would be 100 % of budget to the highest-
iROAS channel.  That is not a useful recommendation.  Instead this module
suggests bounded incremental shifts that respect the safety gates the model
already surfaces.

Recommendation logic
--------------------
- **SCALE** channels: suggest +``scale_pct`` of current spend (default +20%),
  capped per channel.
- **CUT** channels: suggest −``cut_pct`` of current spend (default −50%).
- **HOLD / INCONCLUSIVE** channels: no change.
- The total suggested change conserves budget: cuts fund scales.  If cuts
  exceed scales the residual is held back as "uncommitted" budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from causal_lift.analyzer import AnalysisResult


@dataclass
class ChannelRecommendation:
    channel: str
    current_spend: float
    recommended_spend: float
    delta: float                # recommended_spend - current_spend
    delta_pct: float            # delta / current_spend
    rationale: str
    label: str                  # the underlying SCALE/HOLD/CUT/INCONCLUSIVE label

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "current_spend": round(self.current_spend, 2),
            "recommended_spend": round(self.recommended_spend, 2),
            "delta": round(self.delta, 2),
            "delta_pct": round(self.delta_pct, 3),
            "rationale": self.rationale,
            "label": self.label,
        }


@dataclass
class ReallocationPlan:
    channels: list[ChannelRecommendation]
    total_current: float
    total_recommended: float
    total_delta: float                       # net change in budget
    uncommitted_budget: float                # cuts that didn't get reallocated
    expected_incremental_revenue_change: float  # rough estimate using current iROAS
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "channels": [c.to_dict() for c in self.channels],
            "total_current": round(self.total_current, 2),
            "total_recommended": round(self.total_recommended, 2),
            "total_delta": round(self.total_delta, 2),
            "uncommitted_budget": round(self.uncommitted_budget, 2),
            "expected_incremental_revenue_change": round(
                self.expected_incremental_revenue_change, 2
            ),
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            "Budget reallocation plan",
            f"  current total:    ${self.total_current:,.0f}",
            f"  recommended:      ${self.total_recommended:,.0f}",
            f"  uncommitted:      ${self.uncommitted_budget:,.0f}",
            f"  expected lift:    ${self.expected_incremental_revenue_change:,.0f}",
            "",
            f"  {'channel':<16}{'current':>12}{'recommended':>14}{'delta':>12}{'%':>8}",
            f"  {'-' * 62}",
        ]
        for c in self.channels:
            lines.append(
                f"  {c.channel:<16}${c.current_spend:>10,.0f}"
                f"${c.recommended_spend:>12,.0f}"
                f"${c.delta:>+10,.0f}"
                f"{c.delta_pct:>+7.0%}"
            )
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  * {n}")
        return "\n".join(lines)


def recommend_reallocation(
    result: AnalysisResult,
    current_spend: dict[str, float] | None = None,
    scale_pct: float = 0.20,
    cut_pct: float = 0.50,
    max_channel_delta_pct: float = 0.30,
) -> ReallocationPlan:
    """
    Generate a budget reallocation plan from an :class:`AnalysisResult`.

    Parameters
    ----------
    result : AnalysisResult
        Output from :func:`causal_lift.analyze`.
    current_spend : dict[str, float], optional
        Mapping of channel → current weekly (or daily) spend.  If omitted, the
        per-channel total spend from the analysis window is divided by the
        number of observations.
    scale_pct : float, default 0.20
        Proposed increase for channels labelled ``SCALE``.
    cut_pct : float, default 0.50
        Proposed decrease for channels labelled ``CUT``.
    max_channel_delta_pct : float, default 0.30
        Hard cap on any single channel's recommended Δ as a fraction of its
        current spend.  Prevents the model from suggesting moves the operating
        team can't reasonably execute.

    Returns
    -------
    ReallocationPlan
    """
    if current_spend is None:
        n_periods = max(1, result.observations)
        current_spend = {c.channel: c.total_spend / n_periods for c in result.channels}

    notes: list[str] = []
    recs: list[ChannelRecommendation] = []

    for ch in result.channels:
        cur = float(current_spend.get(ch.channel, 0.0))
        label = ch.recommendation

        if label == "SCALE":
            delta_pct = min(scale_pct, max_channel_delta_pct)
            rationale = (
                f"SCALE: iROAS {ch.incremental_roas:.1f}x with tight CI suggests "
                f"this channel has room to grow. Increase by {delta_pct:.0%} and re-measure."
            )
        elif label == "CUT":
            delta_pct = -min(cut_pct, max_channel_delta_pct)
            rationale = (
                f"CUT: iROAS {ch.incremental_roas:.1f}x is below breakeven and the CI "
                f"gives little chance of profitability. Reduce by {-delta_pct:.0%}."
            )
        elif label == "INCONCLUSIVE":
            delta_pct = 0.0
            rationale = (
                "INCONCLUSIVE: insufficient signal in observational data. Run a budget "
                "holdout (pause for 2–4 weeks) to produce a credible estimate before changing."
            )
        else:  # HOLD
            delta_pct = 0.0
            rationale = "HOLD: estimate exists but uncertainty is too wide to act."

        delta = cur * delta_pct
        recs.append(
            ChannelRecommendation(
                channel=ch.channel,
                current_spend=cur,
                recommended_spend=cur + delta,
                delta=delta,
                delta_pct=delta_pct,
                rationale=rationale,
                label=label,
            )
        )

    # Budget conservation: cuts fund scales.  If cuts exceed scales, hold the surplus back.
    total_current = sum(r.current_spend for r in recs)
    total_recommended = sum(r.recommended_spend for r in recs)
    total_delta = total_recommended - total_current

    uncommitted = max(0.0, -total_delta)
    if uncommitted > 0:
        notes.append(
            f"Net cuts exceed scales by ${uncommitted:,.0f}. Holding this back as "
            "uncommitted budget — consider funding a budget holdout experiment with it."
        )

    # Rough expected lift: Σ iROAS × Δspend  (ignores adstock smoothing)
    by_channel = {c.channel: c for c in result.channels}
    expected_lift = sum(
        by_channel[r.channel].incremental_roas * r.delta
        for r in recs
        if r.channel in by_channel
    )

    if not any(r.label == "SCALE" for r in recs):
        notes.append(
            "No channels qualified for SCALE — the model couldn't confidently identify "
            "any channel as scaleable. Most reallocations are CUT-only. The healthiest "
            "next step is a sequential budget holdout to disambiguate the HOLD/INCONCLUSIVE "
            "channels."
        )

    return ReallocationPlan(
        channels=recs,
        total_current=total_current,
        total_recommended=total_recommended,
        total_delta=total_delta,
        uncommitted_budget=uncommitted,
        expected_incremental_revenue_change=expected_lift,
        notes=notes,
    )
