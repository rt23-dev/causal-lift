"""Base analyzer interface and result dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

# Implausibility thresholds used for safety gates.
ALWAYS_ON_THRESHOLD = 0.85       # channels active in >85% of periods are "always-on"
AGGREGATE_SHARE_CEILING = 0.50   # if Σ(iROAS·spend)/revenue exceeds this, the model is hallucinating


@dataclass
class ChannelResult:
    """Per-channel incrementality result."""

    channel: str
    total_spend: float
    attribution_proxy_roas: float   # proportional daily attribution — NOT platform-reported
    incremental_roas: float         # causal OLS coefficient ($ revenue per $ spend)
    incremental_revenue: float
    confidence_interval: list[float]  # [lower_95, upper_95]
    recommendation: str             # SCALE / HOLD / CUT / INCONCLUSIVE (margin-aware)
    recommendation_reason: str
    model_fit: float = 0.0          # global model R²
    vif_score: float | None = None
    raw_coef: float = 0.0           # unclipped OLS coefficient
    nonzero_share: float = 1.0      # fraction of periods this channel had spend > 0

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "total_spend": round(self.total_spend, 2),
            "attribution_proxy_roas": round(self.attribution_proxy_roas, 2),
            "incremental_roas": round(self.incremental_roas, 2),
            "incremental_revenue": round(self.incremental_revenue, 2),
            "confidence_interval": [round(x, 2) for x in self.confidence_interval],
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
            "model_fit": round(self.model_fit, 3),
            "vif_score": round(self.vif_score, 1) if self.vif_score is not None else None,
            "raw_coef": round(self.raw_coef, 3),
            "nonzero_share": round(self.nonzero_share, 2),
        }


@dataclass
class AnalysisResult:
    """Full analysis result with per-channel breakdown and diagnostics."""

    channels: list[ChannelResult]
    method_used: str
    total_revenue: float
    total_spend: float
    r_squared: float
    observations: int
    contribution_margin: float
    breakeven_roas: float
    durbin_watson: float = 2.0
    cadence: str = "daily"                       # "daily", "weekly", or "irregular"
    implied_incremental_share: float = 0.0       # Σ(iROAS·spend) / total_revenue
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "channels": [ch.to_dict() for ch in self.channels],
            "method_used": self.method_used,
            "total_revenue": round(self.total_revenue, 2),
            "total_spend": round(self.total_spend, 2),
            "r_squared": round(self.r_squared, 3),
            "observations": self.observations,
            "contribution_margin": self.contribution_margin,
            "breakeven_roas": round(self.breakeven_roas, 2),
            "durbin_watson": round(self.durbin_watson, 2),
            "cadence": self.cadence,
            "implied_incremental_share": round(self.implied_incremental_share, 3),
            "warnings": self.warnings,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-channel results as a tidy DataFrame."""
        return pd.DataFrame([ch.to_dict() for ch in self.channels])

    def summary(self) -> str:
        """Plain-text summary suitable for printing (ASCII-only)."""
        lines = [
            f"causal-lift analysis  |  method: {self.method_used}",
            f"  observations: {self.observations} ({self.cadence})  |  R-squared: {self.r_squared:.3f}  |  DW: {self.durbin_watson:.2f}",
            f"  contribution margin: {self.contribution_margin:.0%}  ->  breakeven iROAS = {self.breakeven_roas:.2f}x",
            f"  aggregate implied incremental share: {self.implied_incremental_share:.0%} of revenue",
            "",
            f"  {'channel':<14}{'iROAS':>8}{'CI95':>20}{'VIF':>8}{'on%':>6}{'rec':>14}",
            f"  {'-' * 70}",
        ]
        for ch in self.channels:
            ci = f"[{ch.confidence_interval[0]:.2f}, {ch.confidence_interval[1]:.2f}]"
            vif = f"{ch.vif_score:.1f}" if ch.vif_score is not None else "-"
            on_pct = f"{ch.nonzero_share:.0%}"
            lines.append(
                f"  {ch.channel:<14}{ch.incremental_roas:>7.2f}x{ci:>20}{vif:>8}{on_pct:>6}{ch.recommendation:>14}"
            )
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                clean = w.replace("⚠️", "!").strip()
                clean = clean.encode("ascii", errors="replace").decode("ascii")
                lines.append(f"  * {clean}")
        return "\n".join(lines)


class BaseModel(ABC):
    """Abstract base class for incrementality models."""

    @abstractmethod
    def fit(
        self,
        spend_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        contribution_margin: float = 0.30,
    ) -> AnalysisResult:
        """Fit the model and return analysis results."""

    @staticmethod
    def recommend(
        incremental_roas: float,
        ci: list[float],
        breakeven_roas: float,
        vif: float | None = None,
        nonzero_share: float = 1.0,
    ) -> tuple[str, str]:
        """
        Margin-aware recommendation with safety gates.

        Gates (any one fires → INCONCLUSIVE / HOLD):
        - VIF > 10                  → HOLD (estimate not identified)
        - nonzero_share > 0.85 AND
          would-be SCALE            → INCONCLUSIVE (always-on, likely baseline confound)

        Labels:
        - SCALE          iROAS >= breakeven AND CI lower >= 75% of breakeven
        - CUT            iROAS < 85% of breakeven AND CI upper < breakeven
        - HOLD           otherwise
        - INCONCLUSIVE   estimate exists but cannot be acted on safely
        """
        lower, upper = ci
        be = breakeven_roas

        vif_caveat = ""
        if vif is not None and vif > 10:
            return (
                "HOLD",
                f"High multicollinearity (VIF={vif:.1f}) makes this estimate unreliable — "
                f"channels with correlated spend cannot be individually identified. "
                f"Run a budget experiment to get a clean estimate.",
            )
        if vif is not None and vif > 5:
            vif_caveat = f" (moderate collinearity VIF={vif:.1f} — treat with caution)"

        # Always-on confound check: SCALE on a >85%-on channel is the most dangerous
        # output the library can produce — same-day OLS can't separate its effect
        # from baseline / trend co-movement.
        would_be_scale = incremental_roas >= be and lower >= be * 0.75
        if nonzero_share > ALWAYS_ON_THRESHOLD and would_be_scale:
            return (
                "INCONCLUSIVE",
                f"Channel runs in {nonzero_share:.0%} of periods (>85% always-on). "
                f"Same-day regression cannot separate its effect from baseline/trend "
                f"co-movement, so the {incremental_roas:.1f}x estimate is likely inflated. "
                f"A budget holdout (pause spend in 2-3 random weeks) would give a credible estimate.",
            )

        if would_be_scale:
            return (
                "SCALE",
                f"iROAS {incremental_roas:.1f}x is above your {be:.1f}x breakeven and the CI "
                f"lower bound ({lower:.1f}x ≥ 75% of breakeven) suggests this is likely "
                f"profitable. Scale spend carefully and re-measure.{vif_caveat}",
            )
        if incremental_roas < be * 0.85 and upper < be:
            return (
                "CUT",
                f"iROAS {incremental_roas:.1f}x is below your {be:.1f}x breakeven and the CI "
                f"upper bound ({upper:.1f}x) gives little chance this channel is profitable. "
                f"Redirect spend.{vif_caveat}",
            )
        return (
            "HOLD",
            f"iROAS {incremental_roas:.1f}x vs {be:.1f}x breakeven (CI [{lower:.1f}, {upper:.1f}]). "
            f"Signal exists but uncertainty is too wide to act aggressively — more data or a "
            f"budget experiment will resolve this.{vif_caveat}",
        )
