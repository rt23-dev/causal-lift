"""
Multi-geo MMM analyzer.

When you have **regional disaggregation** in your data (e.g. spend and revenue
broken out by DMA, state, country, or arbitrary marketing region), this
analyzer runs :class:`RegressionMMM` independently per geo and aggregates the
per-channel coefficients across geos via median + cross-geo percentile bands.

This is structurally similar to a panel fixed-effects estimator, with two
practical advantages over a pooled-data fit:

1.  **Cross-geo variance becomes a source of inference.**  If a channel's
    estimate varies wildly across geos, the cross-geo CI is wide — even
    when each individual geo's HAC SE was small.  This catches geo-by-channel
    interactions that pooled regression misses.

2.  **You can spot geo-specific anomalies.**  ``GeoAnalysisResult.per_geo``
    exposes the underlying single-geo results so an operator can drill into
    one DMA that's behaving very differently from the rest.

This is **not** an Abadie-Diamond-Hainmueller synthetic control method.
A proper SCM constructs a weighted combination of untreated geos to serve as
a counterfactual for a specifically-treated unit.  That requires a clear
treatment definition and a substantial implementation (donor pool
selection, optimisation under weight constraints, placebo inference).  It
will land as a separate analyzer in a future release.  For now, the
multi-geo aggregation here covers the common case of "I have regional data
and want better confidence bands than pooled regression."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from causal_lift.analyzer import (
    AGGREGATE_SHARE_CEILING,
    ALWAYS_ON_THRESHOLD,
    AnalysisResult,
    BaseModel,
    ChannelResult,
)
from causal_lift.mmm import RegressionMMM

GEO_COLUMNS = ("geo", "region", "dma", "state", "country")


def detect_geo_column(df: pd.DataFrame) -> str | None:
    """Return the first recognised geo column in ``df``, or None."""
    for col in GEO_COLUMNS:
        if col in df.columns:
            return col
    return None


@dataclass
class GeoAnalysisResult(AnalysisResult):
    """Extends :class:`AnalysisResult` with per-geo breakdown."""

    geos: list[str] = field(default_factory=list)
    per_geo: dict[str, AnalysisResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["geos"] = list(self.geos)
        base["per_geo"] = {g: r.to_dict() for g, r in self.per_geo.items()}
        return base


class GeoMMM(BaseModel):
    """
    Multi-geo regression-based MMM.

    For each geo, fits an independent :class:`RegressionMMM`.  Per-channel
    iROAS estimates are then aggregated across geos via:

    -  point estimate = cross-geo median
    -  95% CI         = cross-geo 2.5th and 97.5th percentiles
    -  VIF            = cross-geo median of per-geo VIFs
    -  nonzero_share  = cross-geo spend-weighted mean

    Channels that appear in fewer than ``min_geos_per_channel`` geos are
    surfaced in warnings but kept in the output.
    """

    def __init__(
        self,
        min_geos_per_channel: int = 2,
        weighted_by_spend: bool = True,
        per_geo_kwargs: dict | None = None,
    ):
        """
        Parameters
        ----------
        min_geos_per_channel : int, default 2
            Channels appearing in fewer geos than this get a warning.
        weighted_by_spend : bool, default True
            If True, geo aggregation weights each geo's estimate by its spend
            share for that channel.  Recommended for unequal-size markets.
            If False, all geos are weighted equally.
        per_geo_kwargs : dict, optional
            Keyword arguments forwarded to the per-geo :class:`RegressionMMM`.
        """
        self.min_geos_per_channel = int(min_geos_per_channel)
        self.weighted_by_spend = bool(weighted_by_spend)
        self.per_geo_kwargs = dict(per_geo_kwargs or {})

    def fit(
        self,
        spend_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        contribution_margin: float = 0.30,
    ) -> GeoAnalysisResult:
        spend_geo_col = detect_geo_column(spend_df)
        sales_geo_col = detect_geo_column(sales_df)
        if spend_geo_col is None:
            raise ValueError(
                "GeoMMM requires a geo column in spend_df. "
                f"Looked for any of: {GEO_COLUMNS}"
            )
        if sales_geo_col is None:
            raise ValueError(
                "GeoMMM requires a geo column in sales_df. "
                f"Looked for any of: {GEO_COLUMNS}"
            )

        warnings_acc: list[str] = []
        per_geo_results: dict[str, AnalysisResult] = {}

        geos = sorted(set(spend_df[spend_geo_col].unique()) & set(sales_df[sales_geo_col].unique()))
        if not geos:
            raise ValueError("No overlapping geos between spend_df and sales_df.")

        for geo in geos:
            sp = spend_df.loc[spend_df[spend_geo_col] == geo].drop(columns=[spend_geo_col])
            sa = sales_df.loc[sales_df[sales_geo_col] == geo].drop(columns=[sales_geo_col])
            try:
                per_geo_results[geo] = RegressionMMM(**self.per_geo_kwargs).fit(
                    sp, sa, contribution_margin=contribution_margin
                )
            except Exception as exc:  # noqa: BLE001
                warnings_acc.append(f"Geo '{geo}' analysis failed: {exc}. Skipped.")
                continue

        if not per_geo_results:
            raise ValueError("No geo produced a successful analysis.")

        warnings_acc.append(
            f"GeoMMM aggregated estimates across {len(per_geo_results)} geos "
            f"({', '.join(list(per_geo_results)[:5])}"
            f"{'…' if len(per_geo_results) > 5 else ''}). "
            f"Per-geo results retained in result.per_geo."
        )

        # Collate per-channel estimates across geos
        all_channels: set[str] = set()
        for r in per_geo_results.values():
            all_channels |= {c.channel for c in r.channels}

        aggregated_channels: list[ChannelResult] = []
        for ch in sorted(all_channels):
            per_geo_for_ch: list[tuple[str, ChannelResult]] = []
            for geo, r in per_geo_results.items():
                for c in r.channels:
                    if c.channel == ch:
                        per_geo_for_ch.append((geo, c))
                        break
            if len(per_geo_for_ch) < self.min_geos_per_channel:
                warnings_acc.append(
                    f"Channel '{ch}' appears in only {len(per_geo_for_ch)} geo(s) "
                    f"(< {self.min_geos_per_channel}); aggregate estimate is unreliable."
                )

            iroas_arr = np.array([c.incremental_roas for _, c in per_geo_for_ch])
            spend_arr = np.array([c.total_spend for _, c in per_geo_for_ch])
            vif_vals = [c.vif_score for _, c in per_geo_for_ch if c.vif_score is not None]
            on_arr = np.array([c.nonzero_share for _, c in per_geo_for_ch])

            # Spend-weighted or unweighted aggregation
            if self.weighted_by_spend and spend_arr.sum() > 0:
                weights = spend_arr / spend_arr.sum()
                point = float(np.sum(iroas_arr * weights))
                share = float(np.sum(on_arr * weights))
            else:
                point = float(np.median(iroas_arr))
                share = float(np.mean(on_arr))

            ci_low, ci_high = (
                (float(np.percentile(iroas_arr, 2.5)), float(np.percentile(iroas_arr, 97.5)))
                if iroas_arr.size >= 3
                else (float(iroas_arr.min()), float(iroas_arr.max()))
            )
            total_spend = float(spend_arr.sum())
            vif = float(np.median(vif_vals)) if vif_vals else None

            display_roas = max(0.0, point)
            inc_rev = display_roas * total_spend
            rec, reason = self.recommend(
                display_roas, [ci_low, ci_high],
                breakeven_roas=1.0 / max(0.05, min(0.95, contribution_margin)),
                vif=vif, nonzero_share=share,
            )

            aggregated_channels.append(
                ChannelResult(
                    channel=ch,
                    total_spend=total_spend,
                    attribution_proxy_roas=float(np.mean([
                        c.attribution_proxy_roas for _, c in per_geo_for_ch
                    ])),
                    incremental_roas=display_roas,
                    incremental_revenue=inc_rev,
                    confidence_interval=[ci_low, ci_high],
                    recommendation=rec,
                    recommendation_reason=reason,
                    model_fit=float(np.mean([c.model_fit for _, c in per_geo_for_ch])),
                    vif_score=vif,
                    raw_coef=point,
                    nonzero_share=share,
                )
            )

        # Aggregate-level totals
        total_revenue = float(sum(r.total_revenue for r in per_geo_results.values()))
        total_spend_all = float(sum(c.total_spend for c in aggregated_channels))
        total_incremental = sum(c.incremental_revenue for c in aggregated_channels)
        implied_share = total_incremental / total_revenue if total_revenue > 0 else 0.0
        if implied_share > AGGREGATE_SHARE_CEILING:
            warnings_acc.append(
                f"Aggregate implied incremental share is {implied_share:.0%} of revenue "
                f"(threshold {AGGREGATE_SHARE_CEILING:.0%}). All SCALE recommendations "
                f"downgraded to INCONCLUSIVE."
            )
            for c in aggregated_channels:
                if c.recommendation == "SCALE":
                    c.recommendation = "INCONCLUSIVE"
                    c.recommendation_reason = (
                        f"Demoted from SCALE: aggregate implied incremental share is "
                        f"{implied_share:.0%}, which is implausibly high."
                    )

        # Aggregate diagnostics
        n_obs = max(r.observations for r in per_geo_results.values())
        avg_r2 = float(np.mean([r.r_squared for r in per_geo_results.values()]))
        avg_dw = float(np.mean([r.durbin_watson for r in per_geo_results.values()]))
        # Always-on aggregation
        always_on = [c.channel for c in aggregated_channels if c.nonzero_share > ALWAYS_ON_THRESHOLD]
        if always_on:
            warnings_acc.append(
                f"Always-on channels across geos: {', '.join(always_on)}. "
                f"SCALE labels were already demoted in per-geo and aggregate passes."
            )

        cadence_modes = [r.cadence for r in per_geo_results.values()]
        cadence = max(set(cadence_modes), key=cadence_modes.count)

        return GeoAnalysisResult(
            channels=aggregated_channels,
            method_used=(
                f"GeoMMM ({len(per_geo_results)} geos; per-geo RegressionMMM, "
                f"{'spend-weighted' if self.weighted_by_spend else 'unweighted'} aggregation)"
            ),
            total_revenue=total_revenue,
            total_spend=total_spend_all,
            r_squared=avg_r2,
            observations=n_obs,
            contribution_margin=max(0.05, min(0.95, contribution_margin)),
            breakeven_roas=round(1.0 / max(0.05, min(0.95, contribution_margin)), 2),
            durbin_watson=avg_dw,
            cadence=cadence,
            implied_incremental_share=implied_share,
            adstock_thetas={},  # per-geo thetas vary; expose via per_geo
            warnings=warnings_acc,
            geos=list(per_geo_results),
            per_geo=per_geo_results,
        )
