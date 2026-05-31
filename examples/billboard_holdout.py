"""
End-to-end geo holdout for a fictional OOH (billboard) campaign.

Scenario
--------
Atlas Athletic Apparel is a $30M-ARR DTC brand considering doubling their
billboard budget in Q1. Before committing $1.2M to a 12-week national OOH
plan, the CMO wants empirical evidence the channel is incremental.

The product layer walks them through:

1.  **Pre-experiment design.**  Given 26 weeks of revenue by DMA, propose
    which 3 markets to test, what statistical power the test will have,
    and the minimum detectable lift.

2.  **Post-experiment analysis.**  Given the actual revenue across treated
    and control DMAs over the 6-week test, compute the causal lift, the
    confidence interval, and the implied iROAS on the spend change.

Run::

    python examples/billboard_holdout.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import causal_lift as cl

# ── Synthetic data setup ──────────────────────────────────────────────────────

def make_atlas_baseline(seed: int = 11) -> pd.DataFrame:
    """26 weeks of weekly revenue across 10 DMAs."""
    rng = np.random.default_rng(seed)
    dmas = [
        "NYC", "LAX", "CHI", "DFW", "PHX", "ATL", "SEA", "MIA", "BOS", "DEN",
    ]
    n_weeks = 26
    dates = pd.date_range("2025-01-06", periods=n_weeks, freq="W-MON")
    # Common national trend (DTC brand growing 3% per quarter)
    trend = 220_000 + np.arange(n_weeks) * 1_800 + rng.normal(0, 6_000, n_weeks)

    rows = []
    for dma in dmas:
        # Each DMA has its own size multiplier and noise level
        size = rng.uniform(0.5, 1.6)
        noise = rng.normal(0, 8_000, n_weeks)
        rev = (trend * size + noise).clip(30_000, None)
        for i, d in enumerate(dates):
            rows.append({"geo": dma, "date": d, "revenue": float(rev[i])})
    return pd.DataFrame(rows)


def add_post_period(
    baseline: pd.DataFrame,
    treated_geos: list[str],
    duration_weeks: int = 6,
    true_lift_pct: float = 0.07,
    seed: int = 22,
) -> pd.DataFrame:
    """Append a post-period where treated DMAs get a real OOH lift."""
    rng = np.random.default_rng(seed)
    last_date = baseline["date"].max()
    post_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1), periods=duration_weeks, freq="W-MON"
    )
    last_per_geo = (
        baseline.sort_values("date").groupby("geo").tail(1).set_index("geo")["revenue"]
    )

    rows = []
    for d in post_dates:
        for g, base_rev in last_per_geo.items():
            # Continue the natural trend
            organic = base_rev * 1.012  # 1.2%/wk growth
            noise = rng.normal(0, base_rev * 0.04)
            rev = organic + noise
            # Treated geos get a real causal lift from OOH spend
            if g in treated_geos:
                rev *= 1 + true_lift_pct
            rows.append({"geo": g, "date": d, "revenue": float(rev)})

    post = pd.DataFrame(rows)
    return pd.concat([baseline, post], ignore_index=True)


# ── The workflow ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print(" Atlas Athletic Apparel — OOH geo holdout")
    print("=" * 72)
    print()

    baseline = make_atlas_baseline()

    # ── STEP 1: pre-experiment design ─────────────────────────────────────────
    print("STEP 1 — Pre-experiment design")
    print("-" * 72)
    print("Question: which 3 DMAs should we treat for a 6-week OOH test,")
    print("and what lift can we credibly detect?")
    print()

    design = cl.design_geo_holdout(
        baseline,
        n_treated=3,
        duration_weeks=6,
        expected_lift_pct=0.07,    # we hope to see 7%+ lift
        significance_level=0.05,
    )
    print(design.summary())
    print()

    # ── STEP 2: simulate running the experiment ───────────────────────────────
    print()
    print("STEP 2 — Experiment runs (simulated here)")
    print("-" * 72)
    print(f"Treated DMAs get a real 7% lift baked in for {6} weeks.")
    print("Other DMAs continue on baseline trajectory.")
    print("Spend uplift in treated DMAs: $480,000 ($80K/wk × 6 wks).")
    print()
    experiment_data = add_post_period(
        baseline,
        treated_geos=design.treated_geos,
        duration_weeks=6,
        true_lift_pct=0.07,
    )

    # ── STEP 3: post-experiment analysis ──────────────────────────────────────
    print()
    print("STEP 3 — Post-experiment analysis")
    print("-" * 72)
    pre_end = baseline["date"].max().strftime("%Y-%m-%d")
    post_start = (baseline["date"].max() + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")

    result = cl.analyze_geo_holdout(
        experiment_data,
        treated_geos=design.treated_geos,
        pre_period_end=pre_end,
        post_period_start=post_start,
        spend_change=480_000,    # the actual $ uplift in treated DMAs
    )
    print(result.summary())
    print()

    # ── Verdict for the CMO ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(" Recommendation to the CMO")
    print("=" * 72)
    if result.verdict == "LIFT_DETECTED":
        print(f"[YES] OOH produced a statistically significant {result.measured_lift_pct:+.1%}")
        print(f"      lift in treated markets. Implied iROAS: {result.implied_iroas:.2f}x.")
        if result.implied_iroas and result.implied_iroas > 1.0:
            print("      Scale the campaign nationally.")
        else:
            print("      But iROAS is below break-even -- do not scale at current creative/spend mix.")
    elif result.verdict == "INCONCLUSIVE":
        print("[?]   Test was inconclusive -- the data couldn't rule out either effect or no effect.")
        print("      Run longer or with a larger spend uplift to narrow the CI.")
    else:
        print(f"[NO]  Verdict: {result.verdict}.")
        print("      Do not scale this channel at current spend levels.")


if __name__ == "__main__":
    main()
