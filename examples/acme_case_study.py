"""
Acme Skincare — case study generator.

Fabricates 78 weeks of plausible data for a fictional $14M-ARR DTC skincare
brand making budget decisions across 6 channels.  Designed to surface the
full range of `causal-lift` behaviour: confident SCALE / CUT calls,
INCONCLUSIVE labels, adstock auto-tuning, and the safety gates firing.

Used by ``docs/case-studies.md``.  Run with::

    python examples/acme_case_study.py

Output is deterministic for a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import causal_lift as cl


def adstock(spend: np.ndarray, theta: float) -> np.ndarray:
    """Geometric adstock with normalised impulse response."""
    if theta == 0:
        return spend
    out = np.zeros_like(spend, dtype=float)
    out[0] = spend[0]
    for t in range(1, len(spend)):
        out[t] = spend[t] + theta * out[t - 1]
    return out * (1.0 - theta)


def burst_spend(rng: np.random.Generator, n: int, base: float, n_bursts: int) -> np.ndarray:
    """Bursty spend pattern (mostly zero, occasional flights)."""
    out = np.zeros(n)
    burst_weeks = rng.choice(n, size=n_bursts, replace=False)
    out[burst_weeks] = base + rng.normal(0, base * 0.15, n_bursts).clip(0, None)
    return out


def generate_acme_data(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate 78 weeks of weekly data for the case study."""
    rng = np.random.default_rng(seed)
    n = 78
    dates = pd.date_range("2024-06-03", periods=n, freq="W-MON")

    # Ground-truth iROAS baked into the data-generating process
    true_iroas = {
        "meta": 3.5,             # decent, mid-funnel performance
        "google_search": 1.5,    # mostly intercepting existing demand
        "tiktok": 4.2,           # winning on creative right now
        "pinterest": 1.8,        # aging out, declining productivity
        "klaviyo": 8.0,          # email is just efficient
        "podcast": 2.8,          # carryover-heavy
    }

    # Channel spend patterns (all in dollars per week)
    meta = (20_000 + rng.normal(0, 4_000, n)).clip(8_000, 35_000)
    google = (15_000 + rng.normal(0, 1_200, n)).clip(12_000, 18_000)        # always-on, low variance
    tiktok = (8_000 + np.arange(n) * 120 + rng.normal(0, 2_000, n)).clip(0, None)  # growing
    pinterest = (12_000 - np.arange(n) * 80 + rng.normal(0, 1_500, n)).clip(2_000, None)  # declining
    klaviyo = (2_000 + rng.normal(0, 200, n)).clip(1_500, 2_500)
    podcast = burst_spend(rng, n, base=16_000, n_bursts=20)                  # bi-weekly bursts

    # Q4 demand spike — Nov + Dec
    month_of = dates.month
    q4_mult = np.where(np.isin(month_of, [11, 12]), 1.35, 1.0)

    # Revenue DGP: growing organic baseline + channel contributions (with adstock for podcast) + noise
    organic = 80_000 + np.arange(n) * 400

    contributions = (
        true_iroas["meta"] * meta
        + true_iroas["google_search"] * google
        + true_iroas["tiktok"] * adstock(tiktok, theta=0.3)
        + true_iroas["pinterest"] * pinterest
        + true_iroas["klaviyo"] * klaviyo
        + true_iroas["podcast"] * adstock(podcast, theta=0.6)
    )
    noise = rng.normal(0, 12_000, n)
    revenue = (organic * q4_mult + contributions * q4_mult + noise).clip(0)

    spend_rows = []
    for i, d in enumerate(dates):
        spend_rows += [
            {"date": d, "channel": "meta",          "spend": float(meta[i])},
            {"date": d, "channel": "google_search", "spend": float(google[i])},
            {"date": d, "channel": "tiktok",        "spend": float(tiktok[i])},
            {"date": d, "channel": "pinterest",     "spend": float(pinterest[i])},
            {"date": d, "channel": "klaviyo",       "spend": float(klaviyo[i])},
            {"date": d, "channel": "podcast",       "spend": float(podcast[i])},
        ]
    spend_df = pd.DataFrame(spend_rows)
    spend_df["date"] = pd.to_datetime(spend_df["date"])

    sales_df = pd.DataFrame({"date": dates, "revenue": revenue})
    return spend_df, sales_df, true_iroas


def main() -> None:
    spend_df, sales_df, truth = generate_acme_data(seed=7)

    print("=" * 72)
    print(" Acme Skincare — 78-week causal-lift analysis")
    print("=" * 72)
    n_weeks = sales_df["date"].nunique()
    avg_weekly_revenue = sales_df["revenue"].mean()
    total_revenue = sales_df["revenue"].sum()
    total_spend = spend_df["spend"].sum()
    annualised = avg_weekly_revenue * 52
    print(f"Date range:        {sales_df['date'].min():%Y-%m-%d} -> {sales_df['date'].max():%Y-%m-%d}")
    print(f"Weeks observed:    {n_weeks}")
    print(f"Avg weekly rev:    ${avg_weekly_revenue:,.0f}  (~${annualised/1e6:.1f}M ARR)")
    print(f"Total revenue:     ${total_revenue/1e6:.2f}M")
    print(f"Total ad spend:    ${total_spend/1e6:.2f}M  ({total_spend/total_revenue:.0%} of revenue)")
    print()

    # Use 35% contribution margin (typical skincare CPG)
    result = cl.analyze(spend_df, sales_df, contribution_margin=0.35)
    print(result.summary())
    print()

    print("Ground truth (true iROAS baked into the DGP):")
    for ch in sorted(truth):
        print(f"  {ch:<16} {truth[ch]:.1f}x")
    print()

    print("Adstock decay auto-selected per channel:")
    for ch in sorted(result.adstock_thetas):
        print(f"  {ch:<16} theta={result.adstock_thetas[ch]:.2f}")


if __name__ == "__main__":
    main()
