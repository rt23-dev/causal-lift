"""
End-to-end SKU-level holdout for a CPG brand on Amazon Ads.

Scenario
--------
RidgeBev is a $120M-revenue functional-beverage brand spending $2.4M/year
on Amazon Sponsored Products across 12 SKUs. The Brand Manager suspects
the top 3 SKUs are over-credited by Amazon's 7-day attribution and the
mid-volume SKUs are doing real incremental work.

The product layer:

1.  **Pre-experiment design.**  Given baseline weekly organic-revenue
    history across 12 ASINs, pick the 3 ASINs to pause for 4 weeks
    (treated set) and the 9 ASINs to keep running (control set).  Report
    the lift this experiment can credibly detect.

2.  **Post-experiment analysis.**  Given actual revenue per ASIN over the
    treated window, compute the causal lift and the implied iROAS on the
    paused spend.  Output a CMO-ready memo.

Run::

    python examples/amazon_sku_holdout.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import causal_lift as cl

# ── Synthetic CPG SKU baseline ────────────────────────────────────────────────

ASIN_PORTFOLIO = [
    "B0A1ENERGY-LMN",   # flagship energy, lemon
    "B0A2ENERGY-CHRY",  # flagship energy, cherry
    "B0A3ENERGY-MNGO",  # flagship energy, mango
    "B0B1HYDRATE-ORG",  # hydration mix, original
    "B0B2HYDRATE-RPB",  # hydration mix, raspberry
    "B0B3HYDRATE-COCO", # hydration mix, coconut
    "B0C1PROTEIN-VAN",  # protein shake, vanilla
    "B0C2PROTEIN-CHC",  # protein shake, chocolate
    "B0D1KETO-MCT",     # MCT oil
    "B0E1BCAA-WTRM",    # BCAAs, watermelon
    "B0E2BCAA-BBRY",    # BCAAs, blueberry
    "B0F1GREENS-MINT",  # greens powder
]


def make_ridgebev_baseline(seed: int = 19) -> pd.DataFrame:
    """16 weeks of weekly revenue across 12 ASINs."""
    rng = np.random.default_rng(seed)
    n_weeks = 16
    dates = pd.date_range("2025-01-06", periods=n_weeks, freq="W-MON")
    # Category-wide demand wobble (Amazon's weekly seasonality + promo cycles)
    common_trend = 60_000 + np.arange(n_weeks) * 800 + rng.normal(0, 3_000, n_weeks)

    rows = []
    for asin in ASIN_PORTFOLIO:
        size = rng.uniform(0.4, 1.3)
        noise = rng.normal(0, 2_500, n_weeks)
        rev = (common_trend * size + noise).clip(8_000, None)
        for i, d in enumerate(dates):
            rows.append({"asin": asin, "date": d, "revenue": float(rev[i])})
    return pd.DataFrame(rows)


def add_post_period(
    baseline: pd.DataFrame,
    treated_skus: list[str],
    duration_weeks: int = 4,
    true_lift_pct: float = -0.08,    # NEGATIVE lift: pausing ads drops sales 8%
    seed: int = 23,
) -> pd.DataFrame:
    """
    Append a post-period where treated ASINs got their Amazon ads paused.

    Negative lift = revenue drops in treated SKUs by the true incremental
    contribution of the paid ads. This is the canonical retail-media
    holdout test: measure the drop, infer the causal lift.
    """
    rng = np.random.default_rng(seed)
    last_date = baseline["date"].max()
    post_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1), periods=duration_weeks, freq="W-MON"
    )
    last_per_sku = (
        baseline.sort_values("date").groupby("asin").tail(1).set_index("asin")["revenue"]
    )

    rows = []
    for d in post_dates:
        for asin, base_rev in last_per_sku.items():
            organic = base_rev * 1.008
            noise = rng.normal(0, base_rev * 0.04)
            rev = organic + noise
            if asin in treated_skus:
                # The paid-ad halt drops revenue by `true_lift_pct` (negative)
                rev *= 1 + true_lift_pct
            rows.append({"asin": asin, "date": d, "revenue": float(rev)})

    post = pd.DataFrame(rows)
    return pd.concat([baseline, post], ignore_index=True)


# ── The workflow ─────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 78)
    print(" RidgeBev — Amazon Sponsored Products SKU holdout")
    print("=" * 78)
    print()
    print("Question: are our top-3 ASINs really incremental, or is Amazon's 7-day")
    print("attribution crediting paid ads for sales we'd have made anyway?")
    print()

    baseline = make_ridgebev_baseline()
    print(f"Baseline:  {baseline['asin'].nunique()} ASINs x "
          f"{baseline['date'].nunique()} weeks of organic-revenue history")
    print()

    # ── STEP 1: pre-experiment design ─────────────────────────────────────────
    print("STEP 1 -- Pre-experiment design")
    print("-" * 78)
    print("Pause Amazon Sponsored Products for 3 ASINs for 4 weeks. Measure the")
    print("revenue drop relative to the 9 control ASINs that keep advertising.")
    print()

    # The geo-holdout machinery generalises to ASINs by passing the ASIN
    # column as the "geo" unit.
    design = cl.design_geo_holdout(
        baseline,
        n_treated=3,
        duration_weeks=4,
        expected_lift_pct=0.08,    # we expect ~8% revenue drop in paused SKUs
        significance_level=0.05,
        geo_column="asin",
    )
    print(design.summary())
    print()

    # ── STEP 2: run the experiment (simulated) ────────────────────────────────
    print()
    print("STEP 2 -- Experiment runs (simulated here)")
    print("-" * 78)
    print(f"Treated ASINs pause Amazon spend for {4} weeks.")
    print("True causal lift baked in: -8% revenue drop in treated SKUs.")
    print("Spend savings during pause: $120,000 ($30K/wk x 4 wks).")
    print()
    experiment = add_post_period(
        baseline,
        treated_skus=design.treated_geos,
        duration_weeks=4,
        true_lift_pct=-0.08,
    )

    # ── STEP 3: post-experiment analysis ──────────────────────────────────────
    print()
    print("STEP 3 -- Post-experiment analysis")
    print("-" * 78)
    pre_end = baseline["date"].max().strftime("%Y-%m-%d")
    post_start = (baseline["date"].max() + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")

    result = cl.analyze_geo_holdout(
        experiment,
        treated_geos=design.treated_geos,
        pre_period_end=pre_end,
        post_period_start=post_start,
        spend_change=-120_000,    # net change is NEGATIVE: we paused spend
        geo_column="asin",
    )
    print(result.summary())
    print()

    # ── Verdict for the Brand Manager ─────────────────────────────────────────
    print()
    print("=" * 78)
    print(" Recommendation to the Brand Manager")
    print("=" * 78)

    measured_drop = -result.measured_lift_pct
    if result.verdict in ("LIFT_DETECTED", "NEGATIVE_LIFT"):
        if measured_drop > 0:
            print(f"[YES] Pausing ads on the treated ASINs dropped revenue by {measured_drop:.1%}.")
            print("      Amazon Sponsored Products IS causally incremental on these SKUs.")
            if result.implied_iroas:
                print(f"      Implied iROAS on paused spend: {result.implied_iroas:.2f}x.")
        else:
            print(f"[?]   Treated ASINs actually grew {measured_drop * -1:+.1%}.")
            print("      Paid spend on these SKUs may have been over-credited by Amazon.")
            print("      Reallocate budget toward higher-performing ASINs.")
    elif result.verdict == "NO_EFFECT":
        print("[?]   No measurable effect from pausing ads.")
        print("      The treated ASINs were probably over-attributed by Amazon's 7-day window.")
        print("      Cut ad spend on these SKUs and watch revenue (it won't move).")
    else:
        print(f"[?]   Verdict: {result.verdict}.")
        print("      Estimate too imprecise. Run longer or with more SKUs in the test.")
    print()
    print("(Next test: re-run on the OTHER 9 ASINs to find ones that actually move.)")


if __name__ == "__main__":
    main()
