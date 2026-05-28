"""
benchmark.py
============

Empirical benchmark of `causal-lift` against Meta's Robyn `dt_simulated_weekly`
dataset — the de-facto industry MMM benchmark.

Sections
--------
1. Headline run on Robyn data (5 paid channels, 208 weeks).
2. "Ground truth" comparison: peer estimators (last-touch share, simple
   bivariate OLS per channel) and plausibility bounds (iROAS ∈ [0, total_rev/total_spend]).
3. Stress tests:
     a) high contribution margin (0.60)
     b) low contribution margin (0.10)
     c) drop half the weeks
     d) inject a near-duplicate fake channel (VIF should explode)
     e) inject an all-zero channel (graceful handling)
     f) reverse the time axis (estimates should be invariant)

Outputs to `examples/benchmark_results/` as CSV + a Markdown summary at
`examples/benchmark.md`.
"""

from __future__ import annotations

import json
import ssl
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import causal_lift as cl

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "examples" / "benchmark_results"
OUT.mkdir(parents=True, exist_ok=True)

ROBYN_CSV = OUT / "dt_simulated_weekly.csv"
ROBYN_URL = (
    "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/"
    "R/data-raw/dt_simulated_weekly.csv"
)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_robyn() -> pd.DataFrame:
    if not ROBYN_CSV.exists():
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(ROBYN_URL, context=ctx) as r:
            ROBYN_CSV.write_bytes(r.read())
    df = pd.read_csv(ROBYN_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def to_causal_lift_format(df: pd.DataFrame):
    """Robyn wide → causal-lift long. Only ad spend channels (`*_S`)."""
    spend_cols = [c for c in df.columns if c.endswith("_S")]
    spend_long = (
        df[["DATE"] + spend_cols]
        .melt(id_vars="DATE", var_name="channel", value_name="spend")
        .rename(columns={"DATE": "date"})
    )
    sales = df[["DATE", "revenue"]].rename(columns={"DATE": "date"})
    return spend_long, sales, spend_cols


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def result_rows(result, label: str) -> list[dict]:
    rows = []
    for ch in result.channels:
        rows.append({
            "scenario": label,
            "channel": ch.channel,
            "total_spend": ch.total_spend,
            "iROAS": ch.incremental_roas,
            "raw_coef": ch.raw_coef,
            "ci_low": ch.confidence_interval[0],
            "ci_high": ch.confidence_interval[1],
            "vif": ch.vif_score,
            "recommendation": ch.recommendation,
        })
    return rows


def print_summary(result, header: str):
    print("\n" + "=" * 78)
    print(header)
    print("=" * 78)
    print(result.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Headline + peer comparison
# ─────────────────────────────────────────────────────────────────────────────
def headline(df: pd.DataFrame) -> tuple[cl.AnalysisResult, list[dict]]:
    spend_long, sales, spend_cols = to_causal_lift_format(df)
    res = cl.analyze(spend_long, sales, contribution_margin=0.30)
    print_summary(res, "1. Robyn weekly  |  causal-lift headline")

    total_rev = float(df["revenue"].sum())
    total_spend = float(df[spend_cols].sum().sum())
    plausibility_ceiling = total_rev / total_spend  # absolute upper bound on weighted-avg iROAS
    print(f"\n  total revenue:  {total_rev:,.0f}")
    print(f"  total spend:    {total_spend:,.0f}")
    print(f"  rev/spend cap:  {plausibility_ceiling:.2f}x  "
          "(no channel's iROAS can exceed this on average)")

    # peer estimators ────────────────────────────────────────────────────────
    import statsmodels.api as sm

    rows = []
    for ch in spend_cols:
        # (a) last-touch attribution by spend share (naive baseline)
        share = df[ch].sum() / total_spend
        attributed_rev = share * total_rev
        last_touch_roas = attributed_rev / df[ch].sum() if df[ch].sum() else np.nan
        # (b) single-channel OLS (revenue ~ const + spend_ch)
        X = sm.add_constant(df[ch].values.astype(float))
        m = sm.OLS(df["revenue"].values.astype(float), X).fit()
        bivar_iroas = float(m.params[1])
        # causal-lift estimate
        causal = next((c for c in res.channels if c.channel == ch), None)
        rows.append({
            "channel": ch,
            "share_of_spend_pct": round(100 * share, 1),
            "last_touch_roas": round(last_touch_roas, 2),
            "bivariate_ols_iROAS": round(bivar_iroas, 2),
            "causal_lift_iROAS": round(causal.incremental_roas, 2) if causal else np.nan,
            "causal_lift_ci": (
                f"[{causal.confidence_interval[0]:.2f}, {causal.confidence_interval[1]:.2f}]"
                if causal else ""
            ),
            "vif": round(causal.vif_score, 1) if causal and causal.vif_score is not None else None,
            "rec": causal.recommendation if causal else "",
        })
    peer = pd.DataFrame(rows)
    peer.to_csv(OUT / "peer_comparison.csv", index=False)
    print("\n  Peer comparison (naive baselines vs causal-lift):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(peer.to_string(index=False))
    return res, rows


# ─────────────────────────────────────────────────────────────────────────────
# Stress tests
# ─────────────────────────────────────────────────────────────────────────────
def stress_tests(df: pd.DataFrame) -> list[dict]:
    spend_long, sales, spend_cols = to_causal_lift_format(df)
    all_rows = []

    # a) high margin
    res = cl.analyze(spend_long, sales, contribution_margin=0.60)
    print_summary(res, "Stress (a) contribution_margin = 0.60  ->  breakeven 1.67x")
    all_rows += result_rows(res, "margin_0.60")

    # b) low margin
    res = cl.analyze(spend_long, sales, contribution_margin=0.10)
    print_summary(res, "Stress (b) contribution_margin = 0.10  ->  breakeven 10.00x")
    all_rows += result_rows(res, "margin_0.10")

    # c) drop half the weeks
    half = df.iloc[::2].reset_index(drop=True)
    sp_h, sa_h, _ = to_causal_lift_format(half)
    res = cl.analyze(sp_h, sa_h, contribution_margin=0.30)
    print_summary(res, f"Stress (c) drop half the weeks  ({len(half)} obs)")
    all_rows += result_rows(res, "half_weeks")

    # d) fake highly-correlated channel
    df_fake = df.copy()
    rng = np.random.default_rng(0)
    df_fake["facebook_S_clone"] = (
        df_fake["facebook_S"] * (1 + 0.02 * rng.standard_normal(len(df_fake)))
    )
    sp_f = df_fake[["DATE", "tv_S", "ooh_S", "print_S", "search_S",
                    "facebook_S", "facebook_S_clone"]].melt(
        id_vars="DATE", var_name="channel", value_name="spend"
    ).rename(columns={"DATE": "date"})
    sa_f = df_fake[["DATE", "revenue"]].rename(columns={"DATE": "date"})
    res = cl.analyze(sp_f, sa_f, contribution_margin=0.30)
    print_summary(res, "Stress (d) add fake near-duplicate of facebook_S")
    all_rows += result_rows(res, "duplicate_channel")

    # e) all-zero channel
    df_zero = df.copy()
    df_zero["zombie_S"] = 0.0
    sp_z = df_zero[["DATE"] + [c for c in df_zero.columns if c.endswith("_S")]].melt(
        id_vars="DATE", var_name="channel", value_name="spend"
    ).rename(columns={"DATE": "date"})
    sa_z = df_zero[["DATE", "revenue"]].rename(columns={"DATE": "date"})
    try:
        res = cl.analyze(sp_z, sa_z, contribution_margin=0.30)
        print_summary(res, "Stress (e) add zombie all-zero channel")
        all_rows += result_rows(res, "zero_channel")
        zero_handled = "zombie_S" not in {c.channel for c in res.channels}
        print(f"  zombie_S dropped from results? {zero_handled}")
    except Exception as e:
        print(f"  Stress (e) raised: {type(e).__name__}: {e}")
        all_rows.append({"scenario": "zero_channel", "channel": "zombie_S",
                         "iROAS": None, "recommendation": f"ERROR:{e}"})

    # f) reverse the time axis (estimates should be invariant up to DOW)
    df_rev = df.copy().iloc[::-1].reset_index(drop=True)
    # Re-stamp dates so they remain monotonic — same calendar but reversed values
    df_rev["DATE"] = df["DATE"].values
    sp_r, sa_r, _ = to_causal_lift_format(df_rev)
    res = cl.analyze(sp_r, sa_r, contribution_margin=0.30)
    print_summary(res, "Stress (f) reverse the data sequence under same dates")
    all_rows += result_rows(res, "reversed_time")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUT / "stress_tests.csv", index=False)
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = load_robyn()
    print(f"Loaded Robyn dataset: {len(df)} weekly rows, "
          f"{df['DATE'].min().date()} .. {df['DATE'].max().date()}")
    print(f"causal-lift version: {cl.__version__}")

    headline_res, peer_rows = headline(df)
    pd.DataFrame(result_rows(headline_res, "headline")).to_csv(
        OUT / "headline_estimates.csv", index=False
    )

    stress_rows = stress_tests(df)

    snapshot = {
        "version": cl.__version__,
        "headline": {
            "r_squared": headline_res.r_squared,
            "durbin_watson": headline_res.durbin_watson,
            "n": headline_res.observations,
            "warnings": headline_res.warnings,
            "channels": [c.to_dict() for c in headline_res.channels],
            "peer_comparison": peer_rows,
        },
        "stress": stress_rows,
    }
    (OUT / "benchmark_snapshot_v2.json").write_text(
        json.dumps(snapshot, indent=2, default=str)
    )
    print("\nDone. Artefacts in examples/benchmark_results/")


if __name__ == "__main__":
    main()
