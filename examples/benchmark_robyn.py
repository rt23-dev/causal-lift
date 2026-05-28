"""
benchmark_robyn.py
==================

Empirical benchmark of `causal-lift` on three datasets:

1. **Robyn simulated weekly data** — Meta's canonical MMM benchmark CSV.
   Real-shaped correlation structure, baked-in adstock; no published
   ground-truth iROAS but useful as a realism check.

2. **Controlled adstock benchmark** — 200 weeks, 3 channels with KNOWN
   true iROAS (2.0x, 3.5x, 1.0x) and increasing geometric adstock decay
   (0.3, 0.6, 0.8). Tests how badly the model's "no adstock" limitation
   biases estimates as a function of carryover strength.

3. **Endogeneity stress test** — algorithmic bidding simulation where
   spend at time t depends on expected demand at time t. Tests the
   README's claim that OLS biases iROAS upward under endogenous spend.

Outputs land in `examples/benchmark_results/`. Re-run with:

    python examples/benchmark_robyn.py

Results are written to:
  - robyn_estimates.csv         per-channel estimates on Robyn data
  - adstock_benchmark.csv       true vs. estimated iROAS, with bias
  - endogeneity_benchmark.csv   true vs. estimated under bidding
  - endogeneity_sweep.csv       bias vs. bidding intensity
  - benchmark_snapshot.json     full structured dump
  - benchmark_run.log           console output capture
"""

from __future__ import annotations

import io
import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "examples" / "benchmark_results"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))
import causal_lift as cl  # noqa: E402

ROBYN_CSV = OUT / "dt_simulated_weekly.csv"
ROBYN_URL = (
    "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/"
    "R/data-raw/dt_simulated_weekly.csv"
)


# ── Logging helper ─────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


LOG_PATH = OUT / "benchmark_run.log"


def log(*args, **kw):
    print(*args, **kw)


# ── 1. Robyn data ──────────────────────────────────────────────────────────
def load_robyn() -> pd.DataFrame:
    """Load Robyn's simulated weekly dataset, downloading if missing."""
    if not ROBYN_CSV.exists():
        import ssl
        import urllib.request

        log(f"Downloading Robyn dataset from {ROBYN_URL} ...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(ROBYN_URL, context=ctx) as r:
            ROBYN_CSV.write_bytes(r.read())
    df = pd.read_csv(ROBYN_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def run_robyn_benchmark() -> dict:
    log("\n" + "=" * 70)
    log("1. ROBYN SIMULATED WEEKLY DATASET")
    log("=" * 70)
    df = load_robyn()
    log(f"  Loaded {len(df)} weekly rows, columns: {list(df.columns)}")
    log(f"  Date range: {df['DATE'].min().date()} .. {df['DATE'].max().date()}")

    spend_cols = [c for c in df.columns if c.endswith("_S")]
    log(f"  Spend channels: {spend_cols}")

    spend_long = df[["DATE"] + spend_cols].melt(
        id_vars="DATE", var_name="channel", value_name="spend"
    )
    spend_long = spend_long.rename(columns={"DATE": "date"})
    sales = df[["DATE", "revenue"]].rename(columns={"DATE": "date"})

    result = cl.analyze(spend_long, sales, contribution_margin=0.30)
    log("\n" + result.summary())

    rows = []
    for ch in result.channels:
        rows.append({
            "channel": ch.channel,
            "total_spend": ch.total_spend,
            "iROAS": ch.incremental_roas,
            "raw_coef": ch.raw_coef,
            "ci_low": ch.confidence_interval[0],
            "ci_high": ch.confidence_interval[1],
            "vif": ch.vif_score,
            "recommendation": ch.recommendation,
        })
    pd.DataFrame(rows).to_csv(OUT / "robyn_estimates.csv", index=False)
    log(f"\n  R^2 = {result.r_squared:.3f}, DW = {result.durbin_watson:.2f}, "
        f"n = {result.observations}")
    log(f"  Warnings: {len(result.warnings)}")
    for w in result.warnings:
        log(f"    - {w[:140]}")
    return {
        "r_squared": result.r_squared,
        "durbin_watson": result.durbin_watson,
        "n_warnings": len(result.warnings),
        "channels": rows,
    }


# ── 2. Adstock benchmark ───────────────────────────────────────────────────
def geometric_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock: adstocked_t = spend_t + decay * adstocked_{t-1}."""
    out = np.zeros_like(spend, dtype=float)
    out[0] = spend[0]
    for t in range(1, len(spend)):
        out[t] = spend[t] + decay * out[t - 1]
    return out


def make_adstock_dataset(n_weeks: int = 200, rng=None):
    """
    Build a synthetic weekly dataset with KNOWN true iROAS and geometric adstock.

    The DGP is:
        adstocked_{ch,t} = spend_{ch,t} + decay_ch * adstocked_{ch,t-1}
        revenue_t        = baseline + trend + season
                           + Σ true_iroas[ch] * adstocked_{ch,t} + noise

    Two notions of "true iROAS" matter:

    * **true_iroas (β)** — the marginal $ revenue per $ of adstocked spend.
      A correctly-specified MMM should recover this.
    * **long-run iROAS (β / (1-d))** — the steady-state $ revenue per $ of
      raw spend if spend is held constant. Under i.i.d. spend, OLS on raw
      spend would (in expectation) estimate this quantity. Same-day OLS
      with auto-correlated spend lies somewhere between the two.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    channels = ["search_low_adstock", "social_mid_adstock", "tv_high_adstock"]
    true_iroas = {"search_low_adstock": 2.0,
                  "social_mid_adstock": 3.5,
                  "tv_high_adstock": 1.0}
    decays = {"search_low_adstock": 0.3,
              "social_mid_adstock": 0.6,
              "tv_high_adstock": 0.8}
    base_levels = {"search_low_adstock": 5000,
                   "social_mid_adstock": 8000,
                   "tv_high_adstock": 12000}

    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

    # Spend kept EXOGENOUS — bias in this dataset comes purely from adstock.
    spend = {}
    for ch in channels:
        drift = np.linspace(1.0, 1.4, n_weeks)
        flight = 1.0 + 0.4 * np.sin(
            2 * np.pi * np.arange(n_weeks) / 26.0 + rng.uniform(0, np.pi)
        )
        noise = rng.lognormal(0, 0.20, n_weeks)
        s = base_levels[ch] * drift * flight * noise
        spend[ch] = np.clip(s, 0, None)

    baseline = 50_000.0
    trend = np.linspace(0, 15_000, n_weeks)
    seasonality = 8_000 * np.sin(2 * np.pi * np.arange(n_weeks) / 52.0)

    adstocked = {ch: geometric_adstock(spend[ch], decays[ch]) for ch in channels}
    incr_per_channel = {ch: true_iroas[ch] * adstocked[ch] for ch in channels}
    incr_total = np.sum(list(incr_per_channel.values()), axis=0)

    rev_mean_est = baseline + incr_total.mean()
    noise = rng.normal(0, 0.12 * rev_mean_est, n_weeks)
    revenue = baseline + trend + seasonality + incr_total + noise

    spend_long = pd.DataFrame({
        "date": np.tile(dates, len(channels)),
        "channel": np.repeat(channels, n_weeks),
        "spend": np.concatenate([spend[ch] for ch in channels]),
    })
    sales = pd.DataFrame({"date": dates, "revenue": revenue})

    truth = {
        "channels": channels,
        "true_iroas": true_iroas,
        "decays": decays,
        "effective_iroas_long_run":
            {ch: true_iroas[ch] / (1 - decays[ch]) for ch in channels},
        "true_incremental_revenue":
            {ch: float(incr_per_channel[ch].sum()) for ch in channels},
        "total_spend": {ch: float(spend[ch].sum()) for ch in channels},
    }
    return spend_long, sales, truth


def run_adstock_benchmark() -> dict:
    log("\n" + "=" * 70)
    log("2. CONTROLLED ADSTOCK BENCHMARK (200 weeks, known truth)")
    log("=" * 70)
    spend_long, sales, truth = make_adstock_dataset()

    log(f"  Channels: {truth['channels']}")
    log(f"  True iROAS (per $ of adstocked spend): {truth['true_iroas']}")
    log(f"  Adstock decays:                        {truth['decays']}")
    long_run = {k: round(v, 2) for k, v in truth["effective_iroas_long_run"].items()}
    log(f"  Long-run iROAS per $ raw spend:        {long_run}")

    result = cl.analyze(spend_long, sales, contribution_margin=0.30)
    log("\n" + result.summary())

    rows = []
    for ch in result.channels:
        true_b = truth["true_iroas"][ch.channel]
        eff_b = truth["effective_iroas_long_run"][ch.channel]
        est = ch.incremental_roas
        rows.append({
            "channel": ch.channel,
            "decay": truth["decays"][ch.channel],
            "true_iroas": true_b,
            "long_run_iroas": eff_b,
            "estimated_iroas": est,
            "ci_low": ch.confidence_interval[0],
            "ci_high": ch.confidence_interval[1],
            "bias_vs_true": est - true_b,
            "bias_pct_vs_true": 100.0 * (est - true_b) / true_b,
            "bias_vs_long_run": est - eff_b,
            "bias_pct_vs_long_run": 100.0 * (est - eff_b) / eff_b,
            "ci_covers_true": ch.confidence_interval[0] <= true_b <= ch.confidence_interval[1],
            "ci_covers_long_run": ch.confidence_interval[0] <= eff_b <= ch.confidence_interval[1],
            "vif": ch.vif_score,
            "recommendation": ch.recommendation,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "adstock_benchmark.csv", index=False)
    log("\n  Per-channel results:")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.3f}".format):
        log(df.to_string(index=False))
    log(f"\n  R^2 = {result.r_squared:.3f}, DW = {result.durbin_watson:.2f}")
    return {"truth": truth, "rows": rows,
            "r_squared": result.r_squared,
            "durbin_watson": result.durbin_watson}


# ── 3. Endogeneity benchmark ───────────────────────────────────────────────
def make_endogenous_dataset(n_weeks=200, bidding_strength=0.6, rng=None):
    """
    Dataset where spend_t responds to expected demand at t (algorithmic
    bidding). NO adstock — isolates the endogeneity effect.

      demand_t      = baseline + trend + seasonality + shock_t   (unobserved)
      spend_{ch,t}  = base + bidding_strength * α_ch * demand_t + ε_{ch,t}
      revenue_t     = demand_t + Σ true_iroas[ch] * spend_{ch,t} + noise
    """
    if rng is None:
        rng = np.random.default_rng(7)

    channels = ["meta_advantage", "google_smart", "tv_brand"]
    true_iroas = {"meta_advantage": 2.5, "google_smart": 3.0, "tv_brand": 1.5}
    alphas = {"meta_advantage": 0.05, "google_smart": 0.04, "tv_brand": 0.02}

    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

    baseline = 80_000.0
    trend = np.linspace(0, 20_000, n_weeks)
    seasonality = 12_000 * np.sin(2 * np.pi * np.arange(n_weeks) / 52.0)
    shock = rng.normal(0, 8_000, n_weeks)
    demand = baseline + trend + seasonality + shock

    spend = {}
    for ch in channels:
        baseline_spend = alphas[ch] * 50_000
        reactive = bidding_strength * alphas[ch] * demand
        noise = rng.normal(0, baseline_spend * 0.3, n_weeks)
        spend[ch] = np.clip(baseline_spend + reactive + noise, 0, None)

    incr = sum(true_iroas[ch] * spend[ch] for ch in channels)
    rev_noise = rng.normal(0, 0.05 * (demand.mean() + incr.mean()), n_weeks)
    revenue = demand + incr + rev_noise

    spend_long = pd.DataFrame({
        "date": np.tile(dates, len(channels)),
        "channel": np.repeat(channels, n_weeks),
        "spend": np.concatenate([spend[ch] for ch in channels]),
    })
    sales = pd.DataFrame({"date": dates, "revenue": revenue})
    truth = {
        "channels": channels,
        "true_iroas": true_iroas,
        "bidding_strength": bidding_strength,
        "total_spend": {ch: float(spend[ch].sum()) for ch in channels},
    }
    return spend_long, sales, truth


def run_endogeneity_benchmark() -> dict:
    log("\n" + "=" * 70)
    log("3. ENDOGENEITY STRESS TEST (algorithmic bidding)")
    log("=" * 70)
    spend_long, sales, truth = make_endogenous_dataset(bidding_strength=0.6)
    log(f"  Channels: {truth['channels']}")
    log(f"  True iROAS:        {truth['true_iroas']}")
    log(f"  Bidding strength:  {truth['bidding_strength']}  "
        f"(0=exogenous, 1=fully reactive)")

    result = cl.analyze(spend_long, sales, contribution_margin=0.30)
    log("\n" + result.summary())

    rows = []
    for ch in result.channels:
        true_b = truth["true_iroas"][ch.channel]
        est = ch.incremental_roas
        rows.append({
            "channel": ch.channel,
            "true_iroas": true_b,
            "estimated_iroas": est,
            "ci_low": ch.confidence_interval[0],
            "ci_high": ch.confidence_interval[1],
            "bias": est - true_b,
            "bias_pct": 100.0 * (est - true_b) / true_b,
            "ci_covers_true":
                ch.confidence_interval[0] <= true_b <= ch.confidence_interval[1],
            "vif": ch.vif_score,
            "recommendation": ch.recommendation,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "endogeneity_benchmark.csv", index=False)
    log("\n  Per-channel results:")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.3f}".format):
        log(df.to_string(index=False))

    # Sweep across bidding strengths
    log("\n  Bias (%) as a function of bidding strength:")
    sweep_rows = []
    for bs in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        sl, sa, tr = make_endogenous_dataset(
            bidding_strength=bs, rng=np.random.default_rng(7)
        )
        r = cl.analyze(sl, sa, contribution_margin=0.30)
        for c in r.channels:
            tb = tr["true_iroas"][c.channel]
            sweep_rows.append({
                "bidding_strength": bs,
                "channel": c.channel,
                "true_iroas": tb,
                "estimated_iroas": c.incremental_roas,
                "bias_pct": 100.0 * (c.incremental_roas - tb) / tb,
            })
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT / "endogeneity_sweep.csv", index=False)
    pivot = sweep.pivot(index="bidding_strength", columns="channel",
                       values="bias_pct").round(1)
    log(pivot.to_string())

    return {"truth": truth, "rows": rows, "sweep": sweep_rows}


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    buf = io.StringIO()
    tee = Tee(sys.stdout, buf)
    with redirect_stdout(tee):
        log("causal-lift empirical benchmark")
        log("================================")
        log(f"  causal-lift version: {cl.__version__}")

        robyn_out = run_robyn_benchmark()
        adstock_out = run_adstock_benchmark()
        endo_out = run_endogeneity_benchmark()

        snapshot = {
            "causal_lift_version": cl.__version__,
            "robyn": robyn_out,
            "adstock": {
                "truth": adstock_out["truth"],
                "rows": adstock_out["rows"],
                "r_squared": adstock_out["r_squared"],
                "durbin_watson": adstock_out["durbin_watson"],
            },
            "endogeneity": endo_out,
        }
        (OUT / "benchmark_snapshot.json").write_text(
            json.dumps(snapshot, indent=2, default=str)
        )
        log("\nDone. Artifacts in examples/benchmark_results/:")
        for p in sorted(OUT.glob("*")):
            log(f"  {p.name}")

    LOG_PATH.write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
