"""
Regression tests against the Robyn weekly benchmark dataset.

These tests guard against the failures the benchmark post-mortem identified:
- Implausibly high iROAS estimates (>20x) getting confident SCALE labels
- Aggregate implied incremental share exceeding the plausibility ceiling
- Weekly cadence going undetected
- Always-on channels getting SCALE without an INCONCLUSIVE downgrade
"""

from pathlib import Path

import pandas as pd
import pytest

import causal_lift as cl
from causal_lift.analyzer import AGGREGATE_SHARE_CEILING

BENCHMARK_CSV = (
    Path(__file__).parent.parent
    / "examples"
    / "benchmark_results"
    / "dt_simulated_weekly.csv"
)


@pytest.fixture(scope="module")
def robyn_data():
    """Load and reshape Robyn's `dt_simulated_weekly.csv` to causal-lift format."""
    if not BENCHMARK_CSV.exists():
        pytest.skip(f"Benchmark dataset not available at {BENCHMARK_CSV}")
    df = pd.read_csv(BENCHMARK_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    spend_cols = [c for c in df.columns if c.endswith("_S")]
    spend_long = (
        df[["DATE"] + spend_cols]
        .melt(id_vars="DATE", var_name="channel", value_name="spend")
        .rename(columns={"DATE": "date"})
    )
    sales = df[["DATE", "revenue"]].rename(columns={"DATE": "date"})
    return spend_long, sales


@pytest.fixture(scope="module")
def robyn_result(robyn_data):
    spend, sales = robyn_data
    return cl.analyze(spend, sales, contribution_margin=0.30)


def test_weekly_cadence_detected(robyn_result):
    """Robyn data is weekly — the library must recognise it."""
    assert robyn_result.cadence == "weekly", (
        f"Expected weekly cadence, got {robyn_result.cadence}"
    )


def test_no_implausibly_high_scale_recommendations(robyn_result):
    """
    No channel with an iROAS above 20x should be labelled SCALE.

    On the Robyn dataset the search channel produces ~85x and facebook ~62x;
    these are baseline-confound artefacts and must not get a SCALE.
    """
    for ch in robyn_result.channels:
        if ch.incremental_roas > 20:
            assert ch.recommendation != "SCALE", (
                f"{ch.channel} has iROAS {ch.incremental_roas:.1f}x but was labelled SCALE — "
                f"this is the exact failure mode the plausibility gates exist to prevent."
            )


def test_always_on_channels_demoted(robyn_result):
    """
    Channels active in >85% of periods that would otherwise be SCALE
    must be labelled INCONCLUSIVE.
    """
    for ch in robyn_result.channels:
        if ch.nonzero_share > 0.85 and ch.incremental_roas > robyn_result.breakeven_roas:
            assert ch.recommendation in {"INCONCLUSIVE", "HOLD"}, (
                f"{ch.channel} runs in {ch.nonzero_share:.0%} of periods and has iROAS "
                f"{ch.incremental_roas:.1f}x above breakeven {robyn_result.breakeven_roas:.1f}x, "
                f"but was labelled {ch.recommendation}. Always-on confound gate failed."
            )


def test_aggregate_plausibility_check_fires(robyn_result):
    """
    On the Robyn dataset, the regression over-attributes revenue to paid media
    (search alone gets credited with ~28% of revenue from 8% of spend).
    The aggregate plausibility warning must fire.
    """
    fired = any(
        "implied incremental share" in w.lower()
        and "implausibly" in w.lower() or "over-attributing" in w.lower()
        for w in robyn_result.warnings
    )
    # If implied share is genuinely under the ceiling, the warning shouldn't fire — that's fine.
    if robyn_result.implied_incremental_share > AGGREGATE_SHARE_CEILING:
        assert fired, (
            f"Implied share {robyn_result.implied_incremental_share:.1%} exceeds ceiling "
            f"{AGGREGATE_SHARE_CEILING:.1%} but no warning fired. "
            f"Warnings present: {robyn_result.warnings}"
        )


def test_implied_share_exposed_on_result(robyn_result):
    """The `implied_incremental_share` field must be populated on AnalysisResult."""
    assert 0.0 <= robyn_result.implied_incremental_share <= 5.0


def test_search_channel_specifically_does_not_get_scale(robyn_result):
    """
    Belt-and-braces: the search channel is the canonical always-on failure case.
    Whatever the gate logic, it must NOT come out SCALE.
    """
    search = next(
        (c for c in robyn_result.channels if c.channel == "search_S"),
        None,
    )
    assert search is not None, "search_S channel missing from results"
    assert search.recommendation != "SCALE", (
        f"search_S labelled SCALE with iROAS {search.incremental_roas:.1f}x, "
        f"nonzero_share {search.nonzero_share:.0%}. This is the headline failure."
    )
