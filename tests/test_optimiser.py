"""Tests for the budget reallocation recommender."""

from dataclasses import replace

import pytest

import causal_lift as cl
from causal_lift import ChannelResult, recommend_reallocation
from causal_lift.optimiser import ReallocationPlan


@pytest.fixture(scope="module")
def baseline_result():
    data = cl.generate_synthetic_data(n_days=90, seed=42)
    return cl.analyze(data.spend_df, data.sales_df, contribution_margin=0.30)


def test_plan_has_one_recommendation_per_channel(baseline_result):
    plan = recommend_reallocation(baseline_result)
    assert {r.channel for r in plan.channels} == {c.channel for c in baseline_result.channels}


def test_scale_channels_get_positive_delta_pct():
    # Construct a synthetic result with one SCALE
    channels = [
        ChannelResult(
            channel="hot", total_spend=10_000.0, attribution_proxy_roas=5.0,
            incremental_roas=4.0, incremental_revenue=40_000.0,
            confidence_interval=[3.5, 4.5], recommendation="SCALE",
            recommendation_reason="", model_fit=0.9, vif_score=1.2, raw_coef=4.0,
            nonzero_share=0.5,
        )
    ]
    result = replace(
        recommend_reallocation.__wrapped__ if hasattr(recommend_reallocation, "__wrapped__")
        else _make_result(channels), channels=channels,
    ) if False else _make_result(channels)
    plan = recommend_reallocation(result)
    rec = next(r for r in plan.channels if r.channel == "hot")
    assert rec.delta_pct > 0
    assert rec.delta > 0


def test_cut_channels_get_negative_delta_pct():
    channels = [
        ChannelResult(
            channel="bad", total_spend=10_000.0, attribution_proxy_roas=1.0,
            incremental_roas=0.5, incremental_revenue=5_000.0,
            confidence_interval=[-0.2, 1.2], recommendation="CUT",
            recommendation_reason="", model_fit=0.5, vif_score=1.0, raw_coef=0.5,
            nonzero_share=0.5,
        )
    ]
    result = _make_result(channels)
    plan = recommend_reallocation(result)
    rec = next(r for r in plan.channels if r.channel == "bad")
    assert rec.delta_pct < 0
    assert rec.delta < 0


def test_hold_and_inconclusive_dont_move():
    channels = [
        ChannelResult(
            channel="hold_ch", total_spend=10_000.0, attribution_proxy_roas=2.0,
            incremental_roas=2.0, incremental_revenue=20_000.0,
            confidence_interval=[1.0, 3.0], recommendation="HOLD",
            recommendation_reason="", model_fit=0.6, vif_score=1.1, raw_coef=2.0,
            nonzero_share=0.6,
        ),
        ChannelResult(
            channel="inc_ch", total_spend=10_000.0, attribution_proxy_roas=3.0,
            incremental_roas=8.0, incremental_revenue=80_000.0,
            confidence_interval=[2.0, 14.0], recommendation="INCONCLUSIVE",
            recommendation_reason="", model_fit=0.7, vif_score=1.0, raw_coef=8.0,
            nonzero_share=0.95,
        ),
    ]
    plan = recommend_reallocation(_make_result(channels))
    for rec in plan.channels:
        assert rec.delta_pct == 0.0
        assert rec.delta == 0.0


def test_max_channel_delta_pct_is_respected():
    channels = [
        ChannelResult(
            channel="hot", total_spend=10_000.0, attribution_proxy_roas=5.0,
            incremental_roas=4.0, incremental_revenue=40_000.0,
            confidence_interval=[3.5, 4.5], recommendation="SCALE",
            recommendation_reason="", model_fit=0.9, vif_score=1.0, raw_coef=4.0,
            nonzero_share=0.5,
        )
    ]
    plan = recommend_reallocation(
        _make_result(channels), scale_pct=0.50, max_channel_delta_pct=0.10
    )
    rec = plan.channels[0]
    assert rec.delta_pct == pytest.approx(0.10, abs=1e-6)


def test_no_scale_channels_emits_note():
    channels = [
        ChannelResult(
            channel="cut1", total_spend=5_000.0, attribution_proxy_roas=0.5,
            incremental_roas=0.3, incremental_revenue=1_500.0,
            confidence_interval=[-0.2, 0.8], recommendation="CUT",
            recommendation_reason="", model_fit=0.5, vif_score=1.0, raw_coef=0.3,
            nonzero_share=0.4,
        )
    ]
    plan = recommend_reallocation(_make_result(channels))
    assert any("disambiguate" in n.lower() for n in plan.notes)


def test_plan_summary_returns_string(baseline_result):
    plan = recommend_reallocation(baseline_result)
    s = plan.summary()
    assert isinstance(s, str)
    assert "reallocation" in s.lower()


def test_plan_to_dict_serialisable(baseline_result):
    plan = recommend_reallocation(baseline_result)
    d = plan.to_dict()
    assert "channels" in d
    assert "total_current" in d
    assert isinstance(d["channels"], list)


def test_returns_reallocation_plan_type(baseline_result):
    plan = recommend_reallocation(baseline_result)
    assert isinstance(plan, ReallocationPlan)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(channels):
    """Build a minimal AnalysisResult for unit testing reallocation logic."""
    from causal_lift.analyzer import AnalysisResult
    total_spend = sum(c.total_spend for c in channels)
    return AnalysisResult(
        channels=channels,
        method_used="test",
        total_revenue=total_spend * 4.0,
        total_spend=total_spend,
        r_squared=0.7,
        observations=52,
        contribution_margin=0.30,
        breakeven_roas=3.33,
        durbin_watson=2.0,
        cadence="weekly",
        implied_incremental_share=0.4,
        adstock_thetas={c.channel: 0.0 for c in channels},
        warnings=[],
    )
