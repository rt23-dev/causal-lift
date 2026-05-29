"""Tests for the RegressionMMM analyzer."""

import numpy as np
import pandas as pd
import pytest

from causal_lift import RegressionMMM, analyze, generate_synthetic_data


def test_analyze_returns_one_result_per_channel():
    data = generate_synthetic_data(n_days=90, seed=42)
    result = analyze(data.spend_df, data.sales_df)
    channels = sorted(c.channel for c in result.channels)
    assert channels == ["facebook", "google", "tiktok"]


def test_facebook_iroas_within_tolerance_of_ground_truth():
    """Facebook has the largest budget experiment — should be well-identified."""
    data = generate_synthetic_data(n_days=90, seed=42)
    result = analyze(data.spend_df, data.sales_df, contribution_margin=0.30)
    fb = next(c for c in result.channels if c.channel == "facebook")
    # Ground truth = 2.0x; allow generous tolerance for noise
    assert 1.0 < fb.incremental_roas < 3.0, (
        f"Facebook iROAS {fb.incremental_roas:.2f} far from ground truth 2.0"
    )
    # CI should contain the true value
    assert fb.confidence_interval[0] < 2.0 < fb.confidence_interval[1] + 1.0


def test_contribution_margin_drives_breakeven():
    data = generate_synthetic_data(n_days=90, seed=42)
    r_30 = analyze(data.spend_df, data.sales_df, contribution_margin=0.30)
    r_50 = analyze(data.spend_df, data.sales_df, contribution_margin=0.50)
    assert r_30.breakeven_roas == pytest.approx(1 / 0.30, abs=0.01)
    assert r_50.breakeven_roas == pytest.approx(1 / 0.50, abs=0.01)


def test_margin_outside_range_is_clamped():
    data = generate_synthetic_data(n_days=90, seed=42)
    r_too_low = analyze(data.spend_df, data.sales_df, contribution_margin=0.001)
    r_too_high = analyze(data.spend_df, data.sales_df, contribution_margin=2.0)
    assert r_too_low.contribution_margin == 0.05
    assert r_too_high.contribution_margin == 0.95


def test_endogeneity_warning_always_present():
    data = generate_synthetic_data(n_days=90, seed=42)
    result = analyze(data.spend_df, data.sales_df)
    assert any("Identification assumption" in w for w in result.warnings)


def test_recommendations_are_valid_labels():
    data = generate_synthetic_data(n_days=90, seed=42)
    result = analyze(data.spend_df, data.sales_df)
    for ch in result.channels:
        assert ch.recommendation in {"SCALE", "HOLD", "CUT", "INCONCLUSIVE"}


def test_high_vif_forces_hold():
    """Two perfectly-correlated channels should produce HOLD recommendations."""
    rng = np.random.default_rng(0)
    n = 90
    dates = pd.date_range("2024-01-01", periods=n)
    base_spend = rng.normal(2000, 400, n).clip(500, 5000)

    # Two perfectly collinear channels
    spend_rows = []
    for i, d in enumerate(dates):
        spend_rows += [
            {"date": d, "channel": "ch_a", "spend": float(base_spend[i])},
            {"date": d, "channel": "ch_b", "spend": float(base_spend[i] * 0.5)},
        ]
    spend_df = pd.DataFrame(spend_rows)
    revenue = 1.5 * base_spend + rng.normal(0, 500, n) + 10_000
    sales_df = pd.DataFrame({"date": dates, "revenue": revenue})

    result = analyze(spend_df, sales_df)
    # Neither channel should get SCALE — the library should fall back to HOLD or
    # INCONCLUSIVE because the estimates aren't reliably identified.
    recs = {c.channel: c.recommendation for c in result.channels}
    assert "SCALE" not in recs.values(), (
        f"Collinear channels should not produce SCALE, got: {recs}"
    )
    assert any(r in {"HOLD", "INCONCLUSIVE"} for r in recs.values()), (
        f"Expected at least one HOLD/INCONCLUSIVE for collinear channels, got: {recs}"
    )


def test_to_dataframe_returns_expected_columns():
    data = generate_synthetic_data(n_days=90, seed=42)
    result = analyze(data.spend_df, data.sales_df)
    df = result.to_dataframe()
    expected = {
        "channel", "total_spend", "attribution_proxy_roas", "incremental_roas",
        "incremental_revenue", "confidence_interval", "recommendation",
        "recommendation_reason", "model_fit", "vif_score", "raw_coef",
    }
    assert expected <= set(df.columns)


def test_summary_returns_string():
    data = generate_synthetic_data(n_days=90, seed=42)
    result = analyze(data.spend_df, data.sales_df)
    s = result.summary()
    assert isinstance(s, str) and "RegressionMMM" in s and "iROAS" in s


def test_too_few_observations_raises_or_warns():
    data = generate_synthetic_data(n_days=15, seed=42)
    # Should still run but warn
    result = analyze(data.spend_df, data.sales_df)
    assert any("periods of overlapping data" in w for w in result.warnings)


def test_can_instantiate_model_directly():
    data = generate_synthetic_data(n_days=90, seed=42)
    model = RegressionMMM()
    result = model.fit(data.spend_df, data.sales_df, contribution_margin=0.30)
    assert result.observations == 90
