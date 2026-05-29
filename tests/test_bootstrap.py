"""Tests for the stationary block bootstrap CI option."""

import pytest

import causal_lift as cl
from causal_lift import RegressionMMM, generate_synthetic_data


def test_bootstrap_returns_finite_cis():
    data = generate_synthetic_data(n_days=90, seed=42)
    mmm = RegressionMMM(inference="bootstrap", n_bootstrap=300, random_state=1)
    result = mmm.fit(data.spend_df, data.sales_df, contribution_margin=0.30)
    for ch in result.channels:
        lo, hi = ch.confidence_interval
        assert lo == lo  # NaN check
        assert hi == hi
        assert hi >= lo


def test_bootstrap_inference_warning_fired():
    data = generate_synthetic_data(n_days=90, seed=42)
    mmm = RegressionMMM(inference="bootstrap", n_bootstrap=200, random_state=1)
    result = mmm.fit(data.spend_df, data.sales_df, contribution_margin=0.30)
    assert any("stationary block bootstrap" in w.lower() for w in result.warnings)


def test_bootstrap_seed_is_deterministic():
    data = generate_synthetic_data(n_days=90, seed=42)
    a = RegressionMMM(inference="bootstrap", n_bootstrap=200, random_state=7).fit(
        data.spend_df, data.sales_df, contribution_margin=0.30
    )
    b = RegressionMMM(inference="bootstrap", n_bootstrap=200, random_state=7).fit(
        data.spend_df, data.sales_df, contribution_margin=0.30
    )
    for ca, cb in zip(a.channels, b.channels):
        assert ca.confidence_interval == cb.confidence_interval


def test_invalid_inference_raises():
    with pytest.raises(ValueError, match="inference must be"):
        RegressionMMM(inference="something-else")


def test_bootstrap_point_estimate_matches_hac():
    """Bootstrap changes CIs but not point estimates."""
    data = generate_synthetic_data(n_days=90, seed=42)
    hac = cl.analyze(data.spend_df, data.sales_df, contribution_margin=0.30)
    boot = RegressionMMM(inference="bootstrap", n_bootstrap=200, random_state=1).fit(
        data.spend_df, data.sales_df, contribution_margin=0.30
    )
    hac_by_ch = {c.channel: c for c in hac.channels}
    for c_boot in boot.channels:
        c_hac = hac_by_ch[c_boot.channel]
        assert c_hac.incremental_roas == pytest.approx(c_boot.incremental_roas, abs=1e-9)
