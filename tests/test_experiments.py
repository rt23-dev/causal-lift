"""Tests for the geo holdout experiment design + analysis module."""

import numpy as np
import pandas as pd
import pytest

from causal_lift import (
    GeoHoldoutDesign,
    GeoHoldoutResult,
    analyze_geo_holdout,
    design_geo_holdout,
)


def _make_baseline(n_geos: int = 8, n_weeks: int = 26, seed: int = 0) -> pd.DataFrame:
    """Build a plausible multi-DMA baseline with correlated revenue series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-06", periods=n_weeks, freq="W-MON")
    # Common trend + per-geo level + per-geo noise
    common_trend = 100_000 + np.arange(n_weeks) * 1500 + rng.normal(0, 5_000, n_weeks)
    rows = []
    for g in range(n_geos):
        level_multiplier = rng.uniform(0.6, 1.4)
        idiosyncratic = rng.normal(0, 3_000, n_weeks)
        rev = (common_trend * level_multiplier + idiosyncratic).clip(10_000, None)
        for i, d in enumerate(dates):
            rows.append({"geo": f"DMA_{g:02d}", "date": d, "revenue": float(rev[i])})
    return pd.DataFrame(rows)


# ── design_geo_holdout ────────────────────────────────────────────────────────


def test_design_returns_design_object():
    baseline = _make_baseline()
    design = design_geo_holdout(baseline, n_treated=3, duration_weeks=4)
    assert isinstance(design, GeoHoldoutDesign)


def test_design_picks_correct_number_of_treated():
    baseline = _make_baseline()
    design = design_geo_holdout(baseline, n_treated=3, duration_weeks=4)
    assert len(design.treated_geos) == 3


def test_design_treated_and_control_partition_universe():
    baseline = _make_baseline(n_geos=8)
    design = design_geo_holdout(baseline, n_treated=3, duration_weeks=4)
    all_geos = set(design.treated_geos) | set(design.control_geos)
    assert len(all_geos) == 8
    assert not (set(design.treated_geos) & set(design.control_geos))


def test_design_respects_user_provided_treated():
    baseline = _make_baseline()
    explicit = ["DMA_01", "DMA_03"]
    design = design_geo_holdout(
        baseline, n_treated=2, duration_weeks=4, treated_geos=explicit
    )
    assert set(design.treated_geos) == set(explicit)


def test_design_raises_when_treated_exceeds_geos():
    baseline = _make_baseline(n_geos=4)
    with pytest.raises(ValueError, match="n_treated"):
        design_geo_holdout(baseline, n_treated=10, duration_weeks=4)


def test_design_raises_when_user_treated_geo_missing():
    baseline = _make_baseline()
    with pytest.raises(ValueError, match="not found"):
        design_geo_holdout(
            baseline, treated_geos=["DMA_99"], n_treated=1, duration_weeks=4
        )


def test_design_power_increases_with_duration():
    baseline = _make_baseline()
    short = design_geo_holdout(baseline, n_treated=3, duration_weeks=2, expected_lift_pct=0.05)
    long = design_geo_holdout(baseline, n_treated=3, duration_weeks=8, expected_lift_pct=0.05)
    assert long.power_at_expected_lift >= short.power_at_expected_lift


def test_design_mde_decreases_with_duration():
    baseline = _make_baseline()
    short = design_geo_holdout(baseline, n_treated=3, duration_weeks=2, expected_lift_pct=0.05)
    long = design_geo_holdout(baseline, n_treated=3, duration_weeks=8, expected_lift_pct=0.05)
    assert long.minimum_detectable_effect <= short.minimum_detectable_effect


def test_design_similarity_score_in_unit_interval():
    baseline = _make_baseline()
    design = design_geo_holdout(baseline, n_treated=3, duration_weeks=4)
    assert -1 <= design.similarity_score <= 1


def test_design_summary_returns_string():
    baseline = _make_baseline()
    design = design_geo_holdout(baseline, n_treated=3, duration_weeks=4)
    s = design.summary()
    assert isinstance(s, str)
    assert "Geo Holdout" in s


def test_design_to_dict_serialisable():
    baseline = _make_baseline()
    design = design_geo_holdout(baseline, n_treated=3, duration_weeks=4)
    d = design.to_dict()
    assert "treated_geos" in d
    assert "power_at_expected_lift" in d


# ── analyze_geo_holdout ───────────────────────────────────────────────────────


def _make_experiment_data(
    treated_lift_pct: float = 0.10, seed: int = 1
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a baseline + post period where treated geos got `treated_lift_pct` boost."""
    baseline = _make_baseline(n_geos=8, n_weeks=20, seed=seed)
    treated = ["DMA_00", "DMA_01", "DMA_02"]

    # Append 4 weeks of post-period; treated geos get the lift, controls don't
    last_date = baseline["date"].max()
    post_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1), periods=4, freq="W-MON"
    )
    rng = np.random.default_rng(seed + 100)

    # Continue the same trend in post period
    baseline_last_per_geo = (
        baseline.sort_values("date").groupby("geo").tail(1).set_index("geo")["revenue"]
    )
    rows = []
    for d in post_dates:
        for g, base_rev in baseline_last_per_geo.items():
            noise = rng.normal(0, base_rev * 0.03)
            growth = base_rev * 1.01  # 1% per week organic
            rev = growth + noise
            if g in treated:
                rev *= 1 + treated_lift_pct
            rows.append({"geo": g, "date": d, "revenue": float(rev)})

    post = pd.DataFrame(rows)
    full = pd.concat([baseline, post], ignore_index=True)
    return full, treated, [g for g in full["geo"].unique() if g not in treated]


def test_analyze_returns_result_object():
    data, treated, _ = _make_experiment_data()
    result = analyze_geo_holdout(
        data, treated_geos=treated,
        pre_period_end="2025-05-05", post_period_start="2025-05-12",
    )
    assert isinstance(result, GeoHoldoutResult)


def test_analyze_detects_real_lift():
    data, treated, _ = _make_experiment_data(treated_lift_pct=0.15, seed=42)
    result = analyze_geo_holdout(
        data, treated_geos=treated,
        pre_period_end="2025-05-05", post_period_start="2025-05-12",
    )
    # Should measure a positive lift roughly in the ballpark
    assert result.measured_lift_pct > 0.05


def test_analyze_no_lift_majority_of_seeds_not_significant():
    """
    With a 5% alpha we expect ~5% false positives. Across 20 seeds with no
    true lift, the majority should come back NO_EFFECT or INCONCLUSIVE.
    """
    significant_count = 0
    for seed in range(20):
        data, treated, _ = _make_experiment_data(treated_lift_pct=0.0, seed=seed)
        result = analyze_geo_holdout(
            data, treated_geos=treated,
            pre_period_end="2025-05-05", post_period_start="2025-05-12",
        )
        if result.verdict == "LIFT_DETECTED":
            significant_count += 1
    # Allow up to 30% false positives (test power lets noise leak through)
    assert significant_count <= 6, (
        f"Got {significant_count}/20 false positives — bootstrap CI may be too narrow."
    )


def test_analyze_with_spend_change_returns_iroas():
    data, treated, _ = _make_experiment_data(treated_lift_pct=0.10, seed=42)
    result = analyze_geo_holdout(
        data, treated_geos=treated,
        pre_period_end="2025-05-05", post_period_start="2025-05-12",
        spend_change=50_000,
    )
    assert result.implied_iroas is not None
    assert result.spend_change == 50_000


def test_analyze_raises_when_treated_geo_missing():
    data, _, _ = _make_experiment_data()
    with pytest.raises(ValueError, match="not present"):
        analyze_geo_holdout(
            data, treated_geos=["DMA_99"],
            pre_period_end="2025-05-05", post_period_start="2025-05-12",
        )


def test_analyze_raises_when_periods_overlap():
    data, treated, _ = _make_experiment_data()
    with pytest.raises(ValueError, match="must be after"):
        analyze_geo_holdout(
            data, treated_geos=treated,
            pre_period_end="2025-05-12", post_period_start="2025-05-05",
        )


def test_analyze_summary_returns_string():
    data, treated, _ = _make_experiment_data(seed=42)
    result = analyze_geo_holdout(
        data, treated_geos=treated,
        pre_period_end="2025-05-05", post_period_start="2025-05-12",
    )
    s = result.summary()
    assert isinstance(s, str) and "Geo Holdout" in s


def test_analyze_to_dict_serialisable():
    data, treated, _ = _make_experiment_data(seed=42)
    result = analyze_geo_holdout(
        data, treated_geos=treated,
        pre_period_end="2025-05-05", post_period_start="2025-05-12",
    )
    d = result.to_dict()
    assert "verdict" in d
    assert "measured_lift_pct" in d
