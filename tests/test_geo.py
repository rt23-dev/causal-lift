"""Tests for the GeoMMM multi-geo analyzer."""

import numpy as np
import pandas as pd
import pytest

import causal_lift as cl
from causal_lift import GeoMMM, generate_synthetic_data
from causal_lift.geo import GeoAnalysisResult, detect_geo_column


def _multigeo_data(n_geos: int = 3, seed: int = 0):
    """Build a tiny multi-geo dataset by replicating the synthetic generator."""
    spend_parts, sales_parts = [], []
    rng = np.random.default_rng(seed)
    for i in range(n_geos):
        d = generate_synthetic_data(n_days=90, seed=int(rng.integers(1, 10_000)))
        sp = d.spend_df.copy()
        sa = d.sales_df.copy()
        sp["geo"] = f"geo_{i}"
        sa["geo"] = f"geo_{i}"
        spend_parts.append(sp)
        sales_parts.append(sa)
    return pd.concat(spend_parts, ignore_index=True), pd.concat(sales_parts, ignore_index=True)


def test_detect_geo_column_recognises_standard_names():
    assert detect_geo_column(pd.DataFrame({"geo": [1], "x": [2]})) == "geo"
    assert detect_geo_column(pd.DataFrame({"dma": [1], "x": [2]})) == "dma"
    assert detect_geo_column(pd.DataFrame({"region": [1]})) == "region"
    assert detect_geo_column(pd.DataFrame({"foo": [1]})) is None


def test_analyze_auto_routes_to_geo_when_geo_column_present():
    spend_df, sales_df = _multigeo_data(n_geos=3)
    result = cl.analyze(spend_df, sales_df, contribution_margin=0.30)
    assert isinstance(result, GeoAnalysisResult)
    assert len(result.geos) == 3


def test_geomm_runs_per_geo():
    spend_df, sales_df = _multigeo_data(n_geos=3)
    result = GeoMMM().fit(spend_df, sales_df, contribution_margin=0.30)
    assert len(result.per_geo) == 3
    # Each per-geo result should look like a normal AnalysisResult
    for _geo, r in result.per_geo.items():
        assert r.observations > 0
        assert 0 <= r.r_squared <= 1


def test_geomm_aggregates_to_one_channelresult_per_unique_channel():
    spend_df, sales_df = _multigeo_data(n_geos=3)
    result = GeoMMM().fit(spend_df, sales_df, contribution_margin=0.30)
    channels = {c.channel for c in result.channels}
    # Synthetic data has 3 channels
    assert channels == {"facebook", "google", "tiktok"}


def test_geomm_no_geo_column_raises_clear_error():
    data = generate_synthetic_data(n_days=90, seed=42)
    with pytest.raises(ValueError, match="requires a geo column"):
        GeoMMM().fit(data.spend_df, data.sales_df, contribution_margin=0.30)


def test_geomm_method_label_mentions_geo_count():
    spend_df, sales_df = _multigeo_data(n_geos=3)
    result = GeoMMM().fit(spend_df, sales_df, contribution_margin=0.30)
    assert "3 geos" in result.method_used


def test_geomm_to_dict_includes_per_geo():
    spend_df, sales_df = _multigeo_data(n_geos=2)
    result = GeoMMM().fit(spend_df, sales_df, contribution_margin=0.30)
    d = result.to_dict()
    assert "geos" in d
    assert "per_geo" in d
    assert len(d["per_geo"]) == 2


def test_geomm_recommendations_valid():
    spend_df, sales_df = _multigeo_data(n_geos=3)
    result = GeoMMM().fit(spend_df, sales_df, contribution_margin=0.30)
    for c in result.channels:
        assert c.recommendation in {"SCALE", "HOLD", "CUT", "INCONCLUSIVE"}
