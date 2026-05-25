"""Tests for the synthetic data generator."""

from causal_lift import generate_synthetic_data


def test_default_shape():
    data = generate_synthetic_data()
    assert len(data.sales_df) == 90
    assert len(data.spend_df) == 90 * 3  # three channels


def test_custom_n_days():
    data = generate_synthetic_data(n_days=30)
    assert len(data.sales_df) == 30
    assert len(data.spend_df) == 30 * 3


def test_reproducible_with_seed():
    a = generate_synthetic_data(n_days=60, seed=123)
    b = generate_synthetic_data(n_days=60, seed=123)
    assert (a.sales_df["revenue"].values == b.sales_df["revenue"].values).all()


def test_different_seeds_differ():
    a = generate_synthetic_data(n_days=60, seed=1)
    b = generate_synthetic_data(n_days=60, seed=2)
    assert not (a.sales_df["revenue"].values == b.sales_df["revenue"].values).all()


def test_ground_truth_exposed():
    data = generate_synthetic_data()
    assert set(data.ground_truth) == {"facebook", "google", "tiktok"}
    assert all(roas > 0 for roas in data.ground_truth.values())


def test_unpack_to_three_values():
    spend_df, sales_df, gt = generate_synthetic_data(n_days=30)
    assert "channel" in spend_df.columns
    assert "revenue" in sales_df.columns
    assert isinstance(gt, dict)


def test_no_negative_revenue():
    data = generate_synthetic_data(n_days=90)
    assert (data.sales_df["revenue"] >= 0).all()


def test_no_negative_spend():
    data = generate_synthetic_data(n_days=90)
    assert (data.spend_df["spend"] >= 0).all()
