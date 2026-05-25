"""Tests for CSV ingestion helpers."""

from io import StringIO

import pytest

from causal_lift.data import CSVValidationError, load_sales_csv, load_spend_csv


def test_load_spend_csv_happy_path():
    content = "date,channel,spend\n2024-01-01,facebook,1000\n2024-01-02,facebook,1200\n"
    df = load_spend_csv(StringIO(content))
    assert list(df.columns) == ["date", "channel", "spend"]
    assert len(df) == 2
    assert df["spend"].sum() == 2200


def test_load_spend_csv_missing_columns():
    content = "date,channel\n2024-01-01,facebook\n"
    with pytest.raises(CSVValidationError, match="missing required columns"):
        load_spend_csv(StringIO(content))


def test_load_spend_csv_negative_spend_rejected():
    content = "date,channel,spend\n2024-01-01,facebook,-50\n"
    with pytest.raises(CSVValidationError, match="negative spend"):
        load_spend_csv(StringIO(content))


def test_load_spend_csv_sanitises_channel_names():
    content = "date,channel,spend\n2024-01-01,Facebook Ads!,500\n"
    df = load_spend_csv(StringIO(content))
    assert df["channel"].iloc[0] == "facebook_ads_"


def test_load_sales_csv_happy_path():
    content = "date,revenue,orders\n2024-01-01,5000,42\n2024-01-02,5500,50\n"
    df = load_sales_csv(StringIO(content))
    assert "orders" in df.columns
    assert df["revenue"].sum() == 10500


def test_load_sales_csv_orders_optional():
    content = "date,revenue\n2024-01-01,5000\n"
    df = load_sales_csv(StringIO(content))
    assert "orders" not in df.columns


def test_load_sales_csv_negative_revenue_rejected():
    content = "date,revenue\n2024-01-01,-100\n"
    with pytest.raises(CSVValidationError, match="negative revenue"):
        load_sales_csv(StringIO(content))


def test_empty_csv_rejected():
    with pytest.raises(CSVValidationError):
        load_spend_csv(StringIO("date,channel,spend\n"))
