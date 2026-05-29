"""Tests for the data loaders module (CSV-based)."""

from io import StringIO

import pytest

from causal_lift.loaders import (
    fetch_google_ads_report_api,
    fetch_meta_ads_insights_api,
    fetch_shopify_orders_api,
    load_google_ads_report_csv,
    load_meta_ads_insights_csv,
    load_shopify_orders_csv,
)


def test_shopify_loader_aggregates_daily():
    csv = (
        "Name,Paid at,Total,Subtotal\n"
        "#1001,2024-01-01T12:00:00+00:00,50.00,45\n"
        "#1001,2024-01-01T12:00:00+00:00,50.00,45\n"  # line-item dup
        "#1002,2024-01-01T14:00:00+00:00,30.00,28\n"
        "#1003,2024-01-02T09:00:00+00:00,80.00,75\n"
    )
    df = load_shopify_orders_csv(StringIO(csv))
    assert set(df.columns) >= {"date", "revenue", "orders"}
    # Two days
    assert len(df) == 2
    # Day-1 collapsed dup line items: 50 + 30 = 80 (one order from #1001 + one from #1002)
    day1 = df.loc[df["date"] == df["date"].min()].iloc[0]
    assert day1["revenue"] == pytest.approx(80.0, abs=0.01)


def test_shopify_loader_missing_column_raises():
    csv = "Name,Paid at,Subtotal\n#1001,2024-01-01T12:00:00+00:00,45\n"
    with pytest.raises(ValueError, match="missing revenue column"):
        load_shopify_orders_csv(StringIO(csv))


def test_meta_loader_day_grouped():
    csv = (
        "Reporting starts,Campaign name,Amount spent (USD)\n"
        "2024-01-01,Brand-Awareness,100.00\n"
        "2024-01-01,Performance-Prospecting,250.00\n"
        "2024-01-02,Brand-Awareness,120.00\n"
    )
    df = load_meta_ads_insights_csv(StringIO(csv), channel_name="meta")
    assert set(df.columns) == {"date", "channel", "spend"}
    assert len(df) == 2
    assert (df["channel"] == "meta").all()
    day1 = df.loc[df["date"] == df["date"].min()].iloc[0]
    assert day1["spend"] == pytest.approx(350.0, abs=0.01)


def test_meta_loader_campaign_grouped():
    csv = (
        "Reporting starts,Campaign name,Amount spent (USD)\n"
        "2024-01-01,Brand-Awareness,100.00\n"
        "2024-01-01,Performance-Prospecting,250.00\n"
    )
    df = load_meta_ads_insights_csv(StringIO(csv), group_by="campaign")
    assert set(df["channel"].unique()) == {"Brand-Awareness", "Performance-Prospecting"}


def test_meta_loader_missing_date_raises():
    csv = "Campaign name,Amount spent (USD)\nBrand,100\n"
    with pytest.raises(ValueError, match="missing a date column"):
        load_meta_ads_insights_csv(StringIO(csv))


def test_google_loader_day_grouped():
    csv = (
        "Day,Campaign,Cost\n"
        "2024-01-01,Search-Brand,300.00\n"
        "2024-01-01,Search-Generic,150.00\n"
        "2024-01-02,Search-Brand,310.00\n"
    )
    df = load_google_ads_report_csv(StringIO(csv), channel_name="google")
    assert len(df) == 2
    assert (df["channel"] == "google").all()


def test_google_loader_campaign_grouped():
    csv = (
        "Day,Campaign,Cost\n"
        "2024-01-01,Search-Brand,300.00\n"
        "2024-01-01,Search-Generic,150.00\n"
    )
    df = load_google_ads_report_csv(StringIO(csv), group_by="campaign")
    assert set(df["channel"].unique()) == {"Search-Brand", "Search-Generic"}


def test_api_stubs_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        fetch_shopify_orders_api()
    with pytest.raises(NotImplementedError):
        fetch_meta_ads_insights_api()
    with pytest.raises(NotImplementedError):
        fetch_google_ads_report_api()
