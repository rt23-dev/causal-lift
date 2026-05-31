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


def test_amazon_loader_day_grouped():
    csv = (
        "Date,Campaign Name,Advertised ASIN,Spend,Sales\n"
        "2025-01-01,SP-Energy-Drink,B0AAA111,500.00,2800\n"
        "2025-01-01,SP-Hydration-Mix,B0BBB222,200.00,1400\n"
        "2025-01-02,SP-Energy-Drink,B0AAA111,550.00,3100\n"
    )
    from causal_lift.loaders import load_amazon_ads_csv
    df = load_amazon_ads_csv(StringIO(csv))
    assert set(df.columns) == {"date", "channel", "spend"}
    assert len(df) == 2
    assert (df["channel"] == "amazon").all()
    day1 = df.loc[df["date"] == df["date"].min()].iloc[0]
    assert day1["spend"] == pytest.approx(700.0, abs=0.01)


def test_amazon_loader_asin_grouped():
    csv = (
        "Date,Campaign Name,Advertised ASIN,Spend,Sales\n"
        "2025-01-01,SP-Energy-Drink,B0AAA111,500.00,2800\n"
        "2025-01-01,SP-Hydration-Mix,B0BBB222,200.00,1400\n"
    )
    from causal_lift.loaders import load_amazon_ads_csv
    df = load_amazon_ads_csv(StringIO(csv), group_by="asin")
    assert "amazon_B0AAA111" in set(df["channel"])
    assert "amazon_B0BBB222" in set(df["channel"])


def test_amazon_loader_campaign_grouped():
    csv = (
        "Date,Campaign Name,Advertised ASIN,Spend,Sales\n"
        "2025-01-01,SP-Energy-Drink,B0AAA111,500.00,2800\n"
        "2025-01-01,SP-Hydration-Mix,B0BBB222,200.00,1400\n"
    )
    from causal_lift.loaders import load_amazon_ads_csv
    df = load_amazon_ads_csv(StringIO(csv), group_by="campaign")
    assert set(df["channel"]) == {"SP-Energy-Drink", "SP-Hydration-Mix"}


def test_amazon_loader_missing_spend_raises():
    csv = "Date,Campaign Name,Advertised ASIN\n2025-01-01,X,B0AAA111\n"
    from causal_lift.loaders import load_amazon_ads_csv
    with pytest.raises(ValueError, match="missing a spend column"):
        load_amazon_ads_csv(StringIO(csv))


def test_amazon_sales_loader_returns_sku_revenue():
    csv = (
        "Date,Advertised ASIN,Sales\n"
        "2025-01-01,B0AAA111,12500\n"
        "2025-01-01,B0BBB222,4200\n"
        "2025-01-02,B0AAA111,13100\n"
    )
    from causal_lift.loaders import load_amazon_sales_csv
    df = load_amazon_sales_csv(StringIO(csv))
    assert set(df.columns) == {"date", "asin", "revenue"}
    assert df["revenue"].sum() == pytest.approx(29_800.0, abs=0.5)


def test_walmart_loader_day_grouped():
    csv = (
        "Date,Campaign Name,Item ID,Ad Spend,Attributed Sales\n"
        "2025-01-01,Sponsored-Search,WM12345,400.00,2200\n"
        "2025-01-01,Sponsored-Search,WM67890,150.00,800\n"
        "2025-01-02,Sponsored-Search,WM12345,425.00,2350\n"
    )
    from causal_lift.loaders import load_walmart_ads_csv
    df = load_walmart_ads_csv(StringIO(csv))
    assert (df["channel"] == "walmart_connect").all()
    day1 = df.loc[df["date"] == df["date"].min()].iloc[0]
    assert day1["spend"] == pytest.approx(550.0, abs=0.01)


def test_walmart_loader_item_grouped():
    csv = (
        "Date,Campaign Name,Item ID,Ad Spend,Attributed Sales\n"
        "2025-01-01,Sponsored-Search,WM12345,400.00,2200\n"
        "2025-01-01,Sponsored-Search,WM67890,150.00,800\n"
    )
    from causal_lift.loaders import load_walmart_ads_csv
    df = load_walmart_ads_csv(StringIO(csv), group_by="item")
    assert "walmart_connect_WM12345" in set(df["channel"])


def test_api_stubs_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        fetch_shopify_orders_api()
    with pytest.raises(NotImplementedError):
        fetch_meta_ads_insights_api()
    with pytest.raises(NotImplementedError):
        fetch_google_ads_report_api()
