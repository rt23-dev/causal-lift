"""
Data loaders for common DTC data sources.

Each loader takes a path (or file-like) to an export from the respective
platform and returns a DataFrame in the canonical causal-lift shape:

- ``spend_df``  columns: ``date``, ``channel``, ``spend``  (+ optional ``geo``)
- ``sales_df``  columns: ``date``, ``revenue``  (+ optional ``orders``, ``geo``)

These loaders handle CSV exports today; full API integrations (OAuth, rate
limiting, automatic refresh) are intentionally deferred — using exports keeps
the library install-light and lets users avoid wiring up credentials just to
try the tool.

Implementation status (as of v0.3.0)
------------------------------------
- ``load_shopify_orders_csv``        — implemented
- ``load_meta_ads_insights_csv``     — implemented
- ``load_google_ads_report_csv``     — implemented
- ``fetch_*`` API variants            — alpha stubs (raise NotImplementedError)

API loaders coming in v0.4.  If you'd like to help build one, open a PR.

Example
-------

    >>> from causal_lift.loaders import load_meta_ads_insights_csv, load_shopify_orders_csv
    >>> meta = load_meta_ads_insights_csv("meta_export.csv")
    >>> shopify = load_shopify_orders_csv("orders_export.csv")
    >>> import causal_lift as cl
    >>> result = cl.analyze(meta, shopify, contribution_margin=0.30)
"""

from __future__ import annotations

from typing import IO, Union

import pandas as pd

PathOrBuffer = Union[str, "IO[str]", "IO[bytes]"]


# ── Shopify ────────────────────────────────────────────────────────────────────

# Standard "Orders export" CSV column names (as of 2024-2025 Shopify admin).
SHOPIFY_DATE_COLS = ("Paid at", "Created at", "Processed at")
SHOPIFY_REVENUE_COL = "Total"
SHOPIFY_ORDERS_GROUP_COL = "Name"


def load_shopify_orders_csv(
    source: PathOrBuffer,
    timezone: str | None = None,
    revenue_col: str | None = None,
) -> pd.DataFrame:
    """
    Load a Shopify Orders CSV export and aggregate to daily revenue.

    Shopify's "Orders export" ships one row per order with multiple line-item
    rows.  We group by order name (or whatever the user supplies) and sum the
    total per day.

    Parameters
    ----------
    source : path or file-like
        CSV export from Shopify Admin → Orders → Export.
    timezone : str, optional
        If provided, parse dates as this timezone, then drop tz info.
    revenue_col : str, optional
        Override the revenue column name.  Defaults to ``"Total"``.

    Returns
    -------
    pandas.DataFrame  with columns ``date``, ``revenue``, ``orders``.
    """
    df = pd.read_csv(source)
    rev_col = revenue_col or SHOPIFY_REVENUE_COL
    if rev_col not in df.columns:
        raise ValueError(
            f"Shopify CSV missing revenue column {rev_col!r}. "
            f"Found columns: {sorted(df.columns.tolist())[:15]}..."
        )

    # Pick the first available date column
    date_col = next((c for c in SHOPIFY_DATE_COLS if c in df.columns), None)
    if date_col is None:
        raise ValueError(
            f"Shopify CSV missing any of the expected date columns: {SHOPIFY_DATE_COLS}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    if timezone:
        df[date_col] = df[date_col].dt.tz_convert(timezone).dt.tz_localize(None)
    else:
        df[date_col] = df[date_col].dt.tz_localize(None)

    # One row per order — collapse line items first.  In Shopify "Orders
    # export" the Total is repeated on every line-item row for an order, so
    # we take the FIRST value per order rather than summing (which would
    # double-count multi-line orders).
    if SHOPIFY_ORDERS_GROUP_COL in df.columns:
        df = df.groupby(SHOPIFY_ORDERS_GROUP_COL, as_index=False).agg(
            **{date_col: (date_col, "first"), rev_col: (rev_col, "first")}
        )

    df["date"] = df[date_col].dt.normalize()
    daily = df.groupby("date", as_index=False).agg(
        revenue=(rev_col, "sum"), orders=(SHOPIFY_ORDERS_GROUP_COL, "count"),
    )
    daily["revenue"] = pd.to_numeric(daily["revenue"], errors="coerce")
    return daily.dropna(subset=["revenue"]).reset_index(drop=True)


# ── Meta Ads ───────────────────────────────────────────────────────────────────

# Common Meta Ads Manager "Reports → Export" column names.  Meta supports many
# breakdowns and the CSV column casing/punctuation varies by region.
META_DATE_COL = "Reporting starts"
META_DATE_COL_ALT = "Day"
META_CAMPAIGN_COL = "Campaign name"
META_SPEND_COL = "Amount spent (USD)"
META_SPEND_COL_FALLBACKS = ("Amount spent", "Spend")


def load_meta_ads_insights_csv(
    source: PathOrBuffer,
    channel_name: str = "meta",
    group_by: str = "day",
    spend_col: str | None = None,
) -> pd.DataFrame:
    """
    Load a Meta Ads Manager export and reshape into causal-lift's spend format.

    The default ``channel_name='meta'`` aggregates all campaigns into a single
    channel.  Pass ``group_by='campaign'`` to split per campaign (the campaign
    name becomes the channel column).

    Parameters
    ----------
    source : path or file-like
    channel_name : str
        Channel label when aggregating all campaigns (``group_by='day'``).
    group_by : str
        Either ``'day'`` (one row per date, channel=channel_name) or
        ``'campaign'`` (one row per (date, campaign), channel=Campaign name).
    spend_col : str, optional
        Override the spend column name.

    Returns
    -------
    pandas.DataFrame with columns ``date``, ``channel``, ``spend``.
    """
    df = pd.read_csv(source)
    date_col = (
        META_DATE_COL if META_DATE_COL in df.columns
        else (META_DATE_COL_ALT if META_DATE_COL_ALT in df.columns else None)
    )
    if date_col is None:
        raise ValueError(
            f"Meta CSV missing a date column ({META_DATE_COL!r} or {META_DATE_COL_ALT!r})"
        )
    sp_col = spend_col or next(
        (c for c in (META_SPEND_COL, *META_SPEND_COL_FALLBACKS) if c in df.columns), None
    )
    if sp_col is None:
        raise ValueError(
            f"Meta CSV missing a spend column. Tried: {META_SPEND_COL!r}, "
            f"{META_SPEND_COL_FALLBACKS}"
        )

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df[sp_col] = pd.to_numeric(df[sp_col], errors="coerce").fillna(0)

    if group_by == "campaign":
        if META_CAMPAIGN_COL not in df.columns:
            raise ValueError(
                f"group_by='campaign' requires column {META_CAMPAIGN_COL!r}"
            )
        out = df.groupby(["date", META_CAMPAIGN_COL], as_index=False)[sp_col].sum()
        out = out.rename(columns={META_CAMPAIGN_COL: "channel", sp_col: "spend"})
    else:
        out = df.groupby("date", as_index=False)[sp_col].sum()
        out["channel"] = channel_name
        out = out.rename(columns={sp_col: "spend"})[["date", "channel", "spend"]]

    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ── Google Ads ─────────────────────────────────────────────────────────────────

# Google Ads "Campaign report" CSV (downloaded from Reports → CSV) headers.
GOOGLE_DATE_COL = "Day"
GOOGLE_CAMPAIGN_COL = "Campaign"
GOOGLE_COST_COL = "Cost"
GOOGLE_COST_COL_FALLBACK = "Cost (currency)"


def load_google_ads_report_csv(
    source: PathOrBuffer,
    channel_name: str = "google_search",
    group_by: str = "day",
    cost_col: str | None = None,
) -> pd.DataFrame:
    """
    Load a Google Ads campaign-level CSV export.

    Behaves identically to :func:`load_meta_ads_insights_csv` — pick a default
    channel name (``'google_search'``) or split by campaign.
    """
    df = pd.read_csv(source, skiprows=_count_google_preamble_rows(source))
    if GOOGLE_DATE_COL not in df.columns:
        raise ValueError(f"Google Ads CSV missing column {GOOGLE_DATE_COL!r}")
    cost_c = cost_col or next(
        (c for c in (GOOGLE_COST_COL, GOOGLE_COST_COL_FALLBACK) if c in df.columns), None
    )
    if cost_c is None:
        raise ValueError(
            f"Google Ads CSV missing a cost column. Tried: {GOOGLE_COST_COL!r}, "
            f"{GOOGLE_COST_COL_FALLBACK!r}"
        )

    df["date"] = pd.to_datetime(df[GOOGLE_DATE_COL], errors="coerce")
    df[cost_c] = pd.to_numeric(df[cost_c], errors="coerce").fillna(0)

    if group_by == "campaign":
        if GOOGLE_CAMPAIGN_COL not in df.columns:
            raise ValueError(
                f"group_by='campaign' requires column {GOOGLE_CAMPAIGN_COL!r}"
            )
        out = df.groupby(["date", GOOGLE_CAMPAIGN_COL], as_index=False)[cost_c].sum()
        out = out.rename(columns={GOOGLE_CAMPAIGN_COL: "channel", cost_c: "spend"})
    else:
        out = df.groupby("date", as_index=False)[cost_c].sum()
        out["channel"] = channel_name
        out = out.rename(columns={cost_c: "spend"})[["date", "channel", "spend"]]

    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _count_google_preamble_rows(source: PathOrBuffer) -> int:
    """
    Google Ads CSV exports have 2 metadata rows above the header (report
    title + account info).  Sniff and skip them.
    """
    try:
        if hasattr(source, "seek"):
            pos = source.tell()
            head = "".join(source.readline() for _ in range(5))
            source.seek(pos)
        else:
            with open(source, encoding="utf-8") as f:
                head = "".join(f.readline() for _ in range(5))
    except Exception:
        return 0
    for i, line in enumerate(head.splitlines()):
        if GOOGLE_DATE_COL in line:
            return i
    return 0


# ── Amazon Ads (Sponsored Products / Sponsored Brands) ────────────────────────

# Amazon Advertising Console "Reports → Custom report" CSV columns.
# Casing matches what the console exports for performance reports.
AMAZON_DATE_COLS = ("Date", "Start Date", "Day")
AMAZON_CAMPAIGN_COL = "Campaign Name"
AMAZON_AD_GROUP_COL = "Ad Group Name"
AMAZON_ASIN_COL = "Advertised ASIN"   # falls back to ASIN, SKU
AMAZON_ASIN_FALLBACKS = ("ASIN", "Advertised SKU", "SKU")
AMAZON_SPEND_COL = "Spend"
AMAZON_SPEND_FALLBACKS = ("Cost", "Total Spend", "Ad Spend")
AMAZON_SALES_COL = "Sales"             # 7-day attributed sales (the inflated number)
AMAZON_SALES_FALLBACKS = ("7 Day Total Sales", "Attributed Sales 7d", "Total Sales")


def load_amazon_ads_csv(
    source: PathOrBuffer,
    group_by: str = "day",
    sku_aware: bool = False,
    channel_prefix: str = "amazon",
) -> pd.DataFrame:
    """
    Load an Amazon Advertising Console report (Sponsored Products /
    Sponsored Brands / Sponsored Display).

    Retail-media spend is best analysed at the **ASIN level** because
    incrementality varies enormously per SKU.  By default each ASIN
    becomes its own ``channel`` (so the analyser can compute per-SKU
    iROAS), but you can collapse to a single "amazon" channel by
    passing ``group_by='day'`` and ``sku_aware=False``.

    Parameters
    ----------
    source : path or file-like
        CSV export from Amazon Advertising Console → Reports.
    group_by : {"day", "campaign", "asin"}, default "day"
        - ``"day"``: one row per date, all spend collapsed to ``channel_prefix``.
        - ``"campaign"``: one row per (date, campaign).
        - ``"asin"``: one row per (date, ASIN) — the canonical retail-media shape.
    sku_aware : bool, default False
        If True, prefix every channel with the ASIN (overrides ``group_by``).
        Useful for downstream SKU-level holdout analysis.
    channel_prefix : str
        Base label when collapsing across products.

    Returns
    -------
    pandas.DataFrame  with columns ``date``, ``channel``, ``spend``.
    """
    df = pd.read_csv(source)
    date_col = next((c for c in AMAZON_DATE_COLS if c in df.columns), None)
    if date_col is None:
        raise ValueError(
            f"Amazon Ads CSV missing a date column. Tried: {AMAZON_DATE_COLS}"
        )
    sp_col = next(
        (c for c in (AMAZON_SPEND_COL, *AMAZON_SPEND_FALLBACKS) if c in df.columns),
        None,
    )
    if sp_col is None:
        raise ValueError(
            f"Amazon Ads CSV missing a spend column. Tried: "
            f"{(AMAZON_SPEND_COL, *AMAZON_SPEND_FALLBACKS)}"
        )

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df[sp_col] = pd.to_numeric(df[sp_col], errors="coerce").fillna(0)

    if sku_aware or group_by == "asin":
        asin_col = next(
            (c for c in (AMAZON_ASIN_COL, *AMAZON_ASIN_FALLBACKS) if c in df.columns),
            None,
        )
        if asin_col is None:
            raise ValueError(
                "ASIN-level grouping requires an ASIN or SKU column. "
                f"Tried: {(AMAZON_ASIN_COL, *AMAZON_ASIN_FALLBACKS)}"
            )
        out = df.groupby(["date", asin_col], as_index=False)[sp_col].sum()
        out["channel"] = channel_prefix + "_" + out[asin_col].astype(str)
        out = out.rename(columns={sp_col: "spend"})[["date", "channel", "spend"]]
    elif group_by == "campaign":
        if AMAZON_CAMPAIGN_COL not in df.columns:
            raise ValueError(
                f"group_by='campaign' requires column {AMAZON_CAMPAIGN_COL!r}"
            )
        out = df.groupby(["date", AMAZON_CAMPAIGN_COL], as_index=False)[sp_col].sum()
        out = out.rename(columns={AMAZON_CAMPAIGN_COL: "channel", sp_col: "spend"})
    else:
        out = df.groupby("date", as_index=False)[sp_col].sum()
        out["channel"] = channel_prefix
        out = out.rename(columns={sp_col: "spend"})[["date", "channel", "spend"]]

    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_amazon_sales_csv(
    source: PathOrBuffer,
    sales_col: str | None = None,
    sku_column: str = "asin",
) -> pd.DataFrame:
    """
    Load Amazon Business Reports → Detail Page Sales and Traffic by ASIN.

    Returns one row per (date, asin) with total ordered product sales.
    This is the *organic + paid* number — when joined to spend data and
    fed into causal-lift, the model separates the causal paid contribution.

    Parameters
    ----------
    source : path or file-like
    sales_col : str, optional
        Override the sales column name (defaults to first hit in
        :data:`AMAZON_SALES_FALLBACKS`).
    sku_column : str, default "asin"
        Name to use for the per-SKU column in the output.

    Returns
    -------
    pandas.DataFrame  with columns ``date``, ``sku_column`` (default "asin"),
    ``revenue``.
    """
    df = pd.read_csv(source)
    date_col = next((c for c in AMAZON_DATE_COLS if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"Amazon Sales CSV missing a date column from {AMAZON_DATE_COLS}")
    sl_col = sales_col or next(
        (c for c in (AMAZON_SALES_COL, *AMAZON_SALES_FALLBACKS) if c in df.columns),
        None,
    )
    if sl_col is None:
        raise ValueError(
            f"Amazon Sales CSV missing a sales column. Tried: "
            f"{(AMAZON_SALES_COL, *AMAZON_SALES_FALLBACKS)}"
        )

    asin_col = next(
        (c for c in (AMAZON_ASIN_COL, *AMAZON_ASIN_FALLBACKS) if c in df.columns),
        None,
    )
    if asin_col is None:
        raise ValueError(
            f"Amazon Sales CSV missing an ASIN/SKU column. Tried: "
            f"{(AMAZON_ASIN_COL, *AMAZON_ASIN_FALLBACKS)}"
        )

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df[sl_col] = pd.to_numeric(df[sl_col], errors="coerce").fillna(0)
    out = df.groupby(["date", asin_col], as_index=False)[sl_col].sum()
    out = out.rename(columns={asin_col: sku_column, sl_col: "revenue"})
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ── Walmart Connect ────────────────────────────────────────────────────────────

# Walmart Connect Ad Center export columns.
WALMART_DATE_COLS = ("Date", "Day")
WALMART_CAMPAIGN_COL = "Campaign Name"
WALMART_ITEM_COL = "Item ID"
WALMART_ITEM_FALLBACKS = ("Walmart Item ID", "WPID", "Product ID")
WALMART_SPEND_COL = "Ad Spend"
WALMART_SPEND_FALLBACKS = ("Spend", "Cost")
WALMART_SALES_COL = "Attributed Sales"
WALMART_SALES_FALLBACKS = ("Sales", "Total Sales")


def load_walmart_ads_csv(
    source: PathOrBuffer,
    group_by: str = "day",
    channel_name: str = "walmart_connect",
) -> pd.DataFrame:
    """
    Load a Walmart Connect Ad Center performance export.

    Parameters mirror :func:`load_amazon_ads_csv`.  Walmart's item-level
    reporting (``group_by="item"``) is the equivalent of ASIN-level on
    Amazon.

    Returns
    -------
    pandas.DataFrame  with columns ``date``, ``channel``, ``spend``.
    """
    df = pd.read_csv(source)
    date_col = next((c for c in WALMART_DATE_COLS if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"Walmart Ads CSV missing a date column from {WALMART_DATE_COLS}")
    sp_col = next(
        (c for c in (WALMART_SPEND_COL, *WALMART_SPEND_FALLBACKS) if c in df.columns),
        None,
    )
    if sp_col is None:
        raise ValueError(
            f"Walmart Ads CSV missing a spend column. Tried: "
            f"{(WALMART_SPEND_COL, *WALMART_SPEND_FALLBACKS)}"
        )

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df[sp_col] = pd.to_numeric(df[sp_col], errors="coerce").fillna(0)

    if group_by == "item":
        item_col = next(
            (c for c in (WALMART_ITEM_COL, *WALMART_ITEM_FALLBACKS) if c in df.columns),
            None,
        )
        if item_col is None:
            raise ValueError(
                f"Item-level grouping requires an Item ID column. Tried: "
                f"{(WALMART_ITEM_COL, *WALMART_ITEM_FALLBACKS)}"
            )
        out = df.groupby(["date", item_col], as_index=False)[sp_col].sum()
        out["channel"] = f"{channel_name}_" + out[item_col].astype(str)
        out = out.rename(columns={sp_col: "spend"})[["date", "channel", "spend"]]
    elif group_by == "campaign":
        if WALMART_CAMPAIGN_COL not in df.columns:
            raise ValueError(
                f"group_by='campaign' requires column {WALMART_CAMPAIGN_COL!r}"
            )
        out = df.groupby(["date", WALMART_CAMPAIGN_COL], as_index=False)[sp_col].sum()
        out = out.rename(columns={WALMART_CAMPAIGN_COL: "channel", sp_col: "spend"})
    else:
        out = df.groupby("date", as_index=False)[sp_col].sum()
        out["channel"] = channel_name
        out = out.rename(columns={sp_col: "spend"})[["date", "channel", "spend"]]

    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ── Live API stubs (v0.5 target) ───────────────────────────────────────────────

def fetch_shopify_orders_api(*_args, **_kwargs) -> pd.DataFrame:
    """Placeholder — direct Shopify REST API integration is v0.4 work."""
    raise NotImplementedError(
        "Direct Shopify API integration ships in v0.4. "
        "For now use load_shopify_orders_csv() with an exported CSV."
    )


def fetch_meta_ads_insights_api(*_args, **_kwargs) -> pd.DataFrame:
    """Placeholder — direct Meta Marketing API integration is v0.4 work."""
    raise NotImplementedError(
        "Direct Meta Ads API integration ships in v0.4. "
        "For now use load_meta_ads_insights_csv() with an exported CSV."
    )


def fetch_google_ads_report_api(*_args, **_kwargs) -> pd.DataFrame:
    """Placeholder — direct Google Ads API integration is v0.4 work."""
    raise NotImplementedError(
        "Direct Google Ads API integration ships in v0.4. "
        "For now use load_google_ads_report_csv() with an exported CSV."
    )
