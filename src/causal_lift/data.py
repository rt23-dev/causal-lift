"""CSV ingestion and validation — framework-agnostic (no FastAPI)."""

from __future__ import annotations

from typing import IO, Union

import pandas as pd

PathOrBuffer = Union[str, "IO[str]", "IO[bytes]"]


class CSVValidationError(ValueError):
    """Raised when a CSV file fails schema or content validation."""


def load_spend_csv(source: PathOrBuffer) -> pd.DataFrame:
    """
    Load and validate a spend CSV.

    Required columns: date, channel, spend.
    Channel names are lowercased, stripped, and sanitised
    (non-word characters → underscore) for OLS column-name safety.

    Parameters
    ----------
    source : path or file-like
        Anything `pandas.read_csv` accepts.

    Returns
    -------
    pandas.DataFrame  with columns [date, channel, spend], sorted by date.
    """
    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise CSVValidationError(f"Could not parse spend CSV: {exc}") from exc

    df.columns = df.columns.str.lower().str.strip()
    missing = {"date", "channel", "spend"} - set(df.columns)
    if missing:
        raise CSVValidationError(
            f"Spend CSV is missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["spend"] = pd.to_numeric(df["spend"], errors="coerce")
    df["channel"] = (
        df["channel"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\-]", "_", regex=True)
    )
    df = df.dropna(subset=["date", "spend"])

    if df.empty:
        raise CSVValidationError("Spend CSV has no valid rows after parsing.")
    if df["spend"].min() < 0:
        raise CSVValidationError("Spend CSV contains negative spend values.")

    return df[["date", "channel", "spend"]].sort_values("date").reset_index(drop=True)


def load_sales_csv(source: PathOrBuffer) -> pd.DataFrame:
    """
    Load and validate a sales CSV.

    Required columns: date, revenue.  Optional: orders.

    Returns
    -------
    pandas.DataFrame  with columns [date, revenue, (orders)], sorted by date.
    """
    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise CSVValidationError(f"Could not parse sales CSV: {exc}") from exc

    df.columns = df.columns.str.lower().str.strip()

    for col in ("date", "revenue"):
        if col not in df.columns:
            raise CSVValidationError(
                f"Sales CSV must have a '{col}' column. "
                f"Found: {sorted(df.columns.tolist())}"
            )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["date", "revenue"])

    if df.empty:
        raise CSVValidationError("Sales CSV has no valid rows after parsing.")
    if df["revenue"].min() < 0:
        raise CSVValidationError("Sales CSV contains negative revenue values.")

    cols = ["date", "revenue"]
    if "orders" in df.columns:
        df["orders"] = pd.to_numeric(df["orders"], errors="coerce")
        cols.append("orders")

    return df[cols].sort_values("date").reset_index(drop=True)
