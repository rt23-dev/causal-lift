"""
Synthetic data generator with known ground-truth incrementality.

Used for testing the library and for the hosted playground demo.  Each channel
includes a deliberate budget experiment (spend scale-up) that creates exogenous
variation — without these experiments, individual channel coefficients in
regression-based MMMs are not reliably identified.  This is by design: the
synthetic data is meant to demonstrate when the model works *and* what
real-world data needs to look like for the estimates to be trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


# Ground-truth incremental ROAS values baked into the data-generating process.
DEFAULT_TRUE_ROAS: Dict[str, float] = {
    "facebook": 2.0,   # below 3.33x breakeven at 30% margin → CUT
    "google": 4.5,     # above 3.33x breakeven → SCALE
    "tiktok": 0.8,     # weak; mostly brand awareness → CUT
}


@dataclass
class SyntheticData:
    """Bundle of generated DataFrames + ground-truth metadata."""

    spend_df: pd.DataFrame
    sales_df: pd.DataFrame
    ground_truth: Dict[str, float]
    note: str

    def __iter__(self):  # allows: spend_df, sales_df, gt = generate_synthetic_data()
        yield self.spend_df
        yield self.sales_df
        yield self.ground_truth


def generate_synthetic_data(
    n_days: int = 90,
    seed: int = 42,
    true_roas: Dict[str, float] | None = None,
) -> SyntheticData:
    """
    Generate `n_days` of synthetic ad spend + sales data with known iROAS.

    Parameters
    ----------
    n_days : int, default 90
    seed : int, default 42
    true_roas : dict, optional
        Override the default true iROAS per channel.

    Returns
    -------
    SyntheticData
        Has .spend_df, .sales_df, .ground_truth, .note.
        Also unpacks: ``spend_df, sales_df, gt = generate_synthetic_data()``.
    """
    rng = np.random.default_rng(seed)
    truth = true_roas if true_roas is not None else DEFAULT_TRUE_ROAS

    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    fb = rng.normal(3000, 600, n_days).clip(400, 9000)
    goog = rng.normal(2000, 350, n_days).clip(300, 5000)
    tiktok = rng.normal(800, 200, n_days).clip(100, 2500)

    # Budget experiments — exogenous spend variation for causal identification
    fb[49:63] *= 2.1      # Facebook scale-up, days 49-62
    goog[14:28] *= 1.7    # Google scale-up, days 14-27
    tiktok[28:36] *= 1.5  # TikTok experiment, days 28-35

    organic = 18_000 * (1.003 ** np.arange(n_days))
    dow_mult = np.where(dates.dayofweek >= 5, 0.75, 1.0)

    revenue = (
        organic * dow_mult
        + truth["facebook"] * fb
        + truth["google"] * goog
        + truth["tiktok"] * tiktok
        + rng.normal(0, 1_800, n_days)
    ).clip(0)

    orders = (revenue / 82).astype(int)

    spend_rows = []
    for i, d in enumerate(dates):
        spend_rows += [
            {"date": d, "channel": "facebook", "spend": round(float(fb[i]), 2)},
            {"date": d, "channel": "google", "spend": round(float(goog[i]), 2)},
            {"date": d, "channel": "tiktok", "spend": round(float(tiktok[i]), 2)},
        ]

    spend_df = pd.DataFrame(spend_rows)
    spend_df["date"] = pd.to_datetime(spend_df["date"])

    sales_df = pd.DataFrame(
        {
            "date": dates,
            "revenue": revenue.round(2),
            "orders": orders,
        }
    )

    return SyntheticData(
        spend_df=spend_df,
        sales_df=sales_df,
        ground_truth=truth,
        note=(
            "Each channel has a deliberate budget experiment (spend scale-up) that "
            "creates exogenous variation for causal identification. Without such "
            "experiments, per-channel iROAS estimates are not reliable."
        ),
    )
