"""
causal-lift — lightweight, transparent marketing mix models for DTC brands.

Public API
----------
>>> import causal_lift as cl
>>> result = cl.analyze(spend_df, sales_df, contribution_margin=0.30)
>>> print(result.summary())

See `RegressionMMM` for the underlying model and `generate_synthetic_data` for
ground-truth-known test data.
"""

from causal_lift._version import __version__
from causal_lift.analyzer import AnalysisResult, BaseModel, ChannelResult
from causal_lift.data import load_sales_csv, load_spend_csv
from causal_lift.mmm import RegressionMMM
from causal_lift.synthetic import generate_synthetic_data

__all__ = [
    "__version__",
    "analyze",
    "AnalysisResult",
    "BaseModel",
    "ChannelResult",
    "RegressionMMM",
    "generate_synthetic_data",
    "load_spend_csv",
    "load_sales_csv",
]


def analyze(spend_df, sales_df, contribution_margin: float = 0.30, model=None) -> AnalysisResult:
    """
    Run incrementality analysis on a (spend_df, sales_df) pair.

    Parameters
    ----------
    spend_df : pandas.DataFrame
        Columns: date, channel, spend.  Long-format daily ad spend.
    sales_df : pandas.DataFrame
        Columns: date, revenue (orders optional).  Daily aggregate revenue.
    contribution_margin : float, default 0.30
        Fraction of revenue that survives COGS, fulfilment, and returns.
        Drives the breakeven iROAS used for SCALE / HOLD / CUT thresholds.
    model : BaseModel, optional
        Pre-instantiated analyzer.  Defaults to `RegressionMMM()`.

    Returns
    -------
    AnalysisResult
        Per-channel iROAS estimates with HAC confidence intervals, VIF
        diagnostics, and margin-aware recommendations.
    """
    if model is None:
        model = RegressionMMM()
    return model.fit(spend_df, sales_df, contribution_margin=contribution_margin)
