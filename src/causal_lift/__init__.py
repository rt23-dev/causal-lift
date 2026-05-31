"""
causal-lift — lightweight, transparent marketing mix models for DTC brands.

Public API
----------
>>> import causal_lift as cl
>>> result = cl.analyze(spend_df, sales_df, contribution_margin=0.30)
>>> print(result.summary())

See ``RegressionMMM`` for the single-population model, ``GeoMMM`` for
multi-geo aggregation, and ``recommend_reallocation`` for budget-shift
suggestions.
"""

from causal_lift._version import __version__
from causal_lift.analyzer import AnalysisResult, BaseModel, ChannelResult
from causal_lift.data import load_sales_csv, load_spend_csv
from causal_lift.experiments import (
    GeoHoldoutDesign,
    GeoHoldoutResult,
    analyze_geo_holdout,
    design_geo_holdout,
)
from causal_lift.geo import GeoAnalysisResult, GeoMMM, detect_geo_column
from causal_lift.mmm import RegressionMMM
from causal_lift.optimiser import (
    ChannelRecommendation,
    ReallocationPlan,
    recommend_reallocation,
)
from causal_lift.synthetic import generate_synthetic_data

__all__ = [
    "__version__",
    # Core
    "analyze",
    "AnalysisResult",
    "BaseModel",
    "ChannelResult",
    "RegressionMMM",
    # Geo
    "GeoMMM",
    "GeoAnalysisResult",
    "detect_geo_column",
    # Budget optimisation
    "recommend_reallocation",
    "ReallocationPlan",
    "ChannelRecommendation",
    # Experiments (geo holdout design + analysis)
    "design_geo_holdout",
    "analyze_geo_holdout",
    "GeoHoldoutDesign",
    "GeoHoldoutResult",
    # Data loading
    "load_spend_csv",
    "load_sales_csv",
    # Demo data
    "generate_synthetic_data",
]


def analyze(
    spend_df,
    sales_df,
    contribution_margin: float = 0.30,
    model=None,
) -> AnalysisResult:
    """
    Run incrementality analysis on a (spend_df, sales_df) pair.

    Auto-detects whether the data is multi-geo and routes to ``GeoMMM`` if so,
    otherwise uses ``RegressionMMM``.

    Parameters
    ----------
    spend_df : pandas.DataFrame
        Columns: date, channel, spend.  Long-format daily / weekly ad spend.
        Optionally includes a geo column (any of: geo, region, dma, state,
        country).  Presence of a geo column auto-routes to ``GeoMMM``.
    sales_df : pandas.DataFrame
        Columns: date, revenue (orders optional, geo optional).
    contribution_margin : float, default 0.30
        Fraction of revenue that survives COGS, fulfilment, and returns.
        Drives the breakeven iROAS used for SCALE / HOLD / CUT / INCONCLUSIVE.
    model : BaseModel, optional
        Pre-instantiated analyzer.  Skips auto-routing.

    Returns
    -------
    AnalysisResult (or ``GeoAnalysisResult`` subclass for multi-geo inputs)
    """
    if model is None:
        if detect_geo_column(spend_df) and detect_geo_column(sales_df):
            model = GeoMMM()
        else:
            model = RegressionMMM()
    return model.fit(spend_df, sales_df, contribution_margin=contribution_margin)
