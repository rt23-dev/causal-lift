"""
Run analysis directly from CSV files.

This example shows the CSV → DataFrame → AnalysisResult flow, which is the
most common path for real-world usage.

Run:
    python -m causal_lift.cli sample --out-dir ./example_data   # writes spend.csv + sales.csv
    python examples/from_csv.py ./example_data/spend.csv ./example_data/sales.csv
"""

import argparse
import sys

import causal_lift as cl
from causal_lift.data import load_sales_csv, load_spend_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spend_csv")
    parser.add_argument("sales_csv")
    parser.add_argument("--margin", type=float, default=0.30)
    args = parser.parse_args(argv)

    spend_df = load_spend_csv(args.spend_csv)
    sales_df = load_sales_csv(args.sales_csv)
    result = cl.analyze(spend_df, sales_df, contribution_margin=args.margin)

    print(result.summary())

    # Optional: dump everything as JSON
    # import json
    # print(json.dumps(result.to_dict(), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
