"""Command-line interface for causal-lift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from causal_lift import __version__, analyze, generate_synthetic_data
from causal_lift.data import CSVValidationError, load_sales_csv, load_spend_csv


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        spend_df = load_spend_csv(args.spend_csv)
        sales_df = load_sales_csv(args.sales_csv)
    except CSVValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        result = analyze(spend_df, sales_df, contribution_margin=args.margin)
    except Exception as exc:  # noqa: BLE001
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() == ".csv":
            result.to_dataframe().to_csv(out_path, index=False)
        else:
            out_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nresults written to {out_path}", file=sys.stderr)

    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    data = generate_synthetic_data(n_days=args.days, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spend_path = out_dir / "spend.csv"
    sales_path = out_dir / "sales.csv"
    data.spend_df.assign(date=data.spend_df["date"].dt.strftime("%Y-%m-%d")).to_csv(
        spend_path, index=False
    )
    data.sales_df.assign(date=data.sales_df["date"].dt.strftime("%Y-%m-%d")).to_csv(
        sales_path, index=False
    )

    print(f"wrote {len(data.spend_df)} spend rows -> {spend_path}")
    print(f"wrote {len(data.sales_df)} sales rows -> {sales_path}")
    print("\nground-truth incremental ROAS (what the model should recover):")
    for ch, roas in data.ground_truth.items():
        print(f"  {ch:<12} {roas:.1f}x")
    print(f"\nnote: {data.note}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-lift",
        description="Lightweight, transparent marketing mix models for DTC brands.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"causal-lift {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_analyze = sub.add_parser(
        "analyze",
        help="Run incrementality analysis on spend + sales CSVs.",
        description="Run a regression-based MMM on the given spend.csv and sales.csv files.",
    )
    p_analyze.add_argument("spend_csv", help="Path to spend CSV (columns: date, channel, spend)")
    p_analyze.add_argument("sales_csv", help="Path to sales CSV (columns: date, revenue [, orders])")
    p_analyze.add_argument(
        "-m", "--margin",
        type=float, default=0.30,
        help="Contribution margin (0.05–0.95). Default 0.30. "
             "Drives breakeven iROAS = 1 / margin.",
    )
    p_analyze.add_argument("--json", action="store_true", help="Output JSON instead of text summary.")
    p_analyze.add_argument("-o", "--output", help="Write results to file (.csv or .json).")
    p_analyze.set_defaults(func=cmd_analyze)

    p_sample = sub.add_parser(
        "sample",
        help="Generate synthetic sample data with known ground-truth iROAS.",
    )
    p_sample.add_argument("--days", type=int, default=90, help="Number of days. Default 90.")
    p_sample.add_argument("--seed", type=int, default=42, help="RNG seed. Default 42.")
    p_sample.add_argument(
        "--out-dir", default=".",
        help="Output directory for spend.csv and sales.csv. Default '.'",
    )
    p_sample.set_defaults(func=cmd_sample)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
