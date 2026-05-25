"""
Quickstart example for causal-lift.

Generate synthetic data, run the analysis, print the summary.
This is the simplest end-to-end usage of the library.

Run:
    python examples/quickstart.py
"""

import causal_lift as cl


def main() -> None:
    # 1. Generate 90 days of synthetic spend + sales data with known ground truth
    data = cl.generate_synthetic_data(n_days=90, seed=42)

    print("Ground truth (true iROAS baked into the DGP):")
    for ch, roas in data.ground_truth.items():
        print(f"  {ch:<10} {roas:.1f}x")
    print()

    # 2. Run the analysis at 30% contribution margin (breakeven iROAS = 3.33x)
    result = cl.analyze(data.spend_df, data.sales_df, contribution_margin=0.30)

    # 3. Print the human-readable summary
    print(result.summary())

    # 4. Per-channel results as a DataFrame for further analysis
    print("\nAs DataFrame:")
    print(result.to_dataframe()[
        ["channel", "incremental_roas", "confidence_interval", "vif_score", "recommendation"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
