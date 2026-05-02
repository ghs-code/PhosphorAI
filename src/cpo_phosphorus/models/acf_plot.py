import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

from cpo_phosphorus.paths import LOCAL_OLS_REPORTS_DIR, LOCAL_PROCESSED_DATA_DIR

DEFAULT_TARGET_COL = "feed_p_ppm"


def run_plot(input_path, output_path, target_col):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_path}")

    target_series = df[target_col].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(target_series, lags=40, alpha=0.05, zero=True, ax=ax)
    ax.set_title(f"Autocorrelation Function (ACF) of Daily {target_col}")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    plt.tight_layout()
    plt.savefig(output, format="jpg", dpi=300)
    plt.close()
    return output


def parse_args():
    default_target_col = os.getenv("CPO_TARGET_COL", DEFAULT_TARGET_COL)
    parser = argparse.ArgumentParser(description="Generate ACF plot for lagged target data")
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_ready_lag.csv"),
        help=f"Path to lagged dataset (default: {LOCAL_PROCESSED_DATA_DIR / 'model_ready_lag.csv'})",
    )
    parser.add_argument(
        "--output",
        default=str(LOCAL_OLS_REPORTS_DIR / "acf_plot.jpg"),
        help=f"Path to output plot (default: {LOCAL_OLS_REPORTS_DIR / 'acf_plot.jpg'})",
    )
    parser.add_argument(
        "--target-col",
        default=default_target_col,
        help=f"Target column to plot (default: {default_target_col})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = run_plot(args.input, args.output, args.target_col)
    print(f"ACF plot successfully saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
