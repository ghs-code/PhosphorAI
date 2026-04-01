import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

from cpo_phosphorus.paths import LOCAL_OLS_REPORTS_DIR, LOCAL_PROCESSED_DATA_DIR


def run_plot(input_path, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    feed_p_ppm = df["feed_p_ppm"].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(feed_p_ppm, lags=40, alpha=0.05, zero=True, ax=ax)
    ax.set_title("Autocorrelation Function (ACF) of Daily Feed Phosphorus Content (ppm)")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    plt.tight_layout()
    plt.savefig(output, format="jpg", dpi=300)
    plt.close()
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ACF plot for lagged feed phosphorus data")
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
    return parser.parse_args()


def main():
    args = parse_args()
    output = run_plot(args.input, args.output)
    print(f"ACF plot successfully saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
