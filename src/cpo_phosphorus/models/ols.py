import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from cpo_phosphorus.paths import LOCAL_OLS_REPORTS_DIR, LOCAL_PROCESSED_DATA_DIR

DEFAULT_TARGET_COL = "feed_p_ppm"
INDEPENDENT_VARS = ["feed_ffa_pct", "feed_mi_pct", "feed_iv", "feed_dobi", "feed_car_pv"]
FIXED_VARS = ["time_trend", "missing_transition_phase"] + [f"month_{i}" for i in range(2, 13)]


def _get_available_fixed_vars(df):
    return [
        col
        for col in FIXED_VARS
        if col in df.columns and df[col].nunique(dropna=False) > 1
    ]


def _write_model_summary(output_path, header, adj_r2, model):
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f'"{header}"\n')
        f.write(f'"Adjusted R-squared: {adj_r2:.4f}"\n\n')
        f.write(model.summary().as_csv())


def run_pipeline(input_path, processed_dir, report_dir, target_col):
    processed_output = Path(processed_dir)
    report_output = Path(report_dir)
    processed_output.mkdir(parents=True, exist_ok=True)
    report_output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_path}")

    independent_vars = [col for col in INDEPENDENT_VARS if col in df.columns and col != target_col]
    if not independent_vars:
        raise ValueError("No available independent variables after excluding the target column.")

    fixed_vars = _get_available_fixed_vars(df)
    df_reg = df[[target_col] + independent_vars + fixed_vars].copy()
    y = df_reg[target_col]

    x_orig = sm.add_constant(df_reg[independent_vars])
    model_orig = sm.OLS(y, x_orig).fit()

    _write_model_summary(
        report_output / "OLSresult_Original.csv",
        f"Target: {target_col}; Original Variables: {', '.join(independent_vars)}",
        model_orig.rsquared_adj,
        model_orig,
    )

    best_adj_r2 = -float("inf")
    best_model = None
    best_vars = []
    run_log = []

    for r in range(0, len(independent_vars) + 1):
        for combo in combinations(independent_vars, r):
            current_vars = fixed_vars + list(combo)
            x_combo = sm.add_constant(df_reg[current_vars])
            model = sm.OLS(y, x_combo).fit()

            if model.rsquared_adj > best_adj_r2:
                best_adj_r2 = model.rsquared_adj
                best_model = model
                best_vars = current_vars

            run_log.append(
                {
                    "variables": current_vars,
                    "rsquared": model.rsquared,
                    "rsquared_adj": model.rsquared_adj,
                    "aic": model.aic,
                    "bic": model.bic,
                }
            )

    _write_model_summary(
        report_output / "OLSresult_Time_Series.csv",
        f"Target: {target_col}; Best Independent Variables Selected: {', '.join(best_vars)}",
        best_adj_r2,
        best_model,
    )

    with (report_output / "OLSresult_Time_Series_log.json").open("w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by="date").reset_index(drop=True)
        lag_col = f"{target_col}_lag1"
        df[lag_col] = df[target_col].shift(1)
        df["date_diff"] = df["date"].diff().dt.days
        df.loc[df["date_diff"] > 2, lag_col] = pd.NA
    else:
        lag_col = f"{target_col}_lag1"
        df[lag_col] = df[target_col].shift(1)

    df_lag = df.dropna(subset=[lag_col]).copy()
    if "date_diff" in df_lag.columns:
        df_lag = df_lag.drop(columns=["date_diff"])

    lag_data_path = processed_output / "model_ready_lag.csv"
    df_lag.to_csv(lag_data_path, index=False)

    lag_independent_vars = best_vars + [lag_col]
    y_lag = df_lag[target_col]
    x_lag = sm.add_constant(df_lag[lag_independent_vars])
    model_lag = sm.OLS(y_lag, x_lag).fit()

    _write_model_summary(
        report_output / "OLSresult_lag.csv",
        f"Target: {target_col}; Model Variables (Optimal + Lag): {', '.join(lag_independent_vars)}",
        model_lag.rsquared_adj,
        model_lag,
    )

    return {
        "dependent_var": target_col,
        "fixed_vars_used": fixed_vars,
        "best_vars": best_vars,
        "best_adj_r2": round(float(best_adj_r2), 6),
        "lag_output": str(lag_data_path),
    }


def parse_args():
    default_target_col = os.getenv("CPO_TARGET_COL", DEFAULT_TARGET_COL)
    parser = argparse.ArgumentParser(description="Run OLS baseline and lagged models")
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_ready.csv"),
        help=f"Path to model_ready.csv (default: {LOCAL_PROCESSED_DATA_DIR / 'model_ready.csv'})",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(LOCAL_PROCESSED_DATA_DIR),
        help=f"Directory for processed datasets (default: {LOCAL_PROCESSED_DATA_DIR})",
    )
    parser.add_argument(
        "--report-dir",
        default=str(LOCAL_OLS_REPORTS_DIR),
        help=f"Directory for OLS reports (default: {LOCAL_OLS_REPORTS_DIR})",
    )
    parser.add_argument(
        "--target-col",
        default=default_target_col,
        help=f"Target column to predict (default: {default_target_col})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_pipeline(
        input_path=args.input,
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        target_col=args.target_col,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
