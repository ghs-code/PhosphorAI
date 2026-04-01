#!/usr/bin/env python3
"""
Random Forest regression pipeline for Wilmar CPO phosphorus capstone project.

Reads the preprocessed model_ready.csv, performs 80/20 train/test split,
tunes hyperparameters via GridSearchCV, trains a RandomForestRegressor,
evaluates on train/test sets, and exports metrics + plots.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from cpo_phosphorus.paths import LOCAL_PROCESSED_DATA_DIR, LOCAL_RF_FULL_REPORTS_DIR


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "feed_p_ppm"
DROP_COLS = [
    "date",
    "rbd_p_ppm",                    # downstream measurement (refined product)
    "acid_dosing_pct",              # process variable (downstream decision)
    "bleaching_earth_dosing_pct",   # process variable (downstream decision)
    "log_feed_p_ppm",               # log of target variable
    "log_feed_ffa_pct",             # duplicate of feed_ffa_pct (avoid splitting importance)
]

PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

CV_FOLDS = 5
CV_SCORING = "neg_mean_squared_error"


def load_data(input_path, target_col):
    """Load model_ready.csv and split into features and target."""
    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_path}")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col] + [c for c in DROP_COLS if c in df.columns])
    return X, y


def split_data(X, y, test_size, random_state):
    """80/20 train/test split with fixed random state."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def tune_hyperparameters(X_train, y_train, param_grid, cv_folds, scoring, random_state):
    """GridSearchCV to find best hyperparameters."""
    base_model = RandomForestRegressor(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(model, X, y, dataset_name):
    """Compute R², RMSE, MAE for a given dataset."""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return {
        "dataset": dataset_name,
        "r2": round(float(r2), 6),
        "rmse": round(float(rmse), 6),
        "mae": round(float(mae), 6),
        "n_samples": int(len(y)),
    }


def diagnose_overfitting(train_metrics, test_metrics):
    """Compare train vs test metrics to diagnose overfitting."""
    r2_gap = train_metrics["r2"] - test_metrics["r2"]
    rmse_gap = test_metrics["rmse"] - train_metrics["rmse"]

    if train_metrics["r2"] < 0.50 and test_metrics["r2"] < 0.50:
        status = "underfitting"
    elif r2_gap > 0.20:
        status = "overfitting"
    elif r2_gap < 0.05:
        status = "good_generalization"
    elif r2_gap < 0.10:
        status = "acceptable_generalization"
    else:
        status = "moderate_gap"

    return {
        "r2_gap_train_minus_test": round(float(r2_gap), 6),
        "rmse_gap_test_minus_train": round(float(rmse_gap), 6),
        "diagnosis": status,
    }


def extract_feature_importance(model, feature_names):
    """Extract and sort feature importances from trained RF model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    rows = []
    for i in indices:
        rows.append({
            "feature": feature_names[i],
            "importance": round(float(importances[i]), 6),
        })
    return pd.DataFrame(rows)


def plot_feature_importance(importance_df, output_path, top_n=15):
    """Bar chart of top-N feature importances."""
    plot_df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#2196F3")
    ax.set_xlabel("Importance")
    ax.set_title(f"Random Forest — Top {min(top_n, len(plot_df))} Feature Importances")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, dataset_name, output_path):
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5, s=30)

    all_vals = np.concatenate([y_true, y_pred])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "r--", linewidth=1, label="Perfect prediction")

    ax.set_xlabel("Actual feed_p_ppm")
    ax.set_ylabel("Predicted feed_p_ppm")
    ax.set_title(f"Actual vs Predicted ({dataset_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, dataset_name, output_path):
    """Residual plot (predicted vs residual)."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5, s=30)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted feed_p_ppm")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residuals ({dataset_name})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_results_summary(best_params, train_metrics, test_metrics, overfitting_diag, grid_search):
    """Assemble full results dict for JSON export."""
    return {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "cv_scoring": CV_SCORING,
        "best_hyperparameters": best_params,
        "cv_best_score_neg_mse": round(float(grid_search.best_score_), 6),
        "cv_best_rmse": round(float(np.sqrt(-grid_search.best_score_)), 6),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "overfitting_diagnosis": overfitting_diag,
    }


def run_pipeline(input_path, output_dir):
    """Execute the full Random Forest pipeline."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X, y = load_data(input_path, TARGET_COL)
    feature_names = list(X.columns)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)

    # 3. Hyperparameter tuning
    grid_search = tune_hyperparameters(
        X_train, y_train, PARAM_GRID, CV_FOLDS, CV_SCORING, RANDOM_STATE
    )
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # 4. Evaluate on train and test
    train_metrics = evaluate_model(best_model, X_train, y_train, "train")
    test_metrics = evaluate_model(best_model, X_test, y_test, "test")

    # 5. Overfitting diagnosis
    overfitting_diag = diagnose_overfitting(train_metrics, test_metrics)

    # 6. Feature importance
    importance_df = extract_feature_importance(best_model, feature_names)

    # 7. Predictions for plots
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # --- Export results ---
    importance_df.to_csv(output / "rf_feature_importance.csv", index=False)

    results = build_results_summary(best_params, train_metrics, test_metrics, overfitting_diag, grid_search)
    with (output / "rf_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # --- Export GridSearchCV details ---
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df.to_csv(output / "rf_gridsearch_cv_results.csv", index=False)

    # --- Plots ---
    plot_feature_importance(importance_df, output / "rf_feature_importance.png")
    plot_actual_vs_predicted(y_test, y_test_pred, "Test Set", output / "rf_actual_vs_predicted_test.png")
    plot_actual_vs_predicted(y_train, y_train_pred, "Train Set", output / "rf_actual_vs_predicted_train.png")
    plot_residuals(y_test, y_test_pred, "Test Set", output / "rf_residuals_test.png")
    plot_residuals(y_train, y_train_pred, "Train Set", output / "rf_residuals_train.png")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest pipeline for CPO phosphorus prediction")
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_ready.csv"),
        help=f"Path to model_ready.csv (default: {LOCAL_PROCESSED_DATA_DIR / 'model_ready.csv'})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_RF_FULL_REPORTS_DIR),
        help=f"Directory for output artifacts (default: {LOCAL_RF_FULL_REPORTS_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_pipeline(input_path=args.input, output_dir=args.output_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
