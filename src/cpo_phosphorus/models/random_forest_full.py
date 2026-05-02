#!/usr/bin/env python3
"""
Random Forest regression pipeline for Wilmar CPO phosphorus capstone project.

Reads model_source.csv, performs 80/20 train/test split, fits preprocessing
inside an sklearn Pipeline during GridSearchCV, trains a RandomForestRegressor,
evaluates on train/test sets, and exports metrics + plots.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from cpo_phosphorus.paths import LOCAL_PROCESSED_DATA_DIR, LOCAL_RF_FULL_REPORTS_DIR
from cpo_phosphorus.pipelines.sklearn_preprocessing import (
    build_leakage_safe_preprocessor,
    get_preprocessed_feature_names,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_TARGET_COL = "feed_p_ppm"
DATE_COL = "date"
NUMERIC_FEATURES = [
    "feed_ffa_pct",
    "feed_mi_pct",
    "feed_iv",
    "feed_dobi",
    "feed_car_pv",
    "log_feed_ffa_pct",
    "time_trend",
]
CATEGORICAL_FEATURES = [
    "feed_tank",
    "feed_type",
]
SOURCE_FEATURES = [
    DATE_COL,
    "feed_ffa_pct",
    "feed_mi_pct",
    "feed_iv",
    "feed_dobi",
    "feed_car_pv",
    "feed_tank",
    "feed_type",
]

PARAM_GRID = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [None, 5, 10, 15, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None],
}

CV_FOLDS = 5
CV_SCORING = "neg_mean_squared_error"


def _strip_model_prefix(params):
    return {key.replace("model__", "", 1): value for key, value in params.items()}


def _target_safe_features(target_col):
    target_derived = {target_col, "log_" + target_col}
    numeric_features = [col for col in NUMERIC_FEATURES if col not in target_derived]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col != target_col]
    source_features = [
        col
        for col in SOURCE_FEATURES
        if col != target_col and col not in target_derived
    ]
    return numeric_features, categorical_features, source_features


def load_data(input_path, target_col):
    """Load model_source.csv and split into raw features and target."""
    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_path}")

    numeric_features, categorical_features, source_features = _target_safe_features(target_col)
    missing = [col for col in source_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna()
    X = df.loc[valid, source_features].copy()
    return X, y.loc[valid], numeric_features, categorical_features


def build_model_pipeline(random_state=RANDOM_STATE, target_col=DEFAULT_TARGET_COL):
    numeric_features, categorical_features, _ = _target_safe_features(target_col)
    log_features = []
    if "log_feed_ffa_pct" in numeric_features:
        log_features.append(("feed_ffa_pct", "log_feed_ffa_pct"))

    preprocessor = build_leakage_safe_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        log_features=log_features,
        include_month=True,
        iqr_columns=["feed_dobi", "feed_ffa_pct", "feed_mi_pct", "feed_iv", "feed_car_pv"],
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(random_state=random_state)),
        ]
    )


def split_data(X, y, test_size, random_state):
    """80/20 train/test split with fixed random state."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def tune_hyperparameters(X_train, y_train, param_grid, cv_folds, scoring, random_state, target_col):
    """GridSearchCV to find best hyperparameters."""
    grid_search = GridSearchCV(
        estimator=build_model_pipeline(random_state, target_col),
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
    rf_model = model.named_steps["model"]
    importances = rf_model.feature_importances_
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


def plot_actual_vs_predicted(y_true, y_pred, dataset_name, output_path, target_col):
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5, s=30)

    all_vals = np.concatenate([y_true, y_pred])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "r--", linewidth=1, label="Perfect prediction")

    ax.set_xlabel(f"Actual {target_col}")
    ax.set_ylabel(f"Predicted {target_col}")
    ax.set_title(f"Actual vs Predicted ({dataset_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, dataset_name, output_path, target_col):
    """Residual plot (predicted vs residual)."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5, s=30)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Predicted {target_col}")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residuals ({dataset_name})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_results_summary(best_params, train_metrics, test_metrics, overfitting_diag, grid_search, target_col):
    """Assemble full results dict for JSON export."""
    return {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "cv_scoring": CV_SCORING,
        "target_col": target_col,
        "preprocessing": "sklearn_pipeline_fit_on_train_folds",
        "best_hyperparameters": best_params,
        "cv_best_score_neg_mse": round(float(grid_search.best_score_), 6),
        "cv_best_rmse": round(float(np.sqrt(-grid_search.best_score_)), 6),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "overfitting_diagnosis": overfitting_diag,
    }


def run_pipeline(input_path, output_dir, target_col):
    """Execute the full Random Forest pipeline."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X, y, numeric_features, categorical_features = load_data(input_path, target_col)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)

    # 3. Hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=build_model_pipeline(RANDOM_STATE, target_col),
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        scoring=CV_SCORING,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_params_clean = _strip_model_prefix(best_params)
    best_model = grid_search.best_estimator_

    # 4. Evaluate on train and test
    train_metrics = evaluate_model(best_model, X_train, y_train, "train")
    test_metrics = evaluate_model(best_model, X_test, y_test, "test")

    # 5. Overfitting diagnosis
    overfitting_diag = diagnose_overfitting(train_metrics, test_metrics)

    # 6. Feature importance
    feature_names = get_preprocessed_feature_names(best_model.named_steps["preprocess"])
    importance_df = extract_feature_importance(best_model, feature_names)

    # 7. Predictions for plots
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # --- Export results ---
    importance_df.to_csv(output / "rf_feature_importance.csv", index=False)

    results = build_results_summary(best_params_clean, train_metrics, test_metrics, overfitting_diag, grid_search, target_col)
    results["features_used"] = numeric_features + categorical_features
    with (output / "rf_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # --- Export GridSearchCV details ---
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df.to_csv(output / "rf_gridsearch_cv_results.csv", index=False)

    # --- Plots ---
    plot_feature_importance(importance_df, output / "rf_feature_importance.png")
    plot_actual_vs_predicted(y_test, y_test_pred, "Test Set", output / "rf_actual_vs_predicted_test.png", target_col)
    plot_actual_vs_predicted(y_train, y_train_pred, "Train Set", output / "rf_actual_vs_predicted_train.png", target_col)
    plot_residuals(y_test, y_test_pred, "Test Set", output / "rf_residuals_test.png", target_col)
    plot_residuals(y_train, y_train_pred, "Train Set", output / "rf_residuals_train.png", target_col)

    return results


def parse_args():
    default_target_col = os.getenv("CPO_TARGET_COL", DEFAULT_TARGET_COL)
    parser = argparse.ArgumentParser(description="Random Forest pipeline for CPO phosphorus prediction")
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_source.csv"),
        help=f"Path to model_source.csv (default: {LOCAL_PROCESSED_DATA_DIR / 'model_source.csv'})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_RF_FULL_REPORTS_DIR),
        help=f"Directory for output artifacts (default: {LOCAL_RF_FULL_REPORTS_DIR})",
    )
    parser.add_argument(
        "--target-col",
        default=default_target_col,
        help=f"Target column to predict (default: {default_target_col})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_pipeline(input_path=args.input, output_dir=args.output_dir, target_col=args.target_col)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
