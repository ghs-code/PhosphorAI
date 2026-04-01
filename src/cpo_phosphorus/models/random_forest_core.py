#!/usr/bin/env python3
"""
Random Forest with only the 5 core quality variables (same as OLS),
for fair comparison between OLS and RF on the same feature set.
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

from cpo_phosphorus.paths import LOCAL_PROCESSED_DATA_DIR, LOCAL_RF_CORE_REPORTS_DIR


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "feed_p_ppm"

SELECTED_FEATURES = [
    "feed_ffa_pct",
    "feed_dobi",
    "feed_iv",
    "feed_car_pv",
    "feed_mi_pct",
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
OUTPUT_PREFIX = "rf_5var"


def run_pipeline(input_path, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Load data — only keep 5 selected features
    df = pd.read_csv(input_path)
    y = df[TARGET_COL].copy()
    X = df[SELECTED_FEATURES].copy()

    # 2. Train/test split (same random_state as full RF)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        scoring=CV_SCORING,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 4. Evaluate
    def eval_metrics(model, X, y, name):
        y_pred = model.predict(X)
        return {
            "dataset": name,
            "r2": round(float(r2_score(y, y_pred)), 6),
            "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 6),
            "mae": round(float(mean_absolute_error(y, y_pred)), 6),
            "n_samples": int(len(y)),
        }

    train_metrics = eval_metrics(best_model, X_train, y_train, "train")
    test_metrics = eval_metrics(best_model, X_test, y_test, "test")

    # 5. Overfitting diagnosis
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

    overfitting_diag = {
        "r2_gap_train_minus_test": round(float(r2_gap), 6),
        "rmse_gap_test_minus_train": round(float(rmse_gap), 6),
        "diagnosis": status,
    }

    # 6. Feature importance
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    importance_df = pd.DataFrame([
        {"feature": SELECTED_FEATURES[i], "importance": round(float(importances[i]), 6)}
        for i in indices
    ])

    # 7. Predictions for plots
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # --- Export results ---
    results = {
        "model": "RandomForest_5var",
        "features_used": SELECTED_FEATURES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "best_hyperparameters": grid_search.best_params_,
        "cv_best_score_neg_mse": round(float(grid_search.best_score_), 6),
        "cv_best_rmse": round(float(np.sqrt(-grid_search.best_score_)), 6),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "overfitting_diagnosis": overfitting_diag,
    }

    with (output / f"{OUTPUT_PREFIX}_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    importance_df.to_csv(output / f"{OUTPUT_PREFIX}_feature_importance.csv", index=False)

    pd.DataFrame(grid_search.cv_results_).to_csv(
        output / f"{OUTPUT_PREFIX}_gridsearch_cv_results.csv", index=False
    )

    # --- Plots ---
    # Feature importance
    plot_df = importance_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#2196F3")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest (5 Variables) — Feature Importances")
    fig.tight_layout()
    fig.savefig(output / f"{OUTPUT_PREFIX}_feature_importance.png", dpi=150)
    plt.close(fig)

    # Actual vs Predicted
    for y_true, y_pred, name, suffix in [
        (y_test, y_test_pred, "Test Set", "test"),
        (y_train, y_train_pred, "Train Set", "train"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5, s=30)
        all_vals = np.concatenate([np.array(y_true), y_pred])
        lo, hi = all_vals.min(), all_vals.max()
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Actual feed_p_ppm")
        ax.set_ylabel("Predicted feed_p_ppm")
        ax.set_title(f"Actual vs Predicted — 5 Variables ({name})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{OUTPUT_PREFIX}_actual_vs_predicted_{suffix}.png", dpi=150)
        plt.close(fig)

    # Residuals
    for y_true, y_pred, name, suffix in [
        (y_test, y_test_pred, "Test Set", "test"),
        (y_train, y_train_pred, "Train Set", "train"),
    ]:
        residuals = np.array(y_true) - y_pred
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5, s=30)
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted feed_p_ppm")
        ax.set_ylabel("Residual")
        ax.set_title(f"Residuals — 5 Variables ({name})")
        fig.tight_layout()
        fig.savefig(output / f"{OUTPUT_PREFIX}_residuals_{suffix}.png", dpi=150)
        plt.close(fig)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run Random Forest on the 5 core variables")
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_ready.csv"),
        help=f"Path to model_ready.csv (default: {LOCAL_PROCESSED_DATA_DIR / 'model_ready.csv'})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_RF_CORE_REPORTS_DIR),
        help=f"Directory for output artifacts (default: {LOCAL_RF_CORE_REPORTS_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_pipeline(args.input, args.output_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
