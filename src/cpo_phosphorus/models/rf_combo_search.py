#!/usr/bin/env python3
"""
Exhaustive variable combination search using Random Forest.

Enumerates all valid combinations of candidate variables (respecting
mutually exclusive raw/log pairs), trains RF with simplified GridSearchCV,
and ranks combinations by CV RMSE.
"""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from cpo_phosphorus.paths import LOCAL_PROCESSED_DATA_DIR, LOCAL_RF_COMBO_REPORTS_DIR
from cpo_phosphorus.pipelines.sklearn_preprocessing import build_leakage_safe_preprocessor


RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_TARGET_COL = "feed_p_ppm"
DATE_COL = "date"
CV_FOLDS = 5
CV_SCORING = "neg_mean_squared_error"

# Simplified param grid (Plan A)
PARAM_GRID = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [None, 5, 10, 15, 20],
}

# Independent variables (always include or exclude freely)
INDEPENDENT_VARS = ["feed_dobi", "feed_iv", "feed_car_pv", "feed_mi_pct"]

# Mutually exclusive pairs: pick one or neither from each pair
EXCLUSIVE_PAIRS = [
    ("feed_ffa_pct", "log_feed_ffa_pct"),
]

OUTPUT_PREFIX = "rf_combo_search"


def _strip_model_prefix(params):
    return {key.replace("model__", "", 1): value for key, value in params.items()}


def generate_combinations():
    """Generate all valid variable combinations.

    For each exclusive pair, options are: var_a, var_b, or neither.
    For each independent var, options are: include or exclude.
    At least 1 variable must be selected.
    """
    # Options for each exclusive pair: [None, var_a, var_b]
    pair_options = []
    for var_a, var_b in EXCLUSIVE_PAIRS:
        pair_options.append([None, var_a, var_b])

    # Options for each independent var: [None, var]
    indep_options = []
    for var in INDEPENDENT_VARS:
        indep_options.append([None, var])

    all_options = pair_options + indep_options
    combos = []

    for selection in itertools.product(*all_options):
        features = [v for v in selection if v is not None]
        if len(features) >= 1:
            combos.append(sorted(features))

    # Remove duplicates (sorting ensures consistent order)
    seen = set()
    unique_combos = []
    for combo in combos:
        key = tuple(combo)
        if key not in seen:
            seen.add(key)
            unique_combos.append(combo)

    return unique_combos


def build_model_pipeline(features, random_state):
    preprocessor = build_leakage_safe_preprocessor(
        numeric_features=features,
        log_features=[("feed_ffa_pct", "log_feed_ffa_pct")],
        iqr_columns=["feed_dobi", "feed_ffa_pct", "feed_mi_pct", "feed_iv", "feed_car_pv"],
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(random_state=random_state)),
        ]
    )


def evaluate_combination(X_train, y_train, features, param_grid, cv_folds, random_state):
    """Run GridSearchCV for a single feature combination, return CV RMSE."""
    grid_search = GridSearchCV(
        estimator=build_model_pipeline(features, random_state),
        param_grid=param_grid,
        cv=cv_folds,
        scoring=CV_SCORING,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    cv_rmse = float(np.sqrt(-grid_search.best_score_))
    return {
        "features": features,
        "n_features": len(features),
        "cv_rmse": round(cv_rmse, 6),
        "cv_neg_mse": round(float(grid_search.best_score_), 6),
        "best_params": _strip_model_prefix(grid_search.best_params_),
    }


def run_combo_search(input_path, output_dir, target_col):
    """Run exhaustive combination search."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Load raw model source. Preprocessing is fitted inside each CV fold.
    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_path}")

    source_features = set(INDEPENDENT_VARS)
    for var_a, var_b in EXCLUSIVE_PAIRS:
        source_features.add(var_a)
        if var_b.startswith("log_"):
            source_features.add(var_a)
        else:
            source_features.add(var_b)
    source_features.add(DATE_COL)
    source_features.discard(target_col)
    source_features.discard("log_" + target_col)

    missing = [col for col in sorted(source_features) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna()
    X = df.loc[valid, sorted(source_features)].copy()
    y = y.loc[valid]

    # 2. Train/test split (same as other RF scripts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Generate all valid combinations
    combos = generate_combinations()
    print(f"Total combinations to evaluate: {len(combos)}")

    # 4. Evaluate each combination
    results = []
    start_time = time.time()

    for i, features in enumerate(combos, 1):
        combo_start = time.time()
        result = evaluate_combination(
            X_train, y_train, features, PARAM_GRID, CV_FOLDS, RANDOM_STATE
        )
        elapsed = time.time() - combo_start
        result["time_seconds"] = round(elapsed, 2)
        results.append(result)

        total_elapsed = time.time() - start_time
        print(
            f"[{i}/{len(combos)}] {', '.join(features)} "
            f"→ CV RMSE = {result['cv_rmse']:.4f} "
            f"({elapsed:.1f}s, total {total_elapsed:.0f}s)"
        )

    # 5. Sort by CV RMSE (ascending = better)
    results.sort(key=lambda r: r["cv_rmse"])

    # 6. Add rank
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    # 7. Export results
    # Full results JSON
    summary = {
        "total_combinations": len(combos),
        "param_grid": PARAM_GRID,
        "cv_folds": CV_FOLDS,
        "target_col": target_col,
        "preprocessing": "sklearn_pipeline_fit_on_train_folds",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "total_time_seconds": round(time.time() - start_time, 2),
        "top_10": results[:10],
        "all_results": results,
    }

    with (output / f"{OUTPUT_PREFIX}_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Ranked CSV (easier to read)
    rows = []
    for r in results:
        rows.append({
            "rank": r["rank"],
            "features": " + ".join(r["features"]),
            "n_features": r["n_features"],
            "cv_rmse": r["cv_rmse"],
            "best_n_estimators": r["best_params"]["n_estimators"],
            "best_max_depth": r["best_params"]["max_depth"],
        })
    ranking_df = pd.DataFrame(rows)
    ranking_df.to_csv(output / f"{OUTPUT_PREFIX}_ranking.csv", index=False)

    # 8. Plot: Top 20 combinations by CV RMSE
    plot_n = min(20, len(results))
    top = results[:plot_n]
    labels = [" + ".join(r["features"]) for r in reversed(top)]
    values = [r["cv_rmse"] for r in reversed(top)]

    fig, ax = plt.subplots(figsize=(10, max(6, plot_n * 0.35)))
    colors = ["#4CAF50" if i >= plot_n - 3 else "#2196F3" for i in range(plot_n)]
    ax.barh(labels, values, color=list(reversed(colors)))
    ax.set_xlabel("CV RMSE (lower is better)")
    ax.set_title(f"Top {plot_n} Variable Combinations — CV RMSE Ranking")
    fig.tight_layout()
    fig.savefig(output / f"{OUTPUT_PREFIX}_ranking.png", dpi=150)
    plt.close(fig)

    # 9. Print top 10
    print("\n" + "=" * 60)
    print("TOP 10 COMBINATIONS BY CV RMSE")
    print("=" * 60)
    for r in results[:10]:
        print(
            f"  #{r['rank']:2d}  RMSE={r['cv_rmse']:.4f}  "
            f"({r['n_features']} vars)  {' + '.join(r['features'])}"
        )
    print("=" * 60)

    return summary


def parse_args():
    default_target_col = os.getenv("CPO_TARGET_COL", DEFAULT_TARGET_COL)
    parser = argparse.ArgumentParser(description="Search valid Random Forest feature combinations")
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_source.csv"),
        help=f"Path to model_source.csv (default: {LOCAL_PROCESSED_DATA_DIR / 'model_source.csv'})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_RF_COMBO_REPORTS_DIR),
        help=f"Directory for output artifacts (default: {LOCAL_RF_COMBO_REPORTS_DIR})",
    )
    parser.add_argument(
        "--target-col",
        default=default_target_col,
        help=f"Target column to predict (default: {default_target_col})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_combo_search(args.input, args.output_dir, args.target_col)
    print(
        f"\nDone. {summary['total_combinations']} combinations evaluated "
        f"in {summary['total_time_seconds']:.0f}s."
    )


if __name__ == "__main__":
    main()
