#!/usr/bin/env python3
"""Optimized feed-oil phosphorus prediction model comparison.

This module keeps the sponsor-confirmed target fixed to feed oil phosphorus
(`feed_p_ppm`) and compares conservative feature groups under a shared
preprocessing and evaluation protocol. It is intended to improve explanatory
power without using post-refining variables or target-derived leakage.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cpo_phosphorus.paths import LOCAL_PROCESSED_DATA_DIR, LOCAL_REPORTS_DIR
from cpo_phosphorus.pipelines.sklearn_preprocessing import (
    build_leakage_safe_preprocessor,
    get_preprocessed_feature_names,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_SCORING = "neg_root_mean_squared_error"
DEFAULT_TARGET_COL = "feed_p_ppm"
DATE_COL = "date"
N_JOBS = int(os.getenv("CPO_MODEL_N_JOBS", "1"))
RISK_QUANTILES = [0.75, 0.80, 0.90]
TOP_K_FRACTIONS = [0.05, 0.10, 0.20]

QUALITY_FEATURES = [
    "feed_ffa_pct",
    "feed_mi_pct",
    "feed_iv",
    "feed_dobi",
    "feed_car_pv",
]
CATEGORICAL_FEATURES = ["feed_tank", "feed_type", "source_file"]
LAG_FEATURES = ["feed_p_ppm_lag1", "feed_p_ppm_roll3"]
IQR_COLUMNS = QUALITY_FEATURES


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    numeric_features: list[str]
    categorical_features: list[str]
    include_month: bool

    @property
    def source_features(self):
        source = {DATE_COL}
        for feature in self.numeric_features:
            if feature == "time_trend":
                continue
            if feature.startswith("log_feed_ffa_pct"):
                source.add("feed_ffa_pct")
            else:
                source.add(feature)
        source.update(self.categorical_features)
        return sorted(source)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict
    scale_features: bool = False


FEATURE_GROUPS = [
    FeatureGroup(
        name="quality_only",
        numeric_features=QUALITY_FEATURES,
        categorical_features=[],
        include_month=False,
    ),
    FeatureGroup(
        name="context_aware",
        numeric_features=QUALITY_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        include_month=True,
    ),
    FeatureGroup(
        name="time_aware",
        numeric_features=QUALITY_FEATURES + ["time_trend"],
        categorical_features=CATEGORICAL_FEATURES,
        include_month=True,
    ),
    FeatureGroup(
        name="lag_aware",
        numeric_features=QUALITY_FEATURES + ["time_trend", "feed_p_ppm_lag1"],
        categorical_features=CATEGORICAL_FEATURES,
        include_month=True,
    ),
    FeatureGroup(
        name="lag_roll3_aware",
        numeric_features=QUALITY_FEATURES + ["time_trend"] + LAG_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        include_month=True,
    ),
]

MODEL_SPECS = [
    ModelSpec(
        name="linear_regression",
        estimator=LinearRegression(),
        param_grid={},
        scale_features=True,
    ),
    ModelSpec(
        name="ridge",
        estimator=Ridge(),
        param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        scale_features=True,
    ),
    ModelSpec(
        name="lasso",
        estimator=Lasso(max_iter=20000, random_state=RANDOM_STATE),
        param_grid={"model__alpha": [0.01, 0.1, 1.0]},
        scale_features=True,
    ),
    ModelSpec(
        name="elastic_net",
        estimator=ElasticNet(max_iter=20000, random_state=RANDOM_STATE),
        param_grid={
            "model__alpha": [0.1, 1.0],
            "model__l1_ratio": [0.5],
        },
        scale_features=True,
    ),
    ModelSpec(
        name="rf_regularized",
        estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
        param_grid={
            "model__n_estimators": [300],
            "model__max_depth": [4, 6],
            "model__min_samples_leaf": [5, 10],
            "model__min_samples_split": [10],
            "model__max_features": ["sqrt"],
        },
    ),
    ModelSpec(
        name="extra_trees_regularized",
        estimator=ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
        param_grid={
            "model__n_estimators": [300],
            "model__max_depth": [4, 6],
            "model__min_samples_leaf": [5, 10],
            "model__min_samples_split": [10],
            "model__max_features": ["sqrt"],
        },
    ),
    ModelSpec(
        name="hist_gradient_boosting",
        estimator=HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        param_grid={
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [100],
            "model__max_leaf_nodes": [15],
            "model__l2_regularization": [0.1],
        },
    ),
]


def _strip_model_prefix(params):
    return {key.replace("model__", "", 1): value for key, value in params.items()}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA:
        return None
    return value


def _add_lag_features(df, target_col):
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data = data[data[DATE_COL].notna()].sort_values(DATE_COL).reset_index(drop=True)
    y = pd.to_numeric(data[target_col], errors="coerce")

    lag1 = y.shift(1)
    date_diff = data[DATE_COL].diff().dt.days
    lag1.loc[date_diff.gt(2)] = np.nan

    data[f"{target_col}_lag1"] = lag1
    data[f"{target_col}_roll3"] = lag1.rolling(window=3, min_periods=2).mean()
    return data


def load_model_frame(input_path, target_col):
    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_path}")
    if DATE_COL not in df.columns:
        raise ValueError(f"Date column '{DATE_COL}' not found in {input_path}")

    data = _add_lag_features(df, target_col)
    data["year"] = data[DATE_COL].dt.year.astype(int)
    y = pd.to_numeric(data[target_col], errors="coerce")
    valid = y.notna()
    return data.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def _available_feature_group(group, df):
    missing = [col for col in group.source_features if col not in df.columns]
    if missing:
        raise ValueError(f"Feature group '{group.name}' missing columns: {missing}")
    return group


def build_pipeline(feature_group, model_spec):
    preprocessor = build_leakage_safe_preprocessor(
        numeric_features=feature_group.numeric_features,
        categorical_features=feature_group.categorical_features,
        include_month=feature_group.include_month,
        iqr_columns=IQR_COLUMNS,
    )
    steps = [("preprocess", preprocessor)]
    if model_spec.scale_features:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model_spec.estimator))
    return Pipeline(steps=steps)


def evaluate_predictions(y_true, y_pred, dataset_name):
    return {
        "dataset": dataset_name,
        "r2": round(float(r2_score(y_true, y_pred)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "n_samples": int(len(y_true)),
    }


def _metric_row(validation, model, feature_group, train_metrics, test_metrics, **metadata):
    row = {
        "validation": validation,
        "feature_group": feature_group,
        "model": model,
        "train_r2": None,
        "train_rmse": None,
        "train_mae": None,
        "test_r2": None,
        "test_rmse": None,
        "test_mae": None,
        "r2_gap_train_minus_test": None,
        "rmse_gap_test_minus_train": None,
        "n_train": 0,
        "n_test": 0,
    }
    row.update(metadata)

    if train_metrics:
        row.update(
            {
                "train_r2": train_metrics["r2"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "n_train": train_metrics["n_samples"],
            }
        )
    if test_metrics:
        row.update(
            {
                "test_r2": test_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "n_test": test_metrics["n_samples"],
            }
        )
    if train_metrics and test_metrics:
        row["r2_gap_train_minus_test"] = round(
            float(train_metrics["r2"] - test_metrics["r2"]), 6
        )
        row["rmse_gap_test_minus_train"] = round(
            float(test_metrics["rmse"] - train_metrics["rmse"]), 6
        )
    return row


def _prediction_rows(df, indices, y_true, y_pred, validation, model, feature_group, **metadata):
    rows = []
    for idx, actual, predicted in zip(indices, y_true, y_pred):
        row = {
            "validation": validation,
            "feature_group": feature_group,
            "model": model,
            "date": str(pd.to_datetime(df.loc[idx, DATE_COL]).date()),
            "actual": float(actual),
            "predicted": float(predicted),
            "residual": float(actual - predicted),
        }
        row.update(metadata)
        rows.append(row)
    return rows


def _append_constant_baseline(
    rows,
    prediction_rows,
    df,
    y,
    train_idx,
    test_idx,
    validation,
    model_name,
    value,
    **metadata,
):
    train_pred = np.repeat(value, len(train_idx))
    test_pred = np.repeat(value, len(test_idx))
    train_metrics = evaluate_predictions(y.loc[train_idx], train_pred, "train")
    test_metrics = evaluate_predictions(y.loc[test_idx], test_pred, "test")
    rows.append(
        _metric_row(
            validation,
            model_name,
            "baseline",
            train_metrics,
            test_metrics,
            **metadata,
        )
    )
    prediction_rows.extend(
        _prediction_rows(
            df,
            test_idx,
            y.loc[test_idx],
            test_pred,
            validation,
            model_name,
            "baseline",
            **metadata,
        )
    )


def _append_previous_value_baseline(
    rows,
    prediction_rows,
    df,
    y,
    train_idx,
    test_idx,
    validation,
    target_col,
    **metadata,
):
    lag_col = f"{target_col}_lag1"
    if lag_col not in df.columns:
        return

    train_idx = pd.Index(train_idx)
    test_idx = pd.Index(test_idx)
    train_idx = train_idx[df.loc[train_idx, lag_col].notna()]
    test_idx = test_idx[df.loc[test_idx, lag_col].notna()]
    if len(train_idx) < 2 or len(test_idx) < 2:
        return

    train_pred = df.loc[train_idx, lag_col].astype(float)
    test_pred = df.loc[test_idx, lag_col].astype(float)
    train_metrics = evaluate_predictions(y.loc[train_idx], train_pred, "train")
    test_metrics = evaluate_predictions(y.loc[test_idx], test_pred, "test")
    rows.append(
        _metric_row(
            validation,
            "previous_value",
            "baseline",
            train_metrics,
            test_metrics,
            **metadata,
        )
    )
    prediction_rows.extend(
        _prediction_rows(
            df,
            test_idx,
            y.loc[test_idx],
            test_pred,
            validation,
            "previous_value",
            "baseline",
            **metadata,
        )
    )


def evaluate_baselines(df, y, target_col):
    rows = []
    prediction_rows = []
    all_idx = df.index

    train_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    _append_constant_baseline(
        rows,
        prediction_rows,
        df,
        y,
        train_idx,
        test_idx,
        "random_split",
        "mean",
        float(y.loc[train_idx].mean()),
    )
    _append_constant_baseline(
        rows,
        prediction_rows,
        df,
        y,
        train_idx,
        test_idx,
        "random_split",
        "median",
        float(y.loc[train_idx].median()),
    )
    _append_previous_value_baseline(
        rows,
        prediction_rows,
        df,
        y,
        train_idx,
        test_idx,
        "random_split",
        target_col,
    )

    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    for train_year in years:
        for test_year in years:
            if train_year == test_year:
                continue
            train_idx = df.index[df["year"].eq(train_year)]
            test_idx = df.index[df["year"].eq(test_year)]
            metadata = {"train_year": int(train_year), "test_year": int(test_year)}
            if len(train_idx) < 20 or len(test_idx) < 20:
                continue
            _append_constant_baseline(
                rows,
                prediction_rows,
                df,
                y,
                train_idx,
                test_idx,
                "year_holdout",
                "mean",
                float(y.loc[train_idx].mean()),
                **metadata,
            )
            _append_constant_baseline(
                rows,
                prediction_rows,
                df,
                y,
                train_idx,
                test_idx,
                "year_holdout",
                "median",
                float(y.loc[train_idx].median()),
                **metadata,
            )
            _append_previous_value_baseline(
                rows,
                prediction_rows,
                df,
                y,
                train_idx,
                test_idx,
                "year_holdout",
                target_col,
                **metadata,
            )

    ordered = df.sort_values(DATE_COL).reset_index()
    split_idx = int(len(ordered) * 0.8)
    if split_idx >= 20 and len(ordered) - split_idx >= 20:
        train_idx = ordered.loc[: split_idx - 1, "index"]
        test_idx = ordered.loc[split_idx:, "index"]
        metadata = {
            "train_date_min": str(pd.to_datetime(df.loc[train_idx, DATE_COL]).min().date()),
            "train_date_max": str(pd.to_datetime(df.loc[train_idx, DATE_COL]).max().date()),
            "test_date_min": str(pd.to_datetime(df.loc[test_idx, DATE_COL]).min().date()),
            "test_date_max": str(pd.to_datetime(df.loc[test_idx, DATE_COL]).max().date()),
        }
        _append_constant_baseline(
            rows,
            prediction_rows,
            df,
            y,
            train_idx,
            test_idx,
            "blocked_time_holdout",
            "mean",
            float(y.loc[train_idx].mean()),
            **metadata,
        )
        _append_constant_baseline(
            rows,
            prediction_rows,
            df,
            y,
            train_idx,
            test_idx,
            "blocked_time_holdout",
            "median",
            float(y.loc[train_idx].median()),
            **metadata,
        )
        _append_previous_value_baseline(
            rows,
            prediction_rows,
            df,
            y,
            train_idx,
            test_idx,
            "blocked_time_holdout",
            target_col,
            **metadata,
        )

    monthly_rows, monthly_predictions = evaluate_monthly_baselines(df, y, target_col)
    rows.extend(monthly_rows)
    prediction_rows.extend(monthly_predictions)

    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def evaluate_monthly_baselines(
    df,
    y,
    target_col,
    min_train_months=3,
    min_train_samples=60,
    min_test_samples=5,
):
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data["_period"] = data[DATE_COL].dt.to_period("M")
    periods = sorted(data["_period"].dropna().unique())
    lag_col = f"{target_col}_lag1"

    prediction_rows = []
    for period_index, period in enumerate(periods):
        if period_index < min_train_months:
            continue
        train_periods = periods[:period_index]
        train_idx = data.index[data["_period"].isin(train_periods)]
        test_idx = data.index[data["_period"].eq(period)]
        if len(train_idx) < min_train_samples or len(test_idx) < min_test_samples:
            continue

        metadata = {
            "test_period": str(period),
            "train_period_start": str(train_periods[0]),
            "train_period_end": str(train_periods[-1]),
        }
        for model_name, value in [
            ("mean", float(y.loc[train_idx].mean())),
            ("median", float(y.loc[train_idx].median())),
        ]:
            pred = np.repeat(value, len(test_idx))
            prediction_rows.extend(
                _prediction_rows(
                    df,
                    test_idx,
                    y.loc[test_idx],
                    pred,
                    "monthly_rolling",
                    model_name,
                    "baseline",
                    **metadata,
                )
            )

        if lag_col in df.columns:
            lag_idx = test_idx[df.loc[test_idx, lag_col].notna()]
            if len(lag_idx) >= min_test_samples:
                prediction_rows.extend(
                    _prediction_rows(
                        df,
                        lag_idx,
                        y.loc[lag_idx],
                        df.loc[lag_idx, lag_col].astype(float),
                        "monthly_rolling",
                        "previous_value",
                        "baseline",
                        **metadata,
                    )
                )

    prediction_df = pd.DataFrame(prediction_rows)
    rows = []
    if prediction_df.empty:
        return rows, prediction_rows

    for model_name, group in prediction_df.groupby("model"):
        metrics = evaluate_predictions(group["actual"], group["predicted"], "test")
        rows.append(
            _metric_row(
                "monthly_rolling",
                model_name,
                "baseline",
                None,
                metrics,
                n_windows=int(group["test_period"].nunique()),
                test_period_start=str(group["test_period"].min()),
                test_period_end=str(group["test_period"].max()),
            )
        )
    return rows, prediction_rows


def build_risk_thresholds(y):
    rows = []
    for quantile in RISK_QUANTILES:
        threshold = float(pd.Series(y).quantile(quantile))
        rows.append(
            {
                "threshold_name": f"p{int(round(quantile * 100))}",
                "quantile": quantile,
                "threshold_ppm": round(threshold, 6),
            }
        )
    return pd.DataFrame(rows)


def evaluate_risk_predictions(y_true, y_score, thresholds_df, **metadata):
    actual = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(score)
    actual = actual[valid]
    score = score[valid]
    rows = []
    if len(actual) == 0:
        return rows

    for threshold_row in thresholds_df.to_dict(orient="records"):
        threshold = float(threshold_row["threshold_ppm"])
        actual_high = actual >= threshold
        predicted_high = score >= threshold

        tp = int(np.sum(predicted_high & actual_high))
        fp = int(np.sum(predicted_high & ~actual_high))
        tn = int(np.sum(~predicted_high & ~actual_high))
        fn = int(np.sum(~predicted_high & actual_high))
        positives = tp + fn
        predicted_positives = tp + fp

        precision = tp / predicted_positives if predicted_positives else 0.0
        recall = tp / positives if positives else None
        false_negative_rate = fn / positives if positives else None
        specificity = tn / (tn + fp) if (tn + fp) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if recall is not None and (precision + recall) > 0
            else None
        )

        row = {
            **metadata,
            "threshold_name": threshold_row["threshold_name"],
            "risk_threshold_ppm": threshold,
            "n_samples": int(len(actual)),
            "actual_high_count": int(positives),
            "predicted_high_count": int(predicted_positives),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": None if precision is None else round(float(precision), 6),
            "recall": None if recall is None else round(float(recall), 6),
            "false_negative_rate": (
                None if false_negative_rate is None else round(float(false_negative_rate), 6)
            ),
            "specificity": None if specificity is None else round(float(specificity), 6),
            "f1": None if f1 is None else round(float(f1), 6),
        }

        order = np.argsort(-score)
        for fraction in TOP_K_FRACTIONS:
            k = max(1, int(np.ceil(len(score) * fraction)))
            top_idx = order[:k]
            top_hits = int(np.sum(actual_high[top_idx]))
            suffix = f"top_{int(round(fraction * 100))}pct"
            row[f"{suffix}_alert_count"] = int(k)
            row[f"{suffix}_precision"] = round(float(top_hits / k), 6)
            row[f"{suffix}_recall"] = (
                None if positives == 0 else round(float(top_hits / positives), 6)
            )

        rows.append(row)
    return rows


def evaluate_risk_prediction_frame(prediction_df, thresholds_df):
    if prediction_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["validation", "feature_group", "model"]
    optional_cols = [
        "train_year",
        "test_year",
        "train_date_min",
        "train_date_max",
        "test_date_min",
        "test_date_max",
    ]
    for col in optional_cols:
        if col in prediction_df.columns:
            group_cols.append(col)

    group_cols = [col for col in group_cols if col in prediction_df.columns]
    for keys, group in prediction_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = {
            col: (None if pd.isna(value) else value)
            for col, value in zip(group_cols, keys)
        }
        rows.extend(
            evaluate_risk_predictions(
                group["actual"], group["predicted"], thresholds_df, **metadata
            )
        )
    return pd.DataFrame(rows)


def fit_model(X_train, y_train, feature_group, model_spec, cv_folds):
    grid = GridSearchCV(
        estimator=build_pipeline(feature_group, model_spec),
        param_grid=model_spec.param_grid,
        cv=cv_folds,
        scoring=CV_SCORING,
        n_jobs=N_JOBS,
        verbose=0,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)
    return grid


def evaluate_configuration(df, y, feature_group, model_spec):
    group = _available_feature_group(feature_group, df)
    X = df[group.source_features].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    grid = fit_model(X_train, y_train, group, model_spec, CV_FOLDS)
    best = grid.best_estimator_
    train_metrics = evaluate_predictions(y_train, best.predict(X_train), "train")
    test_metrics = evaluate_predictions(y_test, best.predict(X_test), "test")

    return {
        "feature_group": group.name,
        "model": model_spec.name,
        "cv_best_rmse": round(float(-grid.best_score_), 6),
        "best_params": _strip_model_prefix(grid.best_params_),
        "train_r2": train_metrics["r2"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "r2_gap_train_minus_test": round(
            float(train_metrics["r2"] - test_metrics["r2"]), 6
        ),
        "rmse_gap_test_minus_train": round(
            float(test_metrics["rmse"] - train_metrics["rmse"]), 6
        ),
        "n_train": train_metrics["n_samples"],
        "n_test": test_metrics["n_samples"],
    }, grid


def _model_spec_by_name(name):
    for spec in MODEL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def _feature_group_by_name(name):
    for group in FEATURE_GROUPS:
        if group.name == name:
            return group
    raise KeyError(name)


def fit_best_with_params(X_train, y_train, feature_group, model_spec, params):
    pipeline = build_pipeline(feature_group, model_spec)
    pipeline.set_params(**{f"model__{key}": value for key, value in params.items()})
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_year_holdouts(df, y, best_row, return_predictions=False):
    group = _feature_group_by_name(best_row["feature_group"])
    spec = _model_spec_by_name(best_row["model"])
    X = df[group.source_features].copy()
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    rows = []
    prediction_rows = []

    for train_year in years:
        for test_year in years:
            if train_year == test_year:
                continue
            train_mask = df["year"].eq(train_year)
            test_mask = df["year"].eq(test_year)
            if train_mask.sum() < 20 or test_mask.sum() < 20:
                continue

            model = fit_best_with_params(
                X.loc[train_mask],
                y.loc[train_mask],
                group,
                spec,
                best_row["best_params"],
            )
            train_metrics = evaluate_predictions(
                y.loc[train_mask], model.predict(X.loc[train_mask]), "train"
            )
            test_metrics = evaluate_predictions(
                y.loc[test_mask], model.predict(X.loc[test_mask]), "test"
            )
            y_pred = model.predict(X.loc[test_mask])
            rows.append(
                {
                    "validation": "year_holdout",
                    "train_year": int(train_year),
                    "test_year": int(test_year),
                    "feature_group": group.name,
                    "model": spec.name,
                    "train_r2": train_metrics["r2"],
                    "train_rmse": train_metrics["rmse"],
                    "test_r2": test_metrics["r2"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "n_train": train_metrics["n_samples"],
                    "n_test": test_metrics["n_samples"],
                }
            )
            if return_predictions:
                test_idx = df.index[test_mask]
                prediction_rows.extend(
                    _prediction_rows(
                        df,
                        test_idx,
                        y.loc[test_idx],
                        y_pred,
                        "year_holdout",
                        spec.name,
                        group.name,
                        train_year=int(train_year),
                        test_year=int(test_year),
                    )
                )
    metrics_df = pd.DataFrame(rows)
    if return_predictions:
        return metrics_df, pd.DataFrame(prediction_rows)
    return metrics_df


def evaluate_blocked_time_holdout(df, y, best_row, train_fraction=0.8, return_predictions=False):
    group = _feature_group_by_name(best_row["feature_group"])
    spec = _model_spec_by_name(best_row["model"])
    ordered = df.sort_values(DATE_COL).reset_index()
    split_idx = int(len(ordered) * train_fraction)
    if split_idx < 20 or len(ordered) - split_idx < 20:
        empty = pd.DataFrame()
        return (empty, empty) if return_predictions else empty

    train_idx = ordered.loc[: split_idx - 1, "index"]
    test_idx = ordered.loc[split_idx:, "index"]
    X = df[group.source_features].copy()
    model = fit_best_with_params(
        X.loc[train_idx],
        y.loc[train_idx],
        group,
        spec,
        best_row["best_params"],
    )
    train_metrics = evaluate_predictions(
        y.loc[train_idx], model.predict(X.loc[train_idx]), "train"
    )
    test_metrics = evaluate_predictions(
        y.loc[test_idx], model.predict(X.loc[test_idx]), "test"
    )
    y_pred = model.predict(X.loc[test_idx])
    row = {
        "validation": "blocked_time_holdout",
        "train_fraction": train_fraction,
        "train_date_min": str(pd.to_datetime(df.loc[train_idx, DATE_COL]).min().date()),
        "train_date_max": str(pd.to_datetime(df.loc[train_idx, DATE_COL]).max().date()),
        "test_date_min": str(pd.to_datetime(df.loc[test_idx, DATE_COL]).min().date()),
        "test_date_max": str(pd.to_datetime(df.loc[test_idx, DATE_COL]).max().date()),
        "feature_group": group.name,
        "model": spec.name,
        "train_r2": train_metrics["r2"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "n_train": train_metrics["n_samples"],
        "n_test": test_metrics["n_samples"],
    }
    metrics_df = pd.DataFrame([row])
    if not return_predictions:
        return metrics_df

    prediction_df = pd.DataFrame(
        _prediction_rows(
            df,
            test_idx,
            y.loc[test_idx],
            y_pred,
            "blocked_time_holdout",
            spec.name,
            group.name,
            train_date_min=row["train_date_min"],
            train_date_max=row["train_date_max"],
            test_date_min=row["test_date_min"],
            test_date_max=row["test_date_max"],
        )
    )
    return metrics_df, prediction_df


def evaluate_year_holdouts_for_comparison(df, y, comparison):
    rows = []
    for row in comparison.to_dict(orient="records"):
        year_df = evaluate_year_holdouts(df, y, row)
        if not year_df.empty:
            year_df["random_split_test_r2"] = row["test_r2"]
            year_df["random_split_test_rmse"] = row["test_rmse"]
            year_df["cv_best_rmse"] = row["cv_best_rmse"]
            rows.extend(year_df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def evaluate_monthly_rolling_validation(
    df,
    y,
    best_row,
    min_train_months=3,
    min_train_samples=60,
    min_test_samples=5,
):
    """Expanding-window monthly validation using the selected model config.

    For each month after the warm-up window, train on all prior months and test
    on the current month. This approximates a monthly model refresh workflow.
    """

    group = _feature_group_by_name(best_row["feature_group"])
    spec = _model_spec_by_name(best_row["model"])
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data["_period"] = data[DATE_COL].dt.to_period("M")
    periods = sorted(data["_period"].dropna().unique())
    X = data[group.source_features].copy()

    monthly_rows = []
    prediction_rows = []

    for period_index, period in enumerate(periods):
        if period_index < min_train_months:
            continue

        train_periods = periods[:period_index]
        train_mask = data["_period"].isin(train_periods)
        test_mask = data["_period"].eq(period)

        if int(train_mask.sum()) < min_train_samples or int(test_mask.sum()) < min_test_samples:
            continue

        train_idx = data.index[train_mask]
        test_idx = data.index[test_mask]
        model = fit_best_with_params(
            X.loc[train_idx],
            y.loc[train_idx],
            group,
            spec,
            best_row["best_params"],
        )
        y_pred = model.predict(X.loc[test_idx])
        train_metrics = evaluate_predictions(
            y.loc[train_idx], model.predict(X.loc[train_idx]), "train"
        )
        test_metrics = evaluate_predictions(y.loc[test_idx], y_pred, "test")

        monthly_rows.append(
            {
                "test_period": str(period),
                "train_period_start": str(train_periods[0]),
                "train_period_end": str(train_periods[-1]),
                "feature_group": group.name,
                "model": spec.name,
                "train_r2": train_metrics["r2"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "test_r2": test_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "n_train": train_metrics["n_samples"],
                "n_test": test_metrics["n_samples"],
            }
        )

        for idx, actual, predicted in zip(test_idx, y.loc[test_idx], y_pred):
            prediction_rows.append(
                {
                    "validation": "monthly_rolling",
                    "feature_group": group.name,
                    "model": spec.name,
                    "date": str(data.loc[idx, DATE_COL].date()),
                    "test_period": str(period),
                    "actual": float(actual),
                    "predicted": float(predicted),
                    "residual": float(actual - predicted),
                }
            )

    monthly_df = pd.DataFrame(monthly_rows)
    prediction_df = pd.DataFrame(prediction_rows)

    if prediction_df.empty:
        summary = {
            "n_windows": 0,
            "n_predictions": 0,
            "r2": None,
            "rmse": None,
            "mae": None,
        }
    else:
        metrics = evaluate_predictions(
            prediction_df["actual"], prediction_df["predicted"], "monthly_rolling"
        )
        summary = {
            "n_windows": int(len(monthly_df)),
            "n_predictions": int(len(prediction_df)),
            "test_period_start": str(monthly_df["test_period"].iloc[0]),
            "test_period_end": str(monthly_df["test_period"].iloc[-1]),
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        }

    return monthly_df, prediction_df, summary


def extract_importance(best_estimator):
    feature_names = get_preprocessed_feature_names(best_estimator.named_steps["preprocess"])
    model = best_estimator.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "model_feature_importance"
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
        importance_type = "scaled_linear_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    rows = []
    for feature, value in zip(feature_names, values):
        rows.append(
            {
                "feature": feature,
                "importance": round(float(value), 6),
                "abs_importance": round(float(abs(value)), 6),
                "importance_type": importance_type,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_importance", ascending=False)
        .reset_index(drop=True)
    )


def extract_permutation_importance(best_estimator, X_test, y_test):
    result = permutation_importance(
        best_estimator,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=N_JOBS,
    )
    rows = []
    for feature, mean, std in zip(X_test.columns, result.importances_mean, result.importances_std):
        rows.append(
            {
                "feature": feature,
                "importance_mean_rmse_reduction": round(float(mean), 6),
                "importance_std": round(float(std), 6),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("importance_mean_rmse_reduction", ascending=False)
        .reset_index(drop=True)
    )


def summarize_feature_groups(comparison, year_all_df):
    random_summary = (
        comparison.groupby("feature_group")
        .agg(
            best_random_split_test_r2=("test_r2", "max"),
            best_random_split_test_rmse=("test_rmse", "min"),
            best_cv_rmse=("cv_best_rmse", "min"),
        )
        .reset_index()
    )

    if year_all_df.empty:
        return random_summary

    year_summary = (
        year_all_df.groupby("feature_group")
        .agg(
            best_cross_year_test_r2=("test_r2", "max"),
            mean_cross_year_test_r2=("test_r2", "mean"),
            worst_cross_year_test_r2=("test_r2", "min"),
            mean_cross_year_test_rmse=("test_rmse", "mean"),
        )
        .reset_index()
    )
    return random_summary.merge(year_summary, on="feature_group", how="left")


def _pick_p80_risk_focus(risk_df, feature_group=None, model=None):
    if risk_df.empty:
        return risk_df

    focus = risk_df[risk_df["threshold_name"].eq("p80")]
    if feature_group is not None:
        focus = focus[focus["feature_group"].eq(feature_group)]
    if model is not None:
        focus = focus[focus["model"].eq(model)]

    monthly = focus[focus["validation"].eq("monthly_rolling")]
    if not monthly.empty:
        return monthly

    random_split = focus[focus["validation"].eq("random_split")]
    if not random_split.empty:
        return random_split

    return focus


def _best_recall_and_fnr(risk_focus):
    if risk_focus.empty or "recall" not in risk_focus.columns:
        return None, None
    best_recall = risk_focus["recall"].dropna().max()
    best_false_negative_rate = risk_focus["false_negative_rate"].dropna().min()
    best_recall = None if pd.isna(best_recall) else float(best_recall)
    best_false_negative_rate = (
        None if pd.isna(best_false_negative_rate) else float(best_false_negative_rate)
    )
    return best_recall, best_false_negative_rate


def build_decision_summary(comparison, feature_group_summary, risk_df, best_row):
    quality_row = feature_group_summary[
        feature_group_summary["feature_group"].eq("quality_only")
    ]
    best_non_lag = comparison[~comparison["feature_group"].str.contains("lag")]
    best_lag = comparison[comparison["feature_group"].str.contains("lag")]

    quality_cross_year_r2 = None
    quality_random_r2 = None
    if not quality_row.empty:
        quality_random_r2 = float(quality_row["best_random_split_test_r2"].iloc[0])
        if "best_cross_year_test_r2" in quality_row.columns:
            value = quality_row["best_cross_year_test_r2"].iloc[0]
            if pd.notna(value):
                quality_cross_year_r2 = float(value)

    if quality_cross_year_r2 is not None:
        quality_signal_reference = quality_cross_year_r2
    else:
        quality_signal_reference = quality_random_r2

    if quality_signal_reference is None:
        quality_conclusion = "not_evaluated"
    elif quality_signal_reference <= 0.10:
        quality_conclusion = "quality_only_features_insufficient_for_precise_ppm_prediction"
    elif quality_signal_reference <= 0.30:
        quality_conclusion = "quality_only_features_have_limited_signal"
    else:
        quality_conclusion = "quality_only_features_have_material_signal"

    lag_gain = None
    if not best_non_lag.empty and not best_lag.empty:
        lag_gain = float(best_lag["test_r2"].max() - best_non_lag["test_r2"].max())

    if lag_gain is None:
        lag_conclusion = "not_evaluated"
    elif lag_gain >= 0.15:
        lag_conclusion = "lag_features_materially_improve_prediction"
    elif lag_gain >= 0.05:
        lag_conclusion = "lag_features_modestly_improve_prediction"
    else:
        lag_conclusion = "lag_features_do_not_materially_improve_prediction"

    selected_risk_focus = _pick_p80_risk_focus(
        risk_df, best_row["feature_group"], best_row["model"]
    )
    selected_recall, selected_false_negative_rate = _best_recall_and_fnr(
        selected_risk_focus
    )
    best_available_recall, best_available_false_negative_rate = _best_recall_and_fnr(
        _pick_p80_risk_focus(risk_df)
    )

    if selected_recall is None:
        risk_conclusion = "risk_alert_not_evaluated"
    elif selected_recall >= 0.80:
        risk_conclusion = "risk_alert_promising_for_manual_review"
    elif selected_recall >= 0.60:
        risk_conclusion = "risk_alert_marginal_requires_business_review"
    else:
        risk_conclusion = "risk_alert_recall_too_low_prioritize_data_or_features"

    return {
        "business_positioning": "ppm_prediction_plus_high_risk_alert_prototype",
        "not_supported": "automatic_dosing_optimization",
        "quality_only_signal_reference_r2": (
            None if quality_signal_reference is None else round(quality_signal_reference, 6)
        ),
        "quality_only_conclusion": quality_conclusion,
        "best_non_lag_random_split_test_r2": (
            None if best_non_lag.empty else round(float(best_non_lag["test_r2"].max()), 6)
        ),
        "best_lag_random_split_test_r2": (
            None if best_lag.empty else round(float(best_lag["test_r2"].max()), 6)
        ),
        "lag_random_split_test_r2_gain": None if lag_gain is None else round(lag_gain, 6),
        "lag_conclusion": lag_conclusion,
        "prototype_risk_threshold": "p80",
        "selected_model_p80_recall": (
            None if selected_recall is None else round(selected_recall, 6)
        ),
        "selected_model_p80_false_negative_rate": (
            None
            if selected_false_negative_rate is None
            else round(selected_false_negative_rate, 6)
        ),
        "best_available_p80_recall": (
            None if best_available_recall is None else round(best_available_recall, 6)
        ),
        "best_available_p80_false_negative_rate": (
            None
            if best_available_false_negative_rate is None
            else round(best_available_false_negative_rate, 6)
        ),
        "risk_alert_conclusion": risk_conclusion,
    }


def write_run_summary_md(
    output_path,
    summary,
    feature_group_summary,
    risk_thresholds_df,
    decision_summary,
):
    best = summary["best_random_split_model"]
    thresholds = ", ".join(
        f"{row['threshold_name']}={row['threshold_ppm']:.2f} ppm"
        for row in risk_thresholds_df.to_dict(orient="records")
    )
    lines = [
        "# Final Feed Oil Phosphorus Experiment Summary",
        "",
        "## Scope",
        f"- Target: `{summary['target_col']}`",
        f"- Date range: {summary['date_min']} to {summary['date_max']}",
        f"- Valid target samples: {summary['n_samples']}",
        "- Excluded from feed prediction: RBD variables, acid dosing, bleaching earth dosing, and target-derived leakage.",
        "",
        "## Best Random Split Model",
        f"- Feature group: `{best['feature_group']}`",
        f"- Model: `{best['model']}`",
        f"- Test R2/RMSE/MAE: {best['test_r2']:.3f} / {best['test_rmse']:.3f} / {best['test_mae']:.3f}",
        f"- Train-test R2 gap: {best['r2_gap_train_minus_test']:.3f}",
        "",
        "## Feature Group Signal",
    ]
    for row in feature_group_summary.to_dict(orient="records"):
        cross_year = row.get("best_cross_year_test_r2")
        cross_year_text = "NA" if pd.isna(cross_year) else f"{cross_year:.3f}"
        lines.append(
            f"- `{row['feature_group']}`: best random R2 {row['best_random_split_test_r2']:.3f}, "
            f"best cross-year R2 {cross_year_text}"
        )

    lines.extend(
        [
            "",
            "## Risk Alert Prototype",
            f"- Quantile thresholds: {thresholds}",
            f"- Selected model p80 recall: {decision_summary['selected_model_p80_recall']}",
            f"- Selected model p80 false negative rate: {decision_summary['selected_model_p80_false_negative_rate']}",
            f"- Best available p80 recall: {decision_summary['best_available_p80_recall']}",
            "",
            "## Decision Summary",
            f"- Quality-only conclusion: {decision_summary['quality_only_conclusion']}",
            f"- Lag conclusion: {decision_summary['lag_conclusion']}",
            f"- Risk alert conclusion: {decision_summary['risk_alert_conclusion']}",
            "- Business position: ppm prediction plus high-risk alert prototype; not automatic dosing optimization.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(input_path, output_dir, target_col):
    if target_col != DEFAULT_TARGET_COL:
        raise ValueError(
            "feed_model_optimized is fixed to feed-oil phosphorus prediction "
            f"({DEFAULT_TARGET_COL}); got target_col={target_col!r}."
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df, y = load_model_frame(input_path, target_col)
    risk_thresholds_df = build_risk_thresholds(y)
    risk_thresholds_df.to_csv(
        output / "feed_model_optimized_risk_thresholds.csv", index=False
    )

    baseline_df, baseline_predictions_df = evaluate_baselines(df, y, target_col)
    baseline_df.to_csv(output / "feed_model_optimized_baselines.csv", index=False)
    baseline_predictions_df.to_csv(
        output / "feed_model_optimized_baseline_predictions.csv", index=False
    )

    rows = []
    grids = {}
    for group in FEATURE_GROUPS:
        for spec in MODEL_SPECS:
            row, grid = evaluate_configuration(df, y, group, spec)
            rows.append(row)
            grids[(group.name, spec.name)] = grid
            print(
                f"{group.name} / {spec.name}: "
                f"test R2={row['test_r2']:.3f}, RMSE={row['test_rmse']:.3f}, "
                f"CV RMSE={row['cv_best_rmse']:.3f}",
                flush=True,
            )

    comparison = pd.DataFrame(rows).sort_values(
        ["test_rmse", "r2_gap_train_minus_test"], ascending=[True, True]
    )
    comparison.to_csv(output / "feed_model_optimized_comparison.csv", index=False)

    year_all_df = evaluate_year_holdouts_for_comparison(df, y, comparison)
    year_all_df.to_csv(
        output / "feed_model_optimized_year_validation_all_configs.csv", index=False
    )

    feature_group_summary = summarize_feature_groups(comparison, year_all_df)
    feature_group_summary.to_csv(
        output / "feed_model_optimized_feature_group_summary.csv", index=False
    )

    best_row = comparison.iloc[0].to_dict()
    best_group = _feature_group_by_name(best_row["feature_group"])
    X = df[best_group.source_features].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    best_grid = grids[(best_row["feature_group"], best_row["model"])]
    best_estimator = best_grid.best_estimator_
    best_test_pred = best_estimator.predict(X_test)
    best_random_prediction_df = pd.DataFrame(
        _prediction_rows(
            df,
            X_test.index,
            y_test,
            best_test_pred,
            "random_split",
            best_row["model"],
            best_row["feature_group"],
        )
    )

    importance_df = extract_importance(best_estimator)
    importance_df.to_csv(output / "feed_model_optimized_best_importance.csv", index=False)

    permutation_df = extract_permutation_importance(best_estimator, X_test, y_test)
    permutation_df.to_csv(
        output / "feed_model_optimized_best_permutation_importance.csv", index=False
    )

    year_df, year_prediction_df = evaluate_year_holdouts(
        df, y, best_row, return_predictions=True
    )
    year_df.to_csv(output / "feed_model_optimized_year_validation.csv", index=False)
    year_prediction_df.to_csv(
        output / "feed_model_optimized_year_predictions.csv", index=False
    )

    blocked_df, blocked_prediction_df = evaluate_blocked_time_holdout(
        df, y, best_row, return_predictions=True
    )
    blocked_df.to_csv(output / "feed_model_optimized_blocked_validation.csv", index=False)
    blocked_prediction_df.to_csv(
        output / "feed_model_optimized_blocked_predictions.csv", index=False
    )

    rolling_df, rolling_predictions_df, rolling_summary = evaluate_monthly_rolling_validation(
        df, y, best_row
    )
    rolling_df.to_csv(output / "feed_model_optimized_monthly_rolling_validation.csv", index=False)
    rolling_predictions_df.to_csv(
        output / "feed_model_optimized_monthly_rolling_predictions.csv", index=False
    )

    model_prediction_df = pd.concat(
        [
            best_random_prediction_df,
            year_prediction_df,
            blocked_prediction_df,
            rolling_predictions_df,
        ],
        ignore_index=True,
    )
    model_prediction_df.to_csv(
        output / "feed_model_optimized_model_predictions.csv", index=False
    )

    risk_prediction_df = pd.concat(
        [model_prediction_df, baseline_predictions_df], ignore_index=True
    )
    risk_df = evaluate_risk_prediction_frame(risk_prediction_df, risk_thresholds_df)
    risk_df.to_csv(output / "feed_model_optimized_risk_metrics.csv", index=False)

    decision_summary = build_decision_summary(
        comparison, feature_group_summary, risk_df, best_row
    )

    summary = {
        "target_col": target_col,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "cv_scoring": CV_SCORING,
        "n_samples": int(len(df)),
        "date_min": str(pd.to_datetime(df[DATE_COL]).min().date()),
        "date_max": str(pd.to_datetime(df[DATE_COL]).max().date()),
        "best_random_split_model": best_row,
        "risk_thresholds": risk_thresholds_df.to_dict(orient="records"),
        "baseline_summary": baseline_df.to_dict(orient="records"),
        "feature_group_summary": feature_group_summary.to_dict(orient="records"),
        "year_validation": year_df.to_dict(orient="records"),
        "blocked_time_validation": blocked_df.to_dict(orient="records"),
        "monthly_rolling_validation": rolling_summary,
        "decision_summary": decision_summary,
        "output_files": {
            "baselines": "feed_model_optimized_baselines.csv",
            "comparison": "feed_model_optimized_comparison.csv",
            "feature_group_summary": "feed_model_optimized_feature_group_summary.csv",
            "importance": "feed_model_optimized_best_importance.csv",
            "permutation_importance": "feed_model_optimized_best_permutation_importance.csv",
            "risk_thresholds": "feed_model_optimized_risk_thresholds.csv",
            "risk_metrics": "feed_model_optimized_risk_metrics.csv",
            "model_predictions": "feed_model_optimized_model_predictions.csv",
            "baseline_predictions": "feed_model_optimized_baseline_predictions.csv",
            "year_validation": "feed_model_optimized_year_validation.csv",
            "year_validation_all_configs": "feed_model_optimized_year_validation_all_configs.csv",
            "blocked_time_validation": "feed_model_optimized_blocked_validation.csv",
            "monthly_rolling_validation": "feed_model_optimized_monthly_rolling_validation.csv",
            "monthly_rolling_predictions": "feed_model_optimized_monthly_rolling_predictions.csv",
            "run_summary": "feed_model_optimized_run_summary.md",
        },
        "notes": [
            "RBD variables and dosing variables are intentionally excluded for feed-oil phosphorus prediction.",
            "Lag-aware features assume recent feed phosphorus lab results are available before the next prediction.",
            "Random 80/20 split can be optimistic for time-dependent data; year-aware validation should guide deployment claims.",
            "Risk thresholds are prototype quantile cutoffs because no sponsor fixed ppm threshold has been provided.",
        ],
    }
    with (output / "feed_model_optimized_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2, allow_nan=False)
    write_run_summary_md(
        output / "feed_model_optimized_run_summary.md",
        summary,
        feature_group_summary,
        risk_thresholds_df,
        decision_summary,
    )

    return summary


def parse_args():
    default_target_col = os.getenv("CPO_TARGET_COL", DEFAULT_TARGET_COL)
    parser = argparse.ArgumentParser(
        description="Run optimized feed-oil phosphorus model comparison"
    )
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_source.csv"),
        help=f"Path to model_source.csv (default: {LOCAL_PROCESSED_DATA_DIR / 'model_source.csv'})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_REPORTS_DIR / "feed_model_optimized"),
        help="Directory for optimized feed model outputs",
    )
    parser.add_argument(
        "--target-col",
        default=default_target_col,
        help=f"Target column to predict (default: {default_target_col})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_pipeline(args.input, args.output_dir, args.target_col)
    print(json.dumps(_json_safe(results), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
