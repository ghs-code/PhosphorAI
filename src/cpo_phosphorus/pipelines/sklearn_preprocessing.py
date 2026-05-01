"""Leakage-safe sklearn preprocessing components for model training.

The EDA preprocessing pipeline exports descriptive artifacts for the whole
dataset. Model training should instead fit imputation, clipping, encoding, and
feature engineering inside sklearn Pipelines so cross-validation folds and test
sets do not influence training transformations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_IQR_COLUMNS = [
    "feed_dobi",
    "feed_ffa_pct",
    "feed_mi_pct",
    "feed_iv",
    "feed_p_ppm",
    "rbd_ffa_pct",
    "rbd_mi_pct",
    "rbd_iv",
    "rbd_p_ppm",
]


def make_one_hot_encoder():
    """Create a dense OneHotEncoder across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_month_encoder():
    """Create a dense month encoder, dropping January as the baseline."""
    kwargs = {
        "categories": [list(range(1, 13))],
        "drop": [1],
        "handle_unknown": "ignore",
    }
    try:
        return OneHotEncoder(**kwargs, sparse_output=False)
    except TypeError:
        return OneHotEncoder(**kwargs, sparse=False)


def _as_dataframe(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X).copy()


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    """Add calendar features using only training-set state."""

    def __init__(self, date_col="date"):
        self.date_col = date_col

    def fit(self, X, y=None):
        data = _as_dataframe(X)
        if self.date_col not in data.columns:
            raise ValueError(f"Date column '{self.date_col}' is required for preprocessing.")

        dates = pd.to_datetime(data[self.date_col], errors="coerce")
        min_date = dates.min()
        if pd.isna(min_date):
            raise ValueError(f"Date column '{self.date_col}' contains no valid dates.")

        self.min_date_ = min_date
        return self

    def transform(self, X):
        data = _as_dataframe(X)
        dates = pd.to_datetime(data[self.date_col], errors="coerce")
        data["month"] = dates.dt.month
        data["time_trend"] = (dates - self.min_date_).dt.days + 1
        return data


class MonthwiseIQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric outliers using month-specific IQR bounds learned in fit."""

    def __init__(self, columns=None, month_col="month", k=1.5):
        self.columns = columns
        self.month_col = month_col
        self.k = k

    def fit(self, X, y=None):
        data = _as_dataframe(X)
        columns = self.columns if self.columns is not None else DEFAULT_IQR_COLUMNS
        self.columns_ = [col for col in columns if col in data.columns]
        self.bounds_ = {}

        if self.month_col not in data.columns:
            return self

        months = pd.to_numeric(data[self.month_col], errors="coerce")
        for col in self.columns_:
            values = pd.to_numeric(data[col], errors="coerce")
            col_bounds = {}

            for month in sorted(months.dropna().unique()):
                mask = months.eq(month)
                series = values.loc[mask].dropna()
                if len(series) < 4:
                    continue

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if pd.isna(iqr) or iqr == 0:
                    continue

                col_bounds[int(month)] = (float(q1 - (self.k * iqr)), float(q3 + (self.k * iqr)))

            self.bounds_[col] = col_bounds

        return self

    def transform(self, X):
        data = _as_dataframe(X)
        if self.month_col not in data.columns:
            return data

        months = pd.to_numeric(data[self.month_col], errors="coerce")
        for col, col_bounds in getattr(self, "bounds_", {}).items():
            if col not in data.columns:
                continue

            values = pd.to_numeric(data[col], errors="coerce")
            for month, (lower, upper) in col_bounds.items():
                mask = months.eq(month)
                values.loc[mask] = values.loc[mask].clip(lower=lower, upper=upper)
            data[col] = values

        return data


@dataclass
class LogFeature:
    source: str
    output: str


class LogFeatureTransformer(BaseEstimator, TransformerMixin):
    """Add log1p features after training-fold clipping."""

    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        features = self.features or []
        self.features_ = [
            LogFeature(source=source, output=output)
            for source, output in features
        ]
        return self

    def transform(self, X):
        data = _as_dataframe(X)
        for feature in getattr(self, "features_", []):
            if feature.source not in data.columns:
                continue

            values = pd.to_numeric(data[feature.source], errors="coerce")
            data[feature.output] = np.log1p(values.clip(lower=0))

        return data


def build_leakage_safe_preprocessor(
    numeric_features,
    categorical_features=None,
    log_features=None,
    include_month=False,
    iqr_columns=None,
):
    """Build a preprocessing pipeline that can be nested inside GridSearchCV."""

    categorical_features = categorical_features or []
    log_features = log_features or []

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    month_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_month_encoder()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipe, list(numeric_features)))
    if categorical_features:
        transformers.append(("categorical", categorical_pipe, list(categorical_features)))
    if include_month:
        transformers.append(("month", month_pipe, ["month"]))

    return Pipeline(
        steps=[
            ("date_features", DateFeatureTransformer()),
            ("iqr_clip", MonthwiseIQRClipper(columns=iqr_columns or DEFAULT_IQR_COLUMNS)),
            ("log_features", LogFeatureTransformer(features=log_features)),
            (
                "select_and_encode",
                ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )


def get_preprocessed_feature_names(preprocessor):
    """Return output feature names from a fitted leakage-safe preprocessor."""
    column_transformer = preprocessor.named_steps["select_and_encode"]
    return list(column_transformer.get_feature_names_out())
