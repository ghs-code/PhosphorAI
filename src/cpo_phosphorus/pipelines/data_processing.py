#!/usr/bin/env python3
"""
Data preprocessing and EDA pipeline for Wilmar CPO phosphorus capstone project.

This script focuses on framework part 3 only:
1) Load and standardize monthly sheets from raw Excel.
2) Handle missing values and plant-stop placeholders.
3) Transition breakpoint detection for missing September.
4) IQR-based outlier clipping by month.
5) Feature engineering (month dummies, transition flag, trend, log features).
6) EDA exports: descriptive stats, normality checks, correlation matrices, VIF diagnostics.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm
from scipy import stats as scipy_stats
from sklearn.preprocessing import OneHotEncoder
from statsmodels import __version__ as statsmodels_version
from statsmodels.stats.outliers_influence import variance_inflation_factor

from cpo_phosphorus.paths import (
    DEFAULT_RAW_EXCEL,
    LOCAL_PREPROCESSING_REPORTS_DIR,
    LOCAL_PROCESSED_DATA_DIR,
)


def _get_env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


RAW_COLUMNS = [
    "date",
    "acid_dosing_pct",
    "bleaching_earth_dosing_pct",
    "feed_tank",
    "feed_ffa_pct",
    "feed_mi_pct",
    "feed_iv",
    "feed_dobi",
    "feed_car_pv",
    "feed_p_ppm",
    "feed_type",
    "rbd_tank",
    "rbd_ffa_pct",
    "rbd_mi_pct",
    "rbd_iv",
    "rbd_pv",
    "rbd_color",
    "rbd_odor",
    "rbd_p_ppm",
    "rbd_type",
]

NUMERIC_COLUMNS = [
    "acid_dosing_pct",
    "bleaching_earth_dosing_pct",
    "feed_ffa_pct",
    "feed_mi_pct",
    "feed_iv",
    "feed_dobi",
    "feed_car_pv",
    "feed_p_ppm",
    "rbd_ffa_pct",
    "rbd_mi_pct",
    "rbd_iv",
    "rbd_pv",
    "rbd_color",
    "rbd_odor",
    "rbd_p_ppm",
]

CATEGORICAL_COLUMNS = ["feed_tank", "feed_type", "rbd_tank", "rbd_type"]

IQR_COLUMNS = [
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

CORR_COLUMNS = [
    "feed_dobi",
    "feed_ffa_pct",
    "feed_mi_pct",
    "feed_iv",
    "feed_car_pv",
    "feed_p_ppm",
    "rbd_p_ppm",
]

VIF_COLUMNS = ["feed_dobi", "feed_ffa_pct", "feed_mi_pct", "feed_iv", "feed_car_pv"]

TRANSITION_BASE_COLUMNS = ["feed_dobi", "feed_ffa_pct", "feed_mi_pct", "feed_iv"]

BOXPLOT_COLUMNS = ["feed_dobi", "feed_ffa_pct", "feed_mi_pct", "feed_iv", "feed_p_ppm", "rbd_p_ppm"]

STOP_TOKENS = {"STOP", "PLANT STOPPED"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}


def _split_env_list(value):
    if not value:
        return []
    parts = []
    for chunk in value.split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _parse_year_filter(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"all", "*"}:
        return None

    years = set()
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        years.add(int(part))

    return years or None


def _discover_excel_files(input_path):
    path = Path(input_path).expanduser()
    if path.is_dir():
        files = [
            p
            for p in sorted(path.iterdir())
            if p.is_file()
            and p.suffix.lower() in EXCEL_SUFFIXES
            and not p.name.startswith("~$")
        ]
        if not files:
            raise FileNotFoundError(f"No Excel files found in directory: {path}")
        return files

    if not path.exists():
        raise FileNotFoundError(f"Raw input path does not exist: {path}")
    if path.suffix.lower() not in EXCEL_SUFFIXES:
        raise ValueError(f"Raw input is not an Excel file: {path}")
    return [path]


def resolve_raw_input_paths(input_paths):
    resolved = []
    seen = set()
    for input_path in input_paths:
        for path in _discover_excel_files(input_path):
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                resolved.append(path)

    if not resolved:
        raise ValueError("At least one raw Excel input is required.")

    return resolved


def load_raw_excel(input_path):
    xls = pd.ExcelFile(input_path)
    frames = []
    for sheet in xls.sheet_names:
        raw = pd.read_excel(input_path, sheet_name=sheet, header=None)
        block = raw.iloc[4:, 1:21].copy()
        block.columns = RAW_COLUMNS
        block["sheet_name"] = sheet
        block["source_file"] = Path(input_path).name
        frames.append(block)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)
    return df


def load_raw_inputs(input_paths, year_filter=None):
    paths = resolve_raw_input_paths(input_paths)
    frames = [load_raw_excel(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    rows_loaded = int(len(df))

    years = _parse_year_filter(year_filter)
    if years is not None:
        df = df[df["date"].dt.year.isin(years)].copy()
        if df.empty:
            requested = ", ".join(str(year) for year in sorted(years))
            raise ValueError(f"No rows found for requested year(s): {requested}")

    df = df.sort_values("date").reset_index(drop=True)
    metadata = {
        "source_files": [str(path) for path in paths],
        "requested_years": "all" if years is None else sorted(years),
        "years_present": sorted(df["date"].dt.year.unique().astype(int).tolist()),
        "rows_loaded_before_year_filter": rows_loaded,
        "rows_after_year_filter": int(len(df)),
    }
    return df, metadata


def normalize_and_cast(df):
    data = df.copy()
    for col in NUMERIC_COLUMNS:
        if data[col].dtype == object:
            values = data[col].astype(str).str.strip()
            values = values.replace("", np.nan)
            values = values.str.upper().replace(list(STOP_TOKENS), np.nan)
            data[col] = values
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        values = data[col].where(data[col].notna(), np.nan)
        values = values.astype(str).str.strip()
        values = values.str.upper().str.replace(r"\s+", "", regex=True)
        values = values.replace(
            {"": np.nan, "nan": np.nan, "NaN": np.nan, "NAN": np.nan, "None": np.nan}
        )
        data[col] = values
    return data


def _extract_transition_windows(year_df):
    month = year_df["date"].dt.month
    day = year_df["date"].dt.day

    pre_mask = ((month == 7) & (day >= 22)) | ((month == 8) & (day <= 15))
    post_mask = (month == 10) & (day <= 10)

    if int(pre_mask.sum()) < 6:
        pre_mask = month == 8
    if int(post_mask.sum()) < 6:
        post_mask = month == 10

    return pre_mask, post_mask


def detect_transition_breakpoint_by_year(
    df, columns, effect_threshold=1.0, rel_change_threshold=0.2, p_threshold=0.01
):
    report = {"jump_detected_any": False, "jump_years": [], "years": {}}
    data = df.copy()

    for year in sorted(data["date"].dt.year.unique()):
        year_df = data[data["date"].dt.year == year].copy()
        months_present = sorted(year_df["date"].dt.month.unique().tolist())
        has_aug = 8 in months_present
        has_sep = 9 in months_present
        has_oct = 10 in months_present

        year_report = {
            "months_present": months_present,
            "has_aug": has_aug,
            "has_sep": has_sep,
            "has_oct": has_oct,
            "has_missing_sep": not has_sep,
            "pre_window_rows": 0,
            "post_window_rows": 0,
            "feature_tests": {},
            "jump_detected": False,
            "apply_transition_dummy": False,
        }

        if has_aug and has_oct:
            pre_mask, post_mask = _extract_transition_windows(year_df)
            pre_df = year_df.loc[pre_mask].copy()
            post_df = year_df.loc[post_mask].copy()
            year_report["pre_window_rows"] = int(len(pre_df))
            year_report["post_window_rows"] = int(len(post_df))

            for col in columns:
                pre = pre_df[col].dropna().astype(float)
                post = post_df[col].dropna().astype(float)
                test = {
                    "pre_n": int(len(pre)),
                    "post_n": int(len(post)),
                    "pre_mean": None,
                    "post_mean": None,
                    "mean_diff_post_minus_pre": None,
                    "relative_change": None,
                    "effect_size_like": None,
                    "ttest_pvalue": None,
                    "jump_flag": False,
                }

                if len(pre) < 3 or len(post) < 3:
                    year_report["feature_tests"][col] = test
                    continue

                pre_mean = float(pre.mean())
                post_mean = float(post.mean())
                diff = post_mean - pre_mean
                rel_change = abs(diff) / (abs(pre_mean) + 1e-9)
                pooled = float(np.nanmean([pre.std(ddof=1), post.std(ddof=1)]))
                if pooled > 0:
                    effect_like = abs(diff) / pooled
                else:
                    effect_like = np.inf if diff != 0 else 0.0

                p_value = None
                if len(pre) >= 8 and len(post) >= 8:
                    _, p = scipy_stats.ttest_ind(pre, post, equal_var=False, nan_policy="omit")
                    p_value = float(p)

                jump_rule_size = (effect_like >= effect_threshold) and (
                    rel_change >= rel_change_threshold
                )
                jump_rule_stat = (p_value is not None) and (p_value < p_threshold) and (
                    rel_change >= rel_change_threshold
                )
                jump_flag = bool(jump_rule_size or jump_rule_stat)

                test.update(
                    {
                        "pre_mean": pre_mean,
                        "post_mean": post_mean,
                        "mean_diff_post_minus_pre": float(diff),
                        "relative_change": float(rel_change),
                        "effect_size_like": float(effect_like),
                        "ttest_pvalue": p_value,
                        "jump_flag": jump_flag,
                    }
                )
                year_report["feature_tests"][col] = test
                if jump_flag:
                    year_report["jump_detected"] = True

        year_report["apply_transition_dummy"] = bool(
            year_report["has_missing_sep"] and year_report["jump_detected"]
        )
        if year_report["apply_transition_dummy"]:
            report["jump_years"].append(int(year))
            report["jump_detected_any"] = True

        report["years"][str(year)] = year_report

    return report


def add_time_features(df, transition_report):
    data = df.copy()
    data["month"] = data["date"].dt.month.astype(int)
    data["time_trend"] = np.arange(1, len(data) + 1)
    data["missing_transition_phase"] = 0

    for year in transition_report.get("jump_years", []):
        year_mask = data["date"].dt.year == int(year)
        data.loc[year_mask & (data["month"] >= 10), "missing_transition_phase"] = 1

    month_cat = pd.Categorical(data["month"], categories=list(range(1, 13)))
    month_dummies = pd.get_dummies(month_cat, prefix="month", dtype=np.int8)
    data = pd.concat([data, month_dummies], axis=1)
    return data


def iqr_clip_by_month(df, columns, k=1.5):
    data = df.copy()
    outlier_counts = {}
    for col in columns:
        total = 0
        for month in range(1, 13):
            mask = data["month"] == month
            series = data.loc[mask, col].dropna()
            if len(series) < 4:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower = q1 - (k * iqr)
            upper = q3 + (k * iqr)
            month_vals = data.loc[mask, col]
            outlier_mask = month_vals.lt(lower) | month_vals.gt(upper)
            total += int(outlier_mask.sum())
            data.loc[mask, col] = month_vals.clip(lower=lower, upper=upper)
        outlier_counts[col] = total
    return data, outlier_counts


def handle_missing_values(df, exclude_numeric_columns=None):
    data = df.copy().sort_values("date").reset_index(drop=True)
    exclude_numeric_columns = set(exclude_numeric_columns or [])
    numeric_columns = [col for col in NUMERIC_COLUMNS if col not in exclude_numeric_columns]

    data[numeric_columns] = data[numeric_columns].interpolate(
        method="linear", limit_direction="both"
    )
    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].median())

    for col in CATEGORICAL_COLUMNS:
        mode = data[col].mode(dropna=True)
        fallback = mode.iloc[0] if not mode.empty else "UNKNOWN"
        data[col] = data[col].ffill().bfill().fillna(fallback)
    return data


def add_log_features(df):
    data = df.copy()
    for col in ["feed_ffa_pct", "feed_p_ppm", "rbd_p_ppm"]:
        data["log_" + col] = np.log1p(data[col].clip(lower=0))
    return data


def _encode_categorical_ohe(df, cat_cols):
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.int8)

    encoded = encoder.fit_transform(df[cat_cols])
    feature_names = encoder.get_feature_names_out(cat_cols)
    return pd.DataFrame(encoded, columns=feature_names, index=df.index)


def calculate_vif(df, feature_cols):
    clean = df[feature_cols].dropna().astype(float)
    if clean.empty:
        return pd.DataFrame(columns=["feature", "vif"])
    if clean.shape[1] == 1:
        return pd.DataFrame([{"feature": feature_cols[0], "vif": 1.0}])

    with_const = sm.add_constant(clean, has_constant="add")
    values = with_const.values
    rows = []
    for i, col in enumerate(with_const.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(values, i)
        rows.append({"feature": col, "vif": float(vif)})

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def select_features_by_vif(df, feature_cols, severe_threshold=10.0):
    remaining = [c for c in feature_cols if c in df.columns]
    removed = []
    history = []

    while len(remaining) >= 2:
        vif_now = calculate_vif(df, remaining)
        if vif_now.empty:
            break

        worst = vif_now.iloc[0]
        worst_feature = str(worst["feature"])
        worst_vif = float(worst["vif"])
        if np.isfinite(worst_vif) and worst_vif < severe_threshold:
            break

        removed.append(worst_feature)
        history.append({"removed_feature": worst_feature, "vif_at_removal": worst_vif})
        remaining = [c for c in remaining if c != worst_feature]

    final_vif = calculate_vif(df, remaining) if remaining else pd.DataFrame(columns=["feature", "vif"])
    moderate_risk = []
    if not final_vif.empty:
        moderate_risk = (
            final_vif[(final_vif["vif"] >= 5.0) & (final_vif["vif"] < severe_threshold)]["feature"]
            .astype(str)
            .tolist()
        )

    return (
        {
            "severe_threshold": severe_threshold,
            "kept_features": remaining,
            "removed_features": removed,
            "removal_history": history,
            "moderate_risk_features_after_filter": moderate_risk,
        },
        final_vif,
    )


def build_model_ready(df, target_col, retained_vif_features):
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' does not exist in dataset.")

    data = df.copy()
    base_features = [
        "acid_dosing_pct",
        "bleaching_earth_dosing_pct",
        "feed_ffa_pct",
        "feed_mi_pct",
        "feed_iv",
        "feed_dobi",
        "feed_car_pv",
        "feed_p_ppm",
        "log_feed_ffa_pct",
        "log_feed_p_ppm",
        "time_trend",
        "missing_transition_phase",
    ]
    target_derived_features = {target_col, "log_" + target_col}
    base_features = [c for c in base_features if c not in target_derived_features]

    retained_set = set(retained_vif_features)
    base_features = [c for c in base_features if c not in VIF_COLUMNS or c in retained_set]

    month_features = [c for c in data.columns if c.startswith("month_") and c != "month_1"]
    encoded_cat = _encode_categorical_ohe(data, ["feed_tank", "feed_type"])
    keep_cols = list(dict.fromkeys(["date", target_col] + base_features + month_features))
    model = pd.concat([data[keep_cols], encoded_cat], axis=1)
    return model


def build_descriptive_stats(df, numeric_cols):
    valid_cols = [c for c in numeric_cols if c in df.columns]
    if not valid_cols:
        return pd.DataFrame()

    base = df[valid_cols]
    desc = base.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["missing_count"] = base.isna().sum()
    desc["skew"] = base.skew(numeric_only=True)
    desc["kurtosis"] = base.kurtosis(numeric_only=True)
    desc = desc.reset_index().rename(columns={"index": "feature", "count": "non_null_count", "50%": "median"})

    ordered_cols = [
        "feature",
        "non_null_count",
        "missing_count",
        "mean",
        "std",
        "min",
        "25%",
        "median",
        "75%",
        "max",
        "skew",
        "kurtosis",
    ]
    return desc[ordered_cols]


def build_monthly_boxplot_stats(df, columns):
    rows = []
    for col in columns:
        for month in range(1, 13):
            series = df.loc[df["month"] == month, col].dropna().astype(float)
            if series.empty:
                continue
            rows.append(
                {
                    "feature": col,
                    "month": month,
                    "n": int(len(series)),
                    "min": float(series.min()),
                    "q1": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "q3": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_normality_report(df, columns):
    rows = []
    for col in columns:
        series = df[col].dropna().astype(float)
        record = {
            "feature": col,
            "n_obs": int(len(series)),
            "test": None,
            "stat": None,
            "p_value": None,
            "is_normal_at_0_05": None,
        }

        if len(series) >= 8:
            if len(series) <= 5000:
                stat, p_value = scipy_stats.shapiro(series)
                test = "shapiro"
            else:
                stat, p_value = scipy_stats.normaltest(series)
                test = "dagostino_k2"

            record.update(
                {
                    "test": test,
                    "stat": float(stat),
                    "p_value": float(p_value),
                    "is_normal_at_0_05": bool(p_value >= 0.05),
                }
            )

        rows.append(record)

    return pd.DataFrame(rows)


def build_correlations(df, columns, normality_df):
    corr_cols = [c for c in columns if c in df.columns]
    pearson_df = df[corr_cols].corr(method="pearson")
    spearman_df = df[corr_cols].corr(method="spearman")

    usable_flags = normality_df["is_normal_at_0_05"].dropna().tolist()
    if usable_flags and all(usable_flags):
        primary_method = "pearson"
        primary_df = pearson_df
    else:
        primary_method = "spearman"
        primary_df = spearman_df

    return pearson_df, spearman_df, primary_df, primary_method


def build_summary(
    before_df,
    after_df,
    outlier_counts,
    transition_report,
    primary_corr_method,
    normality_df,
    vif_before_df,
    vif_after_df,
    vif_selection,
    input_metadata,
):
    missing_before = before_df.isna().sum().to_dict()
    missing_after = after_df.isna().sum().to_dict()
    normality_records = normality_df.to_dict(orient="records")

    summary = {
        "rows_before": int(len(before_df)),
        "rows_after": int(len(after_df)),
        "input_metadata": input_metadata,
        "date_min": str(after_df["date"].min().date()),
        "date_max": str(after_df["date"].max().date()),
        "months_present": sorted(after_df["month"].unique().tolist()),
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "outliers_clipped_by_month_iqr": outlier_counts,
        "transition_breakpoint": transition_report,
        "correlation_primary_method": primary_corr_method,
        "normality_tests": normality_records,
        "vif_before_filter": vif_before_df.to_dict(orient="records"),
        "vif_after_filter": vif_after_df.to_dict(orient="records"),
        "vif_feature_selection": vif_selection,
        "library_runtime": {
            "statsmodels": statsmodels_version,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    return summary


def run_pipeline(input_paths, processed_dir, report_dir, target_col, vif_threshold, year_filter):
    processed_output = Path(processed_dir)
    report_output = Path(report_dir)
    processed_output.mkdir(parents=True, exist_ok=True)
    report_output.mkdir(parents=True, exist_ok=True)

    raw, input_metadata = load_raw_inputs(input_paths, year_filter=year_filter)
    typed = normalize_and_cast(raw)

    transition_check_columns = list(dict.fromkeys(TRANSITION_BASE_COLUMNS + [target_col]))
    transition_report = detect_transition_breakpoint_by_year(
        typed,
        transition_check_columns,
        effect_threshold=1.0,
        rel_change_threshold=0.2,
        p_threshold=0.01,
    )
    with_time = add_time_features(typed, transition_report)
    iqr_columns = [col for col in IQR_COLUMNS if col != target_col]
    clipped, outlier_counts = iqr_clip_by_month(with_time, iqr_columns, k=1.5)
    imputed = handle_missing_values(clipped, exclude_numeric_columns=[target_col])
    featured = add_log_features(imputed)

    normality_df = build_normality_report(featured, CORR_COLUMNS)
    pearson_df, spearman_df, corr_df, primary_corr_method = build_correlations(
        featured, CORR_COLUMNS, normality_df
    )

    vif_before = calculate_vif(featured, VIF_COLUMNS)
    vif_selection, vif_after = select_features_by_vif(
        featured, VIF_COLUMNS, severe_threshold=vif_threshold
    )

    model_ready = build_model_ready(
        featured, target_col=target_col, retained_vif_features=vif_selection["kept_features"]
    )
    model_ready = model_ready[model_ready[target_col].notna()].reset_index(drop=True)
    model_source_cols = RAW_COLUMNS + ["source_file"]
    model_source = typed[[col for col in model_source_cols if col in typed.columns]].copy()

    descriptive_df = build_descriptive_stats(
        featured, NUMERIC_COLUMNS + ["log_feed_ffa_pct", "log_feed_p_ppm", "log_rbd_p_ppm"]
    )
    monthly_boxplot_df = build_monthly_boxplot_stats(featured, BOXPLOT_COLUMNS)

    summary = build_summary(
        before_df=typed,
        after_df=featured,
        outlier_counts=outlier_counts,
        transition_report=transition_report,
        primary_corr_method=primary_corr_method,
        normality_df=normality_df,
        vif_before_df=vif_before,
        vif_after_df=vif_after,
        vif_selection=vif_selection,
        input_metadata=input_metadata,
    )

    featured.to_csv(processed_output / "processed_full.csv", index=False)
    model_source.to_csv(processed_output / "model_source.csv", index=False)
    model_ready.to_csv(processed_output / "model_ready.csv", index=False)
    descriptive_df.to_csv(report_output / "descriptive_stats.csv", index=False)
    monthly_boxplot_df.to_csv(report_output / "monthly_boxplot_stats.csv", index=False)
    normality_df.to_csv(report_output / "normality_core_metrics.csv", index=False)
    corr_df.to_csv(report_output / "correlation_core_metrics.csv")
    pearson_df.to_csv(report_output / "correlation_pearson_core_metrics.csv")
    spearman_df.to_csv(report_output / "correlation_spearman_core_metrics.csv")
    vif_before.to_csv(report_output / "vif_core_features.csv", index=False)
    vif_after.to_csv(report_output / "vif_core_features_after_filter.csv", index=False)

    with (report_output / "transition_breakpoint_report.json").open("w", encoding="utf-8") as f:
        json.dump(transition_report, f, ensure_ascii=False, indent=2)
    with (report_output / "vif_feature_selection.json").open("w", encoding="utf-8") as f:
        json.dump(vif_selection, f, ensure_ascii=False, indent=2)
    with (report_output / "preprocessing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def parse_args():
    default_target_col = os.getenv("CPO_TARGET_COL", "feed_p_ppm")
    default_vif_threshold = _get_env_float("CPO_VIF_THRESHOLD", 10.0)
    default_raw_inputs = _split_env_list(os.getenv("CPO_RAW_INPUTS"))
    if not default_raw_inputs:
        default_raw_input = os.getenv("CPO_RAW_INPUT")
        default_raw_inputs = [default_raw_input] if default_raw_input else [str(DEFAULT_RAW_EXCEL)]
    default_year = os.getenv("CPO_YEAR", "all")

    parser = argparse.ArgumentParser(description="Wilmar capstone preprocessing and EDA pipeline")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help=(
            "Path to a raw Excel file or a directory containing raw Excel files. "
            "Can be passed multiple times. "
            f"Default: {default_raw_inputs}"
        ),
    )
    parser.add_argument(
        "--year",
        default=default_year,
        help=(
            "Year filter by data date. Use 'all', a single year like '2025', "
            "or comma-separated years like '2024,2025'. "
            f"Default: {default_year}"
        ),
    )
    parser.add_argument(
        "--processed-dir",
        default=str(LOCAL_PROCESSED_DATA_DIR),
        help=f"Directory for processed datasets (default: {LOCAL_PROCESSED_DATA_DIR})",
    )
    parser.add_argument(
        "--report-dir",
        default=str(LOCAL_PREPROCESSING_REPORTS_DIR),
        help=f"Directory for preprocessing reports (default: {LOCAL_PREPROCESSING_REPORTS_DIR})",
    )
    parser.add_argument(
        "--target-col",
        default=default_target_col,
        help=f"Target column kept in model-ready output (default: {default_target_col})",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=default_vif_threshold,
        help=f"Severe VIF threshold for iterative feature removal (default: {default_vif_threshold})",
    )
    args = parser.parse_args()
    if args.input is None:
        args.input = default_raw_inputs
    return args


def main():
    args = parse_args()
    summary = run_pipeline(
        input_paths=args.input,
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        target_col=args.target_col,
        vif_threshold=args.vif_threshold,
        year_filter=args.year,
    )
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
