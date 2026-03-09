## 1. 项目目录

```text
IND5005B-AI-Driven-Phosphorus-Content-Prediction-and-Optimization-in-CPO-Refining/
├── data_processing.py
├── data/
│   └── Copy of R3 QUALITY 2025.xlsx
├── outputs/
│   ├── processed_full.csv
│   ├── model_ready.csv
│   ├── descriptive_stats.csv
│   ├── monthly_boxplot_stats.csv
│   ├── normality_core_metrics.csv
│   ├── correlation_core_metrics.csv
│   ├── correlation_pearson_core_metrics.csv
│   ├── correlation_spearman_core_metrics.csv
│   ├── vif_core_features.csv
│   ├── vif_core_features_after_filter.csv
│   ├── transition_breakpoint_report.json
│   ├── vif_feature_selection.json
│   └── preprocessing_summary.json
└── README.md
```

## 2. 脚本处理流程

1. 读取 Excel 全部工作表并标准化字段。
2. 处理数值/类别异常值占位（如 `STOP`、`PLANT STOPPED`）。
3. 进行“9月缺失”断点检验（8月/10月窗口对比），自动决定是否启用 `missing_transition_phase`。
4. 生成时间特征（`month`、`time_trend`、月份哑变量）。
5. 按月进行 IQR 异常值裁剪（`k=1.5`）。
6. 缺失值插补（数值线性插值+中位数，类别前后填充+众数回填）。
7. 构建对数特征（`log_feed_ffa_pct`、`log_feed_p_ppm`、`log_rbd_p_ppm`）。
8. 生成 EDA 结果：描述统计、月度箱线统计、正态性检验、相关性矩阵、VIF 与高共线性特征筛选。

## 3. 环境依赖

建议使用 Python 3.11+。

```bash
python -m pip install -U pip
python -m pip install numpy pandas scipy scikit-learn statsmodels openpyxl
```

## 4. 运行方式

在项目根目录执行：

```bash
python data_processing.py \
  --input "data/Copy of R3 QUALITY 2025.xlsx" \
  --output-dir "outputs" \
  --target-col "rbd_p_ppm" \
  --vif-threshold 10
```

参数说明：

- `--input`：原始 Excel 路径（必填）
- `--output-dir`：输出目录（默认 `./outputs`）
- `--target-col`：在 `model_ready.csv` 中保留的目标列（默认 `rbd_p_ppm`）
- `--vif-threshold`：严重共线性阈值，超过阈值将迭代剔除高 VIF 特征（默认 `10`）

## 5. 输出文件说明（outputs）

- `processed_full.csv`：完整预处理数据（含时间特征、哑变量、log 特征）。
- `model_ready.csv`：建模就绪数据（已应用 VIF 特征筛选结果）。
- `descriptive_stats.csv`：描述性统计（均值、标准差、分位数、偏度、峰度、缺失计数）。
- `monthly_boxplot_stats.csv`：按月箱线统计（`min/q1/median/q3/max`）。
- `normality_core_metrics.csv`：核心变量正态性检验（Shapiro 或 D’Agostino）。
- `correlation_pearson_core_metrics.csv`：Pearson 相关矩阵。
- `correlation_spearman_core_metrics.csv`：Spearman 相关矩阵。
- `correlation_core_metrics.csv`：主相关矩阵（根据正态性自动选择 Pearson 或 Spearman）。
- `vif_core_features.csv`：VIF 筛选前结果。
- `vif_core_features_after_filter.csv`：VIF 筛选后结果。
- `transition_breakpoint_report.json`：9月缺失断点检验明细（每年、每指标）。
- `vif_feature_selection.json`：VIF 剔除过程与最终保留特征。
- `preprocessing_summary.json`：总摘要（行数、日期范围、缺失值变化、异常值裁剪、断点/VIF/相关性策略、库版本）。
