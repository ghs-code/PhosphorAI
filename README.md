# CPO 磷含量预测项目

本项目用于搭建 Wilmar CPO 精炼数据的预处理与建模工作流框架，支持 OLS 与随机森林两类分析流程。

当前仓库按“仅保留工作流框架，不保留任何潜在涉密数据”的原则组织。真实原始数据、处理中间数据、图表和模型结果都应只保存在本地，不进入 Git 仓库。

## 保密数据约定

- 原始输入文件仅允许保存在本地，不得提交到代码仓库。
- 数据处理后的中间数据、分析报告、模型结果和导出图表也应仅保存在本地。
- 仓库中 `data/raw/`、`data/processed/` 和 `reports/` 目录仅作为结构说明与占位，不存放真实业务数据。

## 目录结构

```text
IND5005B-AI-Driven-Phosphorus-Content-Prediction-and-Optimization-in-CPO-Refining/
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
├── local_data/
│   ├── raw/                     # gitignored，本地保密原始输入
│   └── processed/               # gitignored，旧版固定输出目录
├── local_runs/
│   └── YYYYmmdd_HHMMSS/         # gitignored，默认每次运行的独立输出目录
│       ├── processed/
│       └── reports/
├── reports/
│   ├── README.md
│   ├── preprocessing/
│   │   └── README.md
│   ├── ols/
│   │   └── README.md
│   └── random_forest/
│       ├── README.md
│       ├── full_feature/
│       │   └── README.md
│       ├── core_feature/
│       │   └── README.md
│       ├── combo_search/
│       │   └── README.md
│       └── legacy/
│           └── README.md
├── local_reports/
│   ├── preprocessing/           # gitignored，本地保密报告
│   ├── ols/                     # gitignored，本地保密报告
│   └── random_forest/           # gitignored，本地保密报告
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_ols.py
│   ├── run_acf_plot.py
│   ├── run_rf_full.py
│   ├── run_rf_core.py
│   └── run_rf_combo_search.py
├── src/
│   └── cpo_phosphorus/
│       ├── paths.py
│       ├── pipelines/
│       │   └── data_processing.py
│       └── models/
│           ├── acf_plot.py
│           ├── ols.py
│           ├── random_forest_core.py
│           ├── random_forest_full.py
│           └── rf_combo_search.py
├── pyproject.toml
└── README.md
```

## 结构说明

- `src/`：核心源码。
- `scripts/`：运行入口。
- `local_data/raw/`：本地保密原始输入。
- `local_runs/<run_id>/processed/`：默认每次运行生成的本地保密中间数据。
- `local_runs/<run_id>/reports/`：默认每次运行生成的本地保密分析报告与模型产物。
- `local_data/processed/`：旧版固定中间数据目录，仅在关闭归档模式时使用。
- `local_reports/`：旧版固定报告目录，仅在关闭归档模式时使用。
- `data/raw/`：仓库内原始数据占位说明，不存放真实文件。
- `data/processed/`：仓库内中间数据占位说明，不存放真实文件。
- `reports/`：仓库内报告目录占位说明，不存放真实文件。

## 环境准备

建议使用 Python 3.11 及以上版本。

```bash
python3 -m pip install -U pip
python3 -m pip install -e .
```

如果不使用 editable install，也可以直接运行 `scripts/` 下的入口脚本，因为这些脚本会自动引入 `src/`。

## 配置文件

项目提供了示例配置文件 `.env.example`。

推荐先复制一份本地配置：

```bash
cp .env.example .env
```

常用配置项包括：

- `PYTHON`：执行 Python 命令时使用的解释器，默认 `python3`
- `CPO_RAW_INPUT`：本地保密原始文件路径，也可以设置为包含多年份 Excel 的目录
- `CPO_YEAR`：按数据日期选择年份，默认 `all`；可设为 `2024`、`2025` 或 `2024,2025`
- `CPO_ARCHIVE_RUNS`：是否启用每次运行独立归档目录，默认 `1`
- `CPO_RUN_ID`：运行编号，默认按当前时间生成到微秒，如 `20260501_143000_123456`
- `CPO_RUN_ROOT`：当前运行的根目录，默认 `local_runs/<CPO_RUN_ID>`
- `CPO_PROCESSED_DIR`：本地保密中间数据目录；归档模式下自动解析为 `local_runs/<CPO_RUN_ID>/processed`
- `CPO_MODEL_SOURCE`：随机森林模型使用的原始特征源文件；归档模式下自动解析为当前 run 的 `processed/model_source.csv`
- `CPO_MODEL_READY`：OLS 和历史报告流程使用的全量预处理建模文件；归档模式下自动解析为当前 run 的 `processed/model_ready.csv`
- `CPO_PREPROCESSING_REPORT_DIR`：预处理报告目录
- `CPO_OLS_REPORT_DIR`：OLS 报告目录
- `CPO_RF_FULL_REPORT_DIR`：全量特征随机森林报告目录
- `CPO_RF_CORE_REPORT_DIR`：核心变量随机森林报告目录
- `CPO_RF_COMBO_REPORT_DIR`：变量组合搜索报告目录
- `CPO_TARGET_COL`：当前预测目标列，默认 `feed_p_ppm`；如需预测 `rbd_p_ppm`，可在 `.env` 中切换或运行 `make all CPO_TARGET_COL=rbd_p_ppm`
- `CPO_VIF_THRESHOLD`：VIF 严重共线性阈值

代码默认会读取这些环境变量；如果未设置，则使用仓库中约定的本地默认路径。

## 本地数据放置方式

请将保密原始文件放到：

```text
local_data/raw/Copy of R3 QUALITY 2025.xlsx
```

如果有多年份文件，可以都放在 `local_data/raw/` 下，例如：

```text
local_data/raw/R3 QUALITY 2024.xlsx
local_data/raw/Copy of R3 QUALITY 2025.xlsx
```

预处理后的中间数据默认输出到每次运行独立目录：

```text
local_runs/<run_id>/processed/
```

其中 `model_source.csv` 只包含标准化后的原始字段，不做全量补值、IQR 裁剪、one-hot 或 log 特征；随机森林脚本会在 sklearn Pipeline 内对训练折拟合这些预处理步骤，避免验证/测试数据泄漏。`model_ready.csv` 保留给 OLS 和 EDA 报告使用。

分析报告与模型结果默认输出到：

```text
local_runs/<run_id>/reports/
```

这样重复运行 `make all` 不会覆盖旧结果。如需恢复旧版固定输出目录，可运行 `make all CPO_ARCHIVE_RUNS=0`，此时会使用 `.env` 中的 `local_data/processed/` 和 `local_reports/` 等路径，并覆盖同名文件。

## Make 工作流

项目根目录提供了 `Makefile`，可直接驱动常用流程。

查看可用命令：

```bash
make help
```

查看本次运行将使用的输入和输出目录：

```bash
make print-config
```

初始化本地配置：

```bash
make init-config
```

创建本地保密目录：

```bash
make mkdirs
```

安装依赖：

```bash
make install
```

运行完整流程：

```bash
make all
```

默认会生成一个新的运行目录，例如：

```text
local_runs/20260501_143000_123456/
├── processed/
└── reports/
```

选择某一年或合并所有年份：

```bash
make all CPO_RAW_INPUT=local_data/raw CPO_YEAR=2024
make all CPO_RAW_INPUT=local_data/raw CPO_YEAR=2025
make all CPO_RAW_INPUT=local_data/raw CPO_YEAR=all
```

只运行建模部分：

```bash
make models
```

临时切换预测目标，不需要修改代码：

```bash
make all CPO_TARGET_COL=rbd_p_ppm
make models CPO_TARGET_COL=feed_p_ppm
```

单独运行某一步：

```bash
make preprocess
make ols
make acf
make rf-full
make rf-core
make rf-combo
make feed-opt
```

运行最终固定口径实验，输出到 `local_runs/final_feed_2024_2025/`：

```bash
make final-feed
```

如果需要分步执行同一次实验，请手动固定 `CPO_RUN_ID`，保证各步骤读写同一目录：

```bash
make preprocess CPO_RUN_ID=experiment_01
make models CPO_RUN_ID=experiment_01
```

## 运行方式

运行预处理流程：

```bash
python3 scripts/run_preprocessing.py
```

自定义路径示例：

```bash
python3 scripts/run_preprocessing.py \
  --input "local_data/raw" \
  --year "2024,2025" \
  --processed-dir "local_data/processed" \
  --report-dir "local_reports/preprocessing" \
  --target-col "feed_p_ppm" \
  --vif-threshold 10
```

运行 OLS：

```bash
python3 scripts/run_ols.py
python3 scripts/run_acf_plot.py
```

运行随机森林：

```bash
python3 scripts/run_rf_full.py
python3 scripts/run_rf_core.py
python3 scripts/run_rf_combo_search.py
```

## 默认本地输出

预处理：

- `local_runs/<run_id>/processed/processed_full.csv`
- `local_runs/<run_id>/processed/model_ready.csv`
- `local_runs/<run_id>/processed/model_ready_lag.csv`
- `local_runs/<run_id>/reports/preprocessing/`

OLS：

- `local_runs/<run_id>/reports/ols/`

随机森林：

- `local_reports/random_forest/full_feature/`
- `local_reports/random_forest/core_feature/`
- `local_reports/random_forest/combo_search/`

优化后的 feed oil 预测与风险预警：

- `local_runs/<run_id>/reports/feed_model_optimized/feed_model_optimized_baselines.csv`
- `local_runs/<run_id>/reports/feed_model_optimized/feed_model_optimized_comparison.csv`
- `local_runs/<run_id>/reports/feed_model_optimized/feed_model_optimized_year_validation_all_configs.csv`
- `local_runs/<run_id>/reports/feed_model_optimized/feed_model_optimized_risk_metrics.csv`
- `local_runs/<run_id>/reports/feed_model_optimized/feed_model_optimized_run_summary.md`
- `local_runs/<run_id>/reports/feed_model_optimized/feed_model_optimized_summary.json`

## 说明

- 当前最终 feed oil 优化模型固定预测 `feed_p_ppm`，并明确排除 RBD、acid dosing、bleaching earth dosing 和目标衍生泄漏变量；历史 OLS/RF 脚本仍可通过 `CPO_TARGET_COL` 临时切换目标。
- `feed-opt` 会同时输出 ppm 回归指标和高磷风险预警指标。企业阈值未提供时，风险预警默认用 `feed_p_ppm` 的 75/80/90 分位数作为 prototype cutoff。
- 切换目标后请从 `make preprocess` 开始重跑，确保 `model_ready.csv` 的目标列、目标滞后列和 transition 检测保持一致。
- 切换年份范围后也请从 `make preprocess` 开始重跑；`CPO_RAW_INPUT=local_data/raw CPO_YEAR=all` 会合并目录下所有 Excel 文件，`CPO_YEAR=2024` 只保留数据日期为 2024 年的记录。
- OLS 脚本会自动跳过数据中不存在或恒定不变的固定变量，避免在缺失 9 月数据时引入无效的 `month_9` 哑变量。
- 最终业务定位是 ppm prediction + high-risk alert prototype；当前工作流不会输出自动 dosing optimization 建议。
- 若 `local_data/` 或 `local_reports/` 中的文件也不希望出现在本机其他备份链路中，应再结合你本地的磁盘加密、同步和备份策略一并处理。
