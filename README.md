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
│   └── processed/               # gitignored，本地保密中间数据
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
- `local_data/processed/`：本地保密中间数据。
- `local_reports/`：本地保密分析报告与模型产物。
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
- `CPO_RAW_INPUT`：本地保密原始文件路径
- `CPO_PROCESSED_DIR`：本地保密中间数据目录
- `CPO_PREPROCESSING_REPORT_DIR`：预处理报告目录
- `CPO_OLS_REPORT_DIR`：OLS 报告目录
- `CPO_RF_FULL_REPORT_DIR`：全量特征随机森林报告目录
- `CPO_RF_CORE_REPORT_DIR`：核心变量随机森林报告目录
- `CPO_RF_COMBO_REPORT_DIR`：变量组合搜索报告目录
- `CPO_TARGET_COL`：预处理输出中保留的目标列
- `CPO_VIF_THRESHOLD`：VIF 严重共线性阈值

代码默认会读取这些环境变量；如果未设置，则使用仓库中约定的本地默认路径。

## 本地数据放置方式

请将保密原始文件放到：

```text
local_data/raw/Copy of R3 QUALITY 2025.xlsx
```

预处理后的中间数据默认输出到：

```text
local_data/processed/
```

分析报告与模型结果默认输出到：

```text
local_reports/
```

如有需要，也可以在运行命令中通过参数覆盖这些默认路径。

## Make 工作流

项目根目录提供了 `Makefile`，可直接驱动常用流程。

查看可用命令：

```bash
make help
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

只运行建模部分：

```bash
make models
```

单独运行某一步：

```bash
make preprocess
make ols
make acf
make rf-full
make rf-core
make rf-combo
```

## 运行方式

运行预处理流程：

```bash
python3 scripts/run_preprocessing.py
```

自定义路径示例：

```bash
python3 scripts/run_preprocessing.py \
  --input "local_data/raw/Copy of R3 QUALITY 2025.xlsx" \
  --processed-dir "local_data/processed" \
  --report-dir "local_reports/preprocessing" \
  --target-col "rbd_p_ppm" \
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

- `local_data/processed/processed_full.csv`
- `local_data/processed/model_ready.csv`
- `local_data/processed/model_ready_lag.csv`
- `local_reports/preprocessing/`

OLS：

- `local_reports/ols/`

随机森林：

- `local_reports/random_forest/full_feature/`
- `local_reports/random_forest/core_feature/`
- `local_reports/random_forest/combo_search/`

## 说明

- 当前建模脚本默认预测 `feed_p_ppm`；预处理后数据中仍保留 `rbd_p_ppm`，以便后续比较或扩展。
- OLS 脚本会自动跳过数据中不存在或恒定不变的固定变量，避免在缺失 9 月数据时引入无效的 `month_9` 哑变量。
- 若 `local_data/` 或 `local_reports/` 中的文件也不希望出现在本机其他备份链路中，应再结合你本地的磁盘加密、同步和备份策略一并处理。
