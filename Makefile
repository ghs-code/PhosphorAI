ifneq (,$(wildcard .env))
include .env
export
endif

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

CPO_LOCAL_RAW_DIR ?= local_data/raw
CPO_RAW_INPUT ?= local_data/raw/Copy of R3 QUALITY 2025.xlsx
CPO_YEAR ?= all
CPO_PROCESSED_DIR ?= local_data/processed
CPO_MODEL_SOURCE ?= $(CPO_PROCESSED_DIR)/model_source.csv
CPO_MODEL_READY ?= $(CPO_PROCESSED_DIR)/model_ready.csv
CPO_REPORTS_DIR ?= local_reports
CPO_PREPROCESSING_REPORT_DIR ?= local_reports/preprocessing
CPO_OLS_REPORT_DIR ?= local_reports/ols
CPO_RF_REPORTS_DIR ?= local_reports/random_forest
CPO_RF_FULL_REPORT_DIR ?= local_reports/random_forest/full_feature
CPO_RF_CORE_REPORT_DIR ?= local_reports/random_forest/core_feature
CPO_RF_COMBO_REPORT_DIR ?= local_reports/random_forest/combo_search
CPO_FEED_OPT_REPORT_DIR ?= local_reports/feed_model_optimized
CPO_YEARLY_RUN_PREFIX ?= optimized_feed
CPO_TARGET_COL ?= feed_p_ppm
CPO_VIF_THRESHOLD ?= 10
CPO_ARCHIVE_RUNS ?= 1
DEFAULT_CPO_RUN_ID := $(shell python3 -c 'from datetime import datetime; print(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))')
CPO_RUN_ID ?= $(DEFAULT_CPO_RUN_ID)
CPO_RUN_ID := $(CPO_RUN_ID)
CPO_RUN_ROOT ?= local_runs/$(CPO_RUN_ID)
CPO_RUN_ROOT := $(CPO_RUN_ROOT)

ifeq ($(CPO_ARCHIVE_RUNS),1)
CPO_PROCESSED_DIR := $(CPO_RUN_ROOT)/processed
CPO_MODEL_SOURCE := $(CPO_PROCESSED_DIR)/model_source.csv
CPO_MODEL_READY := $(CPO_PROCESSED_DIR)/model_ready.csv
CPO_REPORTS_DIR := $(CPO_RUN_ROOT)/reports
CPO_PREPROCESSING_REPORT_DIR := $(CPO_REPORTS_DIR)/preprocessing
CPO_OLS_REPORT_DIR := $(CPO_REPORTS_DIR)/ols
CPO_RF_REPORTS_DIR := $(CPO_REPORTS_DIR)/random_forest
CPO_RF_FULL_REPORT_DIR := $(CPO_RF_REPORTS_DIR)/full_feature
CPO_RF_CORE_REPORT_DIR := $(CPO_RF_REPORTS_DIR)/core_feature
CPO_RF_COMBO_REPORT_DIR := $(CPO_RF_REPORTS_DIR)/combo_search
CPO_FEED_OPT_REPORT_DIR := $(CPO_REPORTS_DIR)/feed_model_optimized
endif

.PHONY: help init-config mkdirs install preprocess ols acf rf-full rf-core rf-combo feed-opt feed-opt-yearly final-feed models all

help:
	@echo "Available targets:"
	@echo "  make print-config  # show resolved input/output paths"
	@echo "  make init-config   # copy .env.example to .env if missing"
	@echo "  make mkdirs        # create local confidential directories"
	@echo "  make install       # install project dependencies"
	@echo "  make preprocess    # run preprocessing pipeline"
	@echo "  make ols           # run OLS pipeline"
	@echo "  make acf           # generate ACF plot"
	@echo "  make rf-full       # run full-feature random forest"
	@echo "  make rf-core       # run core-feature random forest"
	@echo "  make rf-combo      # run random forest combo search"
	@echo "  make feed-opt      # run optimized feed-oil model comparison"
	@echo "  make feed-opt-yearly # run optimized feed model separately for 2024 and 2025"
	@echo "  make final-feed    # run fixed final 2024-2025 feed-oil experiment"
	@echo "  make models        # run OLS + ACF + all RF workflows"
	@echo "  make all           # run preprocessing and all model workflows"
	@echo ""
	@echo "Run archive mode:"
	@echo "  CPO_ARCHIVE_RUNS=1 writes each run to local_runs/<timestamp>/ (default)"
	@echo "  CPO_ARCHIVE_RUNS=0 restores legacy fixed output directories from .env"

init-config:
	@test -f .env || cp .env.example .env
	@echo "Config file ready: .env"

print-config:
	@echo "CPO_ARCHIVE_RUNS=$(CPO_ARCHIVE_RUNS)"
	@echo "CPO_RUN_ID=$(CPO_RUN_ID)"
	@echo "CPO_RUN_ROOT=$(CPO_RUN_ROOT)"
	@echo "CPO_RAW_INPUT=$(CPO_RAW_INPUT)"
	@echo "CPO_YEAR=$(CPO_YEAR)"
	@echo "CPO_TARGET_COL=$(CPO_TARGET_COL)"
	@echo "CPO_PROCESSED_DIR=$(CPO_PROCESSED_DIR)"
	@echo "CPO_MODEL_SOURCE=$(CPO_MODEL_SOURCE)"
	@echo "CPO_MODEL_READY=$(CPO_MODEL_READY)"
	@echo "CPO_PREPROCESSING_REPORT_DIR=$(CPO_PREPROCESSING_REPORT_DIR)"
	@echo "CPO_OLS_REPORT_DIR=$(CPO_OLS_REPORT_DIR)"
	@echo "CPO_RF_FULL_REPORT_DIR=$(CPO_RF_FULL_REPORT_DIR)"
	@echo "CPO_RF_CORE_REPORT_DIR=$(CPO_RF_CORE_REPORT_DIR)"
	@echo "CPO_RF_COMBO_REPORT_DIR=$(CPO_RF_COMBO_REPORT_DIR)"
	@echo "CPO_FEED_OPT_REPORT_DIR=$(CPO_FEED_OPT_REPORT_DIR)"
	@echo "CPO_YEARLY_RUN_PREFIX=$(CPO_YEARLY_RUN_PREFIX)"

mkdirs:
	@mkdir -p "$(CPO_LOCAL_RAW_DIR)" "$(CPO_PROCESSED_DIR)" \
		"$(CPO_PREPROCESSING_REPORT_DIR)" "$(CPO_OLS_REPORT_DIR)" \
		"$(CPO_RF_FULL_REPORT_DIR)" "$(CPO_RF_CORE_REPORT_DIR)" \
		"$(CPO_RF_COMBO_REPORT_DIR)" "$(CPO_FEED_OPT_REPORT_DIR)"

install:
	$(PIP) install -U pip
	$(PIP) install -e .

preprocess: mkdirs
	$(PYTHON) scripts/run_preprocessing.py \
		--input "$(CPO_RAW_INPUT)" \
		--year "$(CPO_YEAR)" \
		--processed-dir "$(CPO_PROCESSED_DIR)" \
		--report-dir "$(CPO_PREPROCESSING_REPORT_DIR)" \
		--target-col "$(CPO_TARGET_COL)" \
		--vif-threshold "$(CPO_VIF_THRESHOLD)"

ols: mkdirs
	$(PYTHON) scripts/run_ols.py \
		--input "$(CPO_MODEL_READY)" \
		--processed-dir "$(CPO_PROCESSED_DIR)" \
		--report-dir "$(CPO_OLS_REPORT_DIR)" \
		--target-col "$(CPO_TARGET_COL)"

acf: mkdirs
	$(PYTHON) scripts/run_acf_plot.py \
		--input "$(CPO_PROCESSED_DIR)/model_ready_lag.csv" \
		--output "$(CPO_OLS_REPORT_DIR)/acf_plot.jpg" \
		--target-col "$(CPO_TARGET_COL)"

rf-full: mkdirs
	$(PYTHON) scripts/run_rf_full.py \
		--input "$(CPO_MODEL_SOURCE)" \
		--output-dir "$(CPO_RF_FULL_REPORT_DIR)" \
		--target-col "$(CPO_TARGET_COL)"

rf-core: mkdirs
	$(PYTHON) scripts/run_rf_core.py \
		--input "$(CPO_MODEL_SOURCE)" \
		--output-dir "$(CPO_RF_CORE_REPORT_DIR)" \
		--target-col "$(CPO_TARGET_COL)"

rf-combo: mkdirs
	$(PYTHON) scripts/run_rf_combo_search.py \
		--input "$(CPO_MODEL_SOURCE)" \
		--output-dir "$(CPO_RF_COMBO_REPORT_DIR)" \
		--target-col "$(CPO_TARGET_COL)"

feed-opt: mkdirs
	$(PYTHON) scripts/run_feed_model_optimized.py \
		--input "$(CPO_MODEL_SOURCE)" \
		--output-dir "$(CPO_FEED_OPT_REPORT_DIR)" \
		--target-col "$(CPO_TARGET_COL)"

feed-opt-yearly:
	$(MAKE) preprocess feed-opt \
		CPO_RUN_ID="$(CPO_YEARLY_RUN_PREFIX)_2024" \
		CPO_RUN_ROOT="local_runs/$(CPO_YEARLY_RUN_PREFIX)_2024" \
		CPO_YEAR=2024
	$(MAKE) preprocess feed-opt \
		CPO_RUN_ID="$(CPO_YEARLY_RUN_PREFIX)_2025" \
		CPO_RUN_ROOT="local_runs/$(CPO_YEARLY_RUN_PREFIX)_2025" \
		CPO_YEAR=2025

final-feed:
	$(MAKE) preprocess ols acf feed-opt \
		CPO_RUN_ID="final_feed_2024_2025" \
		CPO_RUN_ROOT="local_runs/final_feed_2024_2025" \
		CPO_RAW_INPUT="$(CPO_LOCAL_RAW_DIR)" \
		CPO_YEAR=2024,2025 \
		CPO_TARGET_COL=feed_p_ppm

models: ols acf rf-full rf-core rf-combo feed-opt

all: preprocess models
