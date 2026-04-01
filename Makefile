ifneq (,$(wildcard .env))
include .env
export
endif

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

CPO_LOCAL_RAW_DIR ?= local_data/raw
CPO_RAW_INPUT ?= local_data/raw/Copy of R3 QUALITY 2025.xlsx
CPO_PROCESSED_DIR ?= local_data/processed
CPO_REPORTS_DIR ?= local_reports
CPO_PREPROCESSING_REPORT_DIR ?= local_reports/preprocessing
CPO_OLS_REPORT_DIR ?= local_reports/ols
CPO_RF_REPORTS_DIR ?= local_reports/random_forest
CPO_RF_FULL_REPORT_DIR ?= local_reports/random_forest/full_feature
CPO_RF_CORE_REPORT_DIR ?= local_reports/random_forest/core_feature
CPO_RF_COMBO_REPORT_DIR ?= local_reports/random_forest/combo_search
CPO_TARGET_COL ?= rbd_p_ppm
CPO_VIF_THRESHOLD ?= 10

.PHONY: help init-config mkdirs install preprocess ols acf rf-full rf-core rf-combo models all

help:
	@echo "Available targets:"
	@echo "  make init-config   # copy .env.example to .env if missing"
	@echo "  make mkdirs        # create local confidential directories"
	@echo "  make install       # install project dependencies"
	@echo "  make preprocess    # run preprocessing pipeline"
	@echo "  make ols           # run OLS pipeline"
	@echo "  make acf           # generate ACF plot"
	@echo "  make rf-full       # run full-feature random forest"
	@echo "  make rf-core       # run core-feature random forest"
	@echo "  make rf-combo      # run random forest combo search"
	@echo "  make models        # run OLS + ACF + all RF workflows"
	@echo "  make all           # run preprocessing and all model workflows"

init-config:
	@test -f .env || cp .env.example .env
	@echo "Config file ready: .env"

mkdirs:
	@mkdir -p "$(CPO_LOCAL_RAW_DIR)" "$(CPO_PROCESSED_DIR)" \
		"$(CPO_PREPROCESSING_REPORT_DIR)" "$(CPO_OLS_REPORT_DIR)" \
		"$(CPO_RF_FULL_REPORT_DIR)" "$(CPO_RF_CORE_REPORT_DIR)" \
		"$(CPO_RF_COMBO_REPORT_DIR)"

install:
	$(PIP) install -U pip
	$(PIP) install -e .

preprocess: mkdirs
	$(PYTHON) scripts/run_preprocessing.py \
		--input "$(CPO_RAW_INPUT)" \
		--processed-dir "$(CPO_PROCESSED_DIR)" \
		--report-dir "$(CPO_PREPROCESSING_REPORT_DIR)" \
		--target-col "$(CPO_TARGET_COL)" \
		--vif-threshold "$(CPO_VIF_THRESHOLD)"

ols: mkdirs
	$(PYTHON) scripts/run_ols.py \
		--input "$(CPO_PROCESSED_DIR)/model_ready.csv" \
		--processed-dir "$(CPO_PROCESSED_DIR)" \
		--report-dir "$(CPO_OLS_REPORT_DIR)"

acf: mkdirs
	$(PYTHON) scripts/run_acf_plot.py \
		--input "$(CPO_PROCESSED_DIR)/model_ready_lag.csv" \
		--output "$(CPO_OLS_REPORT_DIR)/acf_plot.jpg"

rf-full: mkdirs
	$(PYTHON) scripts/run_rf_full.py \
		--input "$(CPO_PROCESSED_DIR)/model_ready.csv" \
		--output-dir "$(CPO_RF_FULL_REPORT_DIR)"

rf-core: mkdirs
	$(PYTHON) scripts/run_rf_core.py \
		--input "$(CPO_PROCESSED_DIR)/model_ready.csv" \
		--output-dir "$(CPO_RF_CORE_REPORT_DIR)"

rf-combo: mkdirs
	$(PYTHON) scripts/run_rf_combo_search.py \
		--input "$(CPO_PROCESSED_DIR)/model_ready.csv" \
		--output-dir "$(CPO_RF_COMBO_REPORT_DIR)"

models: ols acf rf-full rf-core rf-combo

all: preprocess models
