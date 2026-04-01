import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOCAL_DATA_DIR = PROJECT_ROOT / "local_data"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREPROCESSING_REPORTS_DIR = REPORTS_DIR / "preprocessing"
OLS_REPORTS_DIR = REPORTS_DIR / "ols"
RANDOM_FOREST_REPORTS_DIR = REPORTS_DIR / "random_forest"
RF_FULL_REPORTS_DIR = RANDOM_FOREST_REPORTS_DIR / "full_feature"
RF_CORE_REPORTS_DIR = RANDOM_FOREST_REPORTS_DIR / "core_feature"
RF_COMBO_REPORTS_DIR = RANDOM_FOREST_REPORTS_DIR / "combo_search"

LOCAL_RAW_DATA_DIR = _env_path("CPO_LOCAL_RAW_DIR", LOCAL_DATA_DIR / "raw")
LOCAL_PROCESSED_DATA_DIR = _env_path("CPO_PROCESSED_DIR", LOCAL_DATA_DIR / "processed")
LOCAL_REPORTS_DIR = _env_path("CPO_REPORTS_DIR", PROJECT_ROOT / "local_reports")
LOCAL_PREPROCESSING_REPORTS_DIR = _env_path(
    "CPO_PREPROCESSING_REPORT_DIR",
    LOCAL_REPORTS_DIR / "preprocessing",
)
LOCAL_OLS_REPORTS_DIR = _env_path("CPO_OLS_REPORT_DIR", LOCAL_REPORTS_DIR / "ols")
LOCAL_RANDOM_FOREST_REPORTS_DIR = _env_path(
    "CPO_RF_REPORTS_DIR",
    LOCAL_REPORTS_DIR / "random_forest",
)
LOCAL_RF_FULL_REPORTS_DIR = _env_path(
    "CPO_RF_FULL_REPORT_DIR",
    LOCAL_RANDOM_FOREST_REPORTS_DIR / "full_feature",
)
LOCAL_RF_CORE_REPORTS_DIR = _env_path(
    "CPO_RF_CORE_REPORT_DIR",
    LOCAL_RANDOM_FOREST_REPORTS_DIR / "core_feature",
)
LOCAL_RF_COMBO_REPORTS_DIR = _env_path(
    "CPO_RF_COMBO_REPORT_DIR",
    LOCAL_RANDOM_FOREST_REPORTS_DIR / "combo_search",
)

DEFAULT_RAW_EXCEL = _env_path(
    "CPO_RAW_INPUT",
    LOCAL_RAW_DATA_DIR / "Copy of R3 QUALITY 2025.xlsx",
)
