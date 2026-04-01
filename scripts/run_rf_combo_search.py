from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpo_phosphorus.models.rf_combo_search import main


if __name__ == "__main__":
    main()

