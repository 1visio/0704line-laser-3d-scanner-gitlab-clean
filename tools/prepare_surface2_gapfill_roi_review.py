"""Prepare geometry-only ROI review for the 33/38/43/48 mm gap-fill capture.

This thin wrapper reuses ``prepare_surface2_roi_review.py``. It changes only
the dataset/truth mapping and the output location; the imported implementation
still performs one Steger extraction per TIFF, five-repeat medians and
geometry-only draft ROI generation. No C0/C1, truth, residual or height value
is read during this stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import prepare_surface2_roi_review as base


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("obs_33mm", "obs_38mm", "obs_43mm", "obs_48mm")
TRUTH_MM = {
    "obs_33mm": 33.0,
    "obs_38mm": 38.0,
    "obs_43mm": 43.0,
    "obs_48mm": 48.0,
}
DATA_ROOT_DEFAULT = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data"
)
CONFIG_DEFAULT = (
    REPO_ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"
)
OUTPUT_DEFAULT = (
    REPO_ROOT
    / "outputs"
    / "daheng_c1_gauge_blocks_20260819_ground4a"
    / "surface2_gapfill_3348_review"
)


def configure_reused_modules() -> None:
    """Inject only the new dataset mapping into the existing implementation."""

    base.DATASETS = DATASETS
    base.TRUTH_MM = TRUTH_MM
    base.DATA_ROOT_DEFAULT = DATA_ROOT_DEFAULT
    base.CONFIG_DEFAULT = CONFIG_DEFAULT
    base.OUTPUT_DEFAULT = OUTPUT_DEFAULT

    base.GAUGE.DATASETS = DATASETS
    base.GAUGE.DATASET_ORDER = {name: index for index, name in enumerate(DATASETS)}
    base.GAUGE.TRUTH_MM = TRUTH_MM
    base.ANNOTATE.DATASETS = DATASETS
    base.ANNOTATE.DATASET_ORDER = {name: index for index, name in enumerate(DATASETS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def main() -> int:
    configure_reused_modules()
    args = parse_args()
    # The existing main owns all audit/CSV/PNG/JSON/report writing. Pass the
    # wrapper's arguments through after dataset injection so that the old
    # 36/40/46 output tree is never touched.
    sys.argv = [
        sys.argv[0],
        "--data-root",
        str(args.data_root),
        "--config",
        str(args.config),
        "--output",
        str(args.output),
    ]
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
