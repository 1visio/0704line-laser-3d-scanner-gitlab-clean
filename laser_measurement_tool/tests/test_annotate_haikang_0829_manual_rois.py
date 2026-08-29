from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import annotate_haikang_0829_manual_rois as manual_roi


class ValidateRangesTest(unittest.TestCase):
    def test_accepts_ordered_non_overlapping_full_sensor_u(self) -> None:
        manual_roi.validate_ranges(
            {
                "baseline_before": [300, 700],
                "height": [850, 950],
                "baseline_after": [1100, 1500],
            },
            200,
            2447,
        )

    def test_rejects_overlap(self) -> None:
        with self.assertRaisesRegex(manual_roi.ManualRoiError, "before height"):
            manual_roi.validate_ranges(
                {
                    "baseline_before": [300, 850],
                    "height": [850, 950],
                    "baseline_after": [1100, 1500],
                },
                200,
                2447,
            )

    def test_rejects_local_coordinate_that_loses_offset(self) -> None:
        with self.assertRaisesRegex(manual_roi.ManualRoiError, "full-sensor"):
            manual_roi.validate_ranges(
                {
                    "baseline_before": [0, 100],
                    "height": [200, 300],
                    "baseline_after": [400, 500],
                },
                200,
                2447,
            )


class DiscoveryTest(unittest.TestCase):
    def test_requires_the_exact_50_condition_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for height_id in manual_roi.HEIGHT_IDS:
                for position_id in manual_roi.POSITION_IDS:
                    (root / height_id / f"{height_id}_{position_id}").mkdir(
                        parents=True
                    )
            conditions = manual_roi.discover_conditions(root)
            self.assertEqual(len(conditions), 50)
            self.assertEqual(conditions[0].condition, "h02_p01")
            self.assertEqual(conditions[-1].condition, "h30_p10")


class FingerprintTest(unittest.TestCase):
    def test_legacy_fingerprint_is_accepted_once_but_new_hash_mismatch_is_not(self) -> None:
        current = {
            "condition": "h02_p01",
            "config_path": "config.yaml",
            "config_sha256": "config",
            "frames_csv": "frames.csv",
            "frames_csv_sha256": "frames",
            "frame_count": 20,
            "representative_index_one_based": 10,
            "centerline_protocol": "protocol",
            "source_frames": [{"filename": "frame.png", "sha256": "new"}],
            "implementation_sha256": {"extractor": "same"},
        }
        legacy = {
            key: value
            for key, value in current.items()
            if key not in {"source_frames", "implementation_sha256"}
        }
        self.assertTrue(manual_roi.compatible_fingerprint(legacy, current))
        stale = dict(current)
        stale["source_frames"] = [{"filename": "frame.png", "sha256": "old"}]
        self.assertFalse(manual_roi.compatible_fingerprint(stale, current))


if __name__ == "__main__":
    unittest.main()
