from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import replay_haikang_manual_roi_c0_0829 as replay


def registry_entry(height_id: str, position_id: str) -> dict[str, object]:
    return {
        "height_id": height_id,
        "position_id": position_id,
        "condition": f"{height_id}_{position_id}",
        "coordinate_system": "full_sensor",
        "baseline_before_u0": 100,
        "baseline_before_u1": 149,
        "height_u0": 170,
        "height_u1": 209,
        "baseline_after_u0": 230,
        "baseline_after_u1": 279,
        "selection_status": "selected",
        "selection_mode": "manual",
        "manual_provenance": {
            "geometry_only": True,
            "truth_values_used": False,
            "height_results_used": False,
            "automatic_roi_used": False,
        },
    }


def synthetic_summaries(scale: float = 1.0, offset: float = 0.0) -> list[dict[str, object]]:
    truth = {"h02": 2.0, "h06": 6.0, "h10": 10.0, "h20": 20.0, "h30": 30.0}
    rows: list[dict[str, object]] = []
    for height_id in replay.HEIGHT_IDS:
        for position_id in replay.POSITION_IDS:
            gt = truth[height_id]
            measured = scale * gt + offset
            rows.append(
                {
                    "height_id": height_id,
                    "position_id": position_id,
                    "height_gt_mm": gt,
                    "h_raw_mm_median": measured,
                    "bias_mm": measured - gt,
                    "valid_frame_count": 20,
                    "valid_frame_ratio": 1.0,
                    "h_raw_temporal_std_mm": 0.01,
                }
            )
    return rows


class RegistryTest(unittest.TestCase):
    def test_accepts_frozen_50_condition_registry(self) -> None:
        entries = [
            registry_entry(height_id, position_id)
            for height_id in replay.HEIGHT_IDS
            for position_id in replay.POSITION_IDS
        ]
        payload = {
            "task": "H0-1M-A",
            "frozen": True,
            "manual_confirmed": True,
            "coordinate_system": "full_sensor",
            "entries": entries,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _, rois = replay.load_registry(path)
        self.assertEqual(len(rois), 50)
        self.assertEqual(rois["h02_p01"].height, (170, 209))


class ManualMaskTest(unittest.TestCase):
    def test_uses_inclusive_full_sensor_u(self) -> None:
        pixels = np.asarray([[99.0, 1.0], [100.0, 2.0], [149.0, 3.0], [150.0, 4.0]])
        mask = replay.inclusive_u_mask(pixels, (100, 149))
        self.assertEqual(mask.tolist(), [False, True, True, False])


class TruthBarrierTest(unittest.TestCase):
    def test_rejects_truth_before_raw_replay(self) -> None:
        with self.assertRaisesRegex(replay.ManualReplayError, "before raw replay"):
            replay.load_ground_truth_after_replay(False)


class AccuracyTest(unittest.TestCase):
    def test_condition_level_exact_response_is_sufficient(self) -> None:
        result = replay.analyze_accuracy(synthetic_summaries(offset=0.05))
        self.assertEqual(result["overall"]["count"], 50)
        self.assertAlmostEqual(result["overall"]["mae_mm"], 0.05)
        self.assertEqual(result["classification"], "C0_SUFFICIENT")
        self.assertEqual(result["monotonic_response"]["monotonic_position_count"], 10)
        self.assertEqual(result["adjacent_height_difference"]["pair_count"], 40)

    def test_scale_response_is_height_trend_without_fitting_correction(self) -> None:
        result = replay.analyze_accuracy(synthetic_summaries(scale=0.9))
        self.assertEqual(result["classification"], "HEIGHT_SCALE_TREND")
        self.assertEqual(result["recommended_next_step"], "H1 feasibility")
        self.assertTrue(
            result["residual_structure"]["decomposition_is_diagnostic_only_not_a_correction_fit"]
        )


if __name__ == "__main__":
    unittest.main()
