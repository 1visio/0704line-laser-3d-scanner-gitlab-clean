"""Tests for the independent Stage-1 scan configuration."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from scan.config import ScanConfigError, load_scan_config


DEFAULT_SCAN_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "scan_stage1.yaml"
)


def _document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "repeat_one",
        "trajectory": {
            "start_angle_deg": -1.0,
            "end_angle_deg": 1.0,
            "step_angle_deg": 0.5,
        },
        "kinematics": {
            "axis_point_scan_mm": [0.0, 0.0, 0.0],
            "axis_direction_scan": [1.0, 0.0, 0.0],
            "zero_offset_deg": 0.0,
            "T_scan_from_camera_zero": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "output": {
            "directory": "../output",
            "save_profiles_csv": True,
            "save_ply": True,
            "save_pcd": True,
        },
    }


class ScanConfigTests(unittest.TestCase):
    def test_default_example_can_be_loaded(self) -> None:
        config = load_scan_config(DEFAULT_SCAN_CONFIG)

        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.mode, "repeat_one")
        self.assertEqual(config.trajectory.start_angle_deg, -6.0)
        self.assertEqual(config.trajectory.end_angle_deg, 6.0)
        self.assertEqual(config.trajectory.step_angle_deg, 0.2)
        self.assertEqual(len(config.trajectory.angles_deg), 61)
        self.assertEqual(config.trajectory.angles_deg[0], -6.0)
        self.assertEqual(config.trajectory.angles_deg[-1], 6.0)
        self.assertEqual(config.kinematics.axis_direction_scan.shape, (3,))
        self.assertEqual(config.kinematics.T_scan_from_camera_zero.shape, (4, 4))
        self.assertTrue(config.output.save_profiles_csv)
        self.assertTrue(config.output.save_ply)
        self.assertTrue(config.output.save_pcd)
        self.assertEqual(
            config.output.directory,
            (DEFAULT_SCAN_CONFIG.parent / "../output/scan_stage1").resolve(),
        )

    def test_zero_step_is_rejected(self) -> None:
        document = _document()
        document["trajectory"] = {
            **document["trajectory"],  # type: ignore[typeddict-item]
            "step_angle_deg": 0.0,
        }
        with self._temporary_config(document) as path:
            with self.assertRaisesRegex(ScanConfigError, "step_angle_deg"):
                load_scan_config(path)

    def test_zero_axis_direction_is_rejected(self) -> None:
        document = _document()
        kinematics = copy.deepcopy(document["kinematics"])
        kinematics["axis_direction_scan"] = [0.0, 0.0, 0.0]
        document["kinematics"] = kinematics
        with self._temporary_config(document) as path:
            with self.assertRaisesRegex(ScanConfigError, "axis_direction_scan"):
                load_scan_config(path)

    def test_invalid_transform_shape_and_last_row_are_rejected(self) -> None:
        document = _document()
        kinematics = copy.deepcopy(document["kinematics"])
        kinematics["T_scan_from_camera_zero"] = [[1.0, 0.0], [0.0, 1.0]]
        document["kinematics"] = kinematics
        with self._temporary_config(document) as path:
            with self.assertRaisesRegex(ScanConfigError, "T_scan_from_camera_zero"):
                load_scan_config(path)

        document = _document()
        kinematics = copy.deepcopy(document["kinematics"])
        transform = kinematics["T_scan_from_camera_zero"]
        transform[3] = [0.0, 0.0, 0.0, 0.0]
        document["kinematics"] = kinematics
        with self._temporary_config(document) as path:
            with self.assertRaisesRegex(ScanConfigError, "最后一行"):
                load_scan_config(path)

    def test_relative_output_path_is_resolved_from_config_directory(self) -> None:
        document = _document()
        document["output"] = {
            **document["output"],  # type: ignore[typeddict-item]
            "directory": "relative/results",
        }
        with TemporaryDirectory() as temporary_directory:
            config_dir = Path(temporary_directory) / "configs"
            config_dir.mkdir()
            path = config_dir / "scan.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")

            config = load_scan_config(path)

        self.assertEqual(config.output.directory, (config_dir / "relative/results").resolve())

    def test_step_direction_must_reach_end_angle(self) -> None:
        for start, end, step in ((-1.0, 1.0, -0.5), (1.0, -1.0, 0.5)):
            with self.subTest(start=start, end=end, step=step):
                document = _document()
                document["trajectory"] = {
                    "start_angle_deg": start,
                    "end_angle_deg": end,
                    "step_angle_deg": step,
                }
                with self._temporary_config(document) as path:
                    with self.assertRaisesRegex(ScanConfigError, "方向"):
                        load_scan_config(path)

    @staticmethod
    def _temporary_config(document: dict[str, object]):
        temporary_directory = TemporaryDirectory()
        path = Path(temporary_directory.name) / "scan.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")

        class _Context:
            def __enter__(self):
                return path

            def __exit__(self, exc_type, exc_value, traceback):
                temporary_directory.cleanup()
                return False

        return _Context()


if __name__ == "__main__":
    unittest.main()
