"""Tests for traceable Stage-1 scan session output."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scan.config import ScanOutputConfig, load_scan_config
from scan.models import ScanProfile, ScanResult
from scan.offline_scan import OfflineScanResult
from scan import session as scan_session_module
from scan.session import write_scan_session


DEFAULT_SCAN_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "scan_stage1.yaml"
)


def _result(
    *,
    angles: tuple[float, ...] = (0.0, 5.0),
    points_per_profile: int = 1,
    mode: str = "repeat_one",
    filenames: tuple[str, ...] | None = None,
) -> OfflineScanResult:
    profiles: list[ScanProfile] = []
    for frame_index, angle in enumerate(angles):
        pixels = np.column_stack(
            (
                np.arange(points_per_profile, dtype=np.float64) + 10.0,
                np.arange(points_per_profile, dtype=np.float64) + 20.0,
            )
        )
        points_camera = np.column_stack(
            (
                np.arange(points_per_profile, dtype=np.float64) + 1.0,
                np.zeros(points_per_profile),
                np.ones(points_per_profile),
            )
        )
        points_scan = points_camera + np.array([0.0, angle, 0.0])
        profiles.append(
            ScanProfile(
                frame_index=frame_index,
                angle_deg=angle,
                pixels_uv=pixels,
                points_camera=points_camera,
                points_scan=points_scan,
            )
        )
    scan_result = ScanResult(
        profiles=tuple(profiles),
        points_scan=(
            np.concatenate([profile.points_scan for profile in profiles], axis=0)
            if profiles
            else np.empty((0, 3), dtype=np.float64)
        ),
    )
    metadata: dict[str, object] = {
        "mode": mode,
        "input_image": "laser.png",
        "profile_filenames": filenames
        if filenames is not None
        else tuple("laser.png" for _ in range(len(profiles))),
    }
    return OfflineScanResult(scan_result, metadata)


class ScanSessionTests(unittest.TestCase):
    def test_two_sessions_never_overwrite_each_other(self) -> None:
        config = load_scan_config(DEFAULT_SCAN_CONFIG)
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = write_scan_session(
                _result(),
                temporary_directory,
                scan_config=config,
                scan_config_path=DEFAULT_SCAN_CONFIG,
            )
            second = write_scan_session(
                _result(),
                temporary_directory,
                scan_config=config,
                scan_config_path=DEFAULT_SCAN_CONFIG,
            )

            self.assertNotEqual(first, second)
            self.assertTrue((first / "result.json").is_file())
            self.assertTrue((second / "result.json").is_file())
            self.assertTrue(first.name.startswith("scan_"))
            self.assertTrue(second.name.startswith("scan_"))

    def test_poses_csv_contains_command_measured_and_valid_counts(self) -> None:
        config = load_scan_config(DEFAULT_SCAN_CONFIG)
        result = _result(
            angles=(0.0, 5.0),
            points_per_profile=2,
            filenames=("first.png", "second.png"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_directory = write_scan_session(
                result,
                temporary_directory,
                scan_config=config,
                scan_config_path=DEFAULT_SCAN_CONFIG,
            )
            with (session_directory / "poses.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(
            rows[0].keys(),
            {
                "frame_index",
                "filename",
                "angle_command_deg",
                "angle_measured_deg",
                "valid_laser_points",
                "points_camera_count",
                "points_scan_count",
                "valid_point_count",
                "warning",
            },
        )
        self.assertEqual(rows[0]["filename"], "first.png")
        self.assertEqual(rows[0]["angle_command_deg"], "0.0")
        self.assertEqual(rows[0]["angle_measured_deg"], "0.0")
        self.assertEqual(rows[0]["valid_laser_points"], "2")
        self.assertEqual(rows[0]["points_camera_count"], "2")
        self.assertEqual(rows[0]["points_scan_count"], "2")
        self.assertEqual(rows[0]["valid_point_count"], "2")
        self.assertEqual(rows[1]["angle_command_deg"], "5.0")

    def test_repeat_one_source_and_result_mark_demo_mode(self) -> None:
        config = load_scan_config(DEFAULT_SCAN_CONFIG)
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_directory = write_scan_session(
                _result(mode="repeat_one"),
                temporary_directory,
                scan_config=config,
                scan_config_path=DEFAULT_SCAN_CONFIG,
                measure_tool_config_path="measure_tool.yaml",
                source_image="laser.png",
            )
            source = json.loads((session_directory / "source.json").read_text())
            result = json.loads((session_directory / "result.json").read_text())
            self.assertEqual(result["coordinate_system"], "scan")
            self.assertEqual(result["units"], "mm")
            self.assertEqual(result["total_point_count"], 2)
            self.assertEqual(source["source_image"], str(Path("laser.png").resolve()))
            self.assertNotIn(
                "laser.png", {path.name for path in session_directory.iterdir()}
            )
            self.assertTrue(source["kinematic_demo_only"])
            self.assertTrue(result["kinematic_demo_only"])
            self.assertEqual(source["mode"], "repeat_one")

    def test_empty_cloud_writes_zero_statistics_and_cloud_headers(self) -> None:
        config = load_scan_config(DEFAULT_SCAN_CONFIG)
        result = _result(angles=(), mode="sequence")
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_directory = write_scan_session(
                result,
                temporary_directory,
                scan_config=config,
                scan_config_path=DEFAULT_SCAN_CONFIG,
                mode="sequence",
            )
            metadata = json.loads((session_directory / "result.json").read_text())
            ply_text = (session_directory / "cloud_scan.ply").read_text()
            pcd_text = (session_directory / "cloud_scan.pcd").read_text()

        self.assertEqual(metadata["frame_count"], 0)
        self.assertEqual(metadata["total_point_count"], 0)
        self.assertFalse(metadata["kinematic_demo_only"])
        self.assertIn("element vertex 0", ply_text)
        self.assertIn("POINTS 0", pcd_text)

    def test_profiles_switch_disables_profile_files(self) -> None:
        config = load_scan_config(DEFAULT_SCAN_CONFIG)
        config = replace(
            config,
            output=ScanOutputConfig(
                directory=config.output.directory,
                save_profiles_csv=False,
                save_ply=True,
                save_pcd=True,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_directory = write_scan_session(
                _result(),
                temporary_directory,
                scan_config=config,
            )
            self.assertTrue((session_directory / "profiles").is_dir())
            self.assertEqual(list((session_directory / "profiles").iterdir()), [])
            metadata = json.loads((session_directory / "result.json").read_text())
            self.assertEqual(metadata["output_files"]["profiles"], [])

    def test_write_failure_does_not_leave_success_result(self) -> None:
        config = load_scan_config(DEFAULT_SCAN_CONFIG)
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(
                scan_session_module,
                "_write_scan_pcd",
                side_effect=OSError("simulated write failure"),
            ):
                with self.assertRaises(OSError):
                    write_scan_session(
                        _result(),
                        temporary_directory,
                        scan_config=config,
                    )

            root = Path(temporary_directory)
            self.assertEqual(list(root.glob("scan_*")), [])
            self.assertEqual(list(root.glob(".scan_*")), [])


if __name__ == "__main__":
    unittest.main()
