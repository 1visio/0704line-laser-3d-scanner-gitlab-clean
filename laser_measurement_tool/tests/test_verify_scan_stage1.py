"""Tests for the read-only Stage-1 acceptance verifier."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scan.config import load_scan_config
from scan.models import ScanProfile, ScanResult
from scan.offline_scan import OfflineScanResult
from scan.session import write_scan_session
from tools.verify_scan_stage1 import main, verify_scan_session


def _write_config(path: Path, output_directory: Path) -> None:
    path.write_text(
        """schema_version: 1
mode: repeat_one
trajectory:
  start_angle_deg: -1.0
  end_angle_deg: 1.0
  step_angle_deg: 1.0
kinematics:
  axis_point_scan_mm: [0.0, 0.0, 0.0]
  axis_direction_scan: [1.0, 0.0, 0.0]
  zero_offset_deg: 0.0
  T_scan_from_camera_zero:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
output:
  directory: %s
  save_profiles_csv: true
  save_ply: true
  save_pcd: true
"""
        % str(output_directory).replace("\\", "/"),
        encoding="utf-8",
    )


def _rotate_x(points: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    output = np.asarray(points, dtype=np.float64).copy()
    y = output[:, 1].copy()
    z = output[:, 2].copy()
    output[:, 1] = cosine * y - sine * z
    output[:, 2] = sine * y + cosine * z
    return output


def _make_result() -> OfflineScanResult:
    points_camera = np.array(
        [[-1.0, 0.0, 700.0], [0.0, 0.0, 700.0], [1.0, 0.0, 700.0]],
        dtype=np.float64,
    )
    pixels_uv = np.column_stack(
        (np.arange(len(points_camera), dtype=np.float64), np.zeros(len(points_camera)))
    )
    angles = (-1.0, 0.0, 1.0)
    profiles = tuple(
        ScanProfile(
            frame_index=index,
            angle_deg=angle,
            pixels_uv=pixels_uv,
            points_camera=points_camera,
            points_scan=_rotate_x(points_camera, angle),
        )
        for index, angle in enumerate(angles)
    )
    result = ScanResult(
        profiles=profiles,
        points_scan=np.concatenate([profile.points_scan for profile in profiles]),
    )
    metadata: dict[str, object] = {
        "mode": "repeat_one",
        "input_image": "laser.png",
        "profile_filenames": ("laser.png", "laser.png", "laser.png"),
        "calibration_package_id": "test-package",
        "calibration_manifest_sha256": "a" * 64,
        "extraction_algorithm_hash": "b" * 64,
        "frame_stats": tuple(
            {
                "frame_index": index,
                "angle_deg": angle,
                "valid_laser_points": len(points_camera),
                "points_camera_count": len(points_camera),
                "points_scan_count": len(points_camera),
            }
            for index, angle in enumerate(angles)
        ),
        "warnings": (),
    }
    return OfflineScanResult(result, metadata)


class VerifyScanStage1Tests(unittest.TestCase):
    def _create_session(self, directory: Path) -> Path:
        config_path = directory / "scan_stage1.yaml"
        output_root = directory / "scan_output"
        _write_config(config_path, output_root)
        config = load_scan_config(config_path)
        return write_scan_session(
            _make_result(),
            output_root,
            scan_config=config,
            scan_config_path=config_path,
        )

    def test_valid_session_passes_without_modifying_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            session = self._create_session(root)
            before = {
                path.relative_to(session): path.read_bytes()
                for path in session.rglob("*")
                if path.is_file()
            }

            with patch(
                "scan.kinematics.transform_points_camera_to_scan",
                side_effect=AssertionError("verifier must not call production kinematics"),
            ):
                report = verify_scan_session(session)

            after = {
                path.relative_to(session): path.read_bytes()
                for path in session.rglob("*")
                if path.is_file()
            }
            self.assertTrue(report.passed, report.checks)
            self.assertLess(report.kinematic_max_error_mm or 1.0, 1.0e-4)
            self.assertEqual(before, after)

    def test_result_point_count_tampering_fails_and_cli_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            session = self._create_session(Path(directory_name))
            result_path = session / "result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["total_point_count"] += 1
            result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            report = verify_scan_session(session)
            exit_code = main([str(session)])

        self.assertFalse(report.passed)
        self.assertEqual(exit_code, 1)
        self.assertTrue(
            any("点数" in check.name or "profiles" in check.detail for check in report.checks)
        )


if __name__ == "__main__":
    unittest.main()
