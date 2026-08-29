"""load_calibration_files 对仓库真实标定格式的兼容性测试。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from calibration.config_loader import (
    CalibrationConfigError,
    CalibrationFileNotFoundError,
    load_calibration_files,
)


_INTRINSICS = """
image_width: 2448
image_height: 2048
camera_matrix:
  - [4658.5, 0, 1209.5]
  - [0, 4653.0, 991.2]
  - [0, 0, 1]
dist_coeffs: [-0.03, 0.005, -0.002, -0.001, 0]
"""

_LASER_PLANE_COEFFICIENTS = """
coordinate_system: camera
coordinate_unit: mm
coefficients:
  a: -0.0148
  b: 0.9429
  c: 0.3327
  d: -205.16
"""

_EXTRINSICS_TRANSFORM = """
units: mm
T_ground_from_camera:
- [1.0, 0.0, 0.0, -22.6]
- [0.0, -1.0, 0.0, 52.4]
- [0.0, 0.0, -1.0, 718.9]
- [0.0, 0.0, 0.0, 1.0]
"""


class LoadCalibrationFilesTest(unittest.TestCase):
    def _write(self, name: str, content: str) -> Path:
        path = self._directory / name
        path.write_text(content, encoding="utf-8")
        return path

    def setUp(self) -> None:
        self._directory = Path(tempfile.mkdtemp())
        self._intrinsics = self._write("intrinsics.yaml", _INTRINSICS)
        self._laser_plane = self._write(
            "laser_plane.yaml", _LASER_PLANE_COEFFICIENTS
        )
        self._extrinsics = self._write(
            "extrinsics.yaml", _EXTRINSICS_TRANSFORM
        )

    def test_repository_formats_load(self) -> None:
        calibration = load_calibration_files(
            intrinsics=self._intrinsics,
            laser_plane=self._laser_plane,
            extrinsics=self._extrinsics,
        )
        self.assertEqual(calibration["K"].shape, (3, 3))
        self.assertEqual(len(calibration["D"]), 5)
        np.testing.assert_allclose(
            calibration["plane_abcd"],
            [-0.0148, 0.9429, 0.3327, -205.16],
        )
        np.testing.assert_allclose(
            calibration["R"],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        )
        np.testing.assert_allclose(
            calibration["t"], [-22.6, 52.4, 718.9]
        )
        self.assertIsNone(calibration["ground_u_compensation"])

    def test_wrong_coordinate_system_raises(self) -> None:
        self._write(
            "laser_plane.yaml",
            _LASER_PLANE_COEFFICIENTS.replace(
                "coordinate_system: camera", "coordinate_system: ground"
            ),
        )
        with self.assertRaises(CalibrationConfigError):
            load_calibration_files(
                intrinsics=self._intrinsics,
                laser_plane=self._laser_plane,
                extrinsics=self._extrinsics,
            )

    def test_bad_transform_last_row_raises(self) -> None:
        self._write(
            "extrinsics.yaml",
            _EXTRINSICS_TRANSFORM.replace(
                "[0.0, 0.0, 0.0, 1.0]", "[0.0, 0.0, 0.0, 2.0]"
            ),
        )
        with self.assertRaises(CalibrationConfigError):
            load_calibration_files(
                intrinsics=self._intrinsics,
                laser_plane=self._laser_plane,
                extrinsics=self._extrinsics,
            )

    def test_non_orthogonal_rotation_raises(self) -> None:
        self._write(
            "extrinsics.yaml",
            _EXTRINSICS_TRANSFORM.replace(
                "[1.0, 0.0, 0.0, -22.6]", "[1.1, 0.0, 0.0, -22.6]"
            ),
        )
        with self.assertRaises(CalibrationConfigError):
            load_calibration_files(
                intrinsics=self._intrinsics,
                laser_plane=self._laser_plane,
                extrinsics=self._extrinsics,
            )

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(CalibrationFileNotFoundError):
            load_calibration_files(
                intrinsics=self._directory / "missing.yaml",
                laser_plane=self._laser_plane,
                extrinsics=self._extrinsics,
            )

    def test_explicit_ground_u_missing_raises(self) -> None:
        with self.assertRaises(CalibrationFileNotFoundError):
            load_calibration_files(
                intrinsics=self._intrinsics,
                laser_plane=self._laser_plane,
                extrinsics=self._extrinsics,
                ground_u_compensation=self._directory / "missing_u.yaml",
            )


if __name__ == "__main__":
    unittest.main()
