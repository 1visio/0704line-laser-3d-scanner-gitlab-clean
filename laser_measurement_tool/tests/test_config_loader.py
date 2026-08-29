"""标定 YAML 加载与校验测试。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from laser_measurement_tool.calibration.config_loader import (
    CalibrationConfigError,
    CalibrationDimensionError,
    CalibrationFileNotFoundError,
    CalibrationUnitError,
    load_calibration,
    load_calibration_files,
)


class ConfigLoaderTests(unittest.TestCase):
    """验证统一 NumPy 输出和错误分类。"""

    def test_load_calibration_with_optional_compensation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            self._write_yaml(
                directory / "ground_u_compensation.yaml",
                {
                    "units": "px",
                    "coefficients": [0.1, -0.02, 0.003],
                    "sample_table": [[0, 0.0], [100, 0.25]],
                },
            )

            calibration = load_calibration(directory)

        self.assertEqual(calibration["K"].shape, (3, 3))
        self.assertEqual(calibration["D"].shape, (5,))
        self.assertEqual(calibration["plane_abcd"].shape, (4,))
        self.assertEqual(calibration["R"].shape, (3, 3))
        self.assertEqual(calibration["t"].shape, (3,))
        for key in ("K", "D", "plane_abcd", "R", "t"):
            self.assertEqual(calibration[key].dtype, np.float64)
        compensation = calibration["ground_u_compensation"]
        self.assertIsInstance(compensation["coefficients"], np.ndarray)
        self.assertIsInstance(compensation["sample_table"], np.ndarray)

    def test_optional_compensation_is_none_when_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)

            calibration = load_calibration(directory)

        self.assertIsNone(calibration["ground_u_compensation"])

    def test_load_ground_bias_validation_v2_csv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            csv_path = directory / "ground_bias_table.csv"
            csv_path.write_text(
                "column_u_px,xg_mm,bias_mm\n"
                "0,-10,1.5\n"
                "100,0,0.25\n"
                "200,10,-0.5\n",
                encoding="utf-8",
            )

            calibration = load_calibration_files(
                intrinsics=directory / "camera_intrinsics.yaml",
                laser_plane=directory / "laser_plane.yaml",
                extrinsics=directory / "camera_ground_extrinsics.yaml",
                ground_u_compensation=csv_path,
            )

        compensation = calibration["ground_u_compensation"]
        np.testing.assert_allclose(compensation["column_u_px"], [0, 100, 200])
        np.testing.assert_allclose(compensation["bias_mm"], [1.5, 0.25, -0.5])
        self.assertEqual(compensation["source_path"], str(csv_path.resolve()))

    def test_load_ground_bias_validation_npy_dict(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            npy_path = directory / "ground_bias_table.npy"
            np.save(
                npy_path,
                {
                    "columns": np.array([0.0, 100.0, 200.0]),
                    "bias_mm": np.array([1.5, 0.25, -0.5]),
                    "trend_z_mm": np.array([0.75, 0.75, 0.75]),
                    "metadata": {"trend_mode": "constant"},
                },
            )

            calibration = load_calibration_files(
                intrinsics=directory / "camera_intrinsics.yaml",
                laser_plane=directory / "laser_plane.yaml",
                extrinsics=directory / "camera_ground_extrinsics.yaml",
                ground_u_compensation=npy_path,
            )

        compensation = calibration["ground_u_compensation"]
        np.testing.assert_allclose(compensation["column_u_px"], [0, 100, 200])
        np.testing.assert_allclose(compensation["bias_mm"], [1.5, 0.25, -0.5])
        self.assertNotIn("z_offset_mm", compensation)
        self.assertEqual(compensation["source_path"], str(npy_path.resolve()))

    def test_load_ground_bias_v_npy_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            npy_path = directory / "ground_bias_table.npy"
            np.save(
                npy_path,
                {
                    "columns": np.array([10.0, 20.0, 30.0]),
                    "bias_mm": np.array([1.0, 2.0, 3.0]),
                    "metadata": {"compensation_axis": "v"},
                },
            )

            calibration = load_calibration_files(
                intrinsics=directory / "camera_intrinsics.yaml",
                laser_plane=directory / "laser_plane.yaml",
                extrinsics=directory / "camera_ground_extrinsics.yaml",
                ground_u_compensation=npy_path,
            )

        compensation = calibration["ground_u_compensation"]
        self.assertEqual(compensation["compensation_axis"], "v")
        np.testing.assert_allclose(compensation["row_v_px"], [10, 20, 30])
        np.testing.assert_allclose(compensation["coordinate_px"], [10, 20, 30])

    def test_load_ground_bias_v_csv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            csv_path = directory / "ground_bias_table.csv"
            csv_path.write_text(
                "row_v_px,yg_mm,bias_mm\n"
                "10,-10,1.5\n"
                "20,0,0.25\n"
                "30,10,-0.5\n",
                encoding="utf-8",
            )

            calibration = load_calibration_files(
                intrinsics=directory / "camera_intrinsics.yaml",
                laser_plane=directory / "laser_plane.yaml",
                extrinsics=directory / "camera_ground_extrinsics.yaml",
                ground_u_compensation=csv_path,
            )

        compensation = calibration["ground_u_compensation"]
        self.assertEqual(compensation["compensation_axis"], "v")
        np.testing.assert_allclose(compensation["row_v_px"], [10, 20, 30])
        np.testing.assert_allclose(compensation["bias_mm"], [1.5, 0.25, -0.5])

    def test_ground_bias_csv_requires_strictly_increasing_columns(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            csv_path = directory / "ground_bias_table.csv"
            csv_path.write_text(
                "column_u_px,bias_mm\n100,0.1\n100,0.2\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(CalibrationConfigError, "严格递增"):
                load_calibration_files(
                    intrinsics=directory / "camera_intrinsics.yaml",
                    laser_plane=directory / "laser_plane.yaml",
                    extrinsics=directory / "camera_ground_extrinsics.yaml",
                    ground_u_compensation=csv_path,
                )

    def test_existing_camera_and_plane_aliases_are_supported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_yaml(
                directory / "camera_intrinsics.yaml",
                {
                    "camera_matrix": np.eye(3).tolist(),
                    "dist_coeffs": [0, 0, 0, 0, 0],
                },
            )
            self._write_yaml(
                directory / "laser_plane.yaml",
                {
                    "units": "mm",
                    "plane": {"a": 0, "b": 0, "c": 1, "d": -200},
                },
            )
            self._write_extrinsics(directory)

            calibration = load_calibration(directory)

        np.testing.assert_array_equal(calibration["K"], np.eye(3))
        np.testing.assert_array_equal(
            calibration["plane_abcd"],
            np.array([0, 0, 1, -200], dtype=np.float64),
        )

    def test_circular_cone_model_is_loaded_without_plane_alias(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            self._write_yaml(
                directory / "circular_cone.yaml",
                {
                    "model_type": "circular_cone",
                    "units": "mm",
                    "axis_unit_camera": [0.0, 0.0, 2.0],
                    "apex_camera_mm": [1.0, 2.0, 200.0],
                    "half_apex_angle_deg": 45.0,
                    "fit_success": True,
                },
            )
            self._write_extrinsics(directory)

            calibration = load_calibration_files(
                intrinsics=directory / "camera_intrinsics.yaml",
                laser_plane=directory / "circular_cone.yaml",
                extrinsics=directory / "camera_ground_extrinsics.yaml",
            )

        self.assertEqual(calibration["laser_model"]["model_type"], "circular_cone")
        np.testing.assert_allclose(
            calibration["laser_model"]["axis_unit_camera"], [0.0, 0.0, 1.0]
        )
        self.assertNotIn("plane_abcd", calibration)

    def test_circular_cone_fit_failure_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            self._write_yaml(
                directory / "laser_plane.yaml",
                {
                    "model_type": "circular_cone",
                    "axis_unit_camera": [0.0, 0.0, 1.0],
                    "apex_camera_mm": [0.0, 0.0, 200.0],
                    "half_apex_angle_deg": 45.0,
                    "fit_success": False,
                },
            )
            with self.assertRaisesRegex(CalibrationConfigError, "fit_success=false"):
                load_calibration(directory)

    def test_missing_required_file_is_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                CalibrationFileNotFoundError,
                "camera_intrinsics.yaml",
            ):
                load_calibration(temporary_directory)

    def test_parameter_dimension_errors_are_reported(self) -> None:
        cases = (
            ("camera_intrinsics.yaml", "K", [[1, 0], [0, 1]]),
            ("laser_plane.yaml", "plane_abcd", [0, 1, -200]),
            ("camera_ground_extrinsics.yaml", "t", [0, 0]),
        )
        for file_name, key, invalid_value in cases:
            with self.subTest(file_name=file_name, key=key):
                with TemporaryDirectory() as temporary_directory:
                    directory = Path(temporary_directory)
                    self._write_valid_required_files(directory)
                    document = yaml.safe_load(
                        (directory / file_name).read_text(encoding="utf-8")
                    )
                    document[key] = invalid_value
                    self._write_yaml(directory / file_name, document)

                    with self.assertRaises(CalibrationDimensionError):
                        load_calibration(directory)

    def test_incompatible_units_are_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            plane_path = directory / "laser_plane.yaml"
            document = yaml.safe_load(plane_path.read_text(encoding="utf-8"))
            document["units"] = "m"
            self._write_yaml(plane_path, document)

            with self.assertRaises(CalibrationUnitError):
                load_calibration(directory)

    def test_conflicting_optional_compensation_units_are_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            self._write_yaml(
                directory / "ground_u_compensation.yaml",
                {
                    "units": "px",
                    "coordinate_unit": "mm",
                    "coefficients": [0.1, 0.2],
                },
            )

            with self.assertRaisesRegex(CalibrationUnitError, "相互冲突"):
                load_calibration(directory)

    def test_missing_parameter_is_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_valid_required_files(directory)
            self._write_yaml(directory / "laser_plane.yaml", {"units": "mm"})

            with self.assertRaisesRegex(CalibrationConfigError, "plane_abcd"):
                load_calibration(directory)

    def _write_valid_required_files(self, directory: Path) -> None:
        self._write_yaml(
            directory / "camera_intrinsics.yaml",
            {
                "units": "px",
                "K": [[1200, 0, 640], [0, 1200, 512], [0, 0, 1]],
                "D": [0.01, -0.02, 0.001, -0.001, 0.0],
            },
        )
        self._write_yaml(
            directory / "laser_plane.yaml",
            {"units": "mm", "plane_abcd": [0.0, 0.2, 0.98, -200.0]},
        )
        self._write_extrinsics(directory)

    def _write_extrinsics(self, directory: Path) -> None:
        self._write_yaml(
            directory / "camera_ground_extrinsics.yaml",
            {
                "units": "mm",
                "rotation_unit": "dimensionless",
                "R": np.eye(3).tolist(),
                "t": [[10.0], [20.0], [30.0]],
            },
        )

    @staticmethod
    def _write_yaml(path: Path, document) -> None:
        path.write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
