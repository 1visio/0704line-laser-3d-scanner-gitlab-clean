"""app_config 统一配置加载的单元测试。"""

import tempfile
import unittest
from pathlib import Path

from app_config import DEFAULT_CONFIG_PATH, AppConfigError, load_app_config
from calibration.config_loader import load_calibration_files


_VALID_CONFIG = """
schema_version: 1
calibration:
  intrinsics: calib/intrinsics.yaml
  laser_plane: calib/laser_plane.yaml
  extrinsics: calib/extrinsics.yaml
  ground_u_compensation: null
extraction:
  method: centroid
  centroid:
    background_kernel: 31
    min_local_contrast_dn: 15.0
  steger: {}
reconstruction:
  min_camera_depth_mm: 200.0
  max_camera_depth_mm: 900.0
measurement:
  outlier_sigma_multiplier: 2.5
  min_baseline_points: 10
  min_height_points: 10
output:
  dir: results
  save_overlay_png: false
  save_full_pointcloud_ply: false
"""


class LoadAppConfigTest(unittest.TestCase):
    def _write_config(self, content: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "measure_tool.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_valid_config_parses_and_resolves_paths(self) -> None:
        path = self._write_config(_VALID_CONFIG)
        config = load_app_config(path)
        base = path.parent
        self.assertEqual(
            config.calibration.intrinsics, (base / "calib/intrinsics.yaml").resolve()
        )
        self.assertIsNone(config.calibration.ground_u_compensation)
        self.assertEqual(config.extraction_method, "centroid")
        self.assertEqual(
            config.extraction_options["background_kernel"], 31
        )
        self.assertIn("steger", config.extraction_options_by_method)
        self.assertEqual(config.reconstruction.min_camera_depth_mm, 200.0)
        self.assertEqual(config.measurement.outlier_sigma_multiplier, 2.5)
        assert config.output is not None
        self.assertEqual(config.output.directory, (base / "results").resolve())
        self.assertFalse(config.output.save_overlay_png)
        self.assertTrue(config.output.save_pointcloud_csv)
        self.assertFalse(config.output.save_full_pointcloud_ply)
        self.assertEqual(config.session_ground_calibration.mode, "optional")
        self.assertEqual(
            config.session_ground_calibration.square_size_mm,
            20.0,
        )
        self.assertEqual(
            config.session_ground_calibration.ground_reference.support_source,
            "pnp_board_mask",
        )
        self.assertEqual(
            config.session_ground_calibration.ground_reference.mask_inset_mm,
            0.0,
        )

    def test_absolute_paths_are_kept(self) -> None:
        directory = Path(tempfile.mkdtemp())
        absolute = directory / "somewhere" / "intrinsics.yaml"
        content = _VALID_CONFIG.replace(
            "calib/intrinsics.yaml", absolute.as_posix()
        )
        path = self._write_config(content)
        config = load_app_config(path)
        self.assertEqual(config.calibration.intrinsics, absolute)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(AppConfigError):
            load_app_config(Path(tempfile.mkdtemp()) / "missing.yaml")

    def test_missing_calibration_section_raises(self) -> None:
        path = self._write_config("schema_version: 1\nextraction:\n  method: centroid\n")
        with self.assertRaises(AppConfigError):
            load_app_config(path)

    def test_unknown_reconstruction_key_raises(self) -> None:
        content = _VALID_CONFIG.replace(
            "min_camera_depth_mm: 200.0", "bogus_key: 1.0"
        )
        path = self._write_config(content)
        with self.assertRaises(AppConfigError):
            load_app_config(path)

    def test_optional_camera_config_parses_without_changing_default(self) -> None:
        camera_section = """camera:
  exposure_us: 2000.0
  gain_db: 0.0
  pixel_format: Mono8
  offset_x: 0
  offset_y: 0
  width: 4096
  height: 3000
  timeout_ms: 3000
"""
        path = self._write_config(camera_section + _VALID_CONFIG)
        config = load_app_config(path)
        assert config.camera is not None
        self.assertEqual(config.camera.width, 4096)
        self.assertEqual(config.camera.height, 3000)
        self.assertEqual(config.camera.timeout_ms, 3000)

        default_config = load_app_config(DEFAULT_CONFIG_PATH)
        assert default_config.camera is not None
        self.assertEqual(default_config.camera.offset_y, 800)
        self.assertEqual(default_config.camera.width, 2440)
        self.assertEqual(default_config.camera.height, 300)

    def test_invalid_measurement_value_raises(self) -> None:
        content = _VALID_CONFIG.replace(
            "outlier_sigma_multiplier: 2.5", "outlier_sigma_multiplier: -1.0"
        )
        path = self._write_config(content)
        with self.assertRaises(AppConfigError):
            load_app_config(path)

    def test_session_ground_calibration_modes_and_output_parse(self) -> None:
        content = _VALID_CONFIG + """
session_ground_calibration:
  mode: required
  pattern_cols: 11
  pattern_rows: 8
  square_size_mm: 20.0
  detector: classic
  output: session/session_ground_calibration.json
  ground_reference:
    support_source: manual_ground_roi
    mask_inset_mm: 3.5
"""
        path = self._write_config(content)
        config = load_app_config(path)
        self.assertEqual(config.session_ground_calibration.mode, "required")
        self.assertEqual(config.session_ground_calibration.detector, "classic")
        self.assertEqual(
            config.session_ground_calibration.output,
            (path.parent / "session/session_ground_calibration.json").resolve(),
        )
        self.assertEqual(
            config.session_ground_calibration.ground_reference.support_source,
            "manual_ground_roi",
        )
        self.assertEqual(
            config.session_ground_calibration.ground_reference.mask_inset_mm,
            3.5,
        )

    def test_invalid_session_ground_calibration_mode_raises(self) -> None:
        content = _VALID_CONFIG + "session_ground_calibration:\n  mode: always\n"
        path = self._write_config(content)
        with self.assertRaises(AppConfigError):
            load_app_config(path)

    def test_session_ground_calibration_accepts_all_modes(self) -> None:
        for mode in ("disabled", "optional", "required"):
            with self.subTest(mode=mode):
                path = self._write_config(
                    _VALID_CONFIG
                    + f"session_ground_calibration:\n  mode: {mode}\n"
                )
                self.assertEqual(
                    load_app_config(path).session_ground_calibration.mode,
                    mode,
                )

    def test_default_config_and_calibration_are_self_contained(self) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(config.extraction_method, "steger")
        self.assertEqual(config.extraction_options["sigma"], 1.5)
        self.assertEqual(config.extraction_options["roi_margin"], 48)
        tool_directory = DEFAULT_CONFIG_PATH.parent.parent.resolve()
        paths = (
            DEFAULT_CONFIG_PATH.parent / "realtime_steger.yaml",
            config.calibration.intrinsics,
            config.calibration.laser_plane,
            config.calibration.extrinsics,
            config.calibration.manifest,
        )
        for path in paths:
            assert path is not None
            path.resolve().relative_to(tool_directory)
            self.assertTrue(path.is_file())

        calibration = load_calibration_files(
            config.calibration.intrinsics,
            config.calibration.laser_plane,
            config.calibration.extrinsics,
            config.calibration.ground_u_compensation,
        )
        self.assertEqual(calibration["K"].shape, (3, 3))
        self.assertEqual(calibration["laser_model"]["model_type"], "circular_cone")
        self.assertNotIn("plane_abcd", calibration)
        self.assertEqual(calibration["R"].shape, (3, 3))
        self.assertIsNone(calibration["ground_u_compensation"])


if __name__ == "__main__":
    unittest.main()
