"""Main-window result packaging tests."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

from app_config import DEFAULT_CONFIG_PATH, load_app_config
from gui.main_window import MainWindow


class MainWindowSaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_extraction_stays_in_memory_until_result_is_saved(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            window = MainWindow(load_app_config())
            self.assertEqual(window.online_button.text(), "在线相机")
            window._output_directory = output_directory
            window._image = np.zeros((200, 200), dtype=np.uint8)
            window._image_path = output_directory / "sample.tif"
            window.image_view.set_image(window._image)
            centres = np.array(
                [[90.0, 100.0], [100.0, 100.2], [110.0, 100.4]],
                dtype=np.float64,
            )

            with patch("gui.main_window.extract_laser_center", return_value=centres):
                window._extract_laser_line()

            self.assertEqual(list(output_directory.iterdir()), [])
            self.assertIsNone(window.last_laser_csv_path)

            window._save_results()
            result_directory = output_directory / "sample_measure"
            names = {path.name for path in result_directory.iterdir()}
            self.assertEqual(
                names,
                {
                    "laser_center.csv",
                    "result.json",
                    "overlay.png",
                    "full_laser_ground.ply",
                },
            )
            payload = json.loads(
                (result_directory / "result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(payload["measurement_performed"])
            self.assertEqual(payload["laser_center_csv"], "laser_center.csv")
            self.assertNotIn("obstacles", payload)
            self.assertNotIn("ground_reference_mode", payload)
            self.assertNotIn("results_mm", payload)
            self.assertIsNone(payload["height_raw"])
            self.assertIsNone(payload["height_stage_a"])
            self.assertFalse(payload["stage_a_enabled"])
            self.assertFalse(payload["stage_a_valid"])
            self.assertEqual(
                window.last_laser_csv_path,
                result_directory / "laser_center.csv",
            )
            window.close()

    def test_hardware_roi_metadata_restores_full_image_offset(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "frame_000001.tiff"
            (directory / "frames.csv").write_text(
                "filename,offset_x,offset_y\n"
                "frame_000001.tiff,0,880\n",
                encoding="utf-8",
            )
            window = MainWindow(load_app_config())
            try:
                offset = window._resolve_loaded_image_offset(
                    image_path,
                    np.zeros((300, 2448), dtype=np.uint8),
                )
                self.assertEqual(offset, (0, 880))
                window._image_offset = offset or (0, 0)
                local = np.array([[100.0, 145.25]], dtype=np.float64)
                np.testing.assert_allclose(
                    window._centers_in_calibration_coordinates(local),
                    np.array([[100.0, 1025.25]], dtype=np.float64),
                )
            finally:
                window.close()

    def test_metric_chain_is_visible_without_changing_measurement_values(self) -> None:
        config = load_app_config(
            DEFAULT_CONFIG_PATH.parent / "measure_tool_daheng_0811.yaml"
        )
        measurement = SimpleNamespace(
            ground_reference_mode="baseline_roi_profile",
            ground_baseline_zg_mm=0.25,
            ground_noise_sigma_mm=0.02,
            baseline_fit=None,
            baseline_point_count=20,
            baseline_inlier_count=19,
            height_mean_mm=12.5,
            height_median_mm=12.4,
            height_std_mm=0.1,
            length_mm=6.0,
            angle_with_baseline_deg=0.5,
            height_fit=SimpleNamespace(rmse_mm=0.01),
            height_inlier_count=30,
            height_point_count=31,
            ground_profile_fit=None,
            endpoints_ground=np.zeros((2, 3), dtype=np.float64),
        )
        window = MainWindow(
            config,
            system="daheng",
            ground_extrinsic_source="session",
        )
        try:
            scroll_area = window._control_panel_scroll_area
            self.assertTrue(scroll_area.widgetResizable())
            self.assertEqual(
                scroll_area.verticalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAsNeeded,
            )
            self.assertEqual(
                scroll_area.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            window._update_results_panel([measurement])
            labels = [
                label.text()
                for label in window._obstacle_result_groups[0].findChildren(QLabel)
            ]
            label_text = "\n".join(labels)
            self.assertIn("原始高度 height_raw (均值): 12.500 mm", label_text)
            self.assertIn("补偿高度 height_stage_a: 12.550 mm", label_text)
            self.assertIn("Stage-A 状态: 已应用", label_text)
            self.assertNotIn("ground_reference_mode", label_text)
            self.assertNotIn("ground_extrinsic_source", label_text)
            self.assertEqual(
                window._result_labels["ground_reference_mode"].text(),
                "基准 ROI 地面拟合",
            )
            self.assertEqual(
                window._result_labels["ground_source"].text(),
                "session",
            )
            self.assertEqual(
                window._result_labels["stage_a_enabled"].text(),
                "开启",
            )
            self.assertEqual(
                window._result_labels["stage_a_domain"].text(),
                "1.0–30.0 mm",
            )
            self.assertEqual(measurement.height_mean_mm, 12.5)

            stage_a = window._stage_a_height_result(measurement.height_mean_mm)
            window._last_obstacle_reconstructions = [
                SimpleNamespace(filtered={})
            ]
            window._last_reconstruction = {
                "height": SimpleNamespace(filtered={})
            }
            window._last_full_reconstruction = SimpleNamespace(
                point_count=31,
                filtered={},
            )
            payload = window._measurement_payload([measurement])
            obstacle_values = payload["obstacles"][0]["results_mm"]
            self.assertEqual(obstacle_values["height_raw"], 12.5)
            self.assertAlmostEqual(
                obstacle_values["height_stage_a"],
                12.5504244891715,
                places=14,
            )
            self.assertEqual(
                obstacle_values["ground_reference_mode"],
                "baseline_roi_profile",
            )
            self.assertEqual(
                obstacle_values["ground_extrinsic_source"],
                "session",
            )
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
