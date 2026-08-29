"""实时单帧导出与离线分析入口回归测试。"""

from __future__ import annotations

import json
import os
import time
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app_config import DEFAULT_CONFIG_PATH, load_app_config
from gui.main_window import MainWindow
from online.window import OnlineCameraWindow


class OnlineExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])
        cls.config = load_app_config(DEFAULT_CONFIG_PATH)

    def test_external_frame_preserves_full_sensor_coordinates(self) -> None:
        window = MainWindow()
        full_centers = np.asarray(
            [[100.5, 880.25], [200.0, 900.0]], dtype=np.float64
        )
        window.load_external_frame(
            np.zeros((40, 300), dtype=np.uint8),
            full_centers,
            image_offset=(0, 880),
        )
        np.testing.assert_allclose(
            window.current_laser_centers,
            full_centers - np.asarray((0, 880), dtype=np.float64),
        )
        np.testing.assert_allclose(
            window._centers_in_calibration_coordinates(window.current_laser_centers),
            full_centers,
        )
        np.testing.assert_allclose(window.current_laser_centers_full, full_centers)
        with self.assertRaises(ValueError):
            window.load_external_frame(
                np.zeros((40, 300), dtype=np.uint8),
                full_centers,
                image_offset=(0,),
            )
        window.close()

    def test_ground_extrinsic_source_is_shown_in_results(self) -> None:
        window = MainWindow(ground_extrinsic_source="session")
        self.assertEqual(
            window._result_labels["ground_source"].text(),
            "session",
        )
        window.close()

    def test_current_frame_export_and_analysis_window(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config = replace(
                self.config,
                output=replace(
                    self.config.output,
                    directory=Path(temporary_directory),
                ),
            )
            window = OnlineCameraWindow(config, simulate=True)
            try:
                window.connect_camera()
                window.start_stream()
                deadline = time.monotonic() + 3.0
                while window._last_result is None and time.monotonic() < deadline:
                    self.application.processEvents()
                    time.sleep(0.01)
                self.assertIsNotNone(window._last_result)

                target = window.export_current_frame()
                self.assertEqual(
                    {
                        "laser_center.csv",
                        "full_points.csv",
                        "full_laser_ground.ply",
                        "overlay.png",
                        "result.json",
                    },
                    {path.name for path in target.iterdir()},
                )
                payload = json.loads(
                    (target / "result.json").read_text(encoding="utf-8")
                )
                self.assertEqual(payload["source"], "online")
                self.assertIn("reconstructed", payload["point_counts"])
                self.assertIn("height_raw", payload)
                self.assertIn("height_stage_a", payload)
                self.assertIn("stage_a_enabled", payload)
                self.assertIn("stage_a_valid", payload)
                self.assertIsNone(payload["height_raw"])
                self.assertIsNone(payload["height_stage_a"])
                self.assertFalse(payload["stage_a_enabled"])
                self.assertFalse(payload["stage_a_valid"])

                window.open_frame_analysis()
                self.application.processEvents()
                self.assertIsNotNone(window._analysis_window)
                self.assertEqual(
                    window._analysis_window._image_offset,
                    (0, 800),
                )
            finally:
                window.close()
                self.application.processEvents()


if __name__ == "__main__":
    unittest.main()
