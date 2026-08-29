"""Session ground calibration runtime and online GUI integration tests."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app_config import DEFAULT_CONFIG_PATH, load_app_config
from calibration.session_ground import SessionGroundExtrinsic
from online.models import CapturedFrame
from online.pipeline import FramePipeline
from online.session_calibration import (
    SessionGroundRepeatability,
    build_session_ground_payload,
    compare_ground_extrinsics,
    save_session_ground_payload,
)
from online.window import OnlineCameraWindow


class SessionGroundRuntimeTests(unittest.TestCase):
    def test_runtime_source_switch_does_not_change_reference_arrays(self) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        pipeline = FramePipeline(config)
        reference_R, reference_t = pipeline.reference_ground_extrinsic
        session_t = reference_t.copy()
        session_t[0] += 1.25

        self.assertEqual(pipeline.ground_extrinsic_source, "reference")
        pipeline.apply_session_ground_extrinsic(reference_R, session_t)
        self.assertEqual(pipeline.ground_extrinsic_source, "session")
        np.testing.assert_array_equal(pipeline.reference_ground_extrinsic[0], reference_R)
        np.testing.assert_array_equal(pipeline.reference_ground_extrinsic[1], reference_t)
        np.testing.assert_array_equal(
            pipeline.calibration_for_reconstruction()["t"], session_t
        )
        pipeline.reset_ground_extrinsic()
        self.assertEqual(pipeline.ground_extrinsic_source, "reference")

    def test_delta_and_session_json_are_json_safe_and_replaceable(self) -> None:
        reference_R = np.eye(3)
        reference_t = np.zeros(3)
        session_R = np.eye(3)
        session_t = np.asarray([3.0, 4.0, 0.0])
        translation_delta, rotation_delta = compare_ground_extrinsics(
            reference_R, reference_t, session_R, session_t
        )
        self.assertAlmostEqual(translation_delta, 5.0)
        self.assertAlmostEqual(rotation_delta, 0.0)

        result = SessionGroundExtrinsic(
            status="success",
            message="ok",
            detected_corners=np.zeros((88, 2), dtype=np.float32),
            detection_method="SB",
            reprojection_rmse_px=0.125,
            R=session_R,
            t=session_t,
            T_ground_from_camera=np.eye(4),
        )
        payload = build_session_ground_payload(
            result,
            load_app_config(DEFAULT_CONFIG_PATH).session_ground_calibration.board_config(),
            frame_number=12,
            frame_offset=(1760, 0),
            reference_R=reference_R,
            reference_t=reference_t,
            runtime_source="session",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session_ground_calibration.json"
            save_session_ground_payload(path, payload)
            payload["runtime"]["ground_extrinsic_source"] = "reference"
            save_session_ground_payload(path, payload)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "VALID")
        self.assertEqual(saved["detection"]["corner_count"], 88)
        self.assertEqual(saved["runtime"]["ground_extrinsic_source"], "reference")
        self.assertEqual(saved["delta"]["translation_mm"], 5.0)


class SessionGroundWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])
        cls.base_config = load_app_config(DEFAULT_CONFIG_PATH)

    def test_tall_daheng_preview_fits_and_control_panel_scrolls(self) -> None:
        daheng_config = load_app_config(
            DEFAULT_CONFIG_PATH.parent / "measure_tool_daheng_0811.yaml"
        )
        window = OnlineCameraWindow(
            daheng_config,
            simulate=True,
            camera_backend="daheng",
        )
        try:
            self.assertEqual(window._image_view_mode, "fit")
            self.assertGreater(
                window.control_scroll_area.verticalScrollBar().maximum(), 0
            )
        finally:
            window.close()
            self.application.processEvents()

    def test_required_mode_calibrates_before_stream_and_updates_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assert self.base_config.output is not None
            config = replace(
                self.base_config,
                output=replace(
                    self.base_config.output,
                    directory=Path(directory),
                ),
                session_ground_calibration=replace(
                    self.base_config.session_ground_calibration,
                    mode="required",
                ),
            )
            window = OnlineCameraWindow(config, simulate=True)
            try:
                window.connect_camera()
                self.assertTrue(window.session_ground_button.isEnabled())
                self.assertFalse(window.start_button.isEnabled())
                reference_R, reference_t = window._pipeline.reference_ground_extrinsic
                board = config.session_ground_calibration.board_config()
                corners = np.asarray(
                    [
                        [20.0 + 4.0 * col, 20.0 + 4.0 * row]
                        for row in range(board.pattern_rows)
                        for col in range(board.pattern_cols)
                    ],
                    dtype=np.float32,
                )
                fake_result = SessionGroundExtrinsic(
                    status="success",
                    message="ok",
                    detected_corners=corners,
                    detection_method="SB",
                    reprojection_rmse_px=0.2,
                    R=reference_R,
                    t=reference_t,
                    T_ground_from_camera=np.eye(4),
                )
                with patch.object(window, "_start_stream_for_session_calibration"):
                    window.calibrate_session_ground()

                final = replace(fake_result, detection_method="5_frame_median")
                repeatability = SessionGroundRepeatability(
                    required_frames=5,
                    accepted_frames=5,
                    translation_deltas_mm=(0.0,) * 5,
                    rotation_deltas_deg=(0.0,) * 5,
                    translation_mean_mm=0.0,
                    translation_std_mm=0.0,
                    translation_max_mm=0.0,
                    rotation_mean_deg=0.0,
                    rotation_std_deg=0.0,
                    rotation_max_deg=0.0,
                )
                quality = {
                    "saturation_ratio": 0.0,
                    "dynamic_range_p95_p5": 100.0,
                    "edge_margin_px": 20.0,
                    "warnings": [],
                }
                window._session_calibration_frames = [
                    (
                        fake_result,
                        CapturedFrame(
                            image=np.full((100, 100), 120, dtype=np.uint8),
                            camera_frame_number=number,
                            camera_timestamp_ticks=number,
                            host_timestamp_ns=number,
                            host_monotonic_ns=1000 + number,
                            offset_x=0,
                            offset_y=0,
                        ),
                        quality,
                    )
                    for number in range(1, 6)
                ]
                with patch(
                    "online.window.aggregate_session_ground_extrinsic",
                    return_value=(final, repeatability),
                ):
                    window._finalize_session_calibration()
                window._update_control_states()
                self.assertEqual(window._pipeline.ground_extrinsic_source, "session")
                self.assertEqual(window.session_ground_valid_label.text(), "VALID")
                self.assertEqual(window.session_ground_corner_label.text(), "88")
                self.assertTrue(window.start_button.isEnabled())
                payload = json.loads(
                    (Path(directory) / "session_ground_calibration.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(payload["runtime"]["ground_extrinsic_source"], "session")
                self.assertEqual(payload["detection"]["corner_count"], 88)
            finally:
                window.close()
                self.application.processEvents()

if __name__ == "__main__":
    unittest.main()
