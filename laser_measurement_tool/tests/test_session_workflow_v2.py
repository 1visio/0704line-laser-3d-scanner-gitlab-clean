"""Focused regression tests for the in-place Session workflow V2."""

from __future__ import annotations

import os
from dataclasses import replace
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app_config import DEFAULT_CONFIG_PATH, load_app_config
from calibration.session_ground import (
    BoardConfig,
    SessionGroundExtrinsic,
    estimate_session_ground_extrinsic_from_corners,
)
from online.models import CameraConfig, CapturedFrame
from online.session_calibration import (
    SessionGroundRepeatability,
    aggregate_session_ground_extrinsic,
)
from online.window import OnlineCameraWindow


class SessionWorkflowCoreV2Tests(unittest.TestCase):
    def test_five_same_order_corners_use_median_then_shared_pnp(self) -> None:
        board = BoardConfig(11, 8, 20.0)
        K = np.asarray(
            [[820.0, 0.0, 320.0], [0.0, 818.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        D = np.zeros(5, dtype=np.float64)
        rvec = np.asarray([[0.045], [-0.035], [0.012]], dtype=np.float64)
        tvec = np.asarray([[35.0], [-18.0], [720.0]], dtype=np.float64)
        projected, _ = cv2.projectPoints(board.object_points(), rvec, tvec, K, D)
        base = projected.reshape(-1, 2)
        results = [
            estimate_session_ground_extrinsic_from_corners(
                base + np.random.default_rng(seed).normal(0.0, 0.01, base.shape),
                {"K": K, "D": D},
                board,
            )
            for seed in range(5)
        ]

        final, repeatability = aggregate_session_ground_extrinsic(
            results, {"K": K, "D": D}, board
        )

        self.assertEqual(final.status, "success")
        self.assertEqual(final.detection_method, "5_frame_median")
        self.assertEqual(final.detected_corners.shape, (88, 2))
        self.assertEqual(repeatability.accepted_frames, 5)
        self.assertGreaterEqual(repeatability.translation_max_mm, 0.0)
        self.assertGreaterEqual(repeatability.rotation_max_deg, 0.0)

class SessionWorkflowWindowV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_enter_exit_restores_full_camera_config_and_live_controls_keep_mode(self) -> None:
        cases = (
            ("daheng", "measure_tool_daheng_0811.yaml"),
            ("mvs", "measure_tool_haikang_0828.yaml"),
        )
        for backend, config_name in cases:
            with self.subTest(camera_backend=backend):
                config = load_app_config(DEFAULT_CONFIG_PATH.parent / config_name)
                window = OnlineCameraWindow(
                    config,
                    simulate=True,
                    camera_backend=backend,
                )
                try:
                    window.connect_camera()
                    assert window._session is not None
                    original = window._session.config
                    with patch.object(window, "_start_stream_for_session_calibration"):
                        window.calibrate_session_ground()
                    self.assertTrue(window._session_calibration_active)
                    self.assertFalse(window._session_calibration_capturing)
                    self.assertEqual(window._session_calibration_attempts, 0)
                    self.assertEqual(window._session_calibration_frames, [])
                    full_width = window._pipeline.package.image_width
                    full_height = window._pipeline.package.image_height
                    self.assertEqual(
                        (window._session.config.width, window._session.config.height),
                        (full_width, full_height),
                    )
                    if backend == "mvs":
                        self.assertEqual((full_width, full_height), (2448, 2048))
                    self.assertEqual(window._session.config.offset_x, 0)
                    self.assertEqual(window._session.config.offset_y, 0)
                    self.assertEqual(
                        window._session.config,
                        CameraConfig(
                            exposure_us=original.exposure_us,
                            gain_db=original.gain_db,
                            pixel_format=original.pixel_format,
                            offset_x=0,
                            offset_y=0,
                            width=full_width,
                            height=full_height,
                            timeout_ms=original.timeout_ms,
                        ),
                    )

                    window.exposure.setValue(777.0)
                    window._apply_session_calibration_camera_controls()
                    self.assertTrue(window._session_calibration_active)
                    self.assertEqual(window._session.config.exposure_us, 777.0)
                    self.assertEqual(
                        (window._session.config.width, window._session.config.height),
                        (full_width, full_height),
                    )

                    window._exit_session_calibration_mode()
                    self.assertFalse(window._session_calibration_active)
                    self.assertEqual(window._session.config, original)
                finally:
                    window.close()
                    self.application.processEvents()

    def test_session_preview_requires_explicit_capture_button_gate(self) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        window = OnlineCameraWindow(config, simulate=True, camera_backend="daheng")
        try:
            window.connect_camera()
            window._session_calibration_active = True
            board = config.session_ground_calibration.board_config()
            corners = np.asarray(
                [
                    [20.0 + 4.0 * col, 20.0 + 4.0 * row]
                    for row in range(board.pattern_rows)
                    for col in range(board.pattern_cols)
                ],
                dtype=np.float32,
            )
            reference_R, reference_t = window._pipeline.reference_ground_extrinsic
            result = SessionGroundExtrinsic(
                status="success",
                message="ok",
                detected_corners=corners,
                detection_method="SB",
                reprojection_rmse_px=0.1,
                R=reference_R,
                t=reference_t,
                T_ground_from_camera=np.eye(4),
            )
            frame = CapturedFrame(
                image=np.full((32, 32), 120, dtype=np.uint8),
                camera_frame_number=1,
                camera_timestamp_ticks=1,
                host_timestamp_ns=1,
                host_monotonic_ns=10,
            )
            quality = {
                "saturation_ratio": 0.0,
                "dynamic_range_p95_p5": 100.0,
                "edge_margin_px": 20.0,
                "warnings": [],
            }
            with patch(
                "online.window.estimate_session_ground_extrinsic",
                return_value=result,
            ) as estimate, patch(
                "online.window.assess_checkerboard_image_quality",
                return_value=quality,
            ):
                window._handle_session_calibration_frame(frame)
                self.assertEqual(window._session_calibration_attempts, 0)
                estimate.assert_not_called()

                window._session_calibration_capturing = True
                window._handle_session_calibration_frame(frame)
                deadline = time.monotonic() + 2.0
                while (
                    window._session_calibration_worker_busy
                    and time.monotonic() < deadline
                ):
                    QApplication.processEvents()
                    time.sleep(0.01)
                QApplication.processEvents()
                estimate.assert_called_once()
                self.assertEqual(window._session_calibration_attempts, 1)
                self.assertEqual(len(window._session_calibration_frames), 1)
        finally:
            window.close()
            self.application.processEvents()

    def test_mvs_five_accepted_frames_update_overlay_and_runtime_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = load_app_config(
                DEFAULT_CONFIG_PATH.parent / "measure_tool_haikang_0828.yaml"
            )
            assert base.output is not None
            config = replace(
                base,
                output=replace(base.output, directory=directory),
            )
            window = OnlineCameraWindow(config, simulate=True, camera_backend="mvs")
            try:
                window.connect_camera()
                window._session_calibration_active = True
                window._session_calibration_capturing = True
                window._session_calibration_capture_generation = 1
                window._session_calibration_capture_started_host_monotonic_ns = None
                window._session_calibration_frames.clear()
                board = config.session_ground_calibration.board_config()
                corners = np.asarray(
                    [
                        [20.0 + 4.0 * col, 20.0 + 4.0 * row]
                        for row in range(board.pattern_rows)
                        for col in range(board.pattern_cols)
                    ],
                    dtype=np.float32,
                )
                reference_R, reference_t = window._pipeline.reference_ground_extrinsic
                frame_result = SessionGroundExtrinsic(
                    status="success",
                    message="ok",
                    detected_corners=corners,
                    detection_method="SB",
                    reprojection_rmse_px=0.1,
                    R=reference_R,
                    t=reference_t,
                    T_ground_from_camera=np.eye(4),
                )
                final = replace(frame_result, detection_method="5_frame_median")
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
                image = np.full((100, 100), 120, dtype=np.uint8)
                with patch(
                    "online.window.estimate_session_ground_extrinsic",
                    return_value=frame_result,
                ), patch(
                    "online.window.assess_checkerboard_image_quality",
                    return_value=quality,
                ), patch(
                    "online.window.aggregate_session_ground_extrinsic",
                    return_value=(final, repeatability),
                ):
                    for number in range(1, 6):
                        window._handle_session_calibration_frame(
                            CapturedFrame(
                                image=image,
                                camera_frame_number=number,
                                camera_timestamp_ticks=number,
                                host_timestamp_ns=number,
                                host_monotonic_ns=1000 + number,
                                offset_x=0,
                                offset_y=0,
                            )
                        )
                        deadline = time.monotonic() + 2.0
                        while (
                            window._session_calibration_worker_busy
                            and time.monotonic() < deadline
                        ):
                            QApplication.processEvents()
                            time.sleep(0.01)
                        QApplication.processEvents()

                self.assertEqual(len(window._session_calibration_frames), 5)
                self.assertIs(window._active_session_ground_result, final)
                self.assertEqual(window.session_ground_frames_label.text(), "5/5")
                self.assertEqual(window.session_ground_corner_label.text(), "88")
                self.assertEqual(len(window.extracted_corner_scatter.points()), 88)
            finally:
                window.close()
                self.application.processEvents()


if __name__ == "__main__":
    unittest.main()
