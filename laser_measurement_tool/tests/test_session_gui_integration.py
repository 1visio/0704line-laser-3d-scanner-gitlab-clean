"""Session ground calibration runtime and online GUI integration tests."""

from __future__ import annotations

import json
import hashlib
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
from online.pipeline import FramePipeline
from online.ground_sanity import GroundSanityResult
from online.session_calibration import (
    build_session_ground_payload,
    compare_ground_extrinsics,
    merge_session_ground_sanity,
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
            sanity_payload = GroundSanityResult(
                status="VALID",
                message="ok",
                ground_extrinsic_source="session",
                frame_number=13,
                session_calibration_frame_number=12,
                input_point_count=88,
                valid_point_count=88,
                bias_zg_mm=0.01,
                rmse_zg_mm=0.02,
                p95_abs_zg_mm=0.03,
                max_abs_zg_mm=0.04,
                ground_slope_mm_per_mm=0.001,
                evaluated_at_utc="2026-01-01T00:00:00+00:00",
            )
            merge_session_ground_sanity(path, sanity_payload.as_dict())
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "VALID")
        self.assertEqual(saved["detection"]["corner_count"], 88)
        self.assertEqual(saved["runtime"]["ground_extrinsic_source"], "session")
        self.assertEqual(saved["delta"]["translation_mm"], 5.0)
        self.assertEqual(saved["laser_ground_sanity"]["status"], "VALID")
        self.assertEqual(saved["session_calibration_status"], "VALID")


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
            self.assertEqual(
                window.session_sanity_status_label.text(),
                "SESSION_CALIBRATION = 未检查",
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
                fake_result = SessionGroundExtrinsic(
                    status="success",
                    message="ok",
                    detected_corners=np.zeros((88, 2), dtype=np.float32),
                    detection_method="SB",
                    reprojection_rmse_px=0.2,
                    R=reference_R,
                    t=reference_t,
                    T_ground_from_camera=np.eye(4),
                )
                with patch(
                    "online.window.estimate_session_ground_extrinsic",
                    return_value=fake_result,
                ):
                    window.calibrate_session_ground()
                self.assertEqual(window._pipeline.ground_extrinsic_source, "session")
                self.assertEqual(window.session_ground_valid_label.text(), "VALID")
                self.assertEqual(window.session_ground_corner_label.text(), "88")
                self.assertTrue(window.ground_sanity_button.isEnabled())
                window._update_ground_sanity_display(
                    GroundSanityResult(
                        status="VALID",
                        message="ok",
                        ground_extrinsic_source="session",
                        frame_number=2,
                        session_calibration_frame_number=1,
                        input_point_count=88,
                        valid_point_count=88,
                        bias_zg_mm=0.01,
                        rmse_zg_mm=0.02,
                        p95_abs_zg_mm=0.03,
                        max_abs_zg_mm=0.04,
                        ground_slope_mm_per_mm=0.001,
                    )
                )
                self.assertEqual(
                    window.session_sanity_status_label.text(),
                    "SESSION_CALIBRATION = VALID",
                )
                self.assertEqual(
                    window.session_sanity_valid_count_label.text(), "88 / 88"
                )
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

    def test_ground5c_frozen_load_is_session_bound_and_reuses_raw_frame(self) -> None:
        frozen_path = (
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "ground5c_frozen_session_linear_0821"
            / "frozen_session_linear.json"
        )
        if not frozen_path.exists():
            self.skipTest("Ground-5C A-2 output is not present in this checkout")

        window = OnlineCameraWindow(self.base_config, simulate=True)
        try:
            window.connect_camera()
            self.assertFalse(window._has_valid_session_pnp())
            reference_R, reference_t = window._pipeline.reference_ground_extrinsic
            window._pipeline.apply_session_ground_extrinsic(
                reference_R, reference_t, generation=1
            )
            window._active_session_ground_result = SessionGroundExtrinsic(
                status="success",
                message="ok",
                R=reference_R,
                t=reference_t,
            )
            window._update_control_states()
            self.assertTrue(window._has_valid_session_pnp())
            self.assertTrue(window.frozen_session_ground_button.isEnabled())

            window._session.start()
            try:
                frame = window._session.get_frame(window._session.config.timeout_ms)
            finally:
                window._session.stop()
            raw_result = window._pipeline.run_frame(frame)
            raw_points = raw_result.points_ground.copy()
            window._last_result = raw_result

            with patch.object(
                window._pipeline,
                "run_frame",
                side_effect=AssertionError("loading must not rerun extraction"),
            ):
                self.assertTrue(window.load_frozen_session_ground(frozen_path))

            self.assertIsNotNone(window._last_result)
            assert window._last_result is not None
            np.testing.assert_array_equal(window._last_result.points_ground_raw, raw_points)
            self.assertEqual(window._last_result.ground_reference_source, "ground5c_frozen_session_linear")
            self.assertEqual(
                window._last_result.ground_reference_frozen_json_sha256,
                hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            )
            self.assertRegex(
                window.ground_reference_runtime_label.text(),
                r"\d+\s*/\s*\d+",
            )
            self.assertLess(len(window.ground_reference_json_sha_label.text()), 40)
            self.assertIn(
                hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                window.ground_reference_json_sha_label.toolTip(),
            )

            window.open_frame_analysis()
            self.application.processEvents()
            self.assertIsNotNone(window._analysis_window)
            self.assertIs(
                window._analysis_window._ground_reference,
                window._pipeline.session_ground_reference,
            )
            window._analysis_window.close()
            window._analysis_window = None

            with tempfile.TemporaryDirectory() as directory:
                assert self.base_config.output is not None
                window._config = replace(
                    window._config,
                    output=replace(
                        window._config.output,
                        directory=Path(directory),
                    ),
                )
                target = window.export_current_frame()
                exported = json.loads(
                    (target / "result.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    exported["ground_reference"]["frozen_json_sha256"],
                    hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    exported["ground_reference_runtime"]["ground_reference_source"],
                    "ground5c_frozen_session_linear",
                )

            # The existing PnP invalidation path clears both the runtime
            # pipeline reference and the GUI's retained reference handle.
            window._pipeline.apply_session_ground_extrinsic(
                reference_R, reference_t, generation=2
            )
            window._invalidate_ground_reference_for_extrinsic_change("PnP changed")
            self.assertIsNone(window._pipeline.session_ground_reference)
            self.assertIsNone(window._last_ground_reference)
        finally:
            window.close()
            self.application.processEvents()


if __name__ == "__main__":
    unittest.main()
