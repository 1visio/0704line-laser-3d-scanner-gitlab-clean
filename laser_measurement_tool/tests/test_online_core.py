from __future__ import annotations

import ctypes
import csv
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QCoreApplication

from app_config import DEFAULT_CONFIG_PATH, load_app_config
from online.fake_camera import SyntheticCameraSession
from online.controller import (
    RESULT_EMIT_INTERVAL_S,
    STATS_EMIT_INTERVAL_S,
    OnlineController,
)
from online.models import CameraConfig, CameraDeviceInfo, CapturedFrame
from online.mvs_camera import _copy_frame_payload
from online.pipeline import FramePipeline
from online.recording import FrameRecorder
from online.runtime import LatestFrameSlot
from online_camera import build_parser
from reconstruction.reconstructor import reconstruct_uv_to_ground


def _frame(number: int, dtype: np.dtype = np.dtype(np.uint8)) -> CapturedFrame:
    return CapturedFrame(
        np.full((8, 12), number, dtype=dtype),
        camera_frame_number=number,
        camera_timestamp_ticks=number * 10,
        host_timestamp_ns=time.time_ns(),
        host_monotonic_ns=time.perf_counter_ns(),
        offset_x=0,
        offset_y=10,
    )


class OnlineCoreTests(unittest.TestCase):
    def test_camera_config_validation(self) -> None:
        defaults = CameraConfig()
        self.assertEqual(defaults.pixel_format, "Mono8")
        self.assertEqual(defaults.exposure_us, 600.0)
        self.assertEqual(
            (defaults.width, defaults.height, defaults.offset_x, defaults.offset_y),
            (2448, 300, 0, 880),
        )
        with self.assertRaises(ValueError):
            CameraConfig(pixel_format="Mono12Packed")
        with self.assertRaises(ValueError):
            CameraConfig(offset_y=-1)

    def test_mvs_payload_copy_owns_memory_after_sdk_buffer_changes(self) -> None:
        source = (ctypes.c_ubyte * 12)(*range(12))
        image = _copy_frame_payload(
            ctypes.addressof(source), 3, 4, np.dtype(np.uint8)
        )
        source[0] = 255

        self.assertTrue(image.flags.owndata)
        self.assertEqual(image.shape, (3, 4))
        self.assertEqual(int(image[0, 0]), 0)

    def test_synthetic_camera_can_reconfigure_while_stopped(self) -> None:
        camera = SyntheticCameraSession(CameraConfig())
        updated = CameraConfig(
            exposure_us=4321.0,
            width=1200,
            height=200,
            offset_x=100,
            offset_y=700,
        )
        self.assertEqual(camera.configure(updated), updated)
        camera.start()
        frame = camera.get_frame()
        self.assertEqual(frame.image.shape, (200, 1200))
        self.assertEqual((frame.offset_x, frame.offset_y), (100, 700))
        with self.assertRaises(RuntimeError):
            camera.configure(CameraConfig())
        camera.stop()

    def test_controller_rolls_back_when_camera_start_fails(self) -> None:
        class StartFailureSession:
            device = CameraDeviceInfo("TEST", "FAIL")
            config = CameraConfig()

            def configure(self, config: CameraConfig) -> CameraConfig:
                self.config = config
                return config

            def start(self) -> None:
                raise RuntimeError("expected start failure")

            def stop(self) -> None:
                pass

            def close(self) -> None:
                pass

        controller = OnlineController()
        with self.assertRaisesRegex(RuntimeError, "expected start failure"):
            controller.start(
                StartFailureSession(),
                FramePipeline(load_app_config(DEFAULT_CONFIG_PATH)),
                FrameRecorder(),
            )
        self.assertFalse(controller.running)

    def test_processing_failure_does_not_stop_camera_acquisition(self) -> None:
        class FailingPipeline:
            def run_frame(self, _frame: CapturedFrame) -> object:
                raise RuntimeError("expected processing failure")

        camera = SyntheticCameraSession(
            CameraConfig(width=32, height=16, offset_y=10), target_fps=200
        )
        controller = OnlineController()
        processing_errors = 0

        def count_processing_errors(_message: str) -> None:
            nonlocal processing_errors
            processing_errors += 1

        controller.processing_failed.connect(count_processing_errors)
        app = QCoreApplication.instance() or QCoreApplication([])
        controller.start(camera, FailingPipeline(), FrameRecorder())
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        try:
            self.assertTrue(controller.running)
            self.assertGreater(controller._captured, 0)
            self.assertEqual(processing_errors, 1)
        finally:
            controller.stop()
            app.processEvents()

    def test_controller_throttles_ui_signals_without_throttling_processing(
        self,
    ) -> None:
        camera = SyntheticCameraSession(
            CameraConfig(width=2448, height=128, offset_y=960), target_fps=1000
        )
        controller = OnlineController()
        result_count = 0
        stats_count = 0

        def count_result(_result: object) -> None:
            nonlocal result_count
            result_count += 1

        def count_stats(_stats: object) -> None:
            nonlocal stats_count
            stats_count += 1

        controller.result_ready.connect(count_result)
        controller.stats_updated.connect(count_stats)
        duration_s = 0.65
        app = QCoreApplication.instance() or QCoreApplication([])
        controller.start(
            camera,
            FramePipeline(load_app_config(DEFAULT_CONFIG_PATH)),
            FrameRecorder(),
        )
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        controller.stop()
        app.processEvents()

        self.assertGreater(controller._processed, result_count * 2)
        self.assertGreater(result_count, 1)
        self.assertGreater(stats_count, 1)
        self.assertLessEqual(
            result_count, int(duration_s / RESULT_EMIT_INTERVAL_S) + 2
        )
        self.assertLessEqual(
            stats_count, int(duration_s / STATS_EMIT_INTERVAL_S) + 3
        )

    def test_latest_slot_replaces_stale_frame(self) -> None:
        slot = LatestFrameSlot()
        slot.put(_frame(1))
        slot.put(_frame(2))
        self.assertEqual(slot.overwritten, 1)
        self.assertEqual(slot.take(0.01).camera_frame_number, 2)
        self.assertIsNone(slot.take(0.01))

    def test_pipeline_restores_full_sensor_roi_coordinates(self) -> None:
        config = CameraConfig(
            pixel_format="Mono8", offset_x=0, offset_y=960, width=2448, height=128
        )
        camera = SyntheticCameraSession(config, target_fps=1000)
        camera.start()
        result = FramePipeline(load_app_config(DEFAULT_CONFIG_PATH)).run_frame(
            camera.get_frame()
        )
        camera.stop()
        self.assertGreater(len(result.centers_uv_full), 2300)
        self.assertGreater(float(np.median(result.centers_uv_full[:, 1])), 1000.0)
        self.assertLess(float(np.median(result.centers_uv_full[:, 1])), 1050.0)
        self.assertIsNone(result._overlay_rgb)
        self.assertEqual(result.overlay_rgb.shape, (128, 2448, 3))
        self.assertIs(result.overlay_rgb, result._overlay_rgb)
        self.assertEqual(result.section_xz.shape[1], 2)

    def test_pipeline_exposes_camera_points_from_reconstruction(self) -> None:
        app_config = load_app_config(DEFAULT_CONFIG_PATH)
        camera_config = CameraConfig(
            pixel_format="Mono8", offset_x=0, offset_y=960, width=2448, height=128
        )
        camera = SyntheticCameraSession(camera_config, target_fps=1000)
        pipeline = FramePipeline(app_config)
        camera.start()
        try:
            frame = camera.get_frame()
            result = pipeline.run_frame(frame)
        finally:
            camera.stop()

        expected = reconstruct_uv_to_ground(
            result.centers_uv_full,
            pipeline.package.calibration,
            app_config.reconstruction,
        )
        np.testing.assert_array_equal(result.pixels_uv, expected.pixels_uv)
        np.testing.assert_array_equal(result.points_camera, expected.points_camera)
        np.testing.assert_array_equal(result.points_ground, expected.points_ground)

    def test_pipeline_accepts_each_configured_extraction_method(self) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        hashes: set[str] = set()
        for method in ("centroid", "steger", "shared_steger"):
            pipeline = FramePipeline(config, method)
            self.assertEqual(pipeline.extraction_method, method)
            self.assertEqual(
                pipeline.extraction_params.options,
                config.extraction_options_by_method[method],
            )
            hashes.add(pipeline.algorithm_config_sha256)
        self.assertEqual(len(hashes), 3)

    def test_online_cli_can_override_extraction_method(self) -> None:
        args = build_parser().parse_args(["--method", "steger", "--simulate"])
        self.assertEqual(args.method, "steger")
        self.assertTrue(args.simulate)

    def test_online_cli_can_select_daheng_backend(self) -> None:
        args = build_parser().parse_args(
            ["--camera-backend", "daheng", "--simulate"]
        )
        self.assertEqual(args.camera_backend, "daheng")
        self.assertTrue(args.simulate)

    def test_recorder_writes_lossless_frames_and_gap_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = FrameRecorder(queue_capacity=4)
            config = CameraConfig(pixel_format="Mono12", width=12, height=8)
            recorder.start(temporary, 2, config)
            self.assertTrue(recorder.enqueue(_frame(10, np.dtype(np.uint16))))
            self.assertTrue(recorder.enqueue(_frame(12, np.dtype(np.uint16))))
            result = recorder.wait(5.0)
            assert result is not None
            self.assertEqual(result.saved_frames, 2)
            self.assertEqual(result.detected_frame_gaps, 1)
            self.assertEqual(len(list(result.output_dir.glob("*.tiff"))), 2)
            with (result.output_dir / "frames.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]["frame_gap"], "1")

            recorder.start(temporary, 1, config)
            self.assertTrue(recorder.enqueue(_frame(20, np.dtype(np.uint16))))
            second = recorder.wait(5.0)
            assert second is not None
            with (second.output_dir / "frames.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                second_rows = list(csv.DictReader(stream))
            self.assertEqual(second_rows[0]["camera_frame_number"], "20")

    def test_recorder_cancel_is_clean_and_non_erroring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = FrameRecorder(queue_capacity=4)
            recorder.start(temporary, 100, CameraConfig(width=12, height=8))
            recorder.enqueue(_frame(1))
            recorder.cancel()
            self.assertIsNone(recorder.wait(5.0))
            self.assertTrue(recorder.cancelled)
            self.assertIsNone(recorder.error)
            self.assertFalse(recorder.active)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_recorder_preserves_temp_data_when_final_rename_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = FrameRecorder(queue_capacity=4)
            config = CameraConfig(width=12, height=8)
            with patch.object(
                Path,
                "rename",
                side_effect=PermissionError(5, "Access is denied"),
            ):
                recorder.start(temporary, 1, config)
                self.assertTrue(recorder.enqueue(_frame(1)))
                with self.assertRaisesRegex(RuntimeError, "临时数据已保留"):
                    recorder.wait(5.0)

            self.assertIsNotNone(recorder.error)
            assert recorder.error is not None
            self.assertIn("临时数据已保留", str(recorder.error))
            temporary_dirs = list(Path(temporary).glob(".recording_*"))
            self.assertEqual(len(temporary_dirs), 1)
            self.assertTrue((temporary_dirs[0] / "frame_000001.png").is_file())


if __name__ == "__main__":
    unittest.main()
