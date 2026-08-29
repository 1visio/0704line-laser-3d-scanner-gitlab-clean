"""Online window state and camera-configuration lifecycle tests."""

from __future__ import annotations

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from app_config import DEFAULT_CONFIG_PATH, load_app_config
from online.window import (
    ConstrainedImageViewBox,
    OnlineCameraWindow,
    OnlineState,
    _fit_image_view,
    _image_preview_transform,
    _section_connection_mask,
    _section_distance,
    _section_height_range,
)


class OnlineWindowLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])
        cls.config = load_app_config(DEFAULT_CONFIG_PATH)

    def _wait_for_state(
        self,
        window: OnlineCameraWindow,
        expected: OnlineState,
        timeout_s: float = 2.0,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while window._online_state is not expected and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(0.005)
        self.assertIs(window._online_state, expected)

    def test_stop_is_non_blocking_and_restores_idle_controls(self) -> None:
        window = OnlineCameraWindow(self.config, simulate=True)
        window.connect_camera()
        window.start_stream()
        self.assertIs(window._online_state, OnlineState.STREAMING)
        self.assertFalse(window.camera_settings_group.isEnabled())

        started = time.perf_counter()
        window.stop_stream()
        self.assertLess(time.perf_counter() - started, 0.1)
        self.assertIs(window._online_state, OnlineState.STOPPING)
        self._wait_for_state(window, OnlineState.CONNECTED)
        self.assertTrue(window.camera_settings_group.isEnabled())
        self.assertEqual(window.capture_fps_label.text(), "0.0")
        window.disconnect_camera()
        window.close()

    def test_restart_applies_changed_camera_config(self) -> None:
        window = OnlineCameraWindow(self.config, simulate=True)
        window.connect_camera()
        window.start_stream()
        window.stop_stream()
        self._wait_for_state(window, OnlineState.CONNECTED)

        window.exposure.setValue(4321.0)
        window.roi_height.setValue(200)
        window.offset_y.setValue(900)
        window.start_stream()
        self.assertEqual(window._session.config.exposure_us, 4321.0)
        self.assertEqual(window._session.config.height, 200)
        self.assertEqual(window._session.config.offset_y, 900)
        window.disconnect_camera()
        self._wait_for_state(window, OnlineState.DISCONNECTED)
        window.close()

    def test_disconnect_while_streaming_finishes_shutdown(self) -> None:
        window = OnlineCameraWindow(self.config, simulate=True)
        window.connect_camera()
        window.start_stream()
        window.disconnect_camera()
        self.assertTrue(window._pending_disconnect)
        self._wait_for_state(window, OnlineState.DISCONNECTED)
        self.assertIsNone(window._session)
        self.assertFalse(window._controller.running)
        window.close()

    def test_failed_close_keeps_session_available_for_retry(self) -> None:
        window = OnlineCameraWindow(self.config, simulate=True)
        window.connect_camera()
        session = window._session
        original_close = session.close

        def fail_close() -> None:
            raise RuntimeError("close failed")

        session.close = fail_close

        self.assertFalse(window._close_camera_session())
        self.assertIs(window._session, session)
        self.assertIs(window._online_state, OnlineState.ERROR)

        session.close = original_close
        self.assertTrue(window._close_camera_session())
        window.close()

    def test_error_dialog_is_single_non_modal_and_does_not_block_window(self) -> None:
        window = OnlineCameraWindow(self.config, simulate=True)
        try:
            window._show_error("录制收尾失败")
            first = window._error_message_box
            self.assertIsNotNone(first)
            assert first is not None
            self.assertFalse(first.isModal())

            window._show_error("同一个错误不应再次创建弹窗")
            self.assertIs(window._error_message_box, first)

            first.close()
            self.application.processEvents()
            self.assertIsNone(window._error_message_box)
        finally:
            window.close()

    def test_recording_commit_failure_keeps_stream_and_allows_retry(self) -> None:
        window = OnlineCameraWindow(self.config, simulate=True)
        window.connect_camera()
        window.start_stream()
        session = window._session
        try:
            window._recorder._error = RuntimeError("临时目录保留")
            window._poll_recording()
            self.application.processEvents()

            self.assertTrue(window._controller.running)
            self.assertIs(window._online_state, OnlineState.STREAMING)
            self.assertTrue(window.stop_button.isEnabled())
            self.assertTrue(window.record_button.isEnabled())
            self.assertEqual(window.state_label.text(), "录制失败，取流仍在运行")

            window.stop_stream()
            self._wait_for_state(window, OnlineState.CONNECTED)
            self.assertIs(window._session, session)
            window.start_stream()
            self.assertIs(window._online_state, OnlineState.STREAMING)
            self.assertIs(window._session, session)
        finally:
            if window._controller.running:
                window.stop_stream()
                self._wait_for_state(window, OnlineState.CONNECTED)
            window.close()

    def test_stream_error_keeps_connected_session_restartable(self) -> None:
        window = OnlineCameraWindow(self.config, simulate=True)
        window.connect_camera()
        session = window._session
        try:
            window.start_stream()
            window._show_error("模拟取流失败")
            self._wait_for_state(window, OnlineState.ERROR)

            self.assertIs(window._session, session)
            self.assertTrue(window.start_button.isEnabled())
            window.start_stream()
            self.assertIs(window._online_state, OnlineState.STREAMING)
        finally:
            if window._controller.running:
                window.stop_stream()
                self._wait_for_state(window, OnlineState.CONNECTED)
            window.close()

    def test_processing_error_keeps_raw_stream_running(self) -> None:
        window = OnlineCameraWindow(self.config, simulate=True)
        window.connect_camera()
        window.start_stream()
        try:
            window._show_processing_error("逐帧处理失败: simulated")
            self.application.processEvents()

            self.assertTrue(window._controller.running)
            self.assertIs(window._online_state, OnlineState.STREAMING)
            self.assertTrue(window.stop_button.isEnabled())
            self.assertEqual(window.state_label.text(), "处理异常，取流仍在运行")
        finally:
            if window._controller.running:
                window.stop_stream()
                self._wait_for_state(window, OnlineState.CONNECTED)
            window.close()

    def test_image_preview_stays_reachable_and_cannot_zoom_beyond_home(self) -> None:
        view_box = ConstrainedImageViewBox()
        view = pg.PlotWidget(viewBox=view_box)
        view.resize(1200, 400)
        view.setAspectLocked(True)
        view.show()
        self.application.processEvents()
        try:
            for height in (300, 2048):
                image = np.zeros((height, 2448), dtype=np.uint8)
                for mode in ("width", "fit"):
                    _fit_image_view(view, image, mode)
                    home_x, home_y = view_box.viewRange()
                    home_span = (
                        home_x[1] - home_x[0],
                        home_y[1] - home_y[0],
                    )
                    if mode == "fit":
                        self.assertLessEqual(home_x[0], 0)
                        self.assertGreaterEqual(home_x[1], 2448)
                        self.assertLessEqual(home_y[0], 0)
                        self.assertGreaterEqual(home_y[1], height)

                    view_box.scaleBy([0.5, 0.5])
                    view_box.translateBy(x=100_000, y=100_000)
                    moved_x, moved_y = view_box.viewRange()
                    self.assertGreater(moved_x[1], 0)
                    self.assertLess(moved_x[0], 2448)
                    self.assertGreater(moved_y[1], 0)
                    self.assertLess(moved_y[0], height)

                    view_box.scaleBy([100.0, 100.0])
                    final_x, final_y = view_box.viewRange()
                    self.assertLessEqual(
                        final_x[1] - final_x[0], home_span[0] + 1e-6
                    )
                    self.assertLessEqual(
                        final_y[1] - final_y[0], home_span[1] + 1e-6
                    )
        finally:
            view.close()

    def test_preview_rotation_is_display_only_and_swaps_view_extents(self) -> None:
        image = np.arange(6, dtype=np.uint8).reshape(2, 3)
        original = image.copy()
        transform = _image_preview_transform(image, True)

        top_left = transform.map(QPointF(0.0, 0.0))
        top_right = transform.map(QPointF(3.0, 0.0))
        bottom_left = transform.map(QPointF(0.0, 2.0))
        self.assertEqual((top_left.x(), top_left.y()), (2.0, 0.0))
        self.assertEqual((top_right.x(), top_right.y()), (2.0, 3.0))
        self.assertEqual((bottom_left.x(), bottom_left.y()), (0.0, 0.0))
        np.testing.assert_array_equal(image, original)

        view_box = ConstrainedImageViewBox()
        view = pg.PlotWidget(viewBox=view_box)
        view.resize(400, 300)
        view.setAspectLocked(True)
        view.show()
        self.application.processEvents()
        try:
            _fit_image_view(view, image, "fit", rotated=True)
            self.assertEqual(view_box._image_size, (2.0, 3.0))
        finally:
            view.close()

    def test_rotation_button_updates_both_preview_items(self) -> None:
        window = OnlineCameraWindow(self.config, simulate=True)
        image = np.arange(6, dtype=np.uint8).reshape(2, 3)
        window.raw_image_item.setImage(image, autoLevels=False)
        window.extracted_image_item.setImage(image, autoLevels=False)

        window.image_rotate_button.setChecked(True)

        self.assertTrue(window._image_preview_rotated)
        self.assertEqual(
            window.raw_image_item.transform(),
            window.extracted_image_item.transform(),
        )
        self.assertEqual(
            window.raw_image_item.transform(),
            window.raw_image_boundary.transform(),
        )
        self.assertEqual(
            window.extracted_image_item.transform(),
            window.extracted_image_boundary.transform(),
        )
        self.assertFalse(window.raw_image_item.transform().isIdentity())
        np.testing.assert_array_equal(window.raw_image_item.image, image)
        np.testing.assert_array_equal(window.extracted_image_item.image, image)

        window.image_rotate_button.setChecked(False)

        self.assertFalse(window._image_preview_rotated)
        self.assertTrue(window.raw_image_item.transform().isIdentity())
        self.assertTrue(window.extracted_image_item.transform().isIdentity())
        window.close()

    def test_section_keeps_points_and_breaks_on_each_distance_threshold(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [4.0, 0.0, 1.0],
                [4.5, 0.0, 5.0],
                [5.0, 4.0, 5.0],
                [5.5, 4.0, 5.5],
            ],
            dtype=np.float64,
        )
        expected = np.array([1, 0, 0, 0, 1, 0], dtype=np.int32)
        section_distance = np.array([0.0, 1.0, 4.0, 4.5, 5.0, 5.5])
        connections = _section_connection_mask(
            points,
            section_distance,
            max_ds=2.0,
            max_dz=3.0,
            max_distance=4.0,
        )
        np.testing.assert_array_equal(connections, expected)

        window = OnlineCameraWindow(self.config, simulate=True)
        window._update_section_view(points)
        scatter_x, scatter_z = window.section_scatter.getData()
        self.assertEqual(len(scatter_x), len(points))
        np.testing.assert_allclose(scatter_x, _section_distance(points[:, :2]))
        np.testing.assert_allclose(scatter_z, points[:, 2])
        np.testing.assert_array_equal(
            window.section_curve.opts["connect"],
            _section_connection_mask(
                points,
                _section_distance(points[:, :2]),
                max_ds=window.section_max_ds.value(),
                max_dz=window.section_max_dz.value(),
                max_distance=window.section_max_distance.value(),
            ),
        )
        self.assertEqual(
            window.section_view.getAxis("bottom").labelText,
            "S（沿激光线）",
        )
        window.close()

    def test_section_distance_supports_horizontal_vertical_and_diagonal_lines(
        self,
    ) -> None:
        parameter = np.arange(4, dtype=np.float64)
        horizontal = np.column_stack((parameter, np.zeros_like(parameter)))
        vertical = np.column_stack((np.zeros_like(parameter), parameter))
        diagonal = np.column_stack((parameter, parameter))

        np.testing.assert_allclose(_section_distance(horizontal), parameter)
        np.testing.assert_allclose(_section_distance(vertical), parameter)
        np.testing.assert_allclose(
            _section_distance(diagonal), parameter * np.sqrt(2.0)
        )

    def test_auto_height_ignores_sparse_outlier_but_fit_includes_all_points(
        self,
    ) -> None:
        normal_heights = np.linspace(-0.3, 0.3, 1000)
        heights = np.append(normal_heights, 50.0)
        robust_lower, robust_upper = _section_height_range(heights, robust=True)
        full_lower, full_upper = _section_height_range(heights, robust=False)
        self.assertLess(robust_lower, normal_heights.min())
        self.assertGreater(robust_upper, normal_heights.max())
        self.assertLess(robust_upper, 2.0)
        self.assertGreater(full_upper, 50.0)

        window = OnlineCameraWindow(self.config, simulate=True)
        points = np.column_stack(
            (
                np.linspace(0.0, 100.0, len(heights)),
                np.zeros(len(heights)),
                heights,
            )
        )
        window._update_section_view(points)
        _, auto_y = window.section_view.getViewBox().viewRange()
        self.assertLess(auto_y[1], 2.0)

        window._fit_section_view()

        _, fitted_y = window.section_view.getViewBox().viewRange()
        self.assertFalse(window.section_auto_height_checkbox.isChecked())
        self.assertLessEqual(fitted_y[0], heights.min())
        self.assertGreaterEqual(fitted_y[1], heights.max())
        window.close()


if __name__ == "__main__":
    unittest.main()
