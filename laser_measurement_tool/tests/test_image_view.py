"""图像视图坐标转换测试。"""

import importlib.util
import os
import unittest

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PYSIDE6_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PYSIDE6_AVAILABLE:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from laser_measurement_tool.gui.image_view import ImageView
    from laser_measurement_tool.measurement.roi_manager import RoiManager


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 未安装")
class ImageViewTests(unittest.TestCase):
    """验证视口坐标与图像坐标的转换。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_image_and_view_coordinates_round_trip(self) -> None:
        view = ImageView()
        view.resize(640, 480)
        view.set_image(np.zeros((100, 200), dtype=np.uint8))
        view.show()
        self.application.processEvents()
        view.scale(1.5, 1.5)
        view.horizontalScrollBar().setValue(
            view.horizontalScrollBar().maximum() // 2
        )
        view.verticalScrollBar().setValue(view.verticalScrollBar().maximum() // 2)

        expected = QPointF(100.5, 50.5)
        view_position = view.map_image_to_view(expected)
        self.assertIsNotNone(view_position)

        actual = view.map_view_to_image(view_position)
        self.assertIsNotNone(actual)
        self.assertAlmostEqual(actual.x(), expected.x(), delta=0.35)
        self.assertAlmostEqual(actual.y(), expected.y(), delta=0.35)
        self.assertEqual(view.image_pixel_at(view_position), (100, 50))

        view.close()

    def test_view_coordinate_outside_image_returns_none(self) -> None:
        view = ImageView()
        view.resize(640, 480)
        view.set_image(np.zeros((20, 20), dtype=np.uint8))
        view.show()
        self.application.processEvents()

        self.assertIsNone(view.map_view_to_image(QPoint(-100, -100)))
        view.close()

    def test_laser_center_overlay_is_replaced_with_image(self) -> None:
        view = ImageView()
        view.set_image(np.zeros((20, 20), dtype=np.uint8))
        view.set_laser_centers(np.array([[1.25, 2.5], [3.75, 4.125]]))
        self.assertEqual(view.laser_center_count, 2)
        self.assertIsNotNone(view._laser_center_cross_item)
        self.assertTrue(view._laser_center_cross_item.pen().isCosmetic())
        self.assertEqual(
            view._laser_center_cross_item.pen().color().getRgb(),
            (255, 0, 0, 255),
        )
        self.assertEqual(view._laser_center_cross_item.path().elementCount(), 8)

        view.set_image(np.zeros((10, 10), dtype=np.uint8))
        self.assertEqual(view.laser_center_count, 0)
        self.assertIsNone(view._laser_center_cross_item)
        view.close()

    def test_rectangle_drag_emits_roi_and_returns_to_pan(self) -> None:
        view = ImageView()
        view.resize(640, 480)
        view.set_image(np.zeros((100, 200), dtype=np.uint8))
        view.show()
        self.application.processEvents()
        selected = []
        view.roi_selected.connect(lambda kind, rectangle: selected.append((kind, rectangle)))

        start = view.map_image_to_view(QPointF(20.0, 25.0))
        end = view.map_image_to_view(QPointF(80.0, 70.0))
        view.begin_roi_selection("baseline")
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0], "baseline")
        self.assertIsNone(view.pending_roi_kind)
        view.close()

    def test_roi_region_colors_are_managed_as_multiple_items(self) -> None:
        view = ImageView()
        view.set_image(np.zeros((20, 20), dtype=np.uint8))
        manager = RoiManager()
        manager.add_region("baseline", 1.0, 1.0, 5.0, 5.0)
        manager.add_region("obstacle", 8.0, 8.0, 12.0, 12.0)

        view.set_roi_regions(manager.regions)

        self.assertEqual(view.roi_region_count, 2)
        view.clear_roi_regions()
        self.assertEqual(view.roi_region_count, 0)
        view.close()


if __name__ == "__main__":
    unittest.main()
