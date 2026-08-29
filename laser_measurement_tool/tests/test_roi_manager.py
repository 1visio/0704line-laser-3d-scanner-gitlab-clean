"""ROI 管理与中心点筛选测试。"""

import unittest

import numpy as np

from laser_measurement_tool.measurement.roi_manager import RoiKind, RoiManager


class RoiManagerTests(unittest.TestCase):
    """验证多区域、删除顺序和两类点输出。"""

    def test_add_multiple_regions_and_filter_points(self) -> None:
        manager = RoiManager()
        manager.add_region(RoiKind.BASELINE, 0.0, 0.0, 3.0, 3.0)
        manager.add_region("baseline", 7.0, 7.0, 10.0, 10.0)
        manager.add_region("obstacle", 4.0, 4.0, 9.0, 9.0)
        points = np.array([[1.0, 1.0], [5.0, 5.0], [8.0, 8.0], [12.0, 1.0]])

        baseline_points, obstacle_points = manager.filter_points(points)

        np.testing.assert_array_equal(
            baseline_points,
            np.array([[1.0, 1.0], [8.0, 8.0]]),
        )
        np.testing.assert_array_equal(
            obstacle_points,
            np.array([[5.0, 5.0], [8.0, 8.0]]),
        )

    def test_remove_last_and_clear_preserve_add_order(self) -> None:
        manager = RoiManager()
        first = manager.add_region("baseline", 5.0, 6.0, 1.0, 2.0)
        last = manager.add_region("obstacle", 7.0, 8.0, 9.0, 10.0)

        self.assertEqual((first.left, first.top, first.right, first.bottom), (1, 2, 5, 6))
        self.assertEqual(manager.remove_last(), last)
        self.assertEqual(manager.regions, (first,))

        manager.clear()
        self.assertEqual(manager.regions, ())
        self.assertIsNone(manager.remove_last())

    def test_filter_without_regions_returns_empty_outputs(self) -> None:
        manager = RoiManager()
        baseline_points, obstacle_points = manager.filter_points(
            np.array([[1.0, 2.0]])
        )

        self.assertEqual(baseline_points.shape, (0, 2))
        self.assertEqual(obstacle_points.shape, (0, 2))

    def test_filter_points_by_obstacle_region_keeps_groups_separate(self) -> None:
        manager = RoiManager()
        manager.add_region("obstacle", 0.0, 0.0, 4.0, 4.0)
        manager.add_region("baseline", 0.0, 8.0, 10.0, 10.0)
        manager.add_region("obstacle", 6.0, 0.0, 10.0, 4.0)
        points = np.array([[1.0, 1.0], [3.0, 2.0], [7.0, 1.0], [9.0, 2.0]])

        groups = manager.filter_points_by_region(points, RoiKind.OBSTACLE)

        self.assertEqual(len(groups), 2)
        np.testing.assert_array_equal(groups[0], points[:2])
        np.testing.assert_array_equal(groups[1], points[2:])


if __name__ == "__main__":
    unittest.main()
