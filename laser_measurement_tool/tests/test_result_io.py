"""激光中心 CSV 输出测试。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from laser_measurement_tool.utils.result_io import (
    next_laser_center_csv_path,
    save_ground_pointcloud_ply,
    save_laser_centers_csv,
)
from laser_measurement_tool.utils.pointcloud_colors import map_zg_to_rgb


class ResultIoTests(unittest.TestCase):
    """验证 CSV 内容及不覆盖策略。"""

    def test_save_laser_centers_csv_and_choose_next_path(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            image_path = output_directory / "sample.tif"
            first_path = next_laser_center_csv_path(image_path, output_directory)
            save_laser_centers_csv(
                first_path,
                np.array([[1.25, 2.5], [3.75, 4.125]]),
            )

            self.assertEqual(
                first_path.read_text(encoding="utf-8").splitlines(),
                ["u,v", "1.250000,2.500000", "3.750000,4.125000"],
            )
            self.assertEqual(
                next_laser_center_csv_path(image_path, output_directory).name,
                "sample_laser_center_001.csv",
            )

    def test_save_ground_pointcloud_ply(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "full_laser_ground.ply"
            save_ground_pointcloud_ply(
                path,
                np.array([[1.25, -2.5, 0.125], [3.0, 4.5, 6.75]]),
            )

            lines = path.read_text(encoding="ascii").splitlines()

        self.assertIn("element vertex 2", lines)
        self.assertIn("property uchar red", lines)
        self.assertIn("property uchar green", lines)
        self.assertIn("property uchar blue", lines)
        self.assertIn(
            "comment color_mapping zg_high_contrast_from_zg", lines
        )
        first_values = lines[-2].split()
        second_values = lines[-1].split()
        self.assertEqual(
            first_values[:3],
            ["1.250000000", "-2.500000000", "0.125000000"],
        )
        self.assertEqual(
            second_values[:3],
            ["3.000000000", "4.500000000", "6.750000000"],
        )
        self.assertEqual(len(first_values), 6)
        self.assertEqual(len(second_values), 6)
        self.assertNotEqual(first_values[3:], second_values[3:])
        for value in first_values[3:] + second_values[3:]:
            self.assertIn(int(value), range(256))

    def test_save_ground_pointcloud_ply_colors_constant_zg(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "constant_zg.ply"
            save_ground_pointcloud_ply(
                path,
                np.array([[0.0, 0.0, 1.25], [1.0, 0.0, 1.25]]),
            )
            lines = path.read_text(encoding="ascii").splitlines()

        self.assertIn(
            "comment color_zg_range_mm 1.250000000 1.250000000", lines
        )
        self.assertEqual(lines[-2].split()[3:], lines[-1].split()[3:])

    def test_zg_color_map_uses_bright_distinct_endpoints(self) -> None:
        rgb, zg_min, zg_max = map_zg_to_rgb([-1.0, 0.0, 1.0])

        self.assertEqual((zg_min, zg_max), (-1.0, 1.0))
        np.testing.assert_array_equal(rgb[0], [0, 229, 255])
        np.testing.assert_array_equal(rgb[-1], [255, 0, 168])
        self.assertTrue(np.all(rgb.max(axis=1) >= 232))

    def test_ground_pointcloud_ply_rejects_non_finite_points(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.ply"
            with self.assertRaisesRegex(ValueError, "NaN"):
                save_ground_pointcloud_ply(
                    path, np.array([[0.0, 1.0, np.nan]])
                )


if __name__ == "__main__":
    unittest.main()
