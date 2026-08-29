"""激光中心提取接口测试。"""

import unittest

import numpy as np

from laser_measurement_tool.laser.laser_extractor import (
    LaserAlgorithmNotConfiguredError,
    LaserExtractionError,
    LaserExtractionParams,
    extract_laser_center,
)


class LaserExtractorTests(unittest.TestCase):
    """使用假 backend 验证接口，不实现具体提取算法。"""

    def test_extract_laser_center_calls_injected_backend(self) -> None:
        image = np.zeros((8, 10), dtype=np.uint16)
        expected = np.array([[1.25, 2.5], [3.75, 4.125]])

        def fake_backend(actual_image, options):
            self.assertIs(actual_image, image)
            self.assertEqual(options["sample"], 7)
            return expected

        actual = extract_laser_center(
            image,
            LaserExtractionParams(
                method="steger",
                backend=fake_backend,
                options={"sample": 7},
            ),
        )

        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(actual, expected)

    def test_extract_laser_center_requires_backend(self) -> None:
        with self.assertRaises(LaserAlgorithmNotConfiguredError):
            extract_laser_center(
                np.zeros((4, 5), dtype=np.uint8),
                {"method": "gaussian"},
            )

    def test_extract_laser_center_rejects_invalid_output(self) -> None:
        def invalid_backend(_image, _options):
            return np.zeros((3, 3), dtype=np.float64)

        with self.assertRaisesRegex(LaserExtractionError, "形状为 \\(N, 2\\)"):
            extract_laser_center(
                np.zeros((4, 5), dtype=np.uint8),
                {"method": "ransac", "backend": invalid_backend},
            )


if __name__ == "__main__":
    unittest.main()
