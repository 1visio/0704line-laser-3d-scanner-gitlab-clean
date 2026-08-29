"""灰度图像读取测试。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from laser_measurement_tool.utils.image_io import load_grayscale_image


def _write_encoded_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix, image)
    assert success
    encoded.tofile(path)


class ImageIoTests(unittest.TestCase):
    """验证灰度图像格式校验和位深保留。"""

    def test_load_supported_uint8_formats(self) -> None:
        expected = np.arange(30, dtype=np.uint8).reshape(5, 6)
        with TemporaryDirectory() as temporary_directory:
            for suffix in (".png", ".bmp"):
                with self.subTest(suffix=suffix):
                    path = Path(temporary_directory) / f"gray{suffix}"
                    _write_encoded_image(path, expected)

                    actual = load_grayscale_image(path)

                    np.testing.assert_array_equal(actual, expected)

    def test_load_grayscale_image_preserves_uint16(self) -> None:
        expected = np.arange(30, dtype=np.uint16).reshape(5, 6) * 100
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "gray.tif"
            _write_encoded_image(path, expected)

            actual = load_grayscale_image(path)

        self.assertEqual(actual.dtype, np.uint16)
        np.testing.assert_array_equal(actual, expected)

    def test_load_grayscale_image_rejects_color_image(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "color.png"
            _write_encoded_image(path, np.zeros((5, 6, 3), dtype=np.uint8))

            with self.assertRaisesRegex(ValueError, "单通道灰度图像"):
                load_grayscale_image(path)


if __name__ == "__main__":
    unittest.main()
