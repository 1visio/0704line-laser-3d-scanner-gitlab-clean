"""灰度图像读取与格式校验。"""

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".png", ".bmp"})


def load_grayscale_image(file_path: str | Path) -> np.ndarray:
    """读取单通道图像，并保留源图像位深。"""
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图像格式: {path.suffix or '无扩展名'}")
    if not path.is_file():
        raise FileNotFoundError(f"图像文件不存在: {path}")

    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法解码图像: {path}")
    if image.ndim != 2:
        raise ValueError("请选择单通道灰度图像")

    return np.ascontiguousarray(image)
