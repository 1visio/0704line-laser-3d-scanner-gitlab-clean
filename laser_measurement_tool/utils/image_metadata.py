"""读取图像旁的硬件 ROI 元数据。

在线录制和 GUI 会把相机 ROI 的局部图像坐标与全幅标定坐标之间的
偏移写入 sidecar JSON、结果 JSON 或 ``frames.csv``。该模块不依赖 GUI，
供离线扫描和 GUI 共同复用，避免把 ROI 局部像素误当作标定全幅坐标。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path


def offset_from_mapping(mapping: object) -> tuple[int, int] | None:
    """从常见的结果/帧元数据映射中读取非负硬件 ROI 偏移。"""
    if not isinstance(mapping, Mapping):
        return None
    for x_name, y_name in (
        ("offset_x", "offset_y"),
        ("offset_x_px", "offset_y_px"),
        ("u", "v"),
    ):
        if x_name not in mapping or y_name not in mapping:
            continue
        try:
            x = int(mapping[x_name])
            y = int(mapping[y_name])
        except (TypeError, ValueError, OverflowError):
            continue
        if x >= 0 and y >= 0:
            return x, y
    return None


def read_image_offset_metadata(image_path: str | Path) -> tuple[int, int] | None:
    """读取图像相邻 JSON 或录制 ``frames.csv`` 中的 ROI 偏移。

    查找顺序与已有在线 GUI 保持一致：图像同名 JSON、目录中的
    ``result.json``，最后是按文件名匹配的 ``frames.csv`` 行。损坏或
    不相关的元数据会被跳过并继续尝试其他来源。
    """
    path = Path(image_path)
    for metadata_path in (
        path.with_suffix(".json"),
        path.parent / "result.json",
    ):
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        candidates = [payload]
        if isinstance(payload, Mapping):
            candidates.extend((payload.get("image_offset"), payload.get("frame")))
        for candidate in candidates:
            offset = offset_from_mapping(candidate)
            if offset is not None:
                return offset

    frames_csv = path.parent / "frames.csv"
    if frames_csv.is_file():
        try:
            with frames_csv.open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    raw_filename = row.get("filename")
                    if (
                        raw_filename is None
                        or str(raw_filename).replace("\\", "/").rsplit("/", 1)[-1]
                        != path.name
                    ):
                        continue
                    offset = offset_from_mapping(row)
                    if offset is not None:
                        return offset
        except (OSError, UnicodeError, csv.Error):
            pass
    return None


__all__ = ["offset_from_mapping", "read_image_offset_metadata"]
