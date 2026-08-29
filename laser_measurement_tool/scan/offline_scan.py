"""Offline scan orchestration built on the existing single-frame pipeline.

The sequence input CSV must contain an ``angle_deg`` column.  It may also
contain ``frame_index`` and one of ``image``, ``filename``, ``image_path`` or
``path``.  When no image column is present, supported images in the input
directory are sorted by filename and paired with CSV rows in order.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from online.models import CapturedFrame
from online.pipeline import FramePipeline
from laser.laser_extractor import LaserExtractionError
from utils.image_io import SUPPORTED_IMAGE_SUFFIXES, load_grayscale_image
from utils.image_metadata import read_image_offset_metadata

from .accumulator import ScanAccumulator
from .kinematics import transform_points_camera_to_scan
from .models import ScanProfile, ScanResult


_ANGLE_COLUMNS = ("angle_deg", "angle", "angle_command_deg")
_FRAME_COLUMNS = (
    "frame_index",
    "frame_idx",
    "frame",
    "frame_number",
    "frame_id",
    "index",
)
_IMAGE_COLUMNS = (
    "image",
    "filename",
    "file_name",
    "file",
    "image_file",
    "image_path",
    "path",
)


@dataclass(frozen=True, slots=True)
class OfflineScanResult:
    """Scan result plus provenance metadata for an offline run."""

    scan_result: ScanResult
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.scan_result, ScanResult):
            raise TypeError("scan_result 必须是 ScanResult")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def profiles(self) -> tuple[ScanProfile, ...]:
        """Expose profiles directly for callers that do not need the wrapper."""
        return self.scan_result.profiles

    @property
    def points_scan(self) -> np.ndarray:
        """Expose the deterministic accumulated scan-frame cloud."""
        return self.scan_result.points_scan


@dataclass(frozen=True, slots=True)
class _SequenceItem:
    frame_index: int
    angle_deg: float
    image_path: Path


class OfflineScanRunner:
    """Connect an existing :class:`FramePipeline` to scan kinematics."""

    __slots__ = (
        "pipeline",
        "axis_point_scan_mm",
        "axis_direction_scan",
        "zero_offset_deg",
        "T_scan_from_camera_zero",
    )

    def __init__(
        self,
        pipeline: FramePipeline,
        axis_point_scan_mm: np.ndarray,
        axis_direction_scan: np.ndarray,
        zero_offset_deg: float,
        T_scan_from_camera_zero: np.ndarray,
    ) -> None:
        if not callable(getattr(pipeline, "run_frame", None)):
            raise TypeError("pipeline 必须提供 run_frame(frame) 方法")
        self.pipeline = pipeline
        self.axis_point_scan_mm = axis_point_scan_mm
        self.axis_direction_scan = axis_direction_scan
        self.zero_offset_deg = zero_offset_deg
        self.T_scan_from_camera_zero = T_scan_from_camera_zero

    def run_repeat_one(
        self,
        image_path: str | Path,
        angle_sequence_deg: Iterable[float],
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> OfflineScanResult:
        """Process one real image repeatedly at the requested angles."""
        angles = _validate_angle_sequence(angle_sequence_deg)
        path = Path(image_path)
        image = load_grayscale_image(path)
        resolved_offset_x, resolved_offset_y = _resolve_image_offset(
            path,
            image,
            self.pipeline,
            offset_x,
            offset_y,
        )
        accumulator = ScanAccumulator()
        frame_stats: list[dict[str, object]] = []
        warnings: list[dict[str, object]] = []
        for frame_index, angle_deg in enumerate(angles):
            frame = _make_captured_frame(
                image.copy(),
                frame_index,
                offset_x=resolved_offset_x,
                offset_y=resolved_offset_y,
            )
            profile, frame_stat = self._process_frame(frame, frame_index, angle_deg)
            accumulator.add_profile(profile)
            frame_stats.append(frame_stat)
            if frame_stat.get("warning"):
                warnings.append(
                    {
                        "frame_index": frame_index,
                        "message": frame_stat["warning"],
                    }
                )
        return _make_result(
            accumulator,
            mode="repeat_one",
            kinematic_demo_only=True,
            input_image=str(path),
            profile_filenames=tuple(path.name for _ in angles),
            image_offset={"offset_x": resolved_offset_x, "offset_y": resolved_offset_y},
            frame_stats=tuple(frame_stats),
            warnings=tuple(warnings),
        )

    def run_sequence(
        self,
        image_dir: str | Path,
        frame_angle_csv: str | Path,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> OfflineScanResult:
        """Process a directory of images bound to angles in a CSV file."""
        items = _read_sequence_items(image_dir, frame_angle_csv)
        accumulator = ScanAccumulator()
        frame_stats: list[dict[str, object]] = []
        warnings: list[dict[str, object]] = []
        for item in items:
            image = load_grayscale_image(item.image_path)
            resolved_offset_x, resolved_offset_y = _resolve_image_offset(
                item.image_path,
                image,
                self.pipeline,
                offset_x,
                offset_y,
            )
            frame = _make_captured_frame(
                image,
                item.frame_index,
                offset_x=resolved_offset_x,
                offset_y=resolved_offset_y,
            )
            profile, frame_stat = self._process_frame(
                frame, item.frame_index, item.angle_deg
            )
            accumulator.add_profile(profile)
            frame_stats.append(frame_stat)
            if frame_stat.get("warning"):
                warnings.append(
                    {
                        "frame_index": item.frame_index,
                        "message": frame_stat["warning"],
                    }
                )
        return _make_result(
            accumulator,
            mode="sequence",
            kinematic_demo_only=False,
            image_dir=str(Path(image_dir)),
            frame_angle_csv=str(Path(frame_angle_csv)),
            profile_filenames=tuple(item.image_path.name for item in items),
            frame_stats=tuple(frame_stats),
            warnings=tuple(warnings),
        )

    def _process_frame(
        self,
        frame: CapturedFrame,
        frame_index: int,
        angle_deg: float,
    ) -> tuple[ScanProfile, dict[str, object]]:
        warning_parts: list[str] = []
        try:
            frame_result = self.pipeline.run_frame(frame)
        except LaserExtractionError as error:
            # 中心提取失败仍保留该姿态的空 profile，保证 warning、frame_index
            # 和输出统计可追溯，而不会把失败帧静默丢掉。
            warning_parts.append(f"激光中心提取失败: {error}")
            empty_profile = ScanProfile(
                frame_index=frame_index,
                angle_deg=angle_deg,
                pixels_uv=np.empty((0, 2), dtype=np.float64),
                points_camera=np.empty((0, 3), dtype=np.float64),
                points_scan=np.empty((0, 3), dtype=np.float64),
            )
            return empty_profile, _frame_stat(
                frame_index,
                angle_deg,
                0,
                0,
                0,
                "; ".join(warning_parts),
            )

        if not hasattr(frame_result, "points_camera"):
            raise TypeError("FramePipeline.run_frame() 结果缺少 points_camera")
        centers_uv = np.asarray(
            getattr(frame_result, "centers_uv_full", np.empty((0, 2))),
            dtype=np.float64,
        ).reshape(-1, 2)
        points_camera = np.asarray(frame_result.points_camera, dtype=np.float64)
        pixels_uv = getattr(frame_result, "pixels_uv", None)
        if pixels_uv is None:
            pixels_uv = centers_uv
        pixels_uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
        if len(pixels_uv) != len(points_camera):
            # FrameResult 目前公开的是完整提取中心和 points_camera；当重建
            # 过滤掉无效交点时，保持 profile 契约的逐行对齐，并将过滤情况
            # 明确记录在 warning 中。点云本身始终只来自 points_camera。
            warning_parts.append(
                "三维恢复过滤了部分激光中心点，profile 像素按有效点数截取"
            )
            pixels_uv = pixels_uv[: len(points_camera)]
        points_scan = transform_points_camera_to_scan(
            points_camera,
            angle_deg,
            self.axis_point_scan_mm,
            self.axis_direction_scan,
            self.zero_offset_deg,
            self.T_scan_from_camera_zero,
        )
        valid_laser_points = len(centers_uv)
        camera_count = len(points_camera)
        scan_count = len(points_scan)
        if valid_laser_points == 0:
            warning_parts.append("激光中心点数为 0")
        if camera_count == 0:
            warning_parts.append("points_camera 点数为 0")
        if scan_count == 0:
            warning_parts.append("points_scan 点数为 0")
        profile = ScanProfile(
            frame_index=frame_index,
            angle_deg=angle_deg,
            pixels_uv=pixels_uv,
            points_camera=points_camera,
            points_scan=points_scan,
        )
        return profile, _frame_stat(
            frame_index,
            angle_deg,
            valid_laser_points,
            camera_count,
            scan_count,
            "; ".join(warning_parts) if warning_parts else None,
        )


def run_repeat_one(
    image_path: str | Path,
    angle_sequence_deg: Iterable[float],
    pipeline: FramePipeline,
    axis_point_scan_mm: np.ndarray,
    axis_direction_scan: np.ndarray,
    zero_offset_deg: float,
    T_scan_from_camera_zero: np.ndarray,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
) -> OfflineScanResult:
    """Convenience wrapper for :meth:`OfflineScanRunner.run_repeat_one`."""
    runner = OfflineScanRunner(
        pipeline,
        axis_point_scan_mm,
        axis_direction_scan,
        zero_offset_deg,
        T_scan_from_camera_zero,
    )
    return runner.run_repeat_one(
        image_path,
        angle_sequence_deg,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def run_sequence(
    image_dir: str | Path,
    frame_angle_csv: str | Path,
    pipeline: FramePipeline,
    axis_point_scan_mm: np.ndarray,
    axis_direction_scan: np.ndarray,
    zero_offset_deg: float,
    T_scan_from_camera_zero: np.ndarray,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
) -> OfflineScanResult:
    """Convenience wrapper for :meth:`OfflineScanRunner.run_sequence`."""
    runner = OfflineScanRunner(
        pipeline,
        axis_point_scan_mm,
        axis_direction_scan,
        zero_offset_deg,
        T_scan_from_camera_zero,
    )
    return runner.run_sequence(
        image_dir,
        frame_angle_csv,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def run_offline_scan(
    mode: str,
    pipeline: FramePipeline,
    *,
    axis_point_scan_mm: np.ndarray,
    axis_direction_scan: np.ndarray,
    zero_offset_deg: float,
    T_scan_from_camera_zero: np.ndarray,
    image_path: str | Path | None = None,
    angle_sequence_deg: Iterable[float] | None = None,
    image_dir: str | Path | None = None,
    frame_angle_csv: str | Path | None = None,
    offset_x: int = 0,
    offset_y: int = 0,
) -> OfflineScanResult:
    """Dispatch either the ``repeat_one`` or ``sequence`` input mode."""
    if mode == "repeat_one":
        if image_path is None or angle_sequence_deg is None:
            raise ValueError("repeat_one 需要 image_path 和 angle_sequence_deg")
        return run_repeat_one(
            image_path,
            angle_sequence_deg,
            pipeline,
            axis_point_scan_mm,
            axis_direction_scan,
            zero_offset_deg,
            T_scan_from_camera_zero,
            offset_x=offset_x,
            offset_y=offset_y,
        )
    if mode == "sequence":
        if image_dir is None or frame_angle_csv is None:
            raise ValueError("sequence 需要 image_dir 和 frame_angle_csv")
        return run_sequence(
            image_dir,
            frame_angle_csv,
            pipeline,
            axis_point_scan_mm,
            axis_direction_scan,
            zero_offset_deg,
            T_scan_from_camera_zero,
            offset_x=offset_x,
            offset_y=offset_y,
        )
    raise ValueError("mode 必须是 repeat_one 或 sequence")


def _resolve_image_offset(
    image_path: Path,
    image: np.ndarray,
    pipeline: FramePipeline,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int]:
    """把 ROI 图像映射回标定全幅，拒绝无偏移的静默猜测。

    显式传入的非零 offset 优先；否则复用在线录制使用的 sidecar JSON、
    ``result.json`` 和 ``frames.csv``。有标定包而图像尺寸小于全幅时，若
    找不到合法偏移则直接报错，避免把局部 ROI 像素送入全幅标定。
    """
    explicit = _coerce_offset(offset_x, offset_y)
    package = getattr(pipeline, "package", None)
    full_width = getattr(package, "image_width", None)
    full_height = getattr(package, "image_height", None)
    has_full_size = full_width is not None and full_height is not None
    if has_full_size:
        try:
            full_width = int(full_width)
            full_height = int(full_height)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("标定包图像尺寸无效") from error
        if full_width <= 0 or full_height <= 0:
            raise ValueError("标定包图像尺寸必须为正数")

    if explicit != (0, 0):
        _validate_offset_bounds(explicit, image, full_width, full_height)
        return explicit

    metadata_offset = read_image_offset_metadata(image_path)
    if metadata_offset is not None:
        _validate_offset_bounds(metadata_offset, image, full_width, full_height)
        return metadata_offset

    if not has_full_size:
        return explicit

    height, width = image.shape
    if (width, height) == (full_width, full_height):
        return (0, 0)
    if width > full_width or height > full_height:
        raise ValueError(
            f"图像 {width} × {height} 超过标定全幅 "
            f"{full_width} × {full_height}，无法进行扫描"
        )
    raise ValueError(
        f"图像 {width} × {height} 小于标定全幅 {full_width} × {full_height}，"
        "且未找到 ROI 偏移；请在同目录提供同名 JSON、result.json 或 frames.csv"
    )


def _coerce_offset(offset_x: int, offset_y: int) -> tuple[int, int]:
    try:
        values = (int(offset_x), int(offset_y))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("ROI 偏移必须是整数") from error
    if min(values) < 0:
        raise ValueError("ROI 偏移不能为负数")
    return values


def _validate_offset_bounds(
    offset: tuple[int, int],
    image: np.ndarray,
    full_width: int | None,
    full_height: int | None,
) -> None:
    if full_width is None or full_height is None:
        return
    height, width = image.shape
    if offset[0] + width > full_width or offset[1] + height > full_height:
        raise ValueError(
            f"ROI 偏移 {offset} 与图像 {width} × {height} 超出标定全幅 "
            f"{full_width} × {full_height}"
        )


def _frame_stat(
    frame_index: int,
    angle_deg: float,
    valid_laser_points: int,
    points_camera_count: int,
    points_scan_count: int,
    warning: str | None,
) -> dict[str, object]:
    stat: dict[str, object] = {
        "frame_index": int(frame_index),
        "angle_deg": float(angle_deg),
        "valid_laser_points": int(valid_laser_points),
        "points_camera_count": int(points_camera_count),
        "points_scan_count": int(points_scan_count),
    }
    if warning:
        stat["warning"] = warning
    return stat


def _make_result(
    accumulator: ScanAccumulator,
    *,
    mode: str,
    kinematic_demo_only: bool,
    **metadata: object,
) -> OfflineScanResult:
    scan_result = accumulator.to_result()
    result_metadata: dict[str, object] = {
        "mode": mode,
        "kinematic_demo_only": kinematic_demo_only,
        "frame_count": len(scan_result.profiles),
        "point_count": len(scan_result.points_scan),
    }
    result_metadata.update(metadata)
    return OfflineScanResult(scan_result, result_metadata)


def _make_captured_frame(
    image: np.ndarray,
    frame_index: int,
    *,
    offset_x: int,
    offset_y: int,
) -> CapturedFrame:
    now_wall = time.time_ns()
    return CapturedFrame(
        image=np.ascontiguousarray(image),
        camera_frame_number=frame_index,
        camera_timestamp_ticks=None,
        host_timestamp_ns=now_wall,
        host_monotonic_ns=time.perf_counter_ns(),
        offset_x=offset_x,
        offset_y=offset_y,
    )


def _validate_angle_sequence(angle_sequence_deg: Iterable[float]) -> tuple[float, ...]:
    if isinstance(angle_sequence_deg, (str, bytes)):
        raise ValueError("angle_sequence_deg 必须是角度序列")
    try:
        values = tuple(angle_sequence_deg)
    except TypeError as error:
        raise ValueError("angle_sequence_deg 必须是角度序列") from error
    angles: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("angle_sequence_deg 必须包含有限数值")
        try:
            angle = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("angle_sequence_deg 必须包含有限数值") from error
        if not math.isfinite(angle):
            raise ValueError("angle_sequence_deg 必须包含有限数值")
        angles.append(angle)
    return tuple(angles)


def _read_sequence_items(
    image_dir: str | Path,
    frame_angle_csv: str | Path,
) -> tuple[_SequenceItem, ...]:
    directory = Path(image_dir)
    csv_path = Path(frame_angle_csv)
    if not directory.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {directory}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"frame-angle CSV 不存在: {csv_path}")

    image_paths = _sorted_image_paths(directory)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        if not fieldnames or any(field is None for field in fieldnames):
            raise ValueError("frame-angle CSV 必须包含有效表头")
        angle_column = _find_column(fieldnames, _ANGLE_COLUMNS)
        if angle_column is None:
            raise ValueError("frame-angle CSV 缺少 angle_deg 列")
        frame_column = _find_column(fieldnames, _FRAME_COLUMNS)
        image_column = _find_column(fieldnames, _IMAGE_COLUMNS)

        rows: list[_SequenceItem] = []
        seen_indices: set[int] = set()
        for row_number, row in enumerate(reader, start=2):
            if row is None or _is_blank_row(row):
                continue
            if None in row:
                raise ValueError(f"frame-angle CSV 第 {row_number} 行字段数量不匹配")
            frame_index = (
                _parse_frame_index(row.get(frame_column), row_number)
                if frame_column is not None
                else len(rows)
            )
            if frame_index in seen_indices:
                raise ValueError(f"frame-angle CSV 重复 frame_index: {frame_index}")
            seen_indices.add(frame_index)
            angle_deg = _parse_finite_angle(row.get(angle_column), row_number)
            if image_column is None:
                image_path = None
            else:
                raw_image = _required_cell(row.get(image_column), row_number, "图像路径")
                image_path = Path(raw_image)
                if not image_path.is_absolute():
                    image_path = directory / image_path
            rows.append(
                _SequenceItem(
                    frame_index=frame_index,
                    angle_deg=angle_deg,
                    image_path=image_path if image_path is not None else Path(),
                )
            )

    if image_column is None:
        if len(rows) != len(image_paths):
            raise ValueError(
                "frame-angle CSV 行数必须与图像目录中的支持图像数量一致"
            )
        rows = [
            _SequenceItem(item.frame_index, item.angle_deg, image_path)
            for item, image_path in zip(rows, image_paths)
        ]
    return tuple(rows)


def _sorted_image_paths(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )


def _find_column(fieldnames: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {
        field.strip().lower(): field for field in fieldnames if field is not None
    }
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _is_blank_row(row: dict[str | None, str | None]) -> bool:
    return all(value is None or not str(value).strip() for value in row.values())


def _required_cell(value: Any, row_number: int, name: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"frame-angle CSV 第 {row_number} 行缺少{name}")
    return str(value).strip()


def _parse_finite_angle(value: Any, row_number: int) -> float:
    raw = _required_cell(value, row_number, "angle_deg")
    try:
        angle = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"frame-angle CSV 第 {row_number} 行 angle_deg 无效") from error
    if not math.isfinite(angle):
        raise ValueError(f"frame-angle CSV 第 {row_number} 行 angle_deg 必须有限")
    return angle


def _parse_frame_index(value: Any, row_number: int) -> int:
    raw = _required_cell(value, row_number, "frame_index")
    try:
        index = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"frame-angle CSV 第 {row_number} 行 frame_index 无效") from error
    if index < 0:
        raise ValueError(f"frame-angle CSV 第 {row_number} 行 frame_index 不能为负数")
    return index


__all__ = [
    "OfflineScanResult",
    "OfflineScanRunner",
    "run_offline_scan",
    "run_repeat_one",
    "run_sequence",
]
