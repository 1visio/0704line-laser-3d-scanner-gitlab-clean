"""Configuration contracts and loader for the Stage-1 offline scan."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SUPPORTED_SCHEMA_VERSION = 1


class ScanConfigError(ValueError):
    """The Stage-1 scan configuration is missing or invalid."""


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ScanConfigError(f"{name} 必须是有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ScanConfigError(f"{name} 必须是有限数值") from error
    if not math.isfinite(number):
        raise ScanConfigError(f"{name} 必须是有限数值")
    return number


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ScanConfigError(f"{name} 必须是数值数组") from error
    if array.shape != shape:
        raise ScanConfigError(f"{name} 必须是形状 {shape} 的数组")
    if not np.isfinite(array).all():
        raise ScanConfigError(f"{name} 不能包含 NaN 或无穷值")
    return np.ascontiguousarray(array).copy()


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScanConfigError(f"{name} 必须是 YAML 映射")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        names = ", ".join(str(item) for item in unknown)
        raise ScanConfigError(f"{name} 包含不支持的字段: {names}")


@dataclass(frozen=True, slots=True)
class ScanTrajectoryConfig:
    """俯仰扫描轨迹的起止角和角度步长。"""

    start_angle_deg: float
    end_angle_deg: float
    step_angle_deg: float

    def __post_init__(self) -> None:
        start = _finite_float(self.start_angle_deg, "trajectory.start_angle_deg")
        end = _finite_float(self.end_angle_deg, "trajectory.end_angle_deg")
        step = _finite_float(self.step_angle_deg, "trajectory.step_angle_deg")
        if step == 0.0:
            raise ScanConfigError("trajectory.step_angle_deg 不能为 0")
        if start < end and step < 0.0:
            raise ScanConfigError(
                "trajectory.step_angle_deg 的方向无法从 start_angle_deg 走向 end_angle_deg"
            )
        if start > end and step > 0.0:
            raise ScanConfigError(
                "trajectory.step_angle_deg 的方向无法从 start_angle_deg 走向 end_angle_deg"
            )
        object.__setattr__(self, "start_angle_deg", start)
        object.__setattr__(self, "end_angle_deg", end)
        object.__setattr__(self, "step_angle_deg", step)

    @property
    def angles_deg(self) -> tuple[float, ...]:
        """Return a deterministic inclusive angle sequence."""
        start = self.start_angle_deg
        end = self.end_angle_deg
        step = self.step_angle_deg
        if start == end:
            return (start,)

        distance = abs(end - start)
        count = int(math.floor(distance / abs(step) + 1e-12))
        values = [start + index * step for index in range(count + 1)]
        if not math.isclose(values[-1], end, rel_tol=0.0, abs_tol=1e-9):
            values.append(end)
        else:
            values[-1] = end
        return tuple(values)


@dataclass(frozen=True, slots=True)
class ScanKinematicsConfig:
    """扫描轴和零位 camera-to-scan 变换参数。"""

    axis_point_scan_mm: np.ndarray
    axis_direction_scan: np.ndarray
    zero_offset_deg: float
    T_scan_from_camera_zero: np.ndarray

    def __post_init__(self) -> None:
        axis_point = _finite_array(
            self.axis_point_scan_mm, (3,), "kinematics.axis_point_scan_mm"
        )
        axis_direction = _finite_array(
            self.axis_direction_scan, (3,), "kinematics.axis_direction_scan"
        )
        if not np.any(axis_direction != 0.0):
            raise ScanConfigError("kinematics.axis_direction_scan 不能是零向量")
        zero_offset = _finite_float(
            self.zero_offset_deg, "kinematics.zero_offset_deg"
        )
        transform = _finite_array(
            self.T_scan_from_camera_zero,
            (4, 4),
            "kinematics.T_scan_from_camera_zero",
        )
        if not np.allclose(
            transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0
        ):
            raise ScanConfigError(
                "kinematics.T_scan_from_camera_zero 最后一行必须接近 [0, 0, 0, 1]"
            )
        object.__setattr__(self, "axis_point_scan_mm", axis_point)
        object.__setattr__(self, "axis_direction_scan", axis_direction)
        object.__setattr__(self, "zero_offset_deg", zero_offset)
        object.__setattr__(self, "T_scan_from_camera_zero", transform)


@dataclass(frozen=True, slots=True)
class ScanOutputConfig:
    """扫描结果输出位置和格式开关。"""

    directory: Path
    save_profiles_csv: bool = True
    save_ply: bool = True
    save_pcd: bool = True

    def __post_init__(self) -> None:
        try:
            directory = Path(self.directory)
        except (TypeError, ValueError) as error:
            raise ScanConfigError("output.directory 必须是非空路径") from error
        if not str(directory).strip():
            raise ScanConfigError("output.directory 必须是非空路径")
        object.__setattr__(self, "directory", directory)
        for field_name in ("save_profiles_csv", "save_ply", "save_pcd"):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise ScanConfigError(f"output.{field_name} 必须是布尔值")


@dataclass(frozen=True, slots=True)
class ScanConfig:
    """完整的、与单帧测量配置分离的扫描配置。"""

    schema_version: int
    mode: str
    trajectory: ScanTrajectoryConfig
    kinematics: ScanKinematicsConfig
    output: ScanOutputConfig

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, (int, np.integer)
        ):
            raise ScanConfigError("schema_version 必须是整数")
        if int(self.schema_version) != SUPPORTED_SCHEMA_VERSION:
            raise ScanConfigError(
                f"不支持的 schema_version: {self.schema_version}"
            )
        mode = _normalize_mode(self.mode)
        if not isinstance(self.trajectory, ScanTrajectoryConfig):
            raise ScanConfigError("trajectory 必须是 ScanTrajectoryConfig")
        if not isinstance(self.kinematics, ScanKinematicsConfig):
            raise ScanConfigError("kinematics 必须是 ScanKinematicsConfig")
        if not isinstance(self.output, ScanOutputConfig):
            raise ScanConfigError("output 必须是 ScanOutputConfig")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "mode", mode)


def _normalize_mode(value: Any) -> str:
    if not isinstance(value, str):
        raise ScanConfigError("mode 必须是 repeat_one 或 sequence")
    mode = value.strip().lower().replace("-", "_")
    if mode not in {"repeat_one", "sequence"}:
        raise ScanConfigError("mode 必须是 repeat_one 或 sequence")
    return mode


def load_scan_config(path: str | Path) -> ScanConfig:
    """Read and strongly validate a Stage-1 scan YAML file.

    Relative output paths are resolved against the directory containing the
    scan configuration file, never against the process working directory.
    """
    try:
        config_path = Path(path).expanduser()
    except (TypeError, ValueError) as error:
        raise ScanConfigError(f"扫描配置路径无效: {path!r}") from error
    if not config_path.is_file():
        raise ScanConfigError(f"扫描配置不存在: {config_path}")
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ScanConfigError(f"无法读取扫描配置 {config_path}: {error}") from error
    if not isinstance(document, Mapping):
        raise ScanConfigError("扫描配置顶层必须是 YAML 映射")

    _reject_unknown(
        document,
        {"schema_version", "mode", "trajectory", "kinematics", "output"},
        "扫描配置",
    )
    schema_version = document.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(
        schema_version, (int, np.integer)
    ):
        raise ScanConfigError("schema_version 必须是整数")
    if int(schema_version) != SUPPORTED_SCHEMA_VERSION:
        raise ScanConfigError(f"不支持的 schema_version: {schema_version}")

    mode = _normalize_mode(document.get("mode"))
    trajectory_mapping = _required_mapping(document.get("trajectory"), "trajectory")
    _reject_unknown(
        trajectory_mapping,
        {"start_angle_deg", "end_angle_deg", "step_angle_deg"},
        "trajectory",
    )
    trajectory = ScanTrajectoryConfig(
        start_angle_deg=trajectory_mapping.get("start_angle_deg"),
        end_angle_deg=trajectory_mapping.get("end_angle_deg"),
        step_angle_deg=trajectory_mapping.get("step_angle_deg"),
    )

    kinematics_mapping = _required_mapping(document.get("kinematics"), "kinematics")
    _reject_unknown(
        kinematics_mapping,
        {
            "axis_point_scan_mm",
            "axis_direction_scan",
            "zero_offset_deg",
            "T_scan_from_camera_zero",
        },
        "kinematics",
    )
    kinematics = ScanKinematicsConfig(
        axis_point_scan_mm=kinematics_mapping.get("axis_point_scan_mm"),
        axis_direction_scan=kinematics_mapping.get("axis_direction_scan"),
        zero_offset_deg=kinematics_mapping.get("zero_offset_deg", 0.0),
        T_scan_from_camera_zero=kinematics_mapping.get("T_scan_from_camera_zero"),
    )

    output_mapping = _required_mapping(document.get("output"), "output")
    _reject_unknown(
        output_mapping,
        {"directory", "save_profiles_csv", "save_ply", "save_pcd"},
        "output",
    )
    output_value = output_mapping.get("directory")
    if not isinstance(output_value, (str, Path)) or not str(output_value).strip():
        raise ScanConfigError("output.directory 必须是非空路径")
    output_path = Path(output_value).expanduser()
    if not output_path.is_absolute():
        output_path = (config_path.resolve().parent / output_path).resolve()
    output = ScanOutputConfig(
        directory=output_path,
        save_profiles_csv=output_mapping.get("save_profiles_csv", True),
        save_ply=output_mapping.get("save_ply", True),
        save_pcd=output_mapping.get("save_pcd", True),
    )
    return ScanConfig(
        schema_version=int(schema_version),
        mode=mode,
        trajectory=trajectory,
        kinematics=kinematics,
        output=output,
    )


__all__ = [
    "SUPPORTED_SCHEMA_VERSION",
    "ScanConfig",
    "ScanConfigError",
    "ScanKinematicsConfig",
    "ScanOutputConfig",
    "ScanTrajectoryConfig",
    "load_scan_config",
]
