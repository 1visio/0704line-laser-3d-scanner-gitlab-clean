"""Read-only acceptance verifier for a Stage-1 scan Session.

The verifier deliberately does not import ``scan.kinematics``.  Its
repeat-one check uses an independent Rodrigues implementation so that a bug in
the production transform cannot make the acceptance check pass by definition.

Usage::

    python tools/verify_scan_stage1.py output/scan_stage1/scan_20260808_001
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_KINEMATIC_TOLERANCE_MM = 1.0e-4


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(slots=True)
class VerificationReport:
    checks: list[VerificationCheck] = field(default_factory=list)
    kinematic_max_error_mm: float | None = None
    kinematic_rmse_mm: float | None = None

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)


@dataclass(frozen=True, slots=True)
class _ScanConfigValues:
    start_angle_deg: float
    end_angle_deg: float
    step_angle_deg: float
    angles_deg: tuple[float, ...]
    axis_point_scan_mm: np.ndarray
    axis_direction_scan: np.ndarray
    zero_offset_deg: float
    transform: np.ndarray


@dataclass(frozen=True, slots=True)
class _PoseRow:
    frame_index: int
    filename: str
    angle_command_deg: float
    angle_measured_deg: float
    valid_laser_points: int
    points_camera_count: int
    points_scan_count: int
    warning: str


@dataclass(frozen=True, slots=True)
class _ProfileData:
    path: Path
    points_camera: np.ndarray
    points_scan: np.ndarray


@dataclass(frozen=True, slots=True)
class _CloudData:
    points: np.ndarray
    declared_count: int
    comments: tuple[str, ...]
    format_name: str


class _VerificationInputError(ValueError):
    """An input artifact is malformed for Stage-1 acceptance."""


def verify_scan_session(
    session_directory: str | Path,
    *,
    kinematic_tolerance_mm: float = DEFAULT_KINEMATIC_TOLERANCE_MM,
) -> VerificationReport:
    """Validate one existing scan directory without modifying it."""
    report = VerificationReport()
    try:
        tolerance = _positive_finite(kinematic_tolerance_mm, "kinematic_tolerance_mm")
    except _VerificationInputError as error:
        _add(report, "输入容差", False, str(error))
        return report

    root = Path(session_directory).expanduser()
    if not root.is_dir():
        _add(report, "Session 目录", False, f"目录不存在: {root}")
        return report

    required_paths = {
        "scan_config.yaml": root / "scan_config.yaml",
        "source.json": root / "source.json",
        "poses.csv": root / "poses.csv",
        "profiles": root / "profiles",
        "cloud_scan.ply": root / "cloud_scan.ply",
        "cloud_scan.pcd": root / "cloud_scan.pcd",
        "result.json": root / "result.json",
    }
    missing = [
        name
        for name, path in required_paths.items()
        if not (path.is_dir() if name == "profiles" else path.is_file())
    ]
    if missing:
        _add(report, "输出文件完整性", False, "缺少: " + ", ".join(missing))
        return report
    _add(report, "输出文件完整性", True, "scan_config/source/poses/profiles/PLY/PCD/result 均存在")

    try:
        config = _read_scan_config(required_paths["scan_config.yaml"])
    except (OSError, UnicodeError, yaml.YAMLError, _VerificationInputError) as error:
        _add(report, "扫描配置", False, str(error))
        return report
    _add(report, "扫描配置", True, "配置字段完整且数值有效")

    try:
        source = _read_json(required_paths["source.json"], "source.json")
        result = _read_json(required_paths["result.json"], "result.json")
        poses = _read_poses(required_paths["poses.csv"])
        profiles = _read_profiles(required_paths["profiles"])
        ply = _read_ply(required_paths["cloud_scan.ply"])
        pcd = _read_pcd(required_paths["cloud_scan.pcd"])
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        csv.Error,
        ValueError,
        TypeError,
        IndexError,
        OverflowError,
    ) as error:
        _add(report, "输出数据格式", False, str(error))
        return report

    try:
        mode = _normalise_mode(result.get("mode"))
    except _VerificationInputError as error:
        _add(report, "result.json mode", False, str(error))
        return report
    _check_source_metadata(report, source, result, mode)
    _check_result_metadata(report, result, mode)
    _check_trajectory_and_alignment(report, config, result, poses, profiles)
    _check_warning_state(report, result, poses)
    _check_clouds_and_counts(report, result, profiles, ply, pcd)

    if mode == "repeat_one":
        _check_repeat_one_kinematics(
            report,
            config,
            poses,
            profiles,
            tolerance,
        )
    else:
        _add(report, "repeat-one 解析运动学", True, "sequence 模式，跳过 repeat-one 专项校验")
    return report


def verify_scan_stage1(
    session_directory: str | Path,
    *,
    kinematic_tolerance_mm: float = DEFAULT_KINEMATIC_TOLERANCE_MM,
) -> VerificationReport:
    """Named public alias for callers treating the tool as a library."""
    return verify_scan_session(
        session_directory,
        kinematic_tolerance_mm=kinematic_tolerance_mm,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="只读验收 Stage-1 scan Session（不修改输入目录）"
    )
    parser.add_argument("session_directory", type=Path, help="已有 scan_xxx 输出目录")
    parser.add_argument(
        "--kinematic-tolerance-mm",
        type=float,
        default=DEFAULT_KINEMATIC_TOLERANCE_MM,
        help="repeat-one 解析运动学允许的最大点误差（mm）",
    )
    args = parser.parse_args(argv)
    report = verify_scan_session(
        args.session_directory,
        kinematic_tolerance_mm=args.kinematic_tolerance_mm,
    )
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    if report.kinematic_max_error_mm is not None:
        print(f"Kinematics max error: {report.kinematic_max_error_mm:.9g} mm")
        print(f"Kinematics RMSE: {report.kinematic_rmse_mm:.9g} mm")
    print("SUMMARY: PASS" if report.passed else "SUMMARY: FAIL")
    return 0 if report.passed else 1


def _add(report: VerificationReport, name: str, passed: bool, detail: str) -> None:
    report.checks.append(VerificationCheck(name, bool(passed), str(detail)))


def _read_json(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise _VerificationInputError(f"{name} 顶层必须是 JSON 对象")
    return payload


def _read_scan_config(path: Path) -> _ScanConfigValues:
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, Mapping):
        raise _VerificationInputError("scan_config.yaml 顶层必须是映射")
    if document.get("schema_version") != 1:
        raise _VerificationInputError("scan_config.yaml schema_version 必须为 1")
    mode = document.get("mode")
    if not isinstance(mode, str) or mode.strip().lower().replace("-", "_") not in {
        "repeat_one",
        "sequence",
    }:
        raise _VerificationInputError("scan_config.yaml mode 必须是 repeat_one 或 sequence")
    trajectory = _mapping(document, "trajectory")
    start = _finite(trajectory.get("start_angle_deg"), "start_angle_deg")
    end = _finite(trajectory.get("end_angle_deg"), "end_angle_deg")
    step = _finite(trajectory.get("step_angle_deg"), "step_angle_deg")
    if step == 0.0:
        raise _VerificationInputError("step_angle_deg 不能为 0")
    if (end - start) * step < 0.0:
        raise _VerificationInputError("step_angle_deg 方向无法从 start 走向 end")

    kinematics = _mapping(document, "kinematics")
    axis_point = _finite_array(
        kinematics.get("axis_point_scan_mm"), (3,), "axis_point_scan_mm"
    )
    axis_direction = _finite_array(
        kinematics.get("axis_direction_scan"), (3,), "axis_direction_scan"
    )
    if np.linalg.norm(axis_direction) <= np.finfo(np.float64).eps:
        raise _VerificationInputError("axis_direction_scan 不能为零向量")
    zero_offset = _finite(kinematics.get("zero_offset_deg", 0.0), "zero_offset_deg")
    transform = _finite_array(
        kinematics.get("T_scan_from_camera_zero"),
        (4, 4),
        "T_scan_from_camera_zero",
    )
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
        raise _VerificationInputError("T_scan_from_camera_zero 最后一行非法")
    output = _mapping(document, "output")
    if not isinstance(output.get("directory"), (str, Path)) or not str(
        output.get("directory")
    ).strip():
        raise _VerificationInputError("scan_config.yaml output.directory 缺失")
    for field_name in ("save_profiles_csv", "save_ply", "save_pcd"):
        if field_name in output and not isinstance(output[field_name], bool):
            raise _VerificationInputError(f"scan_config.yaml output.{field_name} 必须是布尔值")
    return _ScanConfigValues(
        start,
        end,
        step,
        _angle_sequence(start, end, step),
        axis_point,
        axis_direction,
        zero_offset,
        transform,
    )


def _mapping(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, Mapping):
        raise _VerificationInputError(f"scan_config.yaml 缺少 {name} 映射")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise _VerificationInputError(f"{name} 必须是有限数值")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise _VerificationInputError(f"{name} 必须是有限数值") from error
    if not math.isfinite(result):
        raise _VerificationInputError(f"{name} 必须是有限数值")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise _VerificationInputError(f"{name} 必须为正数")
    return result


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise _VerificationInputError(f"{name} 必须是数值数组") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise _VerificationInputError(f"{name} 必须是有限的 {shape} 数组")
    return np.ascontiguousarray(array)


def _angle_sequence(start: float, end: float, step: float) -> tuple[float, ...]:
    if start == end:
        return (start,)
    distance = abs(end - start)
    count = int(math.floor(distance / abs(step) + 1.0e-12))
    values = [start + index * step for index in range(count + 1)]
    if not math.isclose(values[-1], end, rel_tol=0.0, abs_tol=1.0e-9):
        values.append(end)
    else:
        values[-1] = end
    return tuple(values)


def _normalise_mode(value: Any) -> str:
    if not isinstance(value, str):
        raise _VerificationInputError("result.json mode 必须是 repeat_one 或 sequence")
    mode = value.strip().lower().replace("-", "_")
    if mode not in {"repeat_one", "sequence"}:
        raise _VerificationInputError(f"不支持的扫描 mode: {value!r}")
    return mode


def _check_source_metadata(
    report: VerificationReport,
    source: Mapping[str, Any],
    result: Mapping[str, Any],
    mode: str,
) -> None:
    required = (
        "calibration_package_id",
        "calibration_manifest_sha256",
    )
    missing = [name for name in required if not _present_text(source.get(name))]
    algorithm_hash = source.get("extraction_algorithm_hash") or source.get(
        "algorithm_config_sha256"
    )
    if not _present_text(algorithm_hash):
        missing.append("extraction_algorithm_hash/algorithm_config_sha256")
    try:
        source_mode = _normalise_mode(source.get("mode"))
        if source_mode != mode:
            missing.append("source.mode 与 result.mode 不一致")
    except _VerificationInputError:
        missing.append("source.mode 非法")
    source_demo = source.get("kinematic_demo_only")
    result_demo = result.get("kinematic_demo_only")
    if not isinstance(source_demo, bool) or not isinstance(result_demo, bool):
        missing.append("kinematic_demo_only 必须是布尔值")
    elif source_demo != result_demo or source_demo != (mode == "repeat_one"):
        missing.append("kinematic_demo_only 与 mode 不一致")
    _add(
        report,
        "source 追溯元数据",
        not missing,
        "校验通过" if not missing else "缺失或不一致: " + ", ".join(missing),
    )


def _present_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_result_metadata(
    report: VerificationReport,
    result: Mapping[str, Any],
    mode: str,
) -> None:
    failures: list[str] = []
    for key in (
        "frame_count",
        "total_point_count",
        "angle_start_deg",
        "angle_end_deg",
        "coordinate_system",
        "units",
        "kinematic_demo_only",
    ):
        if key not in result:
            failures.append(key)
    if result.get("coordinate_system") != "scan":
        failures.append("coordinate_system != scan")
    if str(result.get("units", "")).lower() not in {"mm", "millimeter", "millimeters"}:
        failures.append("units != mm")
    if result.get("kinematic_demo_only") is not (mode == "repeat_one"):
        failures.append("kinematic_demo_only")
    for key in ("angle_start_deg", "angle_end_deg"):
        if key in result:
            try:
                _finite(result[key], f"result.{key}")
            except _VerificationInputError:
                failures.append(key)
    _add(
        report,
        "result.json 元数据",
        not failures,
        "坐标系 scan、单位 mm、统计字段完整"
        if not failures
        else "非法或缺失: " + ", ".join(dict.fromkeys(failures)),
    )


def _check_trajectory_and_alignment(
    report: VerificationReport,
    config: _ScanConfigValues,
    result: Mapping[str, Any],
    poses: tuple[_PoseRow, ...],
    profiles: tuple[_ProfileData, ...],
) -> None:
    failures: list[str] = []
    expected_angles = config.angles_deg
    try:
        result_frame_count = _nonnegative_int(result.get("frame_count"), "result.frame_count")
        result_start = _finite(result.get("angle_start_deg"), "result.angle_start_deg")
        result_end = _finite(result.get("angle_end_deg"), "result.angle_end_deg")
    except _VerificationInputError as error:
        _add(report, "轨迹与帧对齐", False, str(error))
        return
    if result_frame_count != len(expected_angles):
        failures.append(f"result.frame_count={result_frame_count}，期望 {len(expected_angles)}")
    if len(poses) != len(expected_angles):
        failures.append(f"poses={len(poses)}，期望 {len(expected_angles)}")
    if len(profiles) != len(expected_angles):
        failures.append(f"profiles={len(profiles)}，期望 {len(expected_angles)}")
    if poses:
        pose_angles = np.asarray([row.angle_command_deg for row in poses])
        if len(pose_angles) == len(expected_angles) and not np.allclose(
            pose_angles, expected_angles, atol=1.0e-9, rtol=0.0
        ):
            failures.append("poses.angle_command_deg 与配置轨迹不一致")
        measured = np.asarray([row.angle_measured_deg for row in poses])
        if not np.allclose(measured, pose_angles, atol=1.0e-9, rtol=0.0):
            failures.append("模拟轴 angle_measured_deg != angle_command_deg")
        frame_indices = [row.frame_index for row in poses]
        if len(set(frame_indices)) != len(frame_indices):
            failures.append("poses.frame_index 重复")
    if profiles and len(poses) == len(profiles):
        expected_names = tuple(f"profile_{index:04d}.csv" for index in range(len(profiles)))
        actual_names = tuple(profile.path.name for profile in profiles)
        if actual_names != expected_names:
            failures.append("profiles 文件未按 profile_XXXX.csv 顺序一一对应")
        for row, profile in zip(poses, profiles):
            if row.points_camera_count != len(profile.points_camera):
                failures.append(
                    f"frame_index={row.frame_index} points_camera_count 与 profile 不一致"
                )
            if row.points_scan_count != len(profile.points_scan):
                failures.append(
                    f"frame_index={row.frame_index} points_scan_count 与 profile 不一致"
                )
    output_files = result.get("output_files")
    if not isinstance(output_files, Mapping):
        failures.append("result.output_files 缺失")
    else:
        declared_profiles = output_files.get("profiles")
        if not isinstance(declared_profiles, list):
            failures.append("result.output_files.profiles 必须是数组")
        else:
            declared_names = tuple(Path(str(value)).name for value in declared_profiles)
            actual_names = tuple(profile.path.name for profile in profiles)
            if declared_names != actual_names:
                failures.append("result.output_files.profiles 与实际文件不一致")
        for key, expected_name in (
            ("scan_config", "scan_config.yaml"),
            ("source", "source.json"),
            ("poses", "poses.csv"),
            ("cloud_scan_ply", "cloud_scan.ply"),
            ("cloud_scan_pcd", "cloud_scan.pcd"),
            ("result", "result.json"),
        ):
            if output_files.get(key) != expected_name:
                failures.append(f"result.output_files.{key} 不指向 {expected_name}")
    if not math.isclose(result_start, config.start_angle_deg, abs_tol=1.0e-9, rel_tol=0.0):
        failures.append("result.angle_start_deg 与配置不一致")
    if not math.isclose(result_end, config.end_angle_deg, abs_tol=1.0e-9, rel_tol=0.0):
        failures.append("result.angle_end_deg 与配置不一致")
    _add(
        report,
        "轨迹与 poses/profiles 对齐",
        not failures,
        "起止角、步长、帧数及顺序一致" if not failures else "; ".join(failures),
    )


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise _VerificationInputError(f"{name} 必须是非负整数")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _VerificationInputError(f"{name} 必须是非负整数") from error
    if str(value).strip() not in {str(integer), f"{integer}.0"} and not isinstance(value, int):
        raise _VerificationInputError(f"{name} 必须是非负整数")
    if integer < 0:
        raise _VerificationInputError(f"{name} 必须是非负整数")
    return integer


def _read_poses(path: Path) -> tuple[_PoseRow, ...]:
    required = {
        "frame_index",
        "filename",
        "angle_command_deg",
        "angle_measured_deg",
        "valid_laser_points",
        "points_camera_count",
        "points_scan_count",
    }
    rows: list[_PoseRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise _VerificationInputError("poses.csv 缺少字段: " + ", ".join(sorted(missing)))
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not str(value or "").strip() for value in row.values()):
                continue
            frame_index = _nonnegative_int(row.get("frame_index"), f"poses 第 {row_number} 行 frame_index")
            filename = str(row.get("filename") or "").strip()
            if not filename:
                raise _VerificationInputError(f"poses 第 {row_number} 行 filename 为空")
            rows.append(
                _PoseRow(
                    frame_index,
                    filename,
                    _finite(row.get("angle_command_deg"), f"poses 第 {row_number} 行 command"),
                    _finite(row.get("angle_measured_deg"), f"poses 第 {row_number} 行 measured"),
                    _nonnegative_int(row.get("valid_laser_points"), f"poses 第 {row_number} 行 laser count"),
                    _nonnegative_int(row.get("points_camera_count"), f"poses 第 {row_number} 行 camera count"),
                    _nonnegative_int(row.get("points_scan_count"), f"poses 第 {row_number} 行 scan count"),
                    str(row.get("warning") or "").strip(),
                )
            )
    return tuple(rows)


def _read_profiles(directory: Path) -> tuple[_ProfileData, ...]:
    paths = tuple(sorted(directory.glob("profile_*.csv"), key=lambda item: item.name))
    if not paths:
        raise _VerificationInputError("profiles/ 中没有 profile_XXXX.csv")
    return tuple(_read_profile(path) for path in paths)


def _read_profile(path: Path) -> _ProfileData:
    required = {
        "u_px",
        "v_px",
        "Xc_mm",
        "Yc_mm",
        "Zc_mm",
        "Xs_mm",
        "Ys_mm",
        "Zs_mm",
    }
    values: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise _VerificationInputError(f"{path.name} 缺少字段: " + ", ".join(sorted(missing)))
        names = ("u_px", "v_px", "Xc_mm", "Yc_mm", "Zc_mm", "Xs_mm", "Ys_mm", "Zs_mm")
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not str(value or "").strip() for value in row.values()):
                continue
            try:
                values.append([float(row[name]) for name in names])
            except (TypeError, ValueError, KeyError) as error:
                raise _VerificationInputError(f"{path.name} 第 {row_number} 行含非数值") from error
    data = np.asarray(values, dtype=np.float64).reshape(-1, 8)
    if not np.isfinite(data).all():
        raise _VerificationInputError(f"{path.name} 含 NaN 或无穷值")
    return _ProfileData(path, data[:, 2:5], data[:, 5:8])


def _read_ply(path: Path) -> _CloudData:
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0].strip().lower() != "ply":
        raise _VerificationInputError("cloud_scan.ply 缺少 ply 头")
    try:
        end_header = next(index for index, line in enumerate(lines) if line.strip().lower() == "end_header")
    except StopIteration as error:
        raise _VerificationInputError("cloud_scan.ply 缺少 end_header") from error
    declared = None
    comments: list[str] = []
    for line in lines[1:end_header]:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("element vertex"):
            declared = int(stripped.split()[2])
        elif lower.startswith("comment "):
            comments.append(stripped[8:].strip().lower())
    if declared is None or declared < 0:
        raise _VerificationInputError("cloud_scan.ply 缺少合法 vertex 数")
    points = _parse_xyz_lines(lines[end_header + 1 :], "cloud_scan.ply")
    return _CloudData(points, declared, tuple(comments), "PLY")


def _read_pcd(path: Path) -> _CloudData:
    lines = path.read_text(encoding="ascii").splitlines()
    data_index = None
    declared = None
    width = None
    comments: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("#"):
            comments.append(stripped[1:].strip().lower())
        elif lower.startswith("points "):
            declared = int(stripped.split()[1])
        elif lower.startswith("width "):
            width = int(stripped.split()[1])
        elif lower == "data ascii":
            data_index = index
            break
    if data_index is None or declared is None or width is None or declared < 0:
        raise _VerificationInputError("cloud_scan.pcd 头部不完整或不是 ASCII PCD")
    if width != declared:
        raise _VerificationInputError(f"cloud_scan.pcd WIDTH={width} 与 POINTS={declared} 不一致")
    points = _parse_xyz_lines(lines[data_index + 1 :], "cloud_scan.pcd")
    return _CloudData(points, declared, tuple(comments), "PCD")


def _parse_xyz_lines(lines: Sequence[str], name: str) -> np.ndarray:
    values: list[list[float]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 3:
            raise _VerificationInputError(f"{name} 数据行 {line_number} 不是 XYZ 三元组")
        try:
            values.append([float(part) for part in parts])
        except ValueError as error:
            raise _VerificationInputError(f"{name} 数据行 {line_number} 含非数值") from error
    points = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise _VerificationInputError(f"{name} XYZ 含 NaN 或无穷值")
    return points


def _check_warning_state(
    report: VerificationReport,
    result: Mapping[str, Any],
    poses: tuple[_PoseRow, ...],
) -> None:
    raw_warnings = result.get("warnings")
    failures: list[str] = []
    if not isinstance(raw_warnings, list):
        failures.append("result.warnings 必须是数组")
        raw_warnings = []
    pose_by_index = {row.frame_index: row for row in poses}
    warning_frames: set[int] = set()
    for warning in raw_warnings:
        if not isinstance(warning, Mapping):
            failures.append("warning 项必须是对象")
            continue
        try:
            frame_index = _nonnegative_int(warning.get("frame_index"), "warning.frame_index")
        except _VerificationInputError:
            failures.append("warning.frame_index 非法")
            continue
        message = str(warning.get("message") or "").strip()
        if frame_index not in pose_by_index or not message:
            failures.append(f"warning frame_index={frame_index} 无对应帧或无 message")
        if frame_index in warning_frames:
            failures.append(f"warning frame_index={frame_index} 重复")
        warning_frames.add(frame_index)
    for row in poses:
        has_zero_count = min(
            row.valid_laser_points,
            row.points_camera_count,
            row.points_scan_count,
        ) == 0
        if has_zero_count and not row.warning:
            failures.append(f"frame_index={row.frame_index} 点数为 0 但没有 warning")
        if row.warning and row.frame_index not in warning_frames:
            failures.append(f"frame_index={row.frame_index} poses warning 未登记到 result.warnings")
        if row.frame_index in warning_frames and not row.warning:
            failures.append(f"frame_index={row.frame_index} result warning 未反映到 poses")
    _add(
        report,
        "warning 状态",
        not failures,
        "warning 与零点帧/帧索引一致" if not failures else "; ".join(failures),
    )


def _check_clouds_and_counts(
    report: VerificationReport,
    result: Mapping[str, Any],
    profiles: tuple[_ProfileData, ...],
    ply: _CloudData,
    pcd: _CloudData,
) -> None:
    failures: list[str] = []
    try:
        total = _nonnegative_int(result.get("total_point_count"), "result.total_point_count")
    except _VerificationInputError as error:
        _add(report, "点数与 XYZ", False, str(error))
        return
    profile_points = (
        np.concatenate([profile.points_scan for profile in profiles], axis=0)
        if profiles
        else np.empty((0, 3), dtype=np.float64)
    )
    if len(profile_points) != total:
        failures.append(f"profiles 点数和={len(profile_points)}，result={total}")
    if ply.declared_count != len(ply.points) or len(ply.points) != total:
        failures.append(f"PLY header/data/result 不一致 ({ply.declared_count}/{len(ply.points)}/{total})")
    if pcd.declared_count != len(pcd.points) or len(pcd.points) != total:
        failures.append(f"PCD header/data/result 不一致 ({pcd.declared_count}/{len(pcd.points)}/{total})")
    if ply.points.shape == pcd.points.shape and not np.allclose(
        ply.points, pcd.points, rtol=0.0, atol=1.0e-8
    ):
        failures.append("PLY 与 PCD XYZ 数据不一致")
    elif ply.points.shape != pcd.points.shape:
        failures.append("PLY 与 PCD XYZ 形状不一致")
    if len(profile_points) == len(ply.points) and not np.allclose(
        profile_points, ply.points, rtol=0.0, atol=2.0e-5
    ):
        failures.append("profiles XYZ 与 PLY 数据不一致")
    comments_ok = (
        any("coordinate_system scan" in comment for comment in ply.comments)
        and any("units millimeter" in comment or "units mm" in comment for comment in ply.comments)
        and any("coordinate_system scan" in comment for comment in pcd.comments)
        and any("units millimeter" in comment or "units mm" in comment for comment in pcd.comments)
    )
    if not comments_ok:
        failures.append("PLY/PCD 坐标系或单位注释不是 scan/mm")
    _add(
        report,
        "点数、XYZ 与点云格式",
        not failures,
        "profiles、PLY、PCD 点数一致，XYZ 有限且坐标系为 scan/mm"
        if not failures
        else "; ".join(failures),
    )


def _check_repeat_one_kinematics(
    report: VerificationReport,
    config: _ScanConfigValues,
    poses: tuple[_PoseRow, ...],
    profiles: tuple[_ProfileData, ...],
    tolerance: float,
) -> None:
    zero_indices = [index for index, row in enumerate(poses) if abs(row.angle_command_deg) <= 1.0e-9]
    if len(zero_indices) != 1:
        _add(report, "repeat-one 解析运动学", False, "找不到唯一的 0° profile")
        return
    zero_index = zero_indices[0]
    if zero_index >= len(profiles):
        _add(report, "repeat-one 解析运动学", False, "0° profile 不存在")
        return
    reference_camera = profiles[zero_index].points_camera
    if len(reference_camera) == 0:
        _add(report, "repeat-one 解析运动学", False, "0° profile 没有可验证点")
        return

    errors: list[np.ndarray] = []
    failures: list[str] = []
    for index, (pose, profile) in enumerate(zip(poses, profiles)):
        if len(profile.points_scan) == 0:
            continue
        if len(profile.points_camera) != len(reference_camera):
            failures.append(f"frame_index={pose.frame_index} camera 点数与 0° 不一致")
            continue
        if not np.allclose(
            profile.points_camera,
            reference_camera,
            rtol=0.0,
            atol=tolerance,
        ):
            failures.append(f"frame_index={pose.frame_index} points_camera 与 0° 不一致")
        expected = _independent_rigid_transform(
            reference_camera,
            pose.angle_command_deg,
            config.axis_point_scan_mm,
            config.axis_direction_scan,
            config.zero_offset_deg,
            config.transform,
        )
        errors.append(np.linalg.norm(profile.points_scan - expected, axis=1))
    if not errors:
        _add(report, "repeat-one 解析运动学", False, "没有可比较的非空 profile")
        return
    all_errors = np.concatenate(errors)
    maximum = float(np.max(all_errors))
    rmse = float(np.sqrt(np.mean(all_errors * all_errors)))
    report.kinematic_max_error_mm = maximum
    report.kinematic_rmse_mm = rmse
    if maximum > tolerance:
        failures.append(f"max error={maximum:.9g} mm > tolerance={tolerance:.9g} mm")
    _add(
        report,
        "repeat-one 解析运动学",
        not failures,
        f"max error={maximum:.9g} mm, RMSE={rmse:.9g} mm"
        if not failures
        else "; ".join(failures),
    )


def _independent_rigid_transform(
    points_camera: np.ndarray,
    angle_deg: float,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    zero_offset_deg: float,
    transform: np.ndarray,
) -> np.ndarray:
    """Independent camera-zero transform followed by axis rotation."""
    points_zero = points_camera @ transform[:3, :3].T + transform[:3, 3]
    axis = axis_direction / np.linalg.norm(axis_direction)
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    angle = math.radians(float(angle_deg) + float(zero_offset_deg))
    identity = np.eye(3, dtype=np.float64)
    rotation = (
        math.cos(angle) * identity
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * skew
    )
    return (points_zero - axis_point) @ rotation.T + axis_point


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_KINEMATIC_TOLERANCE_MM",
    "VerificationCheck",
    "VerificationReport",
    "main",
    "verify_scan_session",
    "verify_scan_stage1",
]
