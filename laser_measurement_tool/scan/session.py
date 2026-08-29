"""Transactional, traceable output sessions for Stage-1 offline scans."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import ScanConfig, ScanConfigError, load_scan_config
from .models import ScanResult
from .offline_scan import OfflineScanResult


class ScanSessionError(RuntimeError):
    """A scan session could not be created or committed."""


@dataclass(frozen=True, slots=True)
class ScanSession:
    """Write one scan result into a unique, self-contained session directory.

    The result is first written below a hidden temporary directory.  The
    directory is renamed into the public ``scan_YYYYMMDD_NNN`` name only after
    every requested output, including ``result.json``, has been written.
    """

    output_root: str | Path
    scan_config: ScanConfig | Mapping[str, Any] | None = None
    scan_config_path: str | Path | None = None
    measure_tool_config_path: str | Path | None = None
    mode: str | None = None
    source_image: str | Path | None = None
    frames_directory: str | Path | None = None
    pipeline: object | None = None
    calibration_package_id: str | None = None
    calibration_manifest_sha256: str | None = None
    extraction_algorithm_hash: str | None = None
    created_time: str | None = None

    def write(self, result: OfflineScanResult | ScanResult) -> Path:
        """Commit ``result`` and return its newly-created session directory."""
        scan_result, result_metadata = _unwrap_result(result)
        config = self._load_config()
        mode = _normalize_mode(
            self.mode or result_metadata.get("mode") or config.mode
        )
        kinematic_demo_only = mode == "repeat_one"
        snapshot = self._config_snapshot(config)
        source_filenames = _source_filenames(result_metadata, scan_result)
        profile_filenames = _profile_output_filenames(scan_result)
        output_flags = _output_flags(config)

        output_root = _resolve_directory(self.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        temporary_directory = Path(
            tempfile.mkdtemp(prefix=".scan_", dir=str(output_root))
        )
        try:
            _write_text_exclusive(temporary_directory / "scan_config.yaml", snapshot)
            source_payload = self._source_payload(
                result_metadata,
                mode=mode,
                kinematic_demo_only=kinematic_demo_only,
            )
            _write_json_exclusive(
                temporary_directory / "source.json", source_payload
            )
            _write_poses(
                temporary_directory / "poses.csv",
                scan_result,
                source_filenames,
                result_metadata,
            )

            profile_files: list[str] = []
            profiles_directory = temporary_directory / "profiles"
            profiles_directory.mkdir()
            if output_flags["save_profiles_csv"]:
                for profile, filename in zip(
                    scan_result.profiles, profile_filenames
                ):
                    profile_path = profiles_directory / filename
                    _write_profile_csv(profile_path, profile)
                    profile_files.append(f"profiles/{filename}")

            output_files: dict[str, object] = {
                "scan_config": "scan_config.yaml",
                "source": "source.json",
                "poses": "poses.csv",
                "profiles": profile_files,
                "cloud_scan_ply": None,
                "cloud_scan_pcd": None,
                "result": "result.json",
            }
            if output_flags["save_ply"]:
                _write_scan_ply(
                    temporary_directory / "cloud_scan.ply", scan_result.points_scan
                )
                output_files["cloud_scan_ply"] = "cloud_scan.ply"
            if output_flags["save_pcd"]:
                _write_scan_pcd(
                    temporary_directory / "cloud_scan.pcd", scan_result.points_scan
                )
                output_files["cloud_scan_pcd"] = "cloud_scan.pcd"

            result_payload = _result_payload(
                scan_result,
                config=config,
                mode=mode,
                kinematic_demo_only=kinematic_demo_only,
                output_files=output_files,
                result_metadata=result_metadata,
            )
            _write_json_exclusive(
                temporary_directory / "result.json", result_payload
            )
            return _commit_session_directory(temporary_directory, output_root)
        except BaseException:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise

    def _load_config(self) -> ScanConfig:
        if self.scan_config is not None:
            if not isinstance(self.scan_config, ScanConfig):
                raise ScanSessionError("scan_config 必须是 ScanConfig")
            return self.scan_config
        if self.scan_config_path is None:
            raise ScanSessionError("必须提供 scan_config 或 scan_config_path")
        try:
            return load_scan_config(self.scan_config_path)
        except ScanConfigError as error:
            raise ScanSessionError(f"无法加载扫描配置快照: {error}") from error

    def _config_snapshot(self, config: ScanConfig) -> str:
        if self.scan_config_path is not None:
            path = Path(self.scan_config_path).expanduser()
            if not path.is_file():
                raise ScanSessionError(f"扫描配置不存在: {path}")
            try:
                text = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError) as error:
                raise ScanSessionError(f"无法读取扫描配置快照 {path}: {error}") from error
            if not text.strip():
                raise ScanSessionError(f"扫描配置快照为空: {path}")
            return text if text.endswith("\n") else text + "\n"

        document = _scan_config_document(config)
        return yaml.safe_dump(
            document,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    def _source_payload(
        self,
        result_metadata: Mapping[str, object],
        *,
        mode: str,
        kinematic_demo_only: bool,
    ) -> dict[str, object]:
        pipeline = self.pipeline
        package = getattr(pipeline, "package", None)
        pipeline_config = getattr(pipeline, "config", None)

        source_image = self.source_image
        if source_image is None:
            source_image = result_metadata.get("input_image")
        frames_directory = self.frames_directory
        if frames_directory is None:
            frames_directory = result_metadata.get("image_dir")
        frame_angle_csv = result_metadata.get("frame_angle_csv")

        measure_config = self.measure_tool_config_path
        if measure_config is None:
            measure_config = getattr(pipeline_config, "config_path", None)

        package_id = self.calibration_package_id
        if package_id is None:
            package_id = _first_value(
                result_metadata.get("calibration_package_id"),
                getattr(package, "package_id", None),
            )
        manifest_sha256 = self.calibration_manifest_sha256
        if manifest_sha256 is None:
            manifest_sha256 = _first_value(
                result_metadata.get("calibration_manifest_sha256"),
                getattr(package, "manifest_sha256", None),
            )
        algorithm_hash = self.extraction_algorithm_hash
        if algorithm_hash is None:
            algorithm_hash = _first_value(
                result_metadata.get("extraction_algorithm_hash"),
                result_metadata.get("algorithm_config_sha256"),
                getattr(pipeline, "algorithm_config_sha256", None),
            )

        payload: dict[str, object] = {
            "mode": mode,
            "source_image": _path_value(source_image),
            "frames_directory": _path_value(frames_directory),
            "frame_angle_csv": _path_value(frame_angle_csv),
            "source": {
                "image": _path_value(source_image),
                "frames_directory": _path_value(frames_directory),
            },
            "measure_tool_config_path": _path_value(measure_config),
            "scan_config_path": _path_value(self.scan_config_path),
            "calibration_package_id": package_id,
            "calibration_manifest_sha256": manifest_sha256,
            "extraction_algorithm_hash": algorithm_hash,
            # Keep the runtime name as an alias for existing pipeline metadata.
            "algorithm_config_sha256": algorithm_hash,
            "created_time": self.created_time or _utc_now(),
            "kinematic_demo_only": kinematic_demo_only,
        }
        if "image_offset" in result_metadata:
            payload["image_offset"] = result_metadata["image_offset"]
        return payload


ScanSessionWriter = ScanSession


def write_scan_session(
    result: OfflineScanResult | ScanResult,
    output_root: str | Path,
    **kwargs: Any,
) -> Path:
    """Convenience wrapper around :class:`ScanSession`."""
    return ScanSession(output_root, **kwargs).write(result)


def _unwrap_result(
    result: OfflineScanResult | ScanResult,
) -> tuple[ScanResult, dict[str, object]]:
    if isinstance(result, OfflineScanResult):
        return result.scan_result, dict(result.metadata)
    if isinstance(result, ScanResult):
        return result, {}
    raise ScanSessionError("扫描结果必须是 OfflineScanResult 或 ScanResult")


def _normalize_mode(value: object) -> str:
    if not isinstance(value, str):
        raise ScanSessionError("扫描 mode 必须是 repeat_one 或 sequence")
    mode = value.strip().lower().replace("-", "_")
    if mode not in {"repeat_one", "sequence"}:
        raise ScanSessionError("扫描 mode 必须是 repeat_one 或 sequence")
    return mode


def _resolve_directory(value: str | Path) -> Path:
    try:
        return Path(value).expanduser().resolve()
    except (TypeError, ValueError, OSError) as error:
        raise ScanSessionError(f"扫描输出目录无效: {value!r}") from error


def _first_value(*values: object) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return None


def _path_value(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    try:
        return str(Path(value).expanduser().resolve())
    except (TypeError, ValueError, OSError):
        return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_filenames(
    metadata: Mapping[str, object], scan_result: ScanResult
) -> tuple[str, ...]:
    raw = metadata.get("profile_filenames")
    names: list[str] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        names = [Path(str(value)).name for value in raw]
    if len(names) != len(scan_result.profiles):
        names = [f"profile_{index:04d}.csv" for index in range(len(scan_result.profiles))]
    return tuple(names)


def _profile_output_filenames(scan_result: ScanResult) -> tuple[str, ...]:
    return tuple(
        f"profile_{index:04d}.csv" for index in range(len(scan_result.profiles))
    )


def _output_flags(config: ScanConfig) -> dict[str, bool]:
    output = config.output
    return {
        "save_profiles_csv": bool(output.save_profiles_csv),
        "save_ply": bool(output.save_ply),
        "save_pcd": bool(output.save_pcd),
    }


def _scan_config_document(config: ScanConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "mode": config.mode,
        "trajectory": {
            "start_angle_deg": config.trajectory.start_angle_deg,
            "end_angle_deg": config.trajectory.end_angle_deg,
            "step_angle_deg": config.trajectory.step_angle_deg,
        },
        "kinematics": {
            "axis_point_scan_mm": config.kinematics.axis_point_scan_mm.tolist(),
            "axis_direction_scan": config.kinematics.axis_direction_scan.tolist(),
            "zero_offset_deg": config.kinematics.zero_offset_deg,
            "T_scan_from_camera_zero": config.kinematics.T_scan_from_camera_zero.tolist(),
        },
        "output": {
            "directory": str(config.output.directory),
            "save_profiles_csv": config.output.save_profiles_csv,
            "save_ply": config.output.save_ply,
            "save_pcd": config.output.save_pcd,
        },
    }


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")


def _write_poses(
    path: Path,
    scan_result: ScanResult,
    profile_filenames: tuple[str, ...],
    result_metadata: Mapping[str, object] | None = None,
) -> None:
    stats_by_index = _frame_stats_by_index(result_metadata)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "frame_index",
                "filename",
                "angle_command_deg",
                "angle_measured_deg",
                "valid_laser_points",
                "points_camera_count",
                "points_scan_count",
                "valid_point_count",
                "warning",
            )
        )
        for profile, filename in zip(scan_result.profiles, profile_filenames):
            stat = stats_by_index.get(profile.frame_index, {})
            valid_laser_points = int(
                stat.get("valid_laser_points", len(profile.pixels_uv))
            )
            points_camera_count = int(
                stat.get("points_camera_count", len(profile.points_camera))
            )
            points_scan_count = int(
                stat.get("points_scan_count", len(profile.points_scan))
            )
            warning = stat.get("warning", "")
            writer.writerow(
                (
                    profile.frame_index,
                    filename,
                    profile.angle_deg,
                    profile.angle_deg,
                    valid_laser_points,
                    points_camera_count,
                    points_scan_count,
                    points_scan_count,
                    warning if warning is not None else "",
                )
            )


def _frame_stats_by_index(
    result_metadata: Mapping[str, object] | None,
) -> dict[int, Mapping[str, object]]:
    if result_metadata is None:
        return {}
    raw_stats = result_metadata.get("frame_stats")
    if not isinstance(raw_stats, Sequence) or isinstance(raw_stats, (str, bytes)):
        return {}
    stats: dict[int, Mapping[str, object]] = {}
    for raw in raw_stats:
        if not isinstance(raw, Mapping):
            continue
        try:
            frame_index = int(raw["frame_index"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        stats[frame_index] = raw
    return stats


def _write_profile_csv(path: Path, profile: Any) -> None:
    rows = np.column_stack(
        (profile.pixels_uv, profile.points_camera, profile.points_scan)
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "u_px",
                "v_px",
                "Xc_mm",
                "Yc_mm",
                "Zc_mm",
                "Xs_mm",
                "Ys_mm",
                "Zs_mm",
            )
        )
        writer.writerows(tuple(f"{value:.6f}" for value in row) for row in rows)


def _write_scan_ply(path: Path, points_scan: np.ndarray) -> None:
    points = _validated_cloud(points_scan)
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write("ply\n")
        stream.write("format ascii 1.0\n")
        stream.write("comment coordinate_system scan\n")
        stream.write("comment units millimeter\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property double x\n")
        stream.write("property double y\n")
        stream.write("property double z\n")
        stream.write("end_header\n")
        for point in points:
            stream.write(" ".join(f"{value:.9f}" for value in point) + "\n")


def _write_scan_pcd(path: Path, points_scan: np.ndarray) -> None:
    points = _validated_cloud(points_scan)
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write("# .PCD v0.7 - Point Cloud Data file format\n")
        stream.write("# coordinate_system scan\n")
        stream.write("# units millimeter\n")
        stream.write("VERSION 0.7\n")
        stream.write("FIELDS x y z\n")
        stream.write("SIZE 4 4 4\n")
        stream.write("TYPE F F F\n")
        stream.write("COUNT 1 1 1\n")
        stream.write(f"WIDTH {len(points)}\n")
        stream.write("HEIGHT 1\n")
        stream.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        stream.write(f"POINTS {len(points)}\n")
        stream.write("DATA ascii\n")
        for point in points:
            stream.write(" ".join(f"{value:.9f}" for value in point) + "\n")


def _validated_cloud(points_scan: np.ndarray) -> np.ndarray:
    points = np.asarray(points_scan, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ScanSessionError("scan 点云必须是形状为 (N, 3) 的数组")
    if not np.isfinite(points).all():
        raise ScanSessionError("scan 点云包含 NaN 或无穷值")
    return points


def _result_payload(
    scan_result: ScanResult,
    *,
    config: ScanConfig,
    mode: str,
    kinematic_demo_only: bool,
    output_files: Mapping[str, object],
    result_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if scan_result.profiles:
        angle_start = scan_result.profiles[0].angle_deg
        angle_end = scan_result.profiles[-1].angle_deg
    else:
        angle_start = config.trajectory.start_angle_deg
        angle_end = config.trajectory.end_angle_deg
    files: list[str] = []
    for value in output_files.values():
        if isinstance(value, str):
            files.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            files.extend(str(item) for item in value)
    payload: dict[str, object] = {
        "mode": mode,
        "frame_count": len(scan_result.profiles),
        "total_point_count": len(scan_result.points_scan),
        "angle_start_deg": angle_start,
        "angle_end_deg": angle_end,
        "coordinate_system": "scan",
        "units": "mm",
        "kinematic_demo_only": kinematic_demo_only,
        "output_files": dict(output_files),
        "files": files,
    }
    if result_metadata is not None:
        frame_stats = result_metadata.get("frame_stats")
        warnings = result_metadata.get("warnings")
        if isinstance(frame_stats, Sequence) and not isinstance(
            frame_stats, (str, bytes)
        ):
            payload["frame_stats"] = list(frame_stats)
        if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
            payload["warnings"] = list(warnings)
    payload.setdefault("frame_stats", [])
    payload.setdefault("warnings", [])
    return payload


def _commit_session_directory(temporary_directory: Path, output_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d")
    for index in range(1, 1000000):
        candidate = output_root / f"scan_{stamp}_{index:03d}"
        if candidate.exists():
            continue
        try:
            temporary_directory.rename(candidate)
            return candidate
        except FileExistsError:
            continue
    raise ScanSessionError("无法分配新的扫描 Session 目录")


__all__ = [
    "ScanSession",
    "ScanSessionError",
    "ScanSessionWriter",
    "write_scan_session",
]
