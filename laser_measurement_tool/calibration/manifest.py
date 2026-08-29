"""Runtime calibration package integrity and provenance checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config_loader import CalibrationConfigError, load_calibration_files


class CalibrationManifestError(CalibrationConfigError):
    """The runtime package manifest is missing, unsafe, or inconsistent."""


@dataclass(frozen=True, slots=True)
class CalibrationPackage:
    manifest_path: Path
    package_id: str
    camera_model: str
    image_width: int
    image_height: int
    algorithm: str
    manifest_sha256: str
    calibration: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Hash calibration text consistently across LF and CRLF checkouts."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_calibration_package(manifest_path: str | Path) -> CalibrationPackage:
    """Validate a self-contained manifest and load its calibration arrays."""
    path = Path(manifest_path).resolve()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CalibrationManifestError(f"无法读取标定清单 {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise CalibrationManifestError("标定清单 schema_version 必须为 1")

    package_id = _required_text(document, "package_id")
    camera = document.get("camera")
    if not isinstance(camera, dict):
        raise CalibrationManifestError("标定清单缺少 camera")
    camera_model = _required_text(camera, "model")
    image_width = _positive_int(camera.get("image_width"), "camera.image_width")
    image_height = _positive_int(camera.get("image_height"), "camera.image_height")

    extractor = document.get("extractor")
    if not isinstance(extractor, dict):
        raise CalibrationManifestError("标定清单缺少 extractor")
    algorithm = _required_text(extractor, "algorithm")
    if algorithm not in {"steger", "shared_steger"}:
        raise CalibrationManifestError(
            f"生产标定包必须使用实时 Steger（steger/shared_steger），实际为 {algorithm!r}"
        )

    files = document.get("files")
    required = ("intrinsics", "laser_plane", "extrinsics")
    if (
        not isinstance(files, dict)
        or any(name not in files for name in required)
        or "ground_u_compensation" not in files
    ):
        raise CalibrationManifestError(
            "标定清单 files 必须包含 intrinsics、laser_plane、extrinsics、"
            "ground_u_compensation；补偿项可显式设为 null 用于无补偿试跑"
        )
    resolved: dict[str, Path] = {}
    for name in required:
        entry = files[name]
        if not isinstance(entry, dict):
            raise CalibrationManifestError(f"files.{name} 必须是映射")
        relative = Path(_required_text(entry, "path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise CalibrationManifestError(f"files.{name}.path 必须是包内相对路径")
        file_path = (path.parent / relative).resolve()
        try:
            file_path.relative_to(path.parent)
        except ValueError as error:
            raise CalibrationManifestError(f"files.{name}.path 越出标定包") from error
        expected_hash = _required_text(entry, "sha256").lower()
        if len(expected_hash) != 64:
            raise CalibrationManifestError(f"files.{name}.sha256 格式错误")
        if not file_path.is_file():
            raise CalibrationManifestError(f"标定文件不存在: {file_path}")
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            raise CalibrationManifestError(
                f"标定文件哈希不匹配: {file_path.name}\n"
                f"期望 {expected_hash}\n实际 {actual_hash}"
            )
        resolved[name] = file_path

    laser_ray_entry = files.get("laser_ray_correction")
    if laser_ray_entry is None:
        laser_ray_path: Path | None = None
    else:
        if not isinstance(laser_ray_entry, dict):
            raise CalibrationManifestError(
                "files.laser_ray_correction 必须是映射或 null"
            )
        relative = Path(_required_text(laser_ray_entry, "path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise CalibrationManifestError(
                "files.laser_ray_correction.path 必须是包内相对路径"
            )
        laser_ray_path = (path.parent / relative).resolve()
        try:
            laser_ray_path.relative_to(path.parent)
        except ValueError as error:
            raise CalibrationManifestError(
                "files.laser_ray_correction.path 越出标定包"
            ) from error
        expected_hash = _required_text(laser_ray_entry, "sha256").lower()
        if len(expected_hash) != 64:
            raise CalibrationManifestError(
                "files.laser_ray_correction.sha256 格式错误"
            )
        if not laser_ray_path.is_file():
            raise CalibrationManifestError(f"标定文件不存在: {laser_ray_path}")
        actual_hash = sha256_file(laser_ray_path)
        if actual_hash != expected_hash:
            raise CalibrationManifestError(
                f"标定文件哈希不匹配: {laser_ray_path.name}\n"
                f"期望 {expected_hash}\n实际 {actual_hash}"
            )

    # 生产包通常提供真实补偿表；显式 null 允许在几何标定完成后先做
    # smoke test，而不需要伪造一张“已验收”的补偿 LUT。
    ground_entry = files["ground_u_compensation"]
    if ground_entry is None:
        ground_u_path: Path | None = None
    else:
        if not isinstance(ground_entry, dict):
            raise CalibrationManifestError(
                "files.ground_u_compensation 必须是映射或 null"
            )
        relative = Path(_required_text(ground_entry, "path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise CalibrationManifestError(
                "files.ground_u_compensation.path 必须是包内相对路径"
            )
        file_path = (path.parent / relative).resolve()
        try:
            file_path.relative_to(path.parent)
        except ValueError as error:
            raise CalibrationManifestError(
                "files.ground_u_compensation.path 越出标定包"
            ) from error
        expected_hash = _required_text(ground_entry, "sha256").lower()
        if len(expected_hash) != 64:
            raise CalibrationManifestError(
                "files.ground_u_compensation.sha256 格式错误"
            )
        if not file_path.is_file():
            raise CalibrationManifestError(f"标定文件不存在: {file_path}")
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            raise CalibrationManifestError(
                f"标定文件哈希不匹配: {file_path.name}\n"
                f"期望 {expected_hash}\n实际 {actual_hash}"
            )
        ground_u_path = file_path

    calibration = load_calibration_files(
        resolved["intrinsics"],
        resolved["laser_plane"],
        resolved["extrinsics"],
        ground_u_path,
        laser_ray_correction=laser_ray_path,
    )
    return CalibrationPackage(
        manifest_path=path,
        package_id=package_id,
        camera_model=camera_model,
        image_width=image_width,
        image_height=image_height,
        algorithm=algorithm,
        manifest_sha256=sha256_file(path),
        calibration=calibration,
    )


def _required_text(document: dict[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value.strip():
        raise CalibrationManifestError(f"{name} 必须是非空字符串")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationManifestError(f"{name} 必须是正整数")
    return value
