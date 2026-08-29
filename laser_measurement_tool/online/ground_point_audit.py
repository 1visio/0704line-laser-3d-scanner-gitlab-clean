"""Read-only point-level export for the online Session ground sanity check.

The exporter consumes the arrays and mask indices already produced by the
formal GUI path.  It never calls Steger, reconstruction, PnP, fitting, or a
ground/reference correction.  The raw ground view is the only view used for
the residual replay; a corrected metric view is recorded separately when it
is active.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from calibration.manifest import sha256_file


AUDIT_SCHEMA_VERSION = 1
METRIC_REPLAY_RTOL = 1.0e-8
METRIC_REPLAY_ATOL_MM = 1.0e-9

CSV_COLUMNS = (
    "session_id",
    "session_generation",
    "ground_extrinsic_generation",
    "frame_id",
    "camera_frame_number",
    "frame_host_monotonic_ns",
    "point_id",
    "source_point_index",
    "u_px",
    "v_px",
    "Xc_mm",
    "Yc_mm",
    "Zc_mm",
    "Xg_mm",
    "Yg_mm",
    "Zg_mm",
    "Xg_metric_mm",
    "Yg_metric_mm",
    "Zg_metric_mm",
    "board_mask_selected",
    "ground_plane_a",
    "ground_plane_b",
    "ground_plane_c",
    "ground_plane_d_mm",
    "Z_plane_mm",
    "r_G_mm",
    "ground_extrinsic_source",
    "calibration_package_id",
    "calibration_manifest_sha256",
    "algorithm_config_sha256",
    "frozen_c0_sha256",
    "frozen_c1_sha256",
)


class GroundPointAuditValidationError(ValueError):
    """The point-level linkage or replay contract is not trustworthy."""


@dataclass(frozen=True, slots=True)
class GroundPointAuditExport:
    """Paths and validated manifest returned after a successful export."""

    audit_dir: Path
    csv_path: Path
    manifest_path: Path
    report_path: Path
    manifest: dict[str, Any]


def build_frozen_chain_provenance(
    manifest_path: str | Path,
    *,
    calibration_package_id: str,
    calibration_manifest_sha256: str,
    algorithm_config_sha256: str,
) -> dict[str, Any]:
    """Return verified Frozen C0/C1 file provenance from the loaded package.

    The runtime calibration loader has already validated the package.  This
    helper only reads that same manifest and verifies the two declared files
    again for the audit record; it does not evaluate or refit either model.
    """
    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise GroundPointAuditValidationError(
            f"calibration manifest 不存在: {manifest}"
        )
    manifest_sha = _require_sha256(
        calibration_manifest_sha256, "calibration_manifest_sha256"
    )
    actual_manifest_sha = sha256_file(manifest)
    if actual_manifest_sha != manifest_sha:
        raise GroundPointAuditValidationError(
            "calibration manifest hash 与 FrameResult provenance 不一致"
        )
    package_id = str(calibration_package_id).strip()
    if not package_id:
        raise GroundPointAuditValidationError("calibration_package_id 不能为空")
    algorithm_sha = _require_sha256(
        algorithm_config_sha256, "algorithm_config_sha256"
    )
    try:
        document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise GroundPointAuditValidationError(
            f"无法读取 calibration manifest: {manifest}: {error}"
        ) from error
    if not isinstance(document, Mapping):
        raise GroundPointAuditValidationError("calibration manifest 根节点必须是映射")
    files = document.get("files")
    if not isinstance(files, Mapping):
        raise GroundPointAuditValidationError("calibration manifest 缺少 files")

    c0 = _manifest_file_info(files, "laser_plane", manifest)
    c1 = _manifest_file_info(files, "laser_ray_correction", manifest)
    if c0 is None:
        raise GroundPointAuditValidationError(
            "calibration manifest 缺少 Frozen C0 laser_plane"
        )
    if c1 is None:
        raise GroundPointAuditValidationError(
            "calibration manifest 缺少 Frozen C1 laser_ray_correction"
        )
    c1_document: Mapping[str, Any] | None = None
    try:
        loaded_c1 = json.loads(Path(c1["path"]).read_text(encoding="utf-8"))
        if isinstance(loaded_c1, Mapping):
            c1_document = loaded_c1
    except (OSError, UnicodeError, json.JSONDecodeError):
        # The manifest/file hash was already verified; parameter metadata is
        # optional and must not be confused with the file identity itself.
        c1_document = None
    if c1_document is not None:
        parameter_sha = c1_document.get("parameter_sha256")
        if isinstance(parameter_sha, str) and parameter_sha.strip():
            c1["parameter_sha256"] = parameter_sha.strip().lower()

    return {
        "calibration_package_id": package_id,
        "calibration_manifest": {
            "path": str(manifest),
            "sha256": manifest_sha,
            "verified_by_runtime_loader": True,
        },
        "algorithm_config_sha256": algorithm_sha,
        "frozen_c0": c0,
        "frozen_c1": c1,
    }


def build_session_ground_plane_provenance(
    session_ground_result: Any,
) -> dict[str, Any]:
    """Describe the existing Session PnP plane without applying it again."""
    rotation = np.asarray(
        getattr(session_ground_result, "R", None), dtype=np.float64
    )
    translation = np.asarray(
        getattr(session_ground_result, "t", None), dtype=np.float64
    ).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise GroundPointAuditValidationError(
            "Session PnP result 缺少有限的 camera-to-ground R/t"
        )
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise GroundPointAuditValidationError("Session PnP R/t 必须是 finite")

    normal_camera = np.ascontiguousarray(rotation[2], dtype=np.float64)
    d_camera = float(translation[2])
    origin = getattr(session_ground_result, "ground_origin_in_camera", None)
    if origin is None:
        origin_array = -rotation.T @ translation
    else:
        origin_array = np.asarray(origin, dtype=np.float64).reshape(-1)
    if origin_array.shape != (3,) or not np.isfinite(origin_array).all():
        raise GroundPointAuditValidationError(
            "Session PnP ground origin 必须是 finite 三维坐标"
        )

    ground_plane = {
        "coordinate_system": "ground",
        "equation": "0*Xg + 0*Yg + 1*Zg + 0 = 0",
        "a": 0.0,
        "b": 0.0,
        "c": 1.0,
        "d_mm": 0.0,
        "z_plane_mm": 0.0,
        "source": "Session PnP ground frame (Zg=0)",
    }
    pnp = {
        "status": getattr(session_ground_result, "status", None),
        "reprojection_rmse_px": getattr(
            session_ground_result, "reprojection_rmse_px", None
        ),
        "rvec_board_to_camera": _json_safe(
            getattr(session_ground_result, "rvec", None)
        ),
        "tvec_board_to_camera_mm": _json_safe(
            getattr(session_ground_result, "tvec", None)
        ),
        "R_camera_to_ground": rotation.tolist(),
        "t_camera_to_ground_mm": translation.tolist(),
        "T_ground_from_camera": _json_safe(
            getattr(session_ground_result, "T_ground_from_camera", None)
        ),
        "ground_normal_in_camera": _json_safe(
            getattr(session_ground_result, "ground_normal_in_camera", None)
        ),
        "ground_origin_in_camera": origin_array.tolist(),
        "camera_plane": {
            "coordinate_system": "camera",
            "equation": "a*Xc + b*Yc + c*Zc + d = 0",
            "a": float(normal_camera[0]),
            "b": float(normal_camera[1]),
            "c": float(normal_camera[2]),
            "d_mm": d_camera,
            "normal": normal_camera.tolist(),
        },
    }
    return {"ground_plane": ground_plane, "session_pnp": pnp}


def export_ground_point_audit(
    output_dir: str | Path,
    *,
    session_id: str,
    session_generation: int,
    ground_extrinsic_generation: int,
    frame_id: str,
    camera_frame_number: int,
    frame_host_monotonic_ns: int,
    frame_offset: tuple[int, int],
    ground_extrinsic_source: str,
    calibration_package_id: str,
    calibration_manifest_sha256: str,
    algorithm_config_sha256: str,
    pixels_uv: np.ndarray,
    points_camera: np.ndarray,
    points_ground_raw: np.ndarray,
    points_ground_metric: np.ndarray | None,
    selected_indices: np.ndarray,
    selected_mask: np.ndarray,
    sanity_points: np.ndarray,
    sanity_result: Any,
    mask_metadata: Mapping[str, Any],
    ground_plane: Mapping[str, Any],
    session_pnp: Mapping[str, Any],
    frozen_provenance: Mapping[str, Any],
) -> GroundPointAuditExport:
    """Validate and write one immutable point-level audit snapshot."""
    session_text = str(session_id).strip()
    frame_text = str(frame_id).strip()
    source_text = str(ground_extrinsic_source).strip().lower()
    if not session_text or not frame_text:
        raise GroundPointAuditValidationError("session_id/frame_id 不能为空")
    if source_text != "session":
        raise GroundPointAuditValidationError(
            "point-level ground audit 只接受 Session ground extrinsic"
        )
    session_gen = _require_nonnegative_int(session_generation, "session_generation")
    extrinsic_gen = _require_nonnegative_int(
        ground_extrinsic_generation, "ground_extrinsic_generation"
    )
    if session_gen != extrinsic_gen:
        raise GroundPointAuditValidationError(
            "session_generation 与 ground_extrinsic_generation 不一致"
        )
    camera_frame = _require_nonnegative_int(camera_frame_number, "camera_frame_number")
    host_ns = _require_nonnegative_int(
        frame_host_monotonic_ns, "frame_host_monotonic_ns"
    )
    try:
        offset = tuple(int(value) for value in frame_offset)
    except (TypeError, ValueError) as error:
        raise GroundPointAuditValidationError("frame_offset 必须是两个整数") from error
    if len(offset) != 2:
        raise GroundPointAuditValidationError("frame_offset 必须是两个整数")

    uv = _points_array(pixels_uv, 2, "pixels_uv")
    camera = _points_array(points_camera, 3, "points_camera")
    raw = _points_array(points_ground_raw, 3, "points_ground_raw")
    if len(uv) != len(camera) or len(uv) != len(raw):
        raise GroundPointAuditValidationError(
            "pixels_uv/points_camera/points_ground_raw 长度不一致"
        )
    metric = None if points_ground_metric is None else _points_array(
        points_ground_metric, 3, "points_ground_metric"
    )
    if metric is not None and len(metric) != len(raw):
        raise GroundPointAuditValidationError(
            "points_ground_metric 必须与 raw ground 点逐行对齐"
        )

    indices = np.asarray(selected_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise GroundPointAuditValidationError(
            "selected_indices 必须是保留原顺序的整数索引数组"
        )
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise GroundPointAuditValidationError("sanity 输入点数为 0，不能生成 point audit")
    if np.any(indices < 0) or np.any(indices >= len(raw)):
        raise GroundPointAuditValidationError("selected_indices 越出原始点数组")
    if len(np.unique(indices)) != len(indices):
        raise GroundPointAuditValidationError("selected_indices 含重复点，无法建立唯一回链")

    mask = np.asarray(selected_mask)
    if mask.ndim != 1 or mask.shape != (len(raw),) or mask.dtype != np.bool_:
        raise GroundPointAuditValidationError(
            "selected_mask 必须是与原始点数组同长度的 bool 数组"
        )
    if not bool(np.all(mask[indices])):
        raise GroundPointAuditValidationError(
            "sanity 输入索引中存在未被 board mask 选中的点"
        )
    if isinstance(mask_metadata, Mapping):
        declared_input_count = mask_metadata.get("input_point_count")
        if declared_input_count is not None and int(declared_input_count) != len(raw):
            raise GroundPointAuditValidationError(
                "board-mask input_point_count 与 raw 点数组不一致"
            )
        declared_selected_count = mask_metadata.get("selected_point_count")
        if (
            declared_selected_count is not None
            and int(declared_selected_count) != len(indices)
        ):
            raise GroundPointAuditValidationError(
                "board-mask selected_point_count 与 sanity 输入点数不一致"
            )

    selected_raw = np.ascontiguousarray(raw[indices], dtype=np.float64)
    sanity_array = _points_array(sanity_points, 3, "sanity_points")
    if len(indices) != len(sanity_array):
        raise GroundPointAuditValidationError(
            "point_count 与 sanity 输入点数不一致"
        )
    if not np.array_equal(selected_raw, sanity_array):
        raise GroundPointAuditValidationError(
            "sanity 输入点不是 points_ground_raw 的同索引切片"
        )

    selected_uv = np.ascontiguousarray(uv[indices], dtype=np.float64)
    selected_camera = np.ascontiguousarray(camera[indices], dtype=np.float64)
    selected_metric = None if metric is None else np.ascontiguousarray(
        metric[indices], dtype=np.float64
    )
    for name, values in (
        ("u/v", selected_uv),
        ("camera XYZ", selected_camera),
        ("raw ground XYZ", selected_raw),
    ):
        if not np.isfinite(values).all():
            raise GroundPointAuditValidationError(f"{name} 含非 finite 数值")
    if selected_metric is not None and not np.isfinite(selected_metric).all():
        raise GroundPointAuditValidationError("metric ground XYZ 含非 finite 数值")

    plane = _normalise_ground_plane(ground_plane)
    coefficients = np.asarray(
        [plane["a"], plane["b"], plane["c"], plane["d_mm"]], dtype=np.float64
    )
    z_plane = -(
        coefficients[0] * selected_raw[:, 0]
        + coefficients[1] * selected_raw[:, 1]
        + coefficients[3]
    ) / coefficients[2]
    residual = selected_raw[:, 2] - z_plane
    if not np.isfinite(z_plane).all() or not np.isfinite(residual).all():
        raise GroundPointAuditValidationError("Z_plane/r_G 含非 finite 数值")

    sanity_input_count = _sanity_int(sanity_result, "input_point_count")
    sanity_valid_count = _sanity_int(sanity_result, "valid_point_count")
    if sanity_input_count != len(indices):
        raise GroundPointAuditValidationError(
            "point_count 与当前 sanity aggregate 的输入点数不一致"
        )
    finite_count = int(np.isfinite(sanity_array).all(axis=1).sum())
    if finite_count != sanity_valid_count:
        raise GroundPointAuditValidationError(
            "导出点 finite count 与 sanity valid_point_count 不一致"
        )
    replay = _replay_metrics(residual)
    expected = {
        "bias_zg_mm": _sanity_float(sanity_result, "bias_zg_mm"),
        "rmse_zg_mm": _sanity_float(sanity_result, "rmse_zg_mm"),
        "p95_abs_zg_mm": _sanity_float(sanity_result, "p95_abs_zg_mm"),
    }
    replay_deltas: dict[str, float] = {}
    for name, actual in replay.items():
        target = expected[name]
        if target is None or not np.isclose(
            actual,
            target,
            rtol=METRIC_REPLAY_RTOL,
            atol=METRIC_REPLAY_ATOL_MM,
        ):
            raise GroundPointAuditValidationError(
                f"SANITY_METRIC_REPLAY failed for {name}: {actual} != {target}"
            )
        replay_deltas[name] = float(actual - target)

    manifest_sha = _require_sha256(
        calibration_manifest_sha256, "calibration_manifest_sha256"
    )
    algorithm_sha = _require_sha256(
        algorithm_config_sha256, "algorithm_config_sha256"
    )
    package_id = str(calibration_package_id).strip()
    if not package_id:
        raise GroundPointAuditValidationError("calibration_package_id 不能为空")
    frozen = _validate_frozen_provenance(frozen_provenance)
    mask_metadata_json = _json_safe(dict(mask_metadata))
    pnp_json = _json_safe(dict(session_pnp))
    timestamp = datetime.now(timezone.utc).isoformat()
    csv_path = Path(output_dir) / "ground_residual_points.csv"
    manifest_path = Path(output_dir) / "ground_point_export_manifest.json"
    report_path = Path(output_dir) / "ground_point_audit_export.md"

    rows: list[dict[str, Any]] = []
    c0_sha = str(frozen["frozen_c0"]["sha256"])
    c1_sha = str(frozen["frozen_c1"]["sha256"])
    for row_index, source_index in enumerate(indices.tolist()):
        raw_xyz = selected_raw[row_index]
        camera_xyz = selected_camera[row_index]
        metric_xyz = None if selected_metric is None else selected_metric[row_index]
        rows.append(
            {
                "session_id": session_text,
                "session_generation": session_gen,
                "ground_extrinsic_generation": extrinsic_gen,
                "frame_id": frame_text,
                "camera_frame_number": camera_frame,
                "frame_host_monotonic_ns": host_ns,
                "point_id": f"{frame_text}:point_{source_index:06d}",
                "source_point_index": source_index,
                "u_px": float(selected_uv[row_index, 0]),
                "v_px": float(selected_uv[row_index, 1]),
                "Xc_mm": float(camera_xyz[0]),
                "Yc_mm": float(camera_xyz[1]),
                "Zc_mm": float(camera_xyz[2]),
                "Xg_mm": float(raw_xyz[0]),
                "Yg_mm": float(raw_xyz[1]),
                "Zg_mm": float(raw_xyz[2]),
                "Xg_metric_mm": None if metric_xyz is None else float(metric_xyz[0]),
                "Yg_metric_mm": None if metric_xyz is None else float(metric_xyz[1]),
                "Zg_metric_mm": None if metric_xyz is None else float(metric_xyz[2]),
                "board_mask_selected": True,
                "ground_plane_a": plane["a"],
                "ground_plane_b": plane["b"],
                "ground_plane_c": plane["c"],
                "ground_plane_d_mm": plane["d_mm"],
                "Z_plane_mm": float(z_plane[row_index]),
                "r_G_mm": float(residual[row_index]),
                "ground_extrinsic_source": source_text,
                "calibration_package_id": package_id,
                "calibration_manifest_sha256": manifest_sha,
                "algorithm_config_sha256": algorithm_sha,
                "frozen_c0_sha256": c0_sha,
                "frozen_c1_sha256": c1_sha,
            }
        )

    point_count = len(rows)
    metric_present = selected_metric is not None
    validation = {
        "GROUND_POINT_EXPORT": "PASS",
        "UV_POINT_IDENTITY": "PASS",
        "RAW_GROUND_PRESERVED": "YES",
        "FRAME_PROVENANCE": "PASS",
        "GENERATION_PROVENANCE": "PASS",
        "SANITY_METRIC_REPLAY": "PASS",
        "READY_FOR_SPATIAL_RESIDUAL_AUDIT": "YES",
    }
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "export_kind": "ground_spatial_audit",
        "export_status": "PASS",
        "created_at_utc": timestamp,
        "session": {
            "session_id": session_text,
            "session_generation": session_gen,
            "ground_extrinsic_generation": extrinsic_gen,
        },
        "frame": {
            "frame_id": frame_text,
            "camera_frame_number": camera_frame,
            "host_monotonic_ns": host_ns,
            "offset_x": offset[0],
            "offset_y": offset[1],
            "ground_extrinsic_source": source_text,
        },
        "point_count": point_count,
        "candidate_point_count": int(len(raw)),
        "sanity_input_point_count": sanity_input_count,
        "valid_point_count": sanity_valid_count,
        "board_mask": mask_metadata_json,
        "coordinate_views": {
            "points_ground_raw": {
                "present": True,
                "definition": "C0+C1+Session PnP ground extrinsic, before Session Ground correction",
                "used_for_sanity": True,
                "used_for_residual": True,
            },
            "points_ground_metric": {
                "present": metric_present,
                "definition": "FrameResult.points_ground; optional active Session Ground reference view",
                "used_for_sanity": False,
                "used_for_residual": False,
            },
            "residual_view": "points_ground_raw",
        },
        "residual_definition": {
            "name": "r_G",
            "formula": "r_G = Zg - Z_plane",
            "units": "mm",
            "plane_coordinate_system": "ground",
            "plane": plane,
        },
        "ground_plane": plane,
        "session_pnp": pnp_json,
        "provenance": {
            "formal_chain": [
                "Steger",
                "Frozen C0",
                "Frozen C1",
                "Session ground extrinsic",
            ],
            "calibration_package_id": package_id,
            "calibration_manifest_sha256": manifest_sha,
            "algorithm_config_sha256": algorithm_sha,
            "frozen_c0": frozen["frozen_c0"],
            "frozen_c1": frozen["frozen_c1"],
        },
        "sanity_replay": {
            "status": "PASS",
            "input_point_count": sanity_input_count,
            "valid_point_count": sanity_valid_count,
            "sanity_status": _sanity_value(sanity_result, "status"),
            "replayed": replay,
            "sanity_aggregate": expected,
            "absolute_deltas": replay_deltas,
            "rtol": METRIC_REPLAY_RTOL,
            "atol_mm": METRIC_REPLAY_ATOL_MM,
        },
        "validation": validation,
        "files": {
            "ground_residual_points_csv": csv_path.name,
            "ground_point_export_manifest_json": manifest_path.name,
            "ground_point_audit_report_md": report_path.name,
        },
    }

    audit_dir = csv_path.parent
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_exclusive(csv_path, rows)
    manifest["files"]["ground_residual_points_sha256"] = _sha256_bytes(
        csv_path.read_bytes()
    )
    _write_json_exclusive(manifest_path, manifest)
    _write_report_exclusive(report_path, manifest)
    return GroundPointAuditExport(
        audit_dir=audit_dir,
        csv_path=csv_path,
        manifest_path=manifest_path,
        report_path=report_path,
        manifest=manifest,
    )


def _points_array(value: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise GroundPointAuditValidationError(
            f"{name} 必须是形状为 (N, {width}) 的数组"
        )
    return np.ascontiguousarray(array)


def _normalise_ground_plane(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GroundPointAuditValidationError("ground_plane 必须是映射")
    try:
        coefficients = [
            float(value["a"]),
            float(value["b"]),
            float(value["c"]),
            float(value["d_mm"]),
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise GroundPointAuditValidationError(
            "ground_plane 必须提供 a/b/c/d_mm"
        ) from error
    if not np.isfinite(coefficients).all() or abs(coefficients[2]) <= np.finfo(float).eps:
        raise GroundPointAuditValidationError("ground_plane 参数必须 finite 且 c 非零")
    result = _json_safe(dict(value))
    z_plane_value = float(
        value.get("z_plane_mm", -coefficients[3] / coefficients[2])
    )
    if not np.isfinite(z_plane_value):
        raise GroundPointAuditValidationError("ground_plane.z_plane_mm 必须 finite")
    result.update(
        {
            "a": coefficients[0],
            "b": coefficients[1],
            "c": coefficients[2],
            "d_mm": coefficients[3],
            "z_plane_mm": z_plane_value,
        }
    )
    return result


def _replay_metrics(residual: np.ndarray) -> dict[str, float]:
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise GroundPointAuditValidationError(
            "SANITY_METRIC_REPLAY 需要非空 finite r_G"
        )
    return {
        "bias_zg_mm": float(np.mean(values)),
        "rmse_zg_mm": float(np.sqrt(np.mean(np.square(values)))),
        "p95_abs_zg_mm": float(np.percentile(np.abs(values), 95.0)),
    }


def _sanity_value(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(name)
    return getattr(result, name, None)


def _sanity_int(result: Any, name: str) -> int:
    value = _sanity_value(result, name)
    if isinstance(value, bool) or value is None or int(value) != value:
        raise GroundPointAuditValidationError(f"sanity.{name} 必须是整数")
    return int(value)


def _sanity_float(result: Any, name: str) -> float | None:
    value = _sanity_value(result, name)
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        raise GroundPointAuditValidationError(f"sanity.{name} 必须是 finite")
    return numeric


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or value is None or int(value) != value or int(value) < 0:
        raise GroundPointAuditValidationError(f"{name} 必须是非负整数")
    return int(value)


def _require_sha256(value: Any, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64:
        raise GroundPointAuditValidationError(f"{name} 不是 SHA-256")
    try:
        int(text, 16)
    except ValueError as error:
        raise GroundPointAuditValidationError(f"{name} 不是 SHA-256") from error
    return text


def _manifest_file_info(
    files: Mapping[str, Any], name: str, manifest_path: Path
) -> dict[str, Any] | None:
    entry = files.get(name)
    if entry is None:
        return None
    if not isinstance(entry, Mapping):
        raise GroundPointAuditValidationError(f"manifest.files.{name} 必须是映射")
    relative = Path(str(entry.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise GroundPointAuditValidationError(
            f"manifest.files.{name}.path 必须是包内相对路径"
        )
    file_path = (manifest_path.parent / relative).resolve()
    try:
        file_path.relative_to(manifest_path.parent)
    except ValueError as error:
        raise GroundPointAuditValidationError(
            f"manifest.files.{name}.path 越出标定包"
        ) from error
    expected = _require_sha256(entry.get("sha256"), f"manifest.files.{name}.sha256")
    if not file_path.is_file():
        raise GroundPointAuditValidationError(f"Frozen 文件不存在: {file_path}")
    actual = sha256_file(file_path)
    if actual != expected:
        raise GroundPointAuditValidationError(
            f"Frozen 文件 hash 不一致: {file_path}"
        )
    return {
        "manifest_key": name,
        "path": str(file_path),
        "sha256": expected,
        "verified": True,
    }


def _validate_frozen_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GroundPointAuditValidationError("frozen_provenance 必须是映射")
    result = _json_safe(dict(value))
    for name in ("frozen_c0", "frozen_c1"):
        entry = result.get(name)
        if not isinstance(entry, Mapping):
            raise GroundPointAuditValidationError(f"缺少 {name} provenance")
        _require_sha256(entry.get("sha256"), f"{name}.sha256")
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise GroundPointAuditValidationError("provenance 含非 finite 数值")
        return numeric
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not np.isfinite(value):
            raise GroundPointAuditValidationError("provenance 含非 finite 数值")
        return value
    return str(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_report_exclusive(path: Path, manifest: Mapping[str, Any]) -> None:
    validation = manifest["validation"]
    replay = manifest["sanity_replay"]
    points = manifest["point_count"]
    lines = [
        "# Ground point audit export",
        "",
        "本报告由正式 GUI Session Ground Sanity Check 在同一帧、同一批数组和同一 board mask 结果上生成。",
        "本导出只读，不重新运行 Steger、C0/C1、Session PnP、Ground 拟合或 compensation。",
        "",
        f"- raw residual view: `points_ground_raw`；point count: `{points}`。",
        f"- sanity status: `{replay.get('sanity_status')}`；metric replay: `{replay.get('status')}`。",
        "- `points_ground_metric`（如存在）仅作旁路记录，不参与本报告的 residual 或 sanity replay。",
        "",
        "## Final flags",
        "",
    ]
    for name in (
        "GROUND_POINT_EXPORT",
        "UV_POINT_IDENTITY",
        "RAW_GROUND_PRESERVED",
        "FRAME_PROVENANCE",
        "GENERATION_PROVENANCE",
        "SANITY_METRIC_REPLAY",
        "READY_FOR_SPATIAL_RESIDUAL_AUDIT",
    ):
        lines.append(f"{name} = {validation[name]}")
    lines.extend(
        [
            "",
            "## Metric definition",
            "",
            "`r_G = Zg - Z_plane`; the plane is the current Session PnP ground frame `Zg=0`.",
            "The CSV retains `source_point_index` and does not infer linkage from row order or array length.",
            "",
        ]
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


__all__ = [
    "CSV_COLUMNS",
    "GroundPointAuditExport",
    "GroundPointAuditValidationError",
    "build_frozen_chain_provenance",
    "build_session_ground_plane_provenance",
    "export_ground_point_audit",
]
