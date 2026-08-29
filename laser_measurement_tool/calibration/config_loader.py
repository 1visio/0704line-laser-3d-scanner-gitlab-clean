"""统一读取并校验测量工具使用的标定 YAML。"""

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from ..reconstruction.laser_ray_correction import (
        LaserRayCorrectionError,
        load_frozen_laser_ray_correction,
    )
except ImportError:  # Supports the existing top-level ``calibration`` imports.
    from reconstruction.laser_ray_correction import (
        LaserRayCorrectionError,
        load_frozen_laser_ray_correction,
    )


_CAMERA_FILE = "camera_intrinsics.yaml"
_CALIBRATION_RESULT_FILE = "calibration_result.yaml"
_LASER_MODEL_FILE = "circular_cone.yaml"
_LEGACY_LASER_MODEL_FILE = "laser_plane.yaml"
_EXTRINSICS_FILE = "camera_ground_extrinsics.yaml"
_GROUND_U_FILE = "ground_u_compensation.yaml"
_GROUND_U_CSV_FILE = "ground_u_compensation.csv"
_DISTORTION_LENGTHS = frozenset({4, 5, 8, 12, 14})
_UNIT_ALIASES = {
    "px": "px",
    "pixel": "px",
    "pixels": "px",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "1": "dimensionless",
    "dimensionless": "dimensionless",
    "unitless": "dimensionless",
}


class CalibrationConfigError(ValueError):
    """标定文件内容无效。"""


class CalibrationFileNotFoundError(FileNotFoundError):
    """必需标定文件不存在。"""


class CalibrationDimensionError(CalibrationConfigError):
    """标定参数维度错误。"""


class CalibrationUnitError(CalibrationConfigError):
    """标定参数单位错误。"""


def load_calibration(config_dir: str | Path) -> dict[str, Any]:
    """读取标定目录；优先使用 circular_cone.yaml，兼容旧 laser_plane.yaml。"""
    directory = Path(config_dir)
    if not directory.is_dir():
        raise CalibrationFileNotFoundError(f"标定目录不存在: {directory}")

    intrinsics_path = directory / _CAMERA_FILE
    calibration_result_path = directory / _CALIBRATION_RESULT_FILE
    if not intrinsics_path.is_file() and calibration_result_path.is_file():
        # 当前随工具发布的内参文件名来自标定脚本输出。
        intrinsics_path = calibration_result_path
    model_path = directory / _LASER_MODEL_FILE
    if not model_path.is_file():
        # 旧标定目录仍只提供 laser_plane.yaml。
        model_path = directory / _LEGACY_LASER_MODEL_FILE
    ground_u_path = directory / _GROUND_U_FILE
    if not ground_u_path.is_file():
        csv_path = directory / _GROUND_U_CSV_FILE
        if csv_path.is_file():
            ground_u_path = csv_path
    return load_calibration_files(
        intrinsics=intrinsics_path,
        laser_plane=model_path,
        extrinsics=directory / _EXTRINSICS_FILE,
        ground_u_compensation=ground_u_path,
        ground_u_optional=True,
    )


def load_calibration_files(
    intrinsics: str | Path,
    laser_plane: str | Path,
    extrinsics: str | Path,
    ground_u_compensation: str | Path | None = None,
    laser_ray_correction: str | Path | None = None,
    *,
    ground_u_optional: bool = False,
) -> dict[str, Any]:
    """按显式路径读取相机、激光表面模型、地面外参与可选 U 补偿。

    参数名 ``laser_plane`` 为旧接口兼容名；传入文件可包含旧式平面参数，
    或 ``model_type`` 为 ``global_plane``、``quadratic_graph``、
    ``circular_cone`` 的激光表面模型。

    返回字典固定包含 ``K``、``D``、``laser_model``、``R``、``t``、
    ``ground_u_compensation``、``laser_ray_correction``。全局平面额外保留
    ``plane_abcd``，兼容旧调用方；没有 C1 文件时对应值为 ``None``。
    """
    camera = _load_camera_intrinsics(Path(intrinsics))
    laser_model = _load_laser_model(Path(laser_plane))
    pose = _load_camera_ground_extrinsics(Path(extrinsics))

    ground_u: dict[str, Any] | None = None
    if ground_u_compensation is not None:
        ground_u_path = Path(ground_u_compensation)
        if not ground_u_path.exists() and not ground_u_optional:
            raise CalibrationFileNotFoundError(
                f"标定文件不存在: {ground_u_path}"
            )
        ground_u = _load_optional_ground_u(ground_u_path)

    frozen_c1 = None
    if laser_ray_correction is not None:
        correction_path = Path(laser_ray_correction)
        if not correction_path.is_file():
            raise CalibrationFileNotFoundError(
                f"标定文件不存在: {correction_path}"
            )
        try:
            frozen_c1 = load_frozen_laser_ray_correction(correction_path)
        except LaserRayCorrectionError as error:
            raise CalibrationConfigError(
                f"frozen laser ray correction 无效: {correction_path}: {error}"
            ) from error

    result: dict[str, Any] = {
        "K": camera["K"],
        "D": camera["D"],
        "laser_model": laser_model,
        "R": pose["R"],
        "t": pose["t"],
        "ground_u_compensation": ground_u,
        "laser_ray_correction": frozen_c1,
    }
    if laser_model["model_type"] == "global_plane":
        result["plane_abcd"] = np.ascontiguousarray(
            np.r_[laser_model["normal"], laser_model["d_mm"]],
            dtype=np.float64,
        )
    return result


def _load_camera_intrinsics(path: Path) -> dict[str, np.ndarray]:
    document = _load_required_yaml(path)
    _validate_units(document, path, ("units", "pixel_unit"), {"px"})

    matrix = _required_value(document, ("K", "camera_matrix"), path, "K")
    distortion = _required_value(document, ("D", "dist_coeffs"), path, "D")
    K = _matrix(matrix, (3, 3), path, "K")
    D = _vector(distortion, path, "D")
    if len(D) not in _DISTORTION_LENGTHS:
        allowed = ", ".join(str(length) for length in sorted(_DISTORTION_LENGTHS))
        raise CalibrationDimensionError(
            f"{path.name} 的 D 长度为 {len(D)}，应为 {allowed} 之一"
        )
    return {"K": K, "D": D}


def _load_laser_model(path: Path) -> dict[str, Any]:
    """读取并规范化三类激光表面模型；兼容旧平面 YAML。"""
    document = _load_required_yaml(path)
    model_type_raw = document.get("model_type")

    if model_type_raw is None:
        plane = _parse_legacy_plane_document(document, path)
        return {
            "model_type": "global_plane",
            "normal": plane[:3],
            "d_mm": float(plane[3]),
            "source_path": str(path.resolve()),
        }

    model_type = str(model_type_raw).strip().lower()
    if model_type == "global_plane":
        return _parse_global_plane_model(document, path)
    if model_type == "quadratic_graph":
        return _parse_quadratic_graph_model(document, path)
    if model_type == "circular_cone":
        return _parse_circular_cone_model(document, path)
    raise CalibrationConfigError(
        f"{path.name} 的 model_type={model_type_raw!r} 不受支持；"
        "应为 global_plane / quadratic_graph / circular_cone"
    )


def _parse_legacy_plane_document(
    document: Mapping[str, Any], path: Path
) -> np.ndarray:
    _validate_units(document, path, ("units", "coordinate_unit"), {"mm"})
    coordinate_system = document.get("coordinate_system")
    if coordinate_system is not None and str(coordinate_system).lower() != "camera":
        raise CalibrationConfigError(
            f"{path.name} 的 coordinate_system 必须为 camera"
        )

    if "plane_abcd" in document:
        raw_plane = document["plane_abcd"]
    elif isinstance(document.get("plane"), Mapping) or isinstance(
        document.get("coefficients"), Mapping
    ):
        plane = document.get("plane") or document["coefficients"]
        raw_plane = [
            _required_value(plane, (name,), path, f"plane.{name}")
            for name in ("a", "b", "c", "d")
        ]
    else:
        raise CalibrationConfigError(
            f"{path.name} 缺少 model_type，且没有 plane_abcd / plane / coefficients"
        )

    plane_abcd = _vector(raw_plane, path, "plane_abcd", expected_length=4)
    normal_norm = float(np.linalg.norm(plane_abcd[:3]))
    if normal_norm <= np.finfo(np.float64).eps:
        raise CalibrationConfigError(f"{path.name} 的平面法向量不能为零")
    # 保留旧格式的原始比例，兼容既有 plane_abcd 调用方；求交时会再归一化。
    return np.ascontiguousarray(plane_abcd)


def _parse_global_plane_model(
    document: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    _validate_units(document, path, ("units", "coordinate_unit"), {"mm"})
    normal = _vector(
        _required_value(document, ("normal",), path, "normal"),
        path,
        "normal",
        expected_length=3,
    )
    d_mm = _numeric_scalar(
        _required_value(document, ("d_mm",), path, "d_mm"),
        path,
        "d_mm",
    )
    norm = float(np.linalg.norm(normal))
    if norm <= np.finfo(np.float64).eps:
        raise CalibrationConfigError(f"{path.name} 的 normal 不能为零向量")
    result: dict[str, Any] = {
        "model_type": "global_plane",
        "normal": np.ascontiguousarray(normal / norm),
        "d_mm": float(d_mm / norm),
        "source_path": str(path.resolve()),
    }
    z_range = _optional_z_valid_range(document, path)
    if z_range is not None:
        result["z_valid_range_mm"] = z_range
    return result


def _parse_quadratic_graph_model(
    document: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    _validate_units(document, path, ("units", "coordinate_unit"), {"mm"})
    dependent_axis = str(
        _required_value(document, ("dependent_axis",), path, "dependent_axis")
    ).strip().upper()
    raw_independent = _required_value(
        document, ("independent_axes",), path, "independent_axes"
    )
    if not isinstance(raw_independent, (list, tuple)) or len(raw_independent) != 2:
        raise CalibrationDimensionError(
            f"{path.name} 的 independent_axes 应为两个坐标轴名称"
        )
    independent_axes = [str(value).strip().upper() for value in raw_independent]
    if {dependent_axis, *independent_axes} != {"X", "Y", "Z"}:
        raise CalibrationConfigError(
            f"{path.name} 的 dependent_axis 与 independent_axes 必须恰好覆盖 X/Y/Z"
        )

    normalization = document.get("normalization")
    if not isinstance(normalization, Mapping):
        raise CalibrationConfigError(f"{path.name} 缺少 normalization 映射")
    center = _vector(
        _required_value(
            normalization,
            ("independent_center_mm",),
            path,
            "normalization.independent_center_mm",
        ),
        path,
        "normalization.independent_center_mm",
        expected_length=2,
    )
    scale = _vector(
        _required_value(
            normalization,
            ("independent_scale_mm",),
            path,
            "normalization.independent_scale_mm",
        ),
        path,
        "normalization.independent_scale_mm",
        expected_length=2,
    )
    if np.any(scale <= 0.0):
        raise CalibrationConfigError(
            f"{path.name} 的 independent_scale_mm 必须全部为正数"
        )
    coefficients = _vector(
        _required_value(document, ("coefficients",), path, "coefficients"),
        path,
        "coefficients",
        expected_length=6,
    )
    result: dict[str, Any] = {
        "model_type": "quadratic_graph",
        "dependent_axis": dependent_axis,
        "independent_axes": independent_axes,
        "normalization": {
            "independent_center_mm": center,
            "independent_scale_mm": scale,
        },
        "coefficients": coefficients,
        "source_path": str(path.resolve()),
    }
    z_range = _optional_z_valid_range(document, path)
    if z_range is not None:
        result["z_valid_range_mm"] = z_range
    return result


def _parse_circular_cone_model(
    document: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    _validate_units(document, path, ("units", "coordinate_unit"), {"mm"})
    if document.get("fit_success") is False:
        raise CalibrationConfigError(
            f"{path.name} 标记 fit_success=false，不能用于正式重建"
        )
    axis = _vector(
        _required_value(
            document, ("axis_unit_camera",), path, "axis_unit_camera"
        ),
        path,
        "axis_unit_camera",
        expected_length=3,
    )
    apex = _vector(
        _required_value(document, ("apex_camera_mm",), path, "apex_camera_mm"),
        path,
        "apex_camera_mm",
        expected_length=3,
    )
    half_angle = _numeric_scalar(
        _required_value(
            document, ("half_apex_angle_deg",), path, "half_apex_angle_deg"
        ),
        path,
        "half_apex_angle_deg",
    )
    norm = float(np.linalg.norm(axis))
    if norm <= np.finfo(np.float64).eps:
        raise CalibrationConfigError(
            f"{path.name} 的 axis_unit_camera 不能为零向量"
        )
    if not 0.0 < half_angle < 90.0:
        raise CalibrationConfigError(
            f"{path.name} 的 half_apex_angle_deg 必须位于 (0, 90)"
        )
    result: dict[str, Any] = {
        "model_type": "circular_cone",
        "axis_unit_camera": np.ascontiguousarray(axis / norm),
        "apex_camera_mm": apex,
        "half_apex_angle_deg": float(half_angle),
        "source_path": str(path.resolve()),
    }
    z_range = _optional_z_valid_range(document, path)
    if z_range is not None:
        result["z_valid_range_mm"] = z_range
    return result


def _optional_z_valid_range(
    document: Mapping[str, Any], path: Path
) -> np.ndarray | None:
    if "z_valid_range_mm" not in document:
        return None
    values = _vector(
        document["z_valid_range_mm"],
        path,
        "z_valid_range_mm",
        expected_length=2,
    )
    if values[0] >= values[1]:
        raise CalibrationConfigError(
            f"{path.name} 的 z_valid_range_mm 必须严格递增"
        )
    return values


def _load_laser_plane(path: Path) -> np.ndarray:
    """旧私有接口兼容：仅接受可转换为 global_plane 的文件。"""
    model = _load_laser_model(path)
    if model["model_type"] != "global_plane":
        raise CalibrationConfigError(
            f"{path.name} 是 {model['model_type']}，不能按旧激光平面接口读取"
        )
    return np.ascontiguousarray(
        np.r_[model["normal"], model["d_mm"]], dtype=np.float64
    )


def _load_camera_ground_extrinsics(path: Path) -> dict[str, np.ndarray]:
    document = _load_required_yaml(path)
    _validate_units(
        document,
        path,
        ("units", "translation_unit", "coordinate_unit"),
        {"mm"},
    )
    _validate_units(
        document,
        path,
        ("rotation_unit",),
        {"dimensionless"},
    )

    if "R" in document or "t" in document:
        rotation = _required_value(document, ("R",), path, "R")
        translation = _required_value(document, ("t",), path, "t")
        R = _matrix(rotation, (3, 3), path, "R")
        t = _vector(translation, path, "t", expected_length=3)
    elif "T_ground_from_camera" in document:
        transform = _matrix(
            document["T_ground_from_camera"],
            (4, 4),
            path,
            "T_ground_from_camera",
        )
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
            raise CalibrationConfigError(
                f"{path.name} 的 T_ground_from_camera 最后一行必须为 [0, 0, 0, 1]"
            )
        R = np.ascontiguousarray(transform[:3, :3])
        t = np.ascontiguousarray(transform[:3, 3])
    else:
        raise CalibrationConfigError(
            f"{path.name} 缺少 R/t 或 T_ground_from_camera"
        )

    orthogonality_error = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(R))
    if orthogonality_error > 1.0e-6 or abs(determinant - 1.0) > 1.0e-6:
        raise CalibrationConfigError(
            f"{path.name} 的旋转矩阵无效：正交误差={orthogonality_error:.3e}，"
            f"det={determinant:.9f}"
        )
    return {"R": R, "t": t}


def _load_optional_ground_u(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return _load_ground_u_csv(path)
    if path.suffix.lower() == ".npy":
        return _load_ground_u_npy(path)

    document = _load_required_yaml(path)
    _validate_units(
        document,
        path,
        ("units", "coordinate_unit"),
        {"px", "mm"},
    )
    converted = _convert_numeric_sequences(document)
    axis = _ground_compensation_axis(converted, path)
    coordinate_key = "column_u_px" if axis == "u" else "row_v_px"
    if "sample_table" in converted:
        table = _numeric_array(converted["sample_table"], path, "sample_table")
        if table.ndim != 2 or table.shape[1] != 2:
            raise CalibrationDimensionError(
                f"{path.name} 的 sample_table 维度为 {table.shape}，应为 (N, 2)"
            )
        converted[coordinate_key] = table[:, 0]
        converted["bias_mm"] = table[:, 1]

    if coordinate_key not in converted or "bias_mm" not in converted:
        raise CalibrationConfigError(
            f"{path.name} 缺少 {coordinate_key}/bias_mm 或 sample_table"
        )
    columns, bias = _validate_ground_u_table(
        converted[coordinate_key], converted["bias_mm"], path, coordinate_key
    )
    converted[coordinate_key] = columns
    converted["coordinate_px"] = columns
    converted["compensation_axis"] = axis
    converted["bias_mm"] = bias
    z_offset = _ground_u_z_offset(converted, path)
    if z_offset is not None:
        converted["z_offset_mm"] = z_offset
    converted["source_path"] = str(path.resolve())
    return converted


def _load_ground_u_npy(path: Path) -> dict[str, Any]:
    try:
        loaded = np.load(path, allow_pickle=True)
    except (OSError, ValueError) as error:
        raise CalibrationConfigError(f"Unable to read {path.name}: {error}") from error

    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        loaded = loaded.item()

    if isinstance(loaded, Mapping):
        converted = dict(loaded)
    elif isinstance(loaded, np.ndarray) and loaded.dtype.names is not None:
        converted = {name: loaded[name] for name in loaded.dtype.names}
    else:
        table = _numeric_array(loaded, path, "ground_u_compensation")
        if table.ndim != 2 or table.shape[1] != 2:
            raise CalibrationDimensionError(
                f"{path.name} must be a dict/structured array or an (N, 2) table"
            )
        converted = {
            "compensation_axis": "u",
            "column_u_px": table[:, 0],
            "bias_mm": table[:, 1],
        }

    axis = _ground_compensation_axis(converted, path)
    coordinate_key = "column_u_px" if axis == "u" else "row_v_px"
    if coordinate_key not in converted and "columns" in converted:
        converted[coordinate_key] = converted["columns"]
    if coordinate_key not in converted or "bias_mm" not in converted:
        raise CalibrationConfigError(
            f"{path.name} must contain {coordinate_key}/bias_mm or columns/bias_mm"
        )
    columns, bias = _validate_ground_u_table(
        converted[coordinate_key], converted["bias_mm"], path, coordinate_key
    )
    converted[coordinate_key] = columns
    converted["coordinate_px"] = columns
    converted["compensation_axis"] = axis
    converted["bias_mm"] = bias
    z_offset = _ground_u_z_offset(converted, path)
    if z_offset is not None:
        converted["z_offset_mm"] = z_offset
    converted["source_path"] = str(path.resolve())
    return converted


def _load_ground_u_csv(path: Path) -> dict[str, Any]:
    """读取 ground_bias_validation_results_v2 输出的逐列补偿表。"""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            coordinate_key = (
                "row_v_px" if "row_v_px" in fieldnames else "column_u_px"
            )
            required = {coordinate_key, "bias_mm"}
            if not required.issubset(fieldnames):
                raise CalibrationConfigError(
                    f"{path.name} 必须包含 column_u_px/bias_mm 或 row_v_px/bias_mm"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise CalibrationConfigError(f"无法读取 {path.name}: {error}") from error

    try:
        columns = np.asarray([row[coordinate_key] for row in rows], dtype=np.float64)
        bias = np.asarray([row["bias_mm"] for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CalibrationConfigError(
            f"{path.name} 的 {coordinate_key}/bias_mm 必须是数值"
        ) from error

    columns, bias = _validate_ground_u_table(
        columns, bias, path, coordinate_key
    )
    axis = "v" if coordinate_key == "row_v_px" else "u"
    return {
        coordinate_key: columns,
        "coordinate_px": columns,
        "compensation_axis": axis,
        "bias_mm": bias,
        "source_path": str(path.resolve()),
    }


def _validate_ground_u_table(
    columns: Any,
    bias: Any,
    path: Path,
    coordinate_key: str = "column_u_px",
) -> tuple[np.ndarray, np.ndarray]:
    columns_array = _vector(columns, path, coordinate_key)
    bias_array = _vector(bias, path, "bias_mm")
    if len(columns_array) == 0:
        raise CalibrationConfigError(f"{path.name} 的补偿表不能为空")
    if len(columns_array) != len(bias_array):
        raise CalibrationDimensionError(
            f"{path.name} 的 {coordinate_key} 与 bias_mm 长度不一致"
        )
    if np.any(np.diff(columns_array) <= 0.0):
        raise CalibrationConfigError(
            f"{path.name} 的 {coordinate_key} 必须严格递增且不能重复"
        )
    return columns_array, bias_array


def _ground_compensation_axis(document: Mapping[str, Any], path: Path) -> str:
    raw_axis = document.get("compensation_axis")
    metadata = document.get("metadata")
    if raw_axis is None and isinstance(metadata, Mapping):
        raw_axis = metadata.get("compensation_axis")
    if raw_axis is None:
        raw_axis = "v" if "row_v_px" in document else "u"
    axis = str(raw_axis).strip().lower()
    if axis not in {"u", "v"}:
        raise CalibrationConfigError(
            f"{path.name} 的 compensation_axis 必须是 u 或 v，实际为 {raw_axis!r}"
        )
    return axis


def _ground_u_z_offset(document: Mapping[str, Any], path: Path) -> float | None:
    if "z_offset_mm" in document:
        return _numeric_scalar(document["z_offset_mm"], path, "z_offset_mm")
    if "ground_z_offset_mm" in document:
        return _numeric_scalar(
            document["ground_z_offset_mm"], path, "ground_z_offset_mm"
        )
    return None


def _numeric_scalar(value: Any, path: Path, name: str) -> float:
    array = _numeric_array(value, path, name)
    if array.size != 1:
        raise CalibrationDimensionError(
            f"{path.name} 的 {name} 维度为 {array.shape}，应为单个数值"
        )
    return float(array.reshape(-1)[0])


def _load_required_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CalibrationFileNotFoundError(f"标定文件不存在: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CalibrationConfigError(f"无法读取 {path.name}: {error}") from error
    if not isinstance(document, Mapping):
        raise CalibrationConfigError(f"{path.name} 的顶层必须是 YAML 映射")
    return dict(document)


def _required_value(
    document: Mapping[str, Any],
    aliases: tuple[str, ...],
    path: Path,
    display_name: str,
) -> Any:
    for key in aliases:
        if key in document:
            return document[key]
    raise CalibrationConfigError(f"{path.name} 缺少参数 {display_name}")


def _matrix(
    value: Any,
    expected_shape: tuple[int, int],
    path: Path,
    name: str,
) -> np.ndarray:
    array = _numeric_array(value, path, name)
    if array.shape != expected_shape:
        raise CalibrationDimensionError(
            f"{path.name} 的 {name} 维度为 {array.shape}，应为 {expected_shape}"
        )
    return np.ascontiguousarray(array)


def _vector(
    value: Any,
    path: Path,
    name: str,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    array = _numeric_array(value, path, name)
    if array.ndim == 1:
        vector = array
    elif array.ndim == 2 and 1 in array.shape:
        vector = array.reshape(-1)
    else:
        raise CalibrationDimensionError(
            f"{path.name} 的 {name} 维度为 {array.shape}，应为一维向量"
        )
    if expected_length is not None and len(vector) != expected_length:
        raise CalibrationDimensionError(
            f"{path.name} 的 {name} 长度为 {len(vector)}，应为 {expected_length}"
        )
    return np.ascontiguousarray(vector)


def _numeric_array(value: Any, path: Path, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CalibrationConfigError(
            f"{path.name} 的 {name} 必须由数值组成"
        ) from error
    if not np.isfinite(array).all():
        raise CalibrationConfigError(f"{path.name} 的 {name} 包含 NaN 或无穷值")
    return array


def _validate_units(
    document: Mapping[str, Any],
    path: Path,
    keys: tuple[str, ...],
    allowed: set[str],
) -> None:
    declared_units: set[str] = set()
    for key in keys:
        if key not in document:
            continue
        raw_unit = document[key]
        if not isinstance(raw_unit, str):
            raise CalibrationUnitError(f"{path.name} 的 {key} 必须是单位字符串")
        canonical = _UNIT_ALIASES.get(raw_unit.strip().lower())
        if canonical not in allowed:
            expected = " 或 ".join(sorted(allowed))
            raise CalibrationUnitError(
                f"{path.name} 的 {key}={raw_unit!r}，应为 {expected}"
            )
        declared_units.add(canonical)
    if len(declared_units) > 1:
        raise CalibrationUnitError(f"{path.name} 声明了相互冲突的单位")


def _convert_numeric_sequences(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _convert_numeric_sequences(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return [_convert_numeric_sequences(item) for item in value]
        if np.isfinite(array).all():
            return np.ascontiguousarray(array)
        raise CalibrationConfigError("ground_u_compensation 包含 NaN 或无穷值")
    return value
