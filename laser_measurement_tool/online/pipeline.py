"""Calibration-consistent processing of one acquired camera frame."""

from __future__ import annotations

import hashlib
import threading
import time

import cv2
import numpy as np

from app_config import AppConfig
from calibration.manifest import CalibrationPackage, load_calibration_package
from correction.stage_a_height_scale import (
    normalize_correction_mode,
    resolve_height_correction,
)
from gui.image_view import _to_uint8_display
from laser.backends import create_extraction_params
from laser.laser_extractor import extract_laser_center
from measurement.ground_reference import (
    SUPPORTED_GROUND_SUPPORT_SOURCES,
    SessionGroundReference,
)
from reconstruction.reconstructor import reconstruct_uv_to_ground

from .models import CapturedFrame, FrameResult


def _finite_values(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _finite_mean(values: np.ndarray | None) -> float | None:
    finite = _finite_values(values)
    return None if not len(finite) else float(np.mean(finite))


def _finite_median(values: np.ndarray | None) -> float | None:
    finite = _finite_values(values)
    return None if not len(finite) else float(np.median(finite))


def _finite_min(values: np.ndarray | None) -> float | None:
    finite = _finite_values(values)
    return None if not len(finite) else float(np.min(finite))


def _finite_max(values: np.ndarray | None) -> float | None:
    finite = _finite_values(values)
    return None if not len(finite) else float(np.max(finite))


def _c1_clamp_status(clamped: np.ndarray | None) -> str:
    if clamped is None:
        return "NOT_APPLICABLE"
    flags = np.asarray(clamped, dtype=bool).reshape(-1)
    if not len(flags):
        return "NOT_APPLICABLE"
    if bool(np.all(flags)):
        return "CLAMPED"
    if bool(np.any(flags)):
        return "MIXED"
    return "IN_DOMAIN"


class FramePipeline:
    """Run the selected extraction and reconstruction path for one frame."""

    def __init__(
        self,
        config: AppConfig,
        extraction_method: str | None = None,
        system: str | None = None,
    ) -> None:
        if config.calibration.manifest is None:
            raise ValueError("在线测量配置必须指定 calibration.manifest")
        self.config = config
        self.system = (system or config.system).strip().lower()
        self.extraction_method = extraction_method or config.extraction_method
        self.extraction_options = dict(
            config.extraction_options_by_method.get(self.extraction_method, {})
        )
        self.package: CalibrationPackage = load_calibration_package(
            config.calibration.manifest
        )
        self._calibration_lock = threading.RLock()
        self._ground_extrinsic_source = "reference"
        self._reference_R = np.ascontiguousarray(
            np.asarray(self.package.calibration["R"], dtype=np.float64).copy()
        )
        self._reference_t = np.ascontiguousarray(
            np.asarray(self.package.calibration["t"], dtype=np.float64).copy()
        )
        self._ground_extrinsic_generation = 0
        self._session_ground_reference: SessionGroundReference | None = None
        self._height_correction_mode = normalize_correction_mode(
            config.correction.mode
        )
        self.extraction_params = create_extraction_params(
            self.extraction_method, self.extraction_options
        )
        self.algorithm_config_sha256 = _algorithm_hash(
            self.extraction_method, self.extraction_options
        )

    def run_frame(self, frame: CapturedFrame) -> FrameResult:
        self._validate_frame_bounds(frame)
        total_start = time.perf_counter_ns()
        extraction_start = time.perf_counter_ns()
        centers_local = extract_laser_center(
            frame.image,
            self.extraction_params,
            image_offset=(frame.offset_x, frame.offset_y),
        )
        extraction_ms = (time.perf_counter_ns() - extraction_start) / 1e6

        centers_full = centers_local.copy()
        if centers_full.size:
            centers_full[:, 0] += frame.offset_x
            centers_full[:, 1] += frame.offset_y

        reconstruction_start = time.perf_counter_ns()
        with self._calibration_lock:
            calibration = dict(self.package.calibration)
            ground_extrinsic_source = self._ground_extrinsic_source
            ground_extrinsic_generation = self._ground_extrinsic_generation
            ground_reference = self._session_ground_reference
            height_correction_mode = self._height_correction_mode
        reconstructed = reconstruct_uv_to_ground(
            centers_full,
            calibration,
            self.config.reconstruction,
        )
        reconstruction_ms = (time.perf_counter_ns() - reconstruction_start) / 1e6
        points, ground_reference_metadata = self._apply_ground_reference(
            reconstructed.points_ground, ground_reference
        )
        section = (
            np.ascontiguousarray(points[:, (0, 2)])
            if len(points)
            else np.empty((0, 2), dtype=np.float64)
        )
        height_metadata = self._height_shadow_metadata(
            reconstructed,
            mode_override=height_correction_mode,
        )
        total_ms = (time.perf_counter_ns() - total_start) / 1e6
        return FrameResult(
            frame=frame,
            centers_uv_full=np.ascontiguousarray(centers_full),
            points_camera=reconstructed.points_camera,
            points_ground=points,
            section_xz=section,
            extraction_ms=extraction_ms,
            reconstruction_ms=reconstruction_ms,
            total_ms=total_ms,
            calibration_package_id=self.package.package_id,
            calibration_manifest_sha256=self.package.manifest_sha256,
            algorithm_config_sha256=self.algorithm_config_sha256,
            ground_extrinsic_source=ground_extrinsic_source,
            ground_extrinsic_generation=ground_extrinsic_generation,
            **ground_reference_metadata,
            **height_metadata,
            points_ground_raw=np.ascontiguousarray(
                reconstructed.points_ground
            ),
            filtered=reconstructed.filtered,
            pixels_uv=reconstructed.pixels_uv,
        )

    def _height_shadow_metadata(
        self,
        reconstruction,
        *,
        mode_override: str | None = None,
    ) -> dict[str, object]:
        """Build read-only H1/H-B2 geometry shadow fields for one frame."""
        q1_values = reconstruction.q1_c0
        q2_values = reconstruction.q2_c0
        q1 = _finite_mean(q1_values)
        q2 = _finite_mean(q2_values)
        q2_in_domain: bool | None = None
        hb2_config = self.config.correction.hb2_height_correction
        if q2_values is not None and len(q2_values):
            q2_values = np.asarray(q2_values, dtype=np.float64)
            if hb2_config is None:
                q2_in_domain = None
            else:
                lower, upper = hb2_config.q2_domain
                q2_in_domain = bool(
                    np.isfinite(q2_values).all()
                    and np.all((q2_values >= lower) & (q2_values <= upper))
                )
        height_result = resolve_height_correction(
            None,
            q1=q1,
            q2=q2,
            q2_in_domain=q2_in_domain,
            system=self.system,
            correction=self.config.correction,
            mode_override=mode_override,
        )
        v_values = (
            reconstruction.pixels_uv[:, 1]
            if len(reconstruction.pixels_uv)
            else np.empty(0, dtype=np.float64)
        )
        c1_status = _c1_clamp_status(reconstruction.c1_clamped)
        metadata = height_result.as_dict()
        metadata.update(
            {
                "v_min": _finite_min(v_values),
                "v_median": _finite_median(v_values),
                "v_max": _finite_max(v_values),
                "point_count": int(reconstruction.point_count),
                "c1_clamp_status": c1_status,
            }
        )
        return metadata

    @property
    def height_correction_mode(self) -> str:
        """Return the active mutually exclusive online height mode."""
        with self._calibration_lock:
            return self._height_correction_mode

    def set_height_correction_mode(self, mode: str) -> str:
        """Set the online scalar mode without changing reconstruction data."""
        normalized = normalize_correction_mode(mode)
        with self._calibration_lock:
            self._height_correction_mode = normalized
        return normalized

    @property
    def ground_extrinsic_source(self) -> str:
        """当前运行时 ground 外参来源：``reference`` 或 ``session``。"""
        with self._calibration_lock:
            return self._ground_extrinsic_source

    @property
    def calibration_package_identity(self) -> tuple[str, str]:
        """Return the immutable package identity used by this pipeline."""
        return self.package.package_id, self.package.manifest_sha256

    @property
    def ground_extrinsic_generation(self) -> int:
        """当前 active ground extrinsic 的轻量 generation token。"""
        with self._calibration_lock:
            return self._ground_extrinsic_generation

    @property
    def reference_ground_extrinsic(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 reference R/t 的副本，用于 Session 标定差异比较。"""
        return self._reference_R.copy(), self._reference_t.copy()

    @property
    def session_ground_reference(self) -> SessionGroundReference | None:
        """当前会话冻结的线性 ground reference；与 PnP 外参独立。"""
        with self._calibration_lock:
            return self._session_ground_reference

    def apply_session_ground_reference(
        self, reference: SessionGroundReference
    ) -> None:
        """仅在当前进程启用 Session ground reference，不写 reference 文件。"""
        if not isinstance(reference, SessionGroundReference):
            raise TypeError("Session ground reference 类型不正确")
        if str(reference.status).upper() != "VALID":
            raise ValueError("只能应用 VALID 的 Session ground reference")
        if reference.support_source not in SUPPORTED_GROUND_SUPPORT_SOURCES:
            raise ValueError(
                "Session ground reference 必须带有明确的 ground support source"
            )
        with self._calibration_lock:
            if reference.active_ground_extrinsic_source != self._ground_extrinsic_source:
                raise ValueError("Session ground reference 的 active extrinsic source 已失效")
            if reference.ground_extrinsic_generation != self._ground_extrinsic_generation:
                raise ValueError("Session ground reference 的 extrinsic generation 已失效")
        lower, upper = reference.valid_s_range_mm
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError("Session ground reference 的 S 范围无效")
        with self._calibration_lock:
            self._session_ground_reference = reference

    def reset_session_ground_reference(self) -> None:
        """清除当前进程的 Session ground reference，不修改 reference 文件。"""
        with self._calibration_lock:
            self._session_ground_reference = None

    def apply_ground_reference_to_points(
        self, points_ground: np.ndarray
    ) -> tuple[np.ndarray, dict[str, object]]:
        """对外提供与 ``run_frame`` 相同的 ground reference 应用路径。"""
        with self._calibration_lock:
            reference = self._session_ground_reference
        return self._apply_ground_reference(points_ground, reference)

    def calibration_for_reconstruction(self) -> dict[str, object]:
        """返回线程安全的当前运行时标定快照。"""
        with self._calibration_lock:
            calibration = dict(self.package.calibration)
            calibration["R"] = np.asarray(
                calibration["R"], dtype=np.float64
            ).copy()
            calibration["t"] = np.asarray(
                calibration["t"], dtype=np.float64
            ).copy()
            return calibration

    def apply_session_ground_extrinsic(
        self,
        R_camera_to_ground: np.ndarray,
        t_camera_to_ground: np.ndarray,
        *,
        generation: int | None = None,
    ) -> None:
        """仅替换当前进程内的 ground R/t，不写入 reference 文件。"""
        rotation = np.asarray(R_camera_to_ground, dtype=np.float64)
        translation = np.asarray(t_camera_to_ground, dtype=np.float64).reshape(-1)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("Session ground 外参必须是 R(3x3) 和 t(3)")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Session ground 外参必须包含有限数值")
        if generation is None:
            generation = self._ground_extrinsic_generation + 1
        if isinstance(generation, bool) or int(generation) < 0:
            raise ValueError("ground extrinsic generation 必须是非负整数")
        with self._calibration_lock:
            generation_changed = (
                self._ground_extrinsic_source != "session"
                or int(generation) != self._ground_extrinsic_generation
            )
            self.package.calibration["R"] = np.ascontiguousarray(rotation.copy())
            self.package.calibration["t"] = np.ascontiguousarray(translation.copy())
            self._ground_extrinsic_source = "session"
            self._ground_extrinsic_generation = int(generation)
            # A new PnP generation cannot safely reuse a reference fitted with
            # the previous camera-to-ground transform.
            if generation_changed:
                self._session_ground_reference = None

    def reset_ground_extrinsic(self, *, generation: int | None = None) -> None:
        """恢复当前进程内的 reference R/t；不写入 reference 文件。"""
        if generation is None:
            generation = self._ground_extrinsic_generation + 1
        if isinstance(generation, bool) or int(generation) < 0:
            raise ValueError("ground extrinsic generation 必须是非负整数")
        with self._calibration_lock:
            self.package.calibration["R"] = self._reference_R.copy()
            self.package.calibration["t"] = self._reference_t.copy()
            self._ground_extrinsic_source = "reference"
            self._ground_extrinsic_generation = int(generation)
            self._session_ground_reference = None

    @staticmethod
    def _apply_ground_reference(
        points_ground: np.ndarray,
        reference: SessionGroundReference | None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        points = np.ascontiguousarray(np.asarray(points_ground, dtype=np.float64))
        if reference is None:
            return points, {
                "ground_reference_source": "none",
                "ground_reference_status": "inactive",
                "ground_reference_valid_s_range_mm": None,
                "ground_reference_applied_count": 0,
                "ground_reference_out_of_range_count": 0,
                "ground_reference_coordinate": None,
                "ground_reference_coordinate_units": None,
                "ground_reference_coordinate_formula": None,
                "ground_reference_origin_xy": None,
                "ground_reference_direction_xy": None,
                "ground_reference_slope_z_per_mm": None,
                "ground_reference_intercept_z_mm": None,
                "ground_reference_frozen_json_path": None,
                "ground_reference_frozen_json_sha256": None,
                "ground_reference_fit_pose_ids": (),
            }
        corrected, valid = reference.apply_to_points(points)
        applied_count = int(valid.sum())
        out_of_range_count = int(len(points) - applied_count)
        if not len(points):
            status = "active_no_points"
        elif out_of_range_count:
            status = "partial_out_of_valid_s_domain"
        else:
            status = "applied"
        return corrected, {
            "ground_reference_source": reference.provenance_source,
            "ground_reference_status": status,
            "ground_reference_valid_s_range_mm": reference.valid_s_range_mm,
            "ground_reference_applied_count": applied_count,
            "ground_reference_out_of_range_count": out_of_range_count,
            "ground_reference_coordinate": reference.coordinate,
            "ground_reference_coordinate_units": reference.coordinate_units,
            "ground_reference_coordinate_formula": reference.coordinate_formula,
            "ground_reference_origin_xy": tuple(
                float(value) for value in np.asarray(reference.origin_xy)
            ),
            "ground_reference_direction_xy": tuple(
                float(value) for value in np.asarray(reference.direction_xy)
            ),
            "ground_reference_slope_z_per_mm": float(reference.slope_z_per_mm),
            "ground_reference_intercept_z_mm": float(reference.intercept_z_mm),
            "ground_reference_frozen_json_path": reference.frozen_json_path,
            "ground_reference_frozen_json_sha256": reference.frozen_json_sha256,
            "ground_reference_fit_pose_ids": tuple(reference.fit_pose_ids),
        }

    def _validate_frame_bounds(self, frame: CapturedFrame) -> None:
        height, width = frame.image.shape
        if frame.offset_x + width > self.package.image_width:
            raise ValueError("相机 ROI 横向范围超出标定图像尺寸")
        if frame.offset_y + height > self.package.image_height:
            raise ValueError("相机 ROI 纵向范围超出标定图像尺寸")


def _algorithm_hash(method: str, options: dict[str, object]) -> str:
    import json

    payload = {"method": method, "options": options}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_overlay(image: np.ndarray, centers_local: np.ndarray) -> np.ndarray:
    """Render the extracted centers only when a preview actually needs them."""
    gray = _to_uint8_display(image)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for u, v in centers_local:
        cv2.circle(
            canvas,
            (int(round(u)), int(round(v))),
            1,
            (50, 255, 90),
            -1,
            lineType=cv2.LINE_AA,
        )
    return canvas
