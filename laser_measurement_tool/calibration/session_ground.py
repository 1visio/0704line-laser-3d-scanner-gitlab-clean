"""Session-time checkerboard PnP ground calibration.

This module is the shared implementation for the existing checkerboard PnP
path and the online session-calibration API.  OpenCV's ``solvePnP`` pose is
kept in its native direction (checkerboard -> camera); the returned ground
transform follows the runtime convention used by ``camera_ground_extrinsics``:

``p_ground = T_ground_from_camera @ p_camera``

The ground origin is the intersection of the camera principal axis and the
checkerboard plane.  ``Zg`` points toward the camera, ``Xg`` is camera +X
projected onto the plane, and ``Yg = Zg cross Xg``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


_DETECTOR_MODES = frozenset({"sb_then_classic", "classic"})
_EPSILON = np.finfo(np.float64).eps


@dataclass(frozen=True, slots=True)
class SessionGroundBoardConfig:
    """Checkerboard protocol used for one session calibration image."""

    pattern_cols: int = 11
    pattern_rows: int = 8
    square_size_mm: float = 20.0
    detector: str = "sb_then_classic"

    def __post_init__(self) -> None:
        if self.pattern_cols < 2 or self.pattern_rows < 2:
            raise ValueError("pattern_cols and pattern_rows must be >= 2")
        if not np.isfinite(self.square_size_mm) or self.square_size_mm <= 0.0:
            raise ValueError("square_size_mm must be finite and positive")
        if self.detector not in _DETECTOR_MODES:
            allowed = ", ".join(sorted(_DETECTOR_MODES))
            raise ValueError(f"detector must be one of: {allowed}")

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.pattern_cols, self.pattern_rows

    def object_points(self) -> np.ndarray:
        """Return object points in the configured millimetre unit."""
        return create_object_points(
            self.pattern_cols,
            self.pattern_rows,
            self.square_size_mm,
        )


# Short public alias for callers that prefer the generic board-config name.
BoardConfig = SessionGroundBoardConfig


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Typed form of the ``K``/``D`` mapping returned by calibration loaders."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True, slots=True)
class ChessboardCornerDetection:
    """Corner-detection result before PnP is attempted."""

    found: bool
    corners: np.ndarray | None
    method: str | None


@dataclass(frozen=True, slots=True)
class SessionGroundExtrinsic:
    """Result of one checkerboard session-ground calibration attempt.

    ``rvec``/``tvec`` and ``R_board_to_camera`` describe OpenCV's native
    checkerboard-to-camera pose.  ``R``/``t`` and
    ``T_ground_from_camera`` describe the runtime camera-to-ground transform.
    """

    status: str
    message: str
    detected_corners: np.ndarray | None = None
    detection_method: str | None = None
    reprojection_rmse_px: float | None = None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    R_board_to_camera: np.ndarray | None = None
    T_camera_from_board: np.ndarray | None = None
    ground_normal_in_camera: np.ndarray | None = None
    ground_origin_in_camera: np.ndarray | None = None
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    T_ground_from_camera: np.ndarray | None = None
    T_camera_from_ground: np.ndarray | None = None

    @property
    def rvec_board_to_camera(self) -> np.ndarray | None:
        return self.rvec

    @property
    def tvec_board_to_camera(self) -> np.ndarray | None:
        return self.tvec

    @property
    def R_camera_to_ground(self) -> np.ndarray | None:
        return self.R

    @property
    def t_camera_to_ground(self) -> np.ndarray | None:
        return self.t


def create_object_points(
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
) -> np.ndarray:
    """Create the project's existing x-fast/y-slow object-point order."""
    if pattern_cols < 2 or pattern_rows < 2:
        raise ValueError("pattern_cols and pattern_rows must be >= 2")
    if not np.isfinite(square_size_mm) or square_size_mm <= 0.0:
        raise ValueError("square_size_mm must be finite and positive")

    points = np.zeros((pattern_cols * pattern_rows, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
        * square_size_mm
    )
    return points


def _as_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError("image must be a 2-D grayscale or 3-D color array")
    if array.shape[2] == 1:
        return array[:, :, 0]
    if array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    raise ValueError("image must have 1, 3, or 4 channels")


def _detect_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
    detector: str,
) -> ChessboardCornerDetection:
    if detector == "sb_then_classic" and hasattr(cv2, "findChessboardCornersSB"):
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
        except cv2.error:
            found, corners = False, None
        if found and corners is not None:
            return ChessboardCornerDetection(
                True,
                np.asarray(corners, dtype=np.float32),
                "SB",
            )

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if detector == "classic":
        flags |= cv2.CALIB_CB_FILTER_QUADS
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return ChessboardCornerDetection(False, None, None)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ChessboardCornerDetection(
        True,
        np.asarray(refined, dtype=np.float32),
        "classic+cornerSubPix",
    )


def detect_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
    detector: str = "sb_then_classic",
) -> tuple[bool, np.ndarray | None]:
    """Compatibility wrapper matching the existing intrinsic-calibration API."""
    if detector not in _DETECTOR_MODES:
        raise ValueError(f"unsupported checkerboard detector: {detector}")
    result = _detect_corners(np.asarray(gray), pattern_size, detector)
    return result.found, result.corners


def _normalise_intrinsics(
    intrinsics: CameraIntrinsics | Mapping[str, Any] | tuple[Any, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(intrinsics, CameraIntrinsics):
        matrix = intrinsics.camera_matrix
        distortion = intrinsics.dist_coeffs
    elif isinstance(intrinsics, Mapping):
        matrix = intrinsics.get("K", intrinsics.get("camera_matrix"))
        distortion = intrinsics.get("D", intrinsics.get("dist_coeffs"))
    elif isinstance(intrinsics, tuple) and len(intrinsics) == 2:
        matrix, distortion = intrinsics
    else:
        raise ValueError("intrinsics must provide camera matrix K and distortion D")

    if matrix is None or distortion is None:
        raise ValueError("intrinsics must provide camera matrix K and distortion D")
    K = np.asarray(matrix, dtype=np.float64)
    D = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if K.shape != (3, 3):
        raise ValueError("camera matrix must have shape (3, 3)")
    if D.size == 0:
        raise ValueError("distortion coefficients must not be empty")
    if not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError("intrinsics must contain finite values")
    return np.ascontiguousarray(K), np.ascontiguousarray(D)


def build_camera_to_ground_transform(
    R_board_to_camera: np.ndarray,
    t_board_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build runtime ground transforms from a checkerboard PnP pose.

    Returns ``(R_camera_to_ground, t_camera_to_ground,
    T_ground_from_camera, T_camera_from_ground)``.  The board in-plane axes
    are intentionally not used as ground axes; this follows the existing
    camera-ground YAML convention and makes the result independent of the
    board's in-plane placement.
    """
    R_board_to_camera = np.asarray(R_board_to_camera, dtype=np.float64)
    t_board_to_camera = np.asarray(t_board_to_camera, dtype=np.float64).reshape(3)
    if R_board_to_camera.shape != (3, 3):
        raise ValueError("R_board_to_camera must have shape (3, 3)")
    if not np.isfinite(R_board_to_camera).all() or not np.isfinite(t_board_to_camera).all():
        raise ValueError("PnP pose must contain finite values")

    normal = np.ascontiguousarray(R_board_to_camera[:, 2], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= _EPSILON:
        raise ValueError("checkerboard normal is degenerate")
    normal /= normal_norm

    # OpenCV camera +Z points away from the camera.  Ground +Z is defined as
    # the checkerboard normal toward the camera, hence the negative-Z choice.
    if normal[2] > 0.0:
        normal = -normal
    if abs(float(normal[2])) <= _EPSILON:
        raise ValueError("checkerboard plane is parallel to the camera principal axis")

    x_reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_reference - float(x_reference @ normal) * normal
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= _EPSILON:
        x_reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = x_reference - float(x_reference @ normal) * normal
        x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= _EPSILON:
        raise ValueError("cannot define ground X axis")
    x_axis /= x_norm
    y_axis = np.cross(normal, x_axis)
    y_axis /= float(np.linalg.norm(y_axis))

    # Columns of this matrix are ground axes expressed in camera coordinates.
    R_ground_to_camera = np.column_stack([x_axis, y_axis, normal])
    R_camera_to_ground = R_ground_to_camera.T

    # The plane through t_board_to_camera has normal ``normal``.  Intersect it
    # with the camera principal ray (0, 0, depth).
    principal_depth = float(normal @ t_board_to_camera) / float(normal[2])
    if not np.isfinite(principal_depth) or principal_depth <= 0.0:
        raise ValueError("checkerboard plane is not in front of the camera")
    ground_origin_in_camera = np.array(
        [0.0, 0.0, principal_depth],
        dtype=np.float64,
    )

    t_camera_to_ground = -R_camera_to_ground @ ground_origin_in_camera
    T_ground_from_camera = np.eye(4, dtype=np.float64)
    T_ground_from_camera[:3, :3] = R_camera_to_ground
    T_ground_from_camera[:3, 3] = t_camera_to_ground

    T_camera_from_ground = np.eye(4, dtype=np.float64)
    T_camera_from_ground[:3, :3] = R_ground_to_camera
    T_camera_from_ground[:3, 3] = ground_origin_in_camera
    return (
        np.ascontiguousarray(R_camera_to_ground),
        np.ascontiguousarray(t_camera_to_ground),
        np.ascontiguousarray(T_ground_from_camera),
        np.ascontiguousarray(T_camera_from_ground),
    )


def _failure(
    status: str,
    message: str,
    *,
    corners: np.ndarray | None = None,
    method: str | None = None,
) -> SessionGroundExtrinsic:
    return SessionGroundExtrinsic(
        status=status,
        message=message,
        detected_corners=corners,
        detection_method=method,
    )


def estimate_session_ground_extrinsic_from_corners(
    corners: np.ndarray,
    intrinsics: CameraIntrinsics | Mapping[str, Any] | tuple[Any, Any],
    board_config: SessionGroundBoardConfig | None = None,
    *,
    detection_method: str | None = "provided",
) -> SessionGroundExtrinsic:
    """Solve the existing Session-PnP semantics from ordered image corners.

    This is the solve-only half of :func:`estimate_session_ground_extrinsic`.
    It is intentionally public so the online five-frame workflow can robustly
    aggregate same-order detections before running the exact same solvePnP and
    camera-to-ground transform path.
    """
    board = board_config or SessionGroundBoardConfig()
    try:
        K, D = _normalise_intrinsics(intrinsics)
    except (TypeError, ValueError) as error:
        return _failure("invalid_intrinsics", str(error))

    try:
        array = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError) as error:
        return _failure("invalid_corners", str(error), method=detection_method)
    expected_count = board.pattern_cols * board.pattern_rows
    if len(array) != expected_count:
        return _failure(
            "invalid_corner_count",
            f"expected {expected_count} checkerboard corners, got {len(array)}",
            corners=array,
            method=detection_method,
        )
    if not np.isfinite(array).all():
        return _failure(
            "invalid_corners",
            "checkerboard corners must be finite",
            corners=array,
            method=detection_method,
        )
    corners_array = np.ascontiguousarray(array)

    object_points = board.object_points()
    try:
        solved, rvec, tvec = cv2.solvePnP(
            object_points,
            corners_array,
            K,
            D,
        )
    except cv2.error as error:
        return _failure(
            "solve_pnp_failed",
            str(error),
            corners=corners_array,
            method=detection_method,
        )
    if not solved:
        return _failure(
            "solve_pnp_failed",
            "cv2.solvePnP returned solved=False",
            corners=corners_array,
            method=detection_method,
        )

    # Keep OpenCV's float32 corner representation for solvePnP, matching the
    # legacy implementation; use float64 only for reported residual arithmetic.
    rvec = np.ascontiguousarray(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    tvec = np.ascontiguousarray(np.asarray(tvec, dtype=np.float64).reshape(3, 1))
    R_board_to_camera, _ = cv2.Rodrigues(rvec)
    R_board_to_camera = np.ascontiguousarray(
        np.asarray(R_board_to_camera, dtype=np.float64)
    )
    T_camera_from_board = np.eye(4, dtype=np.float64)
    T_camera_from_board[:3, :3] = R_board_to_camera
    T_camera_from_board[:3, 3] = tvec.reshape(3)

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        D,
    )
    residual = corners_array.astype(np.float64) - projected.reshape(-1, 2)
    reprojection_rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    try:
        (
            R_camera_to_ground,
            t_camera_to_ground,
            T_ground_from_camera,
            T_camera_from_ground,
        ) = build_camera_to_ground_transform(R_board_to_camera, tvec)
    except ValueError as error:
        return _failure(
            "invalid_pose",
            str(error),
            corners=corners_array,
            method=detection_method,
        )

    normal = np.ascontiguousarray(R_board_to_camera[:, 2], dtype=np.float64)
    if normal[2] > 0.0:
        normal = -normal
    origin = np.ascontiguousarray(T_camera_from_ground[:3, 3], dtype=np.float64)
    return SessionGroundExtrinsic(
        status="success",
        message="session ground calibration succeeded",
        detected_corners=np.ascontiguousarray(corners_array),
        detection_method=detection_method,
        reprojection_rmse_px=reprojection_rmse,
        rvec=rvec,
        tvec=tvec,
        R_board_to_camera=R_board_to_camera,
        T_camera_from_board=np.ascontiguousarray(T_camera_from_board),
        ground_normal_in_camera=normal,
        ground_origin_in_camera=origin,
        R=R_camera_to_ground,
        t=t_camera_to_ground,
        T_ground_from_camera=T_ground_from_camera,
        T_camera_from_ground=T_camera_from_ground,
    )


def estimate_session_ground_extrinsic(
    image: np.ndarray,
    intrinsics: CameraIntrinsics | Mapping[str, Any] | tuple[Any, Any],
    board_config: SessionGroundBoardConfig | None = None,
) -> SessionGroundExtrinsic:
    """Estimate one session's camera-to-ground transform from a board image.

    The solve uses the project's existing OpenCV object-point order and
    default ``cv2.solvePnP`` method (``SOLVEPNP_ITERATIVE``).  It does not
    write or modify any calibration file.
    """
    board = board_config or SessionGroundBoardConfig()
    try:
        K, D = _normalise_intrinsics(intrinsics)
    except (TypeError, ValueError) as error:
        return _failure("invalid_intrinsics", str(error))

    try:
        gray = _as_gray(image)
    except (TypeError, ValueError) as error:
        return _failure("invalid_image", str(error))

    try:
        detection = _detect_corners(gray, board.pattern_size, board.detector)
    except cv2.error as error:
        return _failure("corner_detection_failed", str(error))
    if not detection.found or detection.corners is None:
        return _failure(
            "board_not_detected",
            "checkerboard corners were not detected",
            method=detection.method,
        )

    corners = np.ascontiguousarray(
        np.asarray(detection.corners, dtype=np.float32).reshape(-1, 2)
    )
    return estimate_session_ground_extrinsic_from_corners(
        corners,
        {"K": K, "D": D},
        board,
        detection_method=detection.method,
    )


__all__ = [
    "BoardConfig",
    "CameraIntrinsics",
    "ChessboardCornerDetection",
    "SessionGroundBoardConfig",
    "SessionGroundExtrinsic",
    "build_camera_to_ground_transform",
    "create_object_points",
    "detect_corners",
    "estimate_session_ground_extrinsic",
    "estimate_session_ground_extrinsic_from_corners",
]
