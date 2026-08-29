"""Offline scan data contracts and kinematics."""

from .axis import ScanAxis, SimulatedScanAxis
from .accumulator import ScanAccumulator
from .config import (
    ScanConfig,
    ScanConfigError,
    ScanKinematicsConfig,
    ScanOutputConfig,
    ScanTrajectoryConfig,
    load_scan_config,
)
from .kinematics import transform_points_camera_to_scan
from .models import ScanPose, ScanProfile, ScanResult

__all__ = [
    "ScanAxis",
    "ScanAccumulator",
    "ScanConfig",
    "ScanConfigError",
    "ScanKinematicsConfig",
    "ScanOutputConfig",
    "ScanPose",
    "ScanProfile",
    "ScanResult",
    "ScanSession",
    "ScanSessionError",
    "ScanSessionWriter",
    "ScanTrajectoryConfig",
    "SimulatedScanAxis",
    "load_scan_config",
    "transform_points_camera_to_scan",
    "write_scan_session",
]


def __getattr__(name: str):
    """Load session writers lazily to keep core scan imports lightweight."""
    session_names = {
        "ScanSession",
        "ScanSessionError",
        "ScanSessionWriter",
        "write_scan_session",
    }
    if name in session_names:
        from .session import (
            ScanSession,
            ScanSessionError,
            ScanSessionWriter,
            write_scan_session,
        )

        return {
            "ScanSession": ScanSession,
            "ScanSessionError": ScanSessionError,
            "ScanSessionWriter": ScanSessionWriter,
            "write_scan_session": write_scan_session,
        }[name]
    raise AttributeError(name)
