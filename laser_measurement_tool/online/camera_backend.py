"""Small registry that keeps the online UI independent from SDK adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .daheng_camera import DahengCameraSession, list_devices as list_daheng_devices
from .models import CameraConfig, CameraDeviceInfo, CameraSession
from .mvs_camera import MvsCameraSession, list_devices as list_mvs_devices


@dataclass(frozen=True, slots=True)
class CameraBackend:
    name: str
    display_name: str
    list_devices: Callable[[], list[CameraDeviceInfo]]
    open_session: Callable[[str, CameraConfig], CameraSession]


_BACKENDS = {
    "mvs": CameraBackend(
        name="mvs",
        display_name="HIKROBOT MVS",
        list_devices=list_mvs_devices,
        open_session=MvsCameraSession.open,
    ),
    "daheng": CameraBackend(
        name="daheng",
        display_name="大恒 Galaxy USB3",
        list_devices=list_daheng_devices,
        open_session=DahengCameraSession.open,
    ),
}


def get_camera_backend(name: str) -> CameraBackend:
    normalized = name.strip().lower()
    try:
        return _BACKENDS[normalized]
    except KeyError as error:
        available = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"未知相机 backend {name!r}，可选: {available}") from error


def available_camera_backends() -> tuple[str, ...]:
    return tuple(_BACKENDS)

