"""HIKROBOT MVS adapter for the MV-CS050-60GM GigE camera."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from .models import CameraConfig, CameraDeviceInfo, CapturedFrame


EXPECTED_MODEL = "MV-CS050-60GM"
_DLL_HANDLES: list[object] = []


@dataclass(frozen=True, slots=True)
class _DeviceRecord:
    info: CameraDeviceInfo
    raw: object


def _mvs_python_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("MVS_PYTHON_PATH")
    if configured:
        candidates.append(Path(configured))
    for name in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
        root = os.environ.get(name)
        if root:
            candidates.append(
                Path(root) / "MVS" / "Development" / "Samples" / "Python" / "MvImport"
            )
    return candidates


def _configure_dll_path() -> None:
    runtime = "Win64_x64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "Win32_i86"
    for name in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
        root = os.environ.get(name)
        if not root:
            continue
        directory = Path(root) / "Common Files" / "MVS" / "Runtime" / runtime
        if not directory.is_dir():
            continue
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
        elif str(directory) not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"


def load_mvs_sdk() -> ModuleType:
    """Load the official wrapper lazily so help, tests, and simulation still work."""
    _configure_dll_path()
    errors: list[str] = []
    for directory in [None, *_mvs_python_candidates()]:
        if directory is not None:
            if not directory.is_dir():
                continue
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        try:
            sdk = importlib.import_module("MvCameraControl_class")
        except Exception as error:
            errors.append(f"{directory or 'sys.path'}: {error}")
            continue
        required = (
            "MvCamera",
            "MV_CC_DEVICE_INFO_LIST",
            "MV_CC_DEVICE_INFO",
            "MV_FRAME_OUT",
            "MV_GIGE_DEVICE",
            "MV_ACCESS_Exclusive",
        )
        missing = [name for name in required if not hasattr(sdk, name)]
        if missing:
            errors.append(f"{directory or 'sys.path'}: 缺少 {', '.join(missing)}")
            continue
        return sdk
    detail = "; ".join(errors) or "没有找到 MVS Python 目录"
    raise RuntimeError(
        "无法加载海康 MVS Python SDK。请安装包含 Development/Samples 的官方 "
        "MVS，或把环境变量 MVS_PYTHON_PATH 指向 MvImport 目录。"
        f"\n详情: {detail}"
    )


def _check(operation: str, result: int) -> None:
    if int(result) != 0:
        code = int(result) & 0xFFFFFFFF
        raise RuntimeError(f"{operation} 失败，MVS 状态 0x{code:08X}")


def _decode(value: object) -> str:
    raw = bytes(value).split(b"\0", 1)[0]
    for encoding in ("utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1", errors="replace")


def _ipv4(value: int) -> str:
    return ".".join(str((int(value) >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def _enumerate_records(sdk: ModuleType) -> list[_DeviceRecord]:
    devices = sdk.MV_CC_DEVICE_INFO_LIST()
    _check(
        "枚举 GigE 相机",
        sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE, devices),
    )
    records: list[_DeviceRecord] = []
    for index in range(int(devices.nDeviceNum)):
        pointer = devices.pDeviceInfo[index]
        if not pointer:
            continue
        raw = ctypes.cast(pointer, ctypes.POINTER(sdk.MV_CC_DEVICE_INFO)).contents
        if int(raw.nTLayerType) != int(sdk.MV_GIGE_DEVICE):
            continue
        gige = raw.SpecialInfo.stGigEInfo
        records.append(
            _DeviceRecord(
                CameraDeviceInfo(
                    model=_decode(gige.chModelName),
                    serial_number=_decode(gige.chSerialNumber),
                    ip_address=_ipv4(gige.nCurrentIp),
                ),
                raw,
            )
        )
    return records


def list_devices() -> list[CameraDeviceInfo]:
    sdk = load_mvs_sdk()
    return [record.info for record in _enumerate_records(sdk)]


def _select(records: list[_DeviceRecord], serial_number: str) -> _DeviceRecord:
    matching = [
        record
        for record in records
        if record.info.model == EXPECTED_MODEL
        and (not serial_number or record.info.serial_number == serial_number)
    ]
    if len(matching) == 1:
        return matching[0]
    discovered = ", ".join(record.info.display_name for record in records) or "无"
    if not matching:
        raise RuntimeError(f"未找到 {EXPECTED_MODEL}。已发现: {discovered}")
    raise RuntimeError(f"发现多台 {EXPECTED_MODEL}，请选择序列号。已发现: {discovered}")


def _new_struct(sdk: ModuleType, *names: str) -> object:
    for name in names:
        cls = getattr(sdk, name, None)
        if cls is not None:
            return cls()
    raise RuntimeError(f"MVS Python SDK 缺少结构: {' / '.join(names)}")


def _get_int(camera: object, sdk: ModuleType, name: str) -> object:
    value = _new_struct(sdk, "MVCC_INTVALUE_EX", "MVCC_INTVALUE")
    getter = getattr(camera, "MV_CC_GetIntValueEx", None) or getattr(
        camera, "MV_CC_GetIntValue", None
    )
    if getter is None:
        raise RuntimeError("MVS SDK 缺少整数节点读取接口")
    _check(f"读取 {name}", getter(name, value))
    return value


def _set_int(camera: object, sdk: ModuleType, name: str, requested: int) -> int:
    limits = _get_int(camera, sdk, name)
    minimum, maximum = int(limits.nMin), int(limits.nMax)
    increment = max(1, int(limits.nInc))
    if requested < minimum or requested > maximum:
        raise ValueError(f"{name}={requested} 超出 [{minimum}, {maximum}]")
    if (requested - minimum) % increment:
        raise ValueError(f"{name}={requested} 未按步进 {increment} 对齐")
    setter = getattr(camera, "MV_CC_SetIntValueEx", None) or getattr(
        camera, "MV_CC_SetIntValue", None
    )
    if setter is None:
        raise RuntimeError("MVS SDK 缺少整数节点设置接口")
    _check(f"设置 {name}", setter(name, int(requested)))
    return int(_get_int(camera, sdk, name).nCurValue)


def _set_float(camera: object, sdk: ModuleType, name: str, requested: float) -> float:
    value = _new_struct(sdk, "MVCC_FLOATVALUE")
    _check(f"读取 {name}", camera.MV_CC_GetFloatValue(name, value))
    if requested < float(value.fMin) or requested > float(value.fMax):
        raise ValueError(
            f"{name}={requested:g} 超出 [{float(value.fMin):g}, {float(value.fMax):g}]"
        )
    _check(f"设置 {name}", camera.MV_CC_SetFloatValue(name, float(requested)))
    _check(f"回读 {name}", camera.MV_CC_GetFloatValue(name, value))
    return float(value.fCurValue)


def _set_enum(camera: object, name: str, value: str) -> None:
    setter = getattr(camera, "MV_CC_SetEnumValueByString", None)
    if setter is None:
        raise RuntimeError("MVS SDK 缺少枚举节点设置接口")
    _check(f"设置 {name}={value}", setter(name, value))


def _camera_ticks(info: object) -> int | None:
    if hasattr(info, "nDevTimeStampHigh") and hasattr(info, "nDevTimeStampLow"):
        return (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)
    return int(info.nDevTimeStamp) if hasattr(info, "nDevTimeStamp") else None


def _copy_frame_payload(
    address: int, height: int, width: int, dtype: np.dtype
) -> np.ndarray:
    """Copy an SDK-owned image buffer once into NumPy-owned memory."""
    image = np.empty((height, width), dtype=dtype)
    ctypes.memmove(image.ctypes.data, address, image.nbytes)
    return image


def _apply_config(
    camera: object, sdk: ModuleType, config: CameraConfig
) -> CameraConfig:
    _set_enum(camera, "PixelFormat", config.pixel_format)
    exposure = _set_float(camera, sdk, "ExposureTime", config.exposure_us)
    gain = _set_float(camera, sdk, "Gain", config.gain_db)
    _set_int(camera, sdk, "OffsetX", 0)
    _set_int(camera, sdk, "OffsetY", 0)
    width = _set_int(camera, sdk, "Width", config.width)
    height = _set_int(camera, sdk, "Height", config.height)
    offset_x = _set_int(camera, sdk, "OffsetX", config.offset_x)
    offset_y = _set_int(camera, sdk, "OffsetY", config.offset_y)
    return CameraConfig(
        exposure_us=exposure,
        gain_db=gain,
        pixel_format=config.pixel_format,
        offset_x=offset_x,
        offset_y=offset_y,
        width=width,
        height=height,
        timeout_ms=config.timeout_ms,
    )


class MvsCameraSession:
    """Exclusive continuous-acquisition session for one HIKROBOT camera."""

    def __init__(
        self,
        sdk: ModuleType,
        camera: object,
        device: CameraDeviceInfo,
        config: CameraConfig,
    ) -> None:
        self.sdk = sdk
        self.camera = camera
        self.device = device
        self.config = config
        self._started = False
        self._closed = False

    @classmethod
    def open(cls, serial_number: str, config: CameraConfig) -> "MvsCameraSession":
        sdk = load_mvs_sdk()
        selected = _select(_enumerate_records(sdk), serial_number)
        camera = sdk.MvCamera()
        created = opened = False
        try:
            _check("创建相机句柄", camera.MV_CC_CreateHandle(selected.raw))
            created = True
            _check("打开相机", camera.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0))
            opened = True
            packet_size = int(camera.MV_CC_GetOptimalPacketSize())
            if packet_size > 0:
                _check(
                    "设置 GigE 包大小",
                    camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size),
                )
            _set_enum(camera, "AcquisitionMode", "Continuous")
            _set_enum(camera, "TriggerMode", "Off")
            _set_enum(camera, "ExposureAuto", "Off")
            _set_enum(camera, "GainAuto", "Off")
            applied = _apply_config(camera, sdk, config)
            return cls(sdk, camera, selected.info, applied)
        except Exception:
            if opened:
                camera.MV_CC_CloseDevice()
            if created:
                camera.MV_CC_DestroyHandle()
            raise

    def configure(self, config: CameraConfig) -> CameraConfig:
        if self._closed:
            raise RuntimeError("相机已经关闭")
        if self._started:
            raise RuntimeError("请先停止取流，再修改采集参数")
        self.config = _apply_config(self.camera, self.sdk, config)
        return self.config

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("相机已经关闭")
        if not self._started:
            _check("开始取流", self.camera.MV_CC_StartGrabbing())
            self._started = True

    def get_frame(self, timeout_ms: int | None = None) -> CapturedFrame:
        if not self._started:
            raise RuntimeError("相机尚未开始取流")
        output = self.sdk.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(output), 0, ctypes.sizeof(output))
        _check(
            "获取图像",
            self.camera.MV_CC_GetImageBuffer(
                output, timeout_ms if timeout_ms is not None else self.config.timeout_ms
            ),
        )
        try:
            info = output.stFrameInfo
            width, height = int(info.nWidth), int(info.nHeight)
            if (width, height) != (self.config.width, self.config.height):
                raise RuntimeError(
                    f"相机返回 {width}×{height}，期望 "
                    f"{self.config.width}×{self.config.height}"
                )
            dtype = np.dtype(np.uint8 if self.config.pixel_format == "Mono8" else "<u2")
            expected = width * height * dtype.itemsize
            if int(info.nFrameLen) < expected:
                raise RuntimeError(f"图像负载不完整: {int(info.nFrameLen)} < {expected}")
            image = _copy_frame_payload(output.pBufAddr, height, width, dtype)
            if self.config.pixel_format == "Mono12" and int(image.max()) > 4095:
                raise RuntimeError("Mono12 数据超过 4095，请确认未选择 Mono12Packed")
            return CapturedFrame(
                image=image,
                camera_frame_number=int(info.nFrameNum),
                camera_timestamp_ticks=_camera_ticks(info),
                host_timestamp_ns=time.time_ns(),
                host_monotonic_ns=time.perf_counter_ns(),
                offset_x=self.config.offset_x,
                offset_y=self.config.offset_y,
            )
        finally:
            _check("释放图像缓存", self.camera.MV_CC_FreeImageBuffer(output))

    def stop(self) -> None:
        if self._started:
            _check("停止取流", self.camera.MV_CC_StopGrabbing())
            self._started = False

    def close(self) -> None:
        if self._closed:
            return
        if self._started:
            self.stop()
        close_result = self.camera.MV_CC_CloseDevice()
        destroy_result = self.camera.MV_CC_DestroyHandle()
        self._closed = True
        _check("关闭相机", close_result)
        _check("销毁相机句柄", destroy_result)
