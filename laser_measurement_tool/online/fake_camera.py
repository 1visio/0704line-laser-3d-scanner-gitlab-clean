"""Deterministic synthetic camera used when hardware/SDK is unavailable."""

from __future__ import annotations

import time

import numpy as np

from .models import CameraConfig, CameraDeviceInfo, CapturedFrame


class SyntheticCameraSession:
    device = CameraDeviceInfo("SIMULATED-MV-CS050-60GM", "SIM-001", "127.0.0.1")

    def __init__(self, config: CameraConfig, target_fps: float = 15.0) -> None:
        self.config = config
        self._period_s = 1.0 / target_fps
        self._started = False
        self._frame_number = 0
        self._next_frame_time = 0.0
        self._rows = np.arange(config.height, dtype=np.float32)[:, None]
        self._columns = np.arange(config.width, dtype=np.float32)[None, :]

    def configure(self, config: CameraConfig) -> CameraConfig:
        if self._started:
            raise RuntimeError("模拟相机取流时不能修改采集参数")
        self.config = config
        self._rows = np.arange(config.height, dtype=np.float32)[:, None]
        self._columns = np.arange(config.width, dtype=np.float32)[None, :]
        return self.config

    def start(self) -> None:
        self._started = True
        self._next_frame_time = time.monotonic()

    def get_frame(self, timeout_ms: int | None = None) -> CapturedFrame:
        if not self._started:
            raise RuntimeError("模拟相机尚未开始取流")
        delay = self._next_frame_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_frame_time = max(self._next_frame_time + self._period_s, time.monotonic())
        self._frame_number += 1
        height, width = self.config.height, self.config.width
        centre = height * 0.5 + 4.0 * np.sin(
            self._columns / 170.0 + self._frame_number / 14.0
        )
        stripe = np.exp(-0.5 * ((self._rows - centre) / 1.8) ** 2)
        if self.config.pixel_format == "Mono8":
            image = np.clip(12.0 + 230.0 * stripe, 0, 255).astype(np.uint8)
        else:
            image = np.clip(180.0 + 3600.0 * stripe, 0, 4095).astype(np.uint16)
        return CapturedFrame(
            image=np.ascontiguousarray(image),
            camera_frame_number=self._frame_number,
            camera_timestamp_ticks=self._frame_number * 1000,
            host_timestamp_ns=time.time_ns(),
            host_monotonic_ns=time.perf_counter_ns(),
            offset_x=self.config.offset_x,
            offset_y=self.config.offset_y,
        )

    def stop(self) -> None:
        self._started = False

    def close(self) -> None:
        self._started = False
