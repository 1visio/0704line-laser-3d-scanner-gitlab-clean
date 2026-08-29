"""Background lossless fixed-length frame recording with metadata."""

from __future__ import annotations

import csv
import queue
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2

from .models import CameraConfig, CapturedFrame


FRAME_FIELDS = (
    "filename",
    "camera_frame_number",
    "camera_timestamp_ticks",
    "host_timestamp_ns",
    "host_monotonic_ns",
    "frame_gap",
    "exposure_us",
    "gain_db",
    "pixel_format",
    "offset_x",
    "offset_y",
    "width",
    "height",
)


SHADOW_FIELDS = (
    "camera_frame_number",
    "host_timestamp_ns",
    "height_raw",
    "height_h1",
    "height_hb2",
    "active_height_correction",
    "active_height",
    "active_height_valid",
    "active_height_status",
    "q1",
    "q2",
    "q2_in_domain",
    "hb2_q2_status",
    "v_min",
    "v_median",
    "v_max",
    "point_count",
    "c1_clamp_status",
    "ground_reference_status",
)


@dataclass(frozen=True, slots=True)
class RecordingResult:
    output_dir: Path
    saved_frames: int
    detected_frame_gaps: int
    queue_drops: int


class FrameRecorder:
    """Save camera frames away from the acquisition thread."""

    def __init__(self, queue_capacity: int = 64) -> None:
        self._queue: queue.Queue[CapturedFrame | None] = queue.Queue(queue_capacity)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._active = False
        self._target = 0
        self._config: CameraConfig | None = None
        self._temp_dir: Path | None = None
        self._final_dir: Path | None = None
        self._result: RecordingResult | None = None
        self._error: BaseException | None = None
        self._queue_drops = 0
        self._cancelled = False
        self._shadow_lock = threading.Lock()
        self._shadow_rows: list[dict[str, object]] = []

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def result(self) -> RecordingResult | None:
        return self._result

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def start(self, root: str | Path, frame_count: int, config: CameraConfig) -> Path:
        if frame_count <= 0:
            raise ValueError("frame_count 必须为正数")
        with self._lock:
            if self._active:
                raise RuntimeError("已有录制任务正在运行")
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            output_root = Path(root).resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            final_dir = _next_available(output_root / f"recording_{stamp}")
            temp_dir = Path(tempfile.mkdtemp(prefix=".recording_", dir=output_root))
            self._target = frame_count
            self._config = config
            self._temp_dir = temp_dir
            self._final_dir = final_dir
            self._result = None
            self._error = None
            self._queue_drops = 0
            self._cancelled = False
            with self._shadow_lock:
                self._shadow_rows = []
            self._active = True
            self._thread = threading.Thread(
                target=self._writer_loop, name="frame-recorder", daemon=True
            )
            self._thread.start()
            return final_dir

    def enqueue(self, frame: CapturedFrame) -> bool:
        if not self.active:
            return False
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            self._queue_drops += 1
            return False

    def log_shadow(self, shadow: Mapping[str, object]) -> bool:
        """Keep processed height shadow metadata for the active recording.

        Acquisition and reconstruction run on different threads and may not
        produce one result for every captured frame.  Shadow rows therefore
        retain the processed camera frame number and are written as a separate
        best-effort stream instead of being forced into ``frames.csv``.
        """
        with self._lock:
            if not self._active:
                return False
            row = {field: shadow.get(field, "") for field in SHADOW_FIELDS}
            with self._shadow_lock:
                self._shadow_rows.append(row)
        return True

    def cancel(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._cancelled = True
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put_nowait(None)

    def wait(self, timeout_s: float | None = None) -> RecordingResult | None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
        if self._error is not None:
            raise RuntimeError(f"录制失败: {self._error}") from self._error
        return self._result

    def _writer_loop(self) -> None:
        assert self._temp_dir is not None
        assert self._final_dir is not None
        assert self._config is not None
        rows: list[dict[str, object]] = []
        previous_frame_number: int | None = None
        total_gaps = 0
        try:
            while len(rows) < self._target:
                frame = self._queue.get()
                if frame is None:
                    if self.cancelled:
                        return
                    raise RuntimeError("录制在达到目标帧数前被取消")
                suffix = ".png" if frame.image.dtype.name == "uint8" else ".tiff"
                filename = f"frame_{len(rows) + 1:06d}{suffix}"
                path = self._temp_dir / filename
                if not cv2.imwrite(str(path), frame.image):
                    raise OSError(f"无法保存图像: {path}")
                gap = (
                    0
                    if previous_frame_number is None
                    else max(0, frame.camera_frame_number - previous_frame_number - 1)
                )
                total_gaps += gap
                previous_frame_number = frame.camera_frame_number
                rows.append(
                    {
                        "filename": filename,
                        "camera_frame_number": frame.camera_frame_number,
                        "camera_timestamp_ticks": frame.camera_timestamp_ticks or "",
                        "host_timestamp_ns": frame.host_timestamp_ns,
                        "host_monotonic_ns": frame.host_monotonic_ns,
                        "frame_gap": gap,
                        "exposure_us": self._config.exposure_us,
                        "gain_db": self._config.gain_db,
                        "pixel_format": self._config.pixel_format,
                        "offset_x": frame.offset_x,
                        "offset_y": frame.offset_y,
                        "width": frame.image.shape[1],
                        "height": frame.image.shape[0],
                    }
                )
            with (self._temp_dir / "frames.csv").open(
                "x", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=FRAME_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self._shadow_lock:
                shadow_rows = list(self._shadow_rows)
            with (self._temp_dir / "height_shadow.csv").open(
                "x", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=SHADOW_FIELDS)
                writer.writeheader()
                writer.writerows(shadow_rows)
            self._temp_dir.rename(self._final_dir)
            self._result = RecordingResult(
                output_dir=self._final_dir,
                saved_frames=len(rows),
                detected_frame_gaps=total_gaps,
                queue_drops=self._queue_drops,
            )
        except BaseException as error:
            # Keep a failed recording's temporary directory.  In particular,
            # Windows may reject the final directory rename while an antivirus
            # or indexer still holds one of the newly written files.  Removing
            # the temporary directory here would lose otherwise complete
            # frames and metadata.
            if self._temp_dir is not None and not self.cancelled:
                self._error = RuntimeError(
                    "录制未能提交到正式目录；临时数据已保留。"
                    f"\n临时目录: {self._temp_dir}"
                    f"\n目标目录: {self._final_dir}"
                    f"\n原始错误: {error}"
                )
            else:
                self._error = error
        finally:
            if self.cancelled and self._temp_dir is not None:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            with self._lock:
                self._active = False


def _next_available(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("无法分配新的录制目录")
