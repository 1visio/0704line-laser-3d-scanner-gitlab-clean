"""Measure or preview raw camera acquisition without laser processing."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOL_ROOT))

from online.camera_backend import (  # noqa: E402
    available_camera_backends,
    get_camera_backend,
)
from online.models import CameraConfig, CameraSession, CapturedFrame  # noqa: E402
from online.runtime import LatestFrameSlot  # noqa: E402


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test or preview raw camera acquisition without laser processing."
    )
    parser.add_argument("--list", action="store_true", help="list cameras and exit")
    parser.add_argument(
        "--camera-backend",
        choices=available_camera_backends(),
        default="mvs",
        help="camera backend (mvs or daheng)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show a camera-only window; press Q or Esc to exit",
    )
    parser.add_argument("--serial", default="", help="camera serial number")
    parser.add_argument(
        "--duration",
        type=_non_negative_float,
        default=10.0,
        help="run time in seconds; use 0 for an unlimited preview",
    )
    parser.add_argument("--warmup", type=_non_negative_int, default=5)
    parser.add_argument("--pixel-format", choices=("Mono8", "Mono12"), default="Mono8")
    parser.add_argument("--exposure-us", type=_positive_float, default=600.0)
    parser.add_argument("--gain-db", type=float, default=0.0)
    parser.add_argument("--offset-x", type=_non_negative_int, default=0)
    parser.add_argument("--offset-y", type=_non_negative_int, default=880)
    parser.add_argument("--width", type=_positive_int, default=2448, metavar="PX")
    parser.add_argument("--height", type=_positive_int, default=300, metavar="PX")
    parser.add_argument("--timeout-ms", type=_positive_int, default=2000, metavar="MS")
    parser.add_argument("--save-first", type=Path, help="optionally save the first measured frame")
    return parser


def _print_devices(backend) -> list[object]:
    devices = backend.list_devices()
    print(f"Discovered cameras: {len(devices)}")
    for device in devices:
        print(
            f"  model={device.model}  serial={device.serial_number}  "
            f"ip={device.ip_address or '-'}"
        )
    return devices


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _save_frame(path: Path, image: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save frame to {path}")


@dataclass(slots=True)
class _PreviewState:
    started_at: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    captured: int = 0
    camera_gaps: int = 0
    last_frame_number: int | None = None
    total_bytes: int = 0
    latency_sum_ms: float = 0.0
    latency_max_ms: float = 0.0
    error: Exception | None = None

    def add_frame(self, frame: CapturedFrame, latency_ms: float) -> None:
        with self.lock:
            if self.last_frame_number is not None:
                self.camera_gaps += max(
                    0, frame.camera_frame_number - self.last_frame_number - 1
                )
            self.last_frame_number = frame.camera_frame_number
            self.captured += 1
            self.total_bytes += frame.image.nbytes
            self.latency_sum_ms += latency_ms
            self.latency_max_ms = max(self.latency_max_ms, latency_ms)

    def set_error(self, error: Exception) -> None:
        with self.lock:
            self.error = error

    def snapshot(self) -> tuple[int, int, int, float, float, Exception | None]:
        with self.lock:
            return (
                self.captured,
                self.camera_gaps,
                self.total_bytes,
                self.latency_sum_ms,
                self.latency_max_ms,
                self.error,
            )


def _capture_preview_frames(
    session: CameraSession,
    slot: LatestFrameSlot,
    state: _PreviewState,
    stop_event: threading.Event,
) -> None:
    try:
        while not stop_event.is_set():
            before = time.perf_counter_ns()
            frame = session.get_frame()
            latency_ms = (time.perf_counter_ns() - before) / 1e6
            state.add_frame(frame, latency_ms)
            slot.put(frame)
    except Exception as error:
        if not stop_event.is_set():
            state.set_error(error)
    finally:
        slot.close()


def _display_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image.astype(np.float32) * (255.0 / 4095.0), 0, 255).astype(
        np.uint8
    )


def _preview_canvas(
    frame: CapturedFrame,
    capture_fps: float,
    display_fps: float,
    camera_gaps: int,
    display_skips: int,
) -> np.ndarray:
    import cv2

    image = _display_image(frame.image)
    bar_height = 54
    canvas = np.zeros((image.shape[0] + bar_height, image.shape[1]), dtype=np.uint8)
    canvas[: image.shape[0]] = image
    first_line = f"Capture {capture_fps:.2f} fps    Display {display_fps:.2f} fps"
    second_line = (
        f"Camera gaps {camera_gaps}    Display skips {display_skips}"
        "    Q / Esc: quit"
    )
    cv2.putText(
        canvas,
        first_line,
        (12, image.shape[0] + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        255,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        second_line,
        (12, image.shape[0] + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        255,
        1,
        cv2.LINE_AA,
    )
    return canvas


def _run_preview(
    session: CameraSession, args: argparse.Namespace
) -> tuple[np.ndarray | None, float]:
    import cv2

    window_name = f"{args.camera_backend} Camera Preview (raw frames only)"
    slot = LatestFrameSlot()
    stop_event = threading.Event()
    state = _PreviewState(started_at=time.perf_counter())
    worker = threading.Thread(
        target=_capture_preview_frames,
        args=(session, slot, state, stop_event),
        name="camera-preview-acquisition",
        daemon=True,
    )
    displayed = 0
    first_image: np.ndarray | None = None
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    preview_width = min(args.width, 1600)
    preview_height = max(180, round((args.height + 54) * preview_width / args.width))
    cv2.resizeWindow(window_name, preview_width, preview_height)
    worker.start()
    try:
        while True:
            now = time.perf_counter()
            elapsed = max(now - state.started_at, 1e-9)
            if args.duration > 0 and elapsed >= args.duration:
                break
            frame = slot.take(0.05)
            captured, gaps, _, _, _, error = state.snapshot()
            if error is not None:
                raise error
            if frame is not None:
                displayed += 1
                if first_image is None:
                    first_image = frame.image.copy()
                canvas = _preview_canvas(
                    frame,
                    captured / elapsed,
                    displayed / elapsed,
                    gaps,
                    slot.overwritten,
                )
                cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stop_event.set()
        slot.close()
        worker.join((session.config.timeout_ms / 1000.0) + 1.0)
        cv2.destroyWindow(window_name)

    elapsed = max(time.perf_counter() - state.started_at, 1e-9)
    captured, gaps, total_bytes, latency_sum, latency_max, error = state.snapshot()
    if error is not None:
        raise error
    print("\nCamera-only preview result")
    print(f"  captured frames:    {captured}")
    print(f"  displayed frames:   {displayed}")
    print(f"  elapsed:            {elapsed:.3f} s")
    print(f"  acquisition fps:    {captured / elapsed:.2f}")
    print(f"  display fps:        {displayed / elapsed:.2f}")
    print(f"  camera frame gaps:  {gaps}")
    print(f"  display skips:      {slot.overwritten}")
    print(f"  payload throughput: {total_bytes / elapsed / 1e6:.2f} MB/s")
    if captured:
        print(
            "  get_frame latency:  "
            f"mean={latency_sum / captured:.2f} ms, max={latency_max:.2f} ms"
        )
    return first_image, elapsed


def run(args: argparse.Namespace) -> int:
    backend = get_camera_backend(args.camera_backend)
    devices = _print_devices(backend)
    if args.list:
        return 0
    if not devices:
        raise RuntimeError(f"No {args.camera_backend} camera was found")
    if not args.preview and args.duration == 0:
        raise ValueError("--duration must be greater than zero without --preview")

    config = CameraConfig(
        exposure_us=args.exposure_us,
        gain_db=args.gain_db,
        pixel_format=args.pixel_format,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        width=args.width,
        height=args.height,
        timeout_ms=args.timeout_ms,
    )
    session = backend.open_session(args.serial, config)
    try:
        print(
            f"Opened: model={session.device.model}  "
            f"serial={session.device.serial_number}  "
            f"ip={session.device.ip_address or '-'}"
        )
        print(f"Applied config: {session.config}")
        session.start()

        for _ in range(args.warmup):
            session.get_frame()

        if args.preview:
            first_image, _ = _run_preview(session, args)
            if args.save_first is not None and first_image is not None:
                _save_frame(args.save_first, first_image)
                print(f"  saved first frame:  {args.save_first.resolve()}")
            return 0

        latencies_ms: list[float] = []
        frame_numbers: list[int] = []
        host_times_ns: list[int] = []
        total_bytes = 0
        first_image: np.ndarray | None = None
        started = time.perf_counter()
        last_report = started
        while time.perf_counter() - started < args.duration:
            before = time.perf_counter_ns()
            frame = session.get_frame()
            after = time.perf_counter_ns()
            latencies_ms.append((after - before) / 1e6)
            frame_numbers.append(frame.camera_frame_number)
            host_times_ns.append(frame.host_monotonic_ns)
            total_bytes += frame.image.nbytes
            if first_image is None:
                first_image = frame.image.copy()
            now = time.perf_counter()
            if now - last_report >= 1.0:
                print(f"  frames={len(frame_numbers):5d}  fps={len(frame_numbers) / (now - started):6.2f}")
                last_report = now

        elapsed = time.perf_counter() - started
    finally:
        session.close()

    if not frame_numbers:
        raise RuntimeError("No frame was acquired")
    gaps = sum(
        max(0, current - previous - 1)
        for previous, current in zip(frame_numbers, frame_numbers[1:])
    )
    intervals_ms = [
        (current - previous) / 1e6
        for previous, current in zip(host_times_ns, host_times_ns[1:])
    ]

    print("\nRaw acquisition result")
    print(f"  frames:             {len(frame_numbers)}")
    print(f"  elapsed:            {elapsed:.3f} s")
    print(f"  acquisition fps:    {len(frame_numbers) / elapsed:.2f}")
    print(f"  camera frame gaps:  {gaps}")
    print(f"  payload throughput: {total_bytes / elapsed / 1e6:.2f} MB/s")
    print(
        "  get_frame latency:  "
        f"mean={np.mean(latencies_ms):.2f} ms, "
        f"p95={_percentile(latencies_ms, 95):.2f} ms, "
        f"max={max(latencies_ms):.2f} ms"
    )
    if intervals_ms:
        print(
            "  frame interval:     "
            f"mean={np.mean(intervals_ms):.2f} ms, "
            f"p95={_percentile(intervals_ms, 95):.2f} ms, "
            f"max={max(intervals_ms):.2f} ms"
        )
    assert first_image is not None
    print(
        "  image:              "
        f"shape={first_image.shape}, dtype={first_image.dtype}, "
        f"min={first_image.min()}, max={first_image.max()}, "
        f"mean={first_image.mean():.2f}"
    )
    if args.save_first is not None:
        _save_frame(args.save_first, first_image)
        print(f"  saved first frame:  {args.save_first.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nStopped by user", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Camera test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
