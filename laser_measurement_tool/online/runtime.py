"""Small threading primitives for latest-frame real-time processing."""

from __future__ import annotations

import threading
import time

from .models import CapturedFrame


class LatestFrameSlot:
    """Capacity-one slot that replaces stale frames instead of building latency."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: CapturedFrame | None = None
        self._closed = False
        self.overwritten = 0

    def put(self, frame: CapturedFrame) -> None:
        with self._condition:
            if self._closed:
                return
            if self._frame is not None:
                self.overwritten += 1
            self._frame = frame
            self._condition.notify()

    def take(self, timeout_s: float = 0.2) -> CapturedFrame | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._frame is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            frame = self._frame
            self._frame = None
            return frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._frame = None
            self._condition.notify_all()
