"""Deterministic accumulation of scan profiles."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .models import ScanProfile, ScanResult


class ScanAccumulator:
    """Accumulate scan-frame points without registration or filtering.

    Profiles are retained in insertion order.  The combined cloud is formed
    only by concatenating each profile's ``points_scan`` array, so every point
    stays in the scan frame already assigned by the caller's kinematics.
    """

    __slots__ = ("_profiles",)

    def __init__(self, profiles: Iterable[ScanProfile] | None = None) -> None:
        self._profiles: list[ScanProfile] = []
        if profiles is not None:
            for profile in profiles:
                self.add_profile(profile)

    @property
    def profiles(self) -> tuple[ScanProfile, ...]:
        """Return accumulated profiles in their original insertion order."""
        return tuple(self._profiles)

    @property
    def combined_points(self) -> np.ndarray:
        """Return the concatenated scan-frame points, or an empty ``(0, 3)`` cloud."""
        if not self._profiles:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(
            [profile.points_scan for profile in self._profiles],
            axis=0,
        )

    def add_profile(self, profile: ScanProfile) -> None:
        """Append one validated profile while preserving insertion order."""
        if not isinstance(profile, ScanProfile):
            raise TypeError("profile 必须是 ScanProfile")
        self._profiles.append(profile)

    def clear(self) -> None:
        """Remove all profiles and reset the accumulated cloud to empty."""
        self._profiles.clear()

    def to_result(self) -> ScanResult:
        """Build an immutable result snapshot from the accumulated profiles."""
        return ScanResult(self.profiles, self.combined_points)


__all__ = ["ScanAccumulator"]
