"""Synthetic end-to-end tests for Stage-1 scan kinematics and accumulation.

These tests intentionally bypass cameras, image extraction, calibration files,
and reconstruction.  Their expected values come from independent rigid-body
formulas so a kinematics implementation cannot validate itself.
"""

from __future__ import annotations

import unittest

import numpy as np

from scan.accumulator import ScanAccumulator
from scan.kinematics import transform_points_camera_to_scan
from scan.models import ScanProfile


def _ideal_camera_line() -> np.ndarray:
    x = np.linspace(-50.0, 50.0, 101, dtype=np.float64)
    return np.column_stack(
        (x, np.zeros_like(x), np.full_like(x, 700.0))
    )


def _independent_rodrigues(axis_direction: np.ndarray, angle_deg: float) -> np.ndarray:
    """Independent right-handed Rodrigues implementation for test truth."""
    axis = np.asarray(axis_direction, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    angle_rad = np.deg2rad(float(angle_deg))
    identity = np.eye(3, dtype=np.float64)
    return (
        np.cos(angle_rad) * identity
        + (1.0 - np.cos(angle_rad)) * np.outer(axis, axis)
        + np.sin(angle_rad) * skew
    )


def _independent_expected(
    points_camera: np.ndarray,
    angle_deg: float,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    zero_offset_deg: float = 0.0,
    transform: np.ndarray | None = None,
) -> np.ndarray:
    """Compute expected points without calling the production kinematics."""
    points = np.asarray(points_camera, dtype=np.float64)
    axis_point = np.asarray(axis_point, dtype=np.float64)
    zero_transform = np.eye(4, dtype=np.float64) if transform is None else transform
    points_zero = points @ zero_transform[:3, :3].T + zero_transform[:3, 3]
    rotation = _independent_rodrigues(
        axis_direction,
        float(angle_deg) + float(zero_offset_deg),
    )
    return (points_zero - axis_point) @ rotation.T + axis_point


def _build_profiles(
    points_camera: np.ndarray,
    angles_deg: tuple[float, ...],
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    transform: np.ndarray,
) -> tuple[ScanAccumulator, tuple[ScanProfile, ...]]:
    pixels_uv = np.column_stack(
        (
            points_camera[:, 0],
            np.zeros(len(points_camera), dtype=np.float64),
        )
    )
    accumulator = ScanAccumulator()
    for frame_index, angle_deg in enumerate(angles_deg):
        points_scan = transform_points_camera_to_scan(
            points_camera,
            angle_deg,
            axis_point,
            axis_direction,
            0.0,
            transform,
        )
        accumulator.add_profile(
            ScanProfile(
                frame_index=frame_index,
                angle_deg=angle_deg,
                pixels_uv=pixels_uv,
                points_camera=points_camera,
                points_scan=points_scan,
            )
        )
    return accumulator, accumulator.profiles


class SyntheticScanIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points_camera = _ideal_camera_line()
        self.angles_deg = tuple(
            float(value) for value in np.linspace(-6.0, 6.0, 61)
        )
        self.identity = np.eye(4, dtype=np.float64)

    def _assert_integrity(
        self,
        accumulator: ScanAccumulator,
        profiles: tuple[ScanProfile, ...],
    ) -> None:
        self.assertEqual(len(profiles), 61)
        self.assertEqual(accumulator.combined_points.shape, (61 * 101, 3))
        self.assertTrue(np.isfinite(accumulator.combined_points).all())
        for frame_index, profile in enumerate(profiles):
            self.assertEqual(profile.frame_index, frame_index)
            self.assertEqual(profile.points_camera.shape, (101, 3))
            self.assertEqual(profile.points_scan.shape, (101, 3))
            self.assertAlmostEqual(profile.angle_deg, self.angles_deg[frame_index])
            self.assertTrue(np.isfinite(profile.points_scan).all())
            self.assertEqual(len(profile.pixels_uv), 101)
            np.testing.assert_allclose(
                np.linalg.norm(np.diff(profile.points_scan, axis=0), axis=1),
                np.linalg.norm(np.diff(self.points_camera, axis=0), axis=1),
                rtol=0.0,
                atol=1.0e-10,
            )

    def test_origin_x_axis_full_scan_and_independent_truth(self) -> None:
        axis_point = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        axis_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        accumulator, profiles = _build_profiles(
            self.points_camera,
            self.angles_deg,
            axis_point,
            axis_direction,
            self.identity,
        )

        self._assert_integrity(accumulator, profiles)
        self.assertEqual(len(accumulator.profiles), 61)
        np.testing.assert_allclose(
            accumulator.combined_points,
            np.concatenate([profile.points_scan for profile in profiles], axis=0),
            rtol=0.0,
            atol=0.0,
        )
        for frame_index, point_index in ((0, 0), (30, 50), (60, 100)):
            expected = _independent_expected(
                self.points_camera[[point_index]],
                self.angles_deg[frame_index],
                axis_point,
                axis_direction,
            )
            np.testing.assert_allclose(
                profiles[frame_index].points_scan[[point_index]],
                expected,
                rtol=0.0,
                atol=1.0e-12,
            )

    def test_non_origin_x_axis_keeps_integrity_and_matches_analytic_point(self) -> None:
        axis_point = np.array([20.0, 30.0, 10.0], dtype=np.float64)
        axis_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        accumulator, profiles = _build_profiles(
            self.points_camera,
            self.angles_deg,
            axis_point,
            axis_direction,
            self.identity,
        )

        self._assert_integrity(accumulator, profiles)
        point_index = 50
        frame_index = 60
        expected = _independent_expected(
            self.points_camera[[point_index]],
            6.0,
            axis_point,
            axis_direction,
        )
        np.testing.assert_allclose(
            profiles[frame_index].points_scan[[point_index]],
            expected,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_zero_transform_precedes_axis_rotation(self) -> None:
        axis_point = np.zeros(3, dtype=np.float64)
        axis_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        zero_transform = np.array(
            [
                [0.0, -1.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        accumulator, profiles = _build_profiles(
            self.points_camera,
            self.angles_deg,
            axis_point,
            axis_direction,
            zero_transform,
        )

        self._assert_integrity(accumulator, profiles)
        point_index = 50
        frame_index = 45  # +3 degrees
        expected = _independent_expected(
            self.points_camera[[point_index]],
            self.angles_deg[frame_index],
            axis_point,
            axis_direction,
            transform=zero_transform,
        )
        np.testing.assert_allclose(
            profiles[frame_index].points_scan[[point_index]],
            expected,
            rtol=0.0,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
