"""Tests for the offline scan data contracts."""

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from scan.models import ScanPose, ScanProfile, ScanResult


class ScanPoseTests(unittest.TestCase):
    def test_pose_accepts_finite_angles_and_normalizes_integer_index(self) -> None:
        pose = ScanPose(np.int64(3), np.float32(1.25), np.float64(1.2))

        self.assertEqual(pose.frame_index, 3)
        self.assertEqual(pose.angle_command_deg, 1.25)
        self.assertEqual(pose.angle_measured_deg, 1.2)

    def test_pose_rejects_invalid_index_or_angle(self) -> None:
        with self.assertRaises(ValueError):
            ScanPose(-1, 0.0, 0.0)
        with self.assertRaises(ValueError):
            ScanPose(0, np.nan, 0.0)
        with self.assertRaises(ValueError):
            ScanPose(0, 0.0, np.inf)
        with self.assertRaises(ValueError):
            ScanPose(0, 0.0, None)


class ScanProfileTests(unittest.TestCase):
    def _profile(self) -> ScanProfile:
        return ScanProfile(
            frame_index=2,
            angle_deg=5.0,
            pixels_uv=np.array([[10.0, 20.0], [11.0, 21.0]]),
            points_camera=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            points_scan=np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        )

    def test_profile_validates_shapes_and_keeps_points_aligned(self) -> None:
        profile = self._profile()

        self.assertEqual(profile.pixels_uv.shape, (2, 2))
        self.assertEqual(profile.points_camera.shape, (2, 3))
        self.assertEqual(profile.points_scan.shape, (2, 3))
        self.assertTrue(profile.points_camera.flags.c_contiguous)
        self.assertFalse(hasattr(profile, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            profile.angle_deg = 6.0  # type: ignore[misc]

    def test_profile_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            ScanProfile(
                0,
                0.0,
                np.zeros((2, 2)),
                np.zeros((1, 3)),
                np.zeros((2, 3)),
            )
        with self.assertRaises(ValueError):
            ScanProfile(
                0,
                0.0,
                np.zeros((2, 3)),
                np.zeros((2, 3)),
                np.zeros((2, 3)),
            )

    def test_profile_rejects_nonfinite_values(self) -> None:
        pixels = np.array([[10.0, np.nan]])
        with self.assertRaises(ValueError):
            ScanProfile(0, 0.0, pixels, np.zeros((1, 3)), np.zeros((1, 3)))


class ScanResultTests(unittest.TestCase):
    def test_result_normalizes_profile_sequence_and_validates_cloud(self) -> None:
        profile = ScanProfile(
            0,
            0.0,
            np.zeros((0, 2)),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
        )
        result = ScanResult(
            [profile],
            np.array([[1.0, 2.0, 3.0]]),
        )

        self.assertEqual(result.profiles, (profile,))
        self.assertEqual(result.points_scan.shape, (1, 3))

    def test_result_rejects_non_scan_profiles_or_nonfinite_cloud(self) -> None:
        with self.assertRaises(ValueError):
            ScanResult([object()], np.zeros((0, 3)))
        with self.assertRaises(ValueError):
            ScanResult([], np.array([[0.0, 1.0, np.inf]]))


if __name__ == "__main__":
    unittest.main()
