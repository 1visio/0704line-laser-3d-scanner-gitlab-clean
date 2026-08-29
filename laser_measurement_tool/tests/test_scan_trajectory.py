"""Analytic tests for Stage-1 scan trajectory generation."""

import unittest

from scan.config import ScanConfigError, ScanTrajectoryConfig


class ScanTrajectoryTests(unittest.TestCase):
    def test_forward_trajectory_has_inclusive_61_angles(self) -> None:
        trajectory = ScanTrajectoryConfig(-6.0, 6.0, 0.2)

        angles = trajectory.angles_deg

        self.assertEqual(len(angles), 61)
        self.assertEqual(angles[0], -6.0)
        self.assertEqual(angles[-1], 6.0)
        self.assertTrue(all(left < right for left, right in zip(angles, angles[1:])))

    def test_reverse_trajectory_has_inclusive_61_angles(self) -> None:
        trajectory = ScanTrajectoryConfig(6.0, -6.0, -0.2)

        angles = trajectory.angles_deg

        self.assertEqual(len(angles), 61)
        self.assertEqual(angles[0], 6.0)
        self.assertEqual(angles[-1], -6.0)
        self.assertTrue(all(left > right for left, right in zip(angles, angles[1:])))

    def test_wrong_step_direction_is_rejected(self) -> None:
        with self.assertRaises(ScanConfigError):
            ScanTrajectoryConfig(-6.0, 6.0, -0.2)


if __name__ == "__main__":
    unittest.main()
