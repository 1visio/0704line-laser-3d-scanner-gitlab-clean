"""Tests for the hardware-neutral scan-axis abstraction."""

import unittest

from scan.axis import ScanAxis, SimulatedScanAxis


class SimulatedScanAxisTests(unittest.TestCase):
    def test_simulated_axis_implements_protocol_and_accepts_boundaries(self) -> None:
        axis = SimulatedScanAxis(min_angle_deg=-10.0, max_angle_deg=15.0)

        self.assertIsInstance(axis, ScanAxis)
        self.assertEqual(axis.get_angle(), 0.0)
        axis.move_to(-10.0)
        self.assertEqual(axis.get_angle(), -10.0)
        axis.move_to(15.0)
        self.assertEqual(axis.get_angle(), 15.0)

    def test_home_sets_zero_and_stop_keeps_last_angle(self) -> None:
        axis = SimulatedScanAxis(-20.0, 20.0)
        axis.move_to(8.5)
        axis.stop()
        self.assertEqual(axis.get_angle(), 8.5)

        axis.home()
        self.assertEqual(axis.get_angle(), 0.0)

    def test_out_of_range_command_is_rejected_without_changing_angle(self) -> None:
        axis = SimulatedScanAxis(-5.0, 5.0)
        axis.move_to(2.0)

        with self.assertRaises(ValueError):
            axis.move_to(-5.1)
        self.assertEqual(axis.get_angle(), 2.0)
        with self.assertRaises(ValueError):
            axis.move_to(5.1)
        self.assertEqual(axis.get_angle(), 2.0)

    def test_invalid_limits_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SimulatedScanAxis(5.0, -5.0)
        with self.assertRaises(ValueError):
            SimulatedScanAxis(1.0, 5.0)

    def test_measured_angle_matches_commanded_angle(self) -> None:
        axis = SimulatedScanAxis(-30.0, 30.0)
        commanded = -12.5
        axis.move_to(commanded)

        self.assertEqual(axis.get_angle(), commanded)


if __name__ == "__main__":
    unittest.main()
