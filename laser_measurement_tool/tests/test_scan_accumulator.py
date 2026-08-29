"""Tests for deterministic scan-profile accumulation."""

import unittest

import numpy as np

from scan.accumulator import ScanAccumulator
from scan.models import ScanProfile


def _profile(frame_index: int, points_scan: np.ndarray) -> ScanProfile:
    points = np.asarray(points_scan, dtype=np.float64)
    return ScanProfile(
        frame_index=frame_index,
        angle_deg=float(frame_index),
        pixels_uv=np.zeros((len(points), 2), dtype=np.float64),
        points_camera=points.copy(),
        points_scan=points,
    )


class ScanAccumulatorTests(unittest.TestCase):
    def test_empty_accumulator_has_empty_three_column_cloud(self) -> None:
        accumulator = ScanAccumulator()

        self.assertEqual(accumulator.profiles, ())
        self.assertEqual(accumulator.combined_points.shape, (0, 3))
        self.assertEqual(accumulator.combined_points.dtype, np.float64)

    def test_profiles_and_points_keep_insertion_order(self) -> None:
        first = _profile(3, np.array([[1.0, 2.0, 3.0]]))
        second = _profile(7, np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        accumulator = ScanAccumulator()

        accumulator.add_profile(first)
        accumulator.add_profile(second)

        self.assertEqual(accumulator.profiles, (first, second))
        np.testing.assert_array_equal(
            accumulator.combined_points,
            np.concatenate((first.points_scan, second.points_scan), axis=0),
        )

    def test_constructor_accepts_profiles_in_order(self) -> None:
        first = _profile(0, np.zeros((0, 3)))
        second = _profile(1, np.array([[1.0, 0.0, 0.0]]))

        accumulator = ScanAccumulator([first, second])

        self.assertEqual(accumulator.profiles, (first, second))
        np.testing.assert_array_equal(accumulator.combined_points, second.points_scan)

    def test_clear_removes_profiles_and_points(self) -> None:
        accumulator = ScanAccumulator([_profile(0, np.ones((1, 3)))])

        accumulator.clear()

        self.assertEqual(accumulator.profiles, ())
        self.assertEqual(accumulator.combined_points.shape, (0, 3))

    def test_to_result_contains_ordered_profiles_and_combined_points(self) -> None:
        first = _profile(0, np.array([[1.0, 2.0, 3.0]]))
        second = _profile(1, np.array([[4.0, 5.0, 6.0]]))
        accumulator = ScanAccumulator([first, second])

        result = accumulator.to_result()

        self.assertEqual(result.profiles, (first, second))
        np.testing.assert_array_equal(
            result.points_scan,
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )

    def test_non_profile_is_rejected(self) -> None:
        accumulator = ScanAccumulator()

        with self.assertRaises(TypeError):
            accumulator.add_profile(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
