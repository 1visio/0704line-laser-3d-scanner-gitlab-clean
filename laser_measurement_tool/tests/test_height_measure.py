"""Synthetic unit tests for measurement.height_measure."""

import unittest

import numpy as np

from measurement.height_measure import (
    MeasurementError,
    MeasurementParams,
    measure_height_line,
    measure_height_lines,
)


def _make_baseline(count: int = 100, noise: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(7)
    y = np.linspace(-40.0, 40.0, count)
    x = np.full(count, 1.0) + rng.normal(0.0, noise, count)
    z = rng.normal(0.0, noise, count)
    return np.column_stack([x, y, z])


def _make_height_line(
    count: int = 80,
    height: float = 12.5,
    length: float = 30.0,
    noise: float = 0.02,
) -> np.ndarray:
    rng = np.random.default_rng(11)
    s = np.linspace(0.0, length, count)
    x = 5.0 + s + rng.normal(0.0, noise, count)
    y = np.full(count, 8.0) + rng.normal(0.0, noise, count)
    z = np.full(count, height) + rng.normal(0.0, noise, count)
    return np.column_stack([x, y, z])


class MeasureHeightLineTest(unittest.TestCase):
    def test_recovers_height_and_length(self) -> None:
        result = measure_height_line(
            _make_baseline(), _make_height_line(height=12.5, length=30.0)
        )
        self.assertAlmostEqual(result.height_mean_mm, 12.5, delta=0.05)
        self.assertAlmostEqual(result.height_median_mm, 12.5, delta=0.05)
        self.assertAlmostEqual(result.length_mm, 30.0, delta=0.3)
        self.assertLess(result.height_fit.rmse_mm, 0.1)
        self.assertEqual(result.ground_reference_mode, "baseline_roi_profile")

    def test_uses_zg_zero_when_baseline_is_not_provided(self) -> None:
        result = measure_height_line(
            None, _make_height_line(height=12.5, length=30.0)
        )

        self.assertEqual(result.ground_reference_mode, "zg_zero")
        self.assertEqual(result.ground_baseline_zg_mm, 0.0)
        self.assertIsNone(result.ground_noise_sigma_mm)
        self.assertIsNone(result.baseline_fit)
        self.assertIsNone(result.ground_profile_fit)
        self.assertIsNone(result.angle_with_baseline_deg)
        self.assertEqual(result.baseline_point_count, 0)
        self.assertEqual(result.baseline_inlier_count, 0)
        self.assertAlmostEqual(result.height_mean_mm, 12.5, delta=0.05)

    def test_session_reference_mode_does_not_fit_baseline_again(self) -> None:
        baseline = _make_baseline()
        height = _make_height_line(height=12.5, length=30.0)
        result = measure_height_line(
            baseline,
            height,
            ground_correction_mode="session_reference",
        )

        self.assertEqual(result.ground_reference_mode, "session_reference")
        self.assertEqual(result.ground_baseline_zg_mm, 0.0)
        self.assertIsNone(result.ground_profile_fit)
        self.assertIsNone(result.baseline_fit)
        self.assertEqual(result.baseline_point_count, len(baseline))
        self.assertEqual(result.baseline_inlier_count, 0)
        self.assertAlmostEqual(result.height_mean_mm, 12.5, delta=0.05)

    def test_session_reference_mode_without_baseline_uses_session_zg_zero(self) -> None:
        result = measure_height_line(
            None,
            _make_height_line(height=12.5, length=30.0),
            ground_correction_mode="session_reference",
        )

        self.assertEqual(result.ground_reference_mode, "session_reference")
        self.assertEqual(result.ground_baseline_zg_mm, 0.0)
        self.assertIsNone(result.ground_profile_fit)
        self.assertIsNone(result.baseline_fit)
        self.assertEqual(result.baseline_point_count, 0)
        self.assertEqual(result.baseline_inlier_count, 0)
        self.assertAlmostEqual(result.height_mean_mm, 12.5, delta=0.05)

    def test_empty_baseline_array_is_not_treated_as_zg_zero(self) -> None:
        with self.assertRaisesRegex(
            MeasurementError, "baseline line has too few points"
        ):
            measure_height_line(np.empty((0, 3)), _make_height_line())

    def test_local_ground_profile_removes_baseline_slope(self) -> None:
        rng = np.random.default_rng(19)
        x_ground = np.linspace(-10.0, 50.0, 160)
        y_ground = rng.normal(0.0, 0.02, len(x_ground))
        ground_z = (
            2.0 + 0.02 * x_ground + rng.normal(0.0, 0.005, len(x_ground))
        )
        baseline = np.column_stack([x_ground, y_ground, ground_z])

        x_top = np.linspace(5.0, 35.0, 90)
        y_top = np.full_like(x_top, 8.0) + rng.normal(
            0.0, 0.02, len(x_top)
        )
        top_z = 2.0 + 0.02 * x_top + 10.0 + rng.normal(
            0.0, 0.005, len(x_top)
        )
        height = np.column_stack([x_top, y_top, top_z])

        result = measure_height_line(baseline, height)

        self.assertEqual(result.ground_reference_mode, "baseline_roi_profile")
        self.assertAlmostEqual(result.height_mean_mm, 10.0, delta=0.03)
        self.assertAlmostEqual(
            result.ground_profile_fit.slope_z_per_mm, 0.02, delta=0.002
        )

    def test_multiple_obstacle_groups_are_measured_independently(self) -> None:
        results = measure_height_lines(
            None,
            [
                _make_height_line(height=8.0, length=20.0),
                _make_height_line(height=25.0, length=35.0),
            ],
        )

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].height_mean_mm, 8.0, delta=0.05)
        self.assertAlmostEqual(results[0].length_mm, 20.0, delta=0.3)
        self.assertAlmostEqual(results[1].height_mean_mm, 25.0, delta=0.05)
        self.assertAlmostEqual(results[1].length_mm, 35.0, delta=0.3)

    def test_outliers_are_rejected(self) -> None:
        height_line = _make_height_line(height=10.0, length=20.0)
        contaminated = np.vstack(
            [
                height_line,
                np.array([[100.0, 60.0, 55.0], [-80.0, -70.0, -20.0]]),
            ]
        )
        result = measure_height_line(_make_baseline(), contaminated)
        self.assertAlmostEqual(result.height_mean_mm, 10.0, delta=0.1)
        self.assertAlmostEqual(result.length_mm, 20.0, delta=0.5)
        self.assertLess(
            result.height_inlier_count, result.height_point_count
        )

    def test_angle_between_perpendicular_lines(self) -> None:
        result = measure_height_line(_make_baseline(), _make_height_line())
        self.assertAlmostEqual(
            result.angle_with_baseline_deg, 90.0, delta=1.0
        )

    def test_endpoint_geometry(self) -> None:
        result = measure_height_line(
            _make_baseline(), _make_height_line(length=30.0)
        )
        endpoints = result.endpoints_ground
        self.assertEqual(endpoints.shape, (2, 3))
        span = float(np.linalg.norm(endpoints[1, :2] - endpoints[0, :2]))
        self.assertAlmostEqual(span, result.length_mm, delta=1e-6)

    def test_too_few_points_raises(self) -> None:
        with self.assertRaises(MeasurementError):
            measure_height_line(
                _make_baseline(count=5), _make_height_line()
            )
        with self.assertRaises(MeasurementError):
            measure_height_line(
                _make_baseline(), _make_height_line(count=5)
            )

    def test_min_points_configurable(self) -> None:
        params = MeasurementParams(
            min_baseline_points=5, min_height_points=5
        )
        result = measure_height_line(
            _make_baseline(count=8),
            _make_height_line(count=8),
            params,
        )
        self.assertGreater(result.height_mean_mm, 0.0)

    def test_twenty_points_is_allowed_and_nineteen_rejected(self) -> None:
        params = MeasurementParams()
        result = measure_height_line(
            _make_baseline(count=20),
            _make_height_line(count=20),
            params,
        )
        self.assertGreater(result.height_mean_mm, 0.0)

        with self.assertRaisesRegex(
            MeasurementError, r"height line has too few points: 19 < 20"
        ):
            measure_height_line(
                _make_baseline(count=20),
                _make_height_line(count=19),
                params,
            )

        with self.assertRaisesRegex(
            MeasurementError, r"baseline line has too few points: 19 < 20"
        ):
            measure_height_line(
                _make_baseline(count=19),
                _make_height_line(count=20),
                params,
            )

    def test_degenerate_points_raise(self) -> None:
        identical = np.tile(np.array([[1.0, 2.0, 3.0]]), (50, 1))
        with self.assertRaises(MeasurementError):
            measure_height_line(_make_baseline(), identical)

    def test_nan_input_raises(self) -> None:
        bad = _make_height_line()
        bad[0, 2] = np.nan
        with self.assertRaises(MeasurementError):
            measure_height_line(_make_baseline(), bad)


if __name__ == "__main__":
    unittest.main()
