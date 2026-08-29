"""Unit tests for the diagnostic Session laser-ground sanity check."""

from __future__ import annotations

import unittest

import numpy as np

from online.ground_sanity import (
    evaluate_ground_sanity,
    full_board_physical_polygon,
    select_board_ground_points,
    select_points_inside_board_mask,
)
from measurement.board_mask import select_manual_ground_roi_points


class GroundSanityTests(unittest.TestCase):
    def test_ground_sanity_uses_shared_board_selector(self) -> None:
        self.assertIs(select_points_inside_board_mask, select_board_ground_points)

    def test_valid_metrics_use_raw_zg_and_report_slope(self) -> None:
        distance = np.linspace(0.0, 100.0, 25)
        zg = 0.01 * distance + 0.1
        points = np.column_stack((distance, np.zeros_like(distance), zg))

        result = evaluate_ground_sanity(
            points,
            ground_extrinsic_source="session",
            frame_number=101,
            session_calibration_frame_number=100,
        )

        self.assertEqual(result.status, "VALID")
        self.assertAlmostEqual(result.bias_zg_mm, 0.6)
        self.assertAlmostEqual(result.rmse_zg_mm, np.sqrt(np.mean(zg**2)))
        self.assertAlmostEqual(result.p95_abs_zg_mm, np.percentile(np.abs(zg), 95))
        self.assertAlmostEqual(result.max_abs_zg_mm, 1.1)
        self.assertAlmostEqual(result.ground_slope_mm_per_mm, 0.01)
        self.assertEqual(result.valid_point_count, 25)
        payload = result.as_dict()
        self.assertFalse(payload["correction_applied"])
        self.assertFalse(payload["surface_correction_applied"])
        self.assertFalse(payload["stage_a_applied"])

    def test_full_physical_board_mask_includes_outer_squares(self) -> None:
        camera_matrix = np.asarray(
            [[1000.0, 0.0, 100.0], [0.0, 1000.0, 100.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros(5, dtype=np.float64)
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.asarray([0.0, 0.0, 1000.0], dtype=np.float64)
        corners = np.zeros((6, 2), dtype=np.float64)
        pixels = np.asarray(
            [[90.0, 90.0], [150.0, 130.0], [70.0, 90.0], [170.0, 90.0]],
            dtype=np.float64,
        )
        points = np.column_stack(
            (pixels, np.zeros(len(pixels), dtype=np.float64))
        )
        image_offset = (100, 200)
        polygon = full_board_physical_polygon(
            rvec,
            tvec,
            pattern_cols=3,
            pattern_rows=2,
            square_size_mm=20.0,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_offset=image_offset,
        )
        np.testing.assert_allclose(
            polygon,
            [[180.0, 280.0], [260.0, 280.0], [260.0, 340.0], [180.0, 340.0]],
        )

        selected, mask = select_points_inside_board_mask(
            pixels + np.asarray(image_offset, dtype=np.float64),
            points,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            pattern_cols=3,
            pattern_rows=2,
            square_size_mm=20.0,
            image_offset=image_offset,
            detected_corners=corners,
        )

        np.testing.assert_allclose(selected[:, :2], [[90.0, 90.0], [150.0, 130.0]])
        self.assertEqual(mask["status"], "applied")
        self.assertEqual(mask["mask_mode"], "full_board_physical")
        self.assertEqual(mask["inset_mm"], 0.0)
        self.assertEqual(mask["input_point_count"], 4)
        self.assertEqual(mask["selected_point_count"], 2)
        sanity = evaluate_ground_sanity(
            selected,
            ground_extrinsic_source="session",
            frame_number=2,
            session_calibration_frame_number=1,
            thresholds={"min_valid_points": 1},
            mask=mask,
        )
        self.assertEqual(sanity.status, "VALID")
        self.assertEqual(sanity.as_dict()["mask"]["selected_point_count"], 2)

    def test_manual_ground_roi_is_explicit_support_source(self) -> None:
        pixels = np.asarray(
            [[10.0, 10.0], [20.0, 20.0], [40.0, 40.0]], dtype=np.float64
        )
        points = np.column_stack(
            (pixels, np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
        )
        selected, metadata = select_manual_ground_roi_points(
            pixels,
            points,
            [(5.0, 5.0, 25.0, 25.0)],
        )
        np.testing.assert_allclose(selected[:, 2], [1.0, 2.0])
        self.assertEqual(metadata["source"], "manual_ground_roi")
        self.assertEqual(metadata["status"], "applied")
        self.assertEqual(metadata["selected_point_count"], 2)

    def test_threshold_failure_does_not_subtract_bias(self) -> None:
        points = np.column_stack(
            (
                np.arange(10, dtype=np.float64),
                np.zeros(10, dtype=np.float64),
                np.full(10, 3.0, dtype=np.float64),
            )
        )

        result = evaluate_ground_sanity(
            points,
            ground_extrinsic_source="session",
            frame_number=8,
            session_calibration_frame_number=8,
        )

        self.assertEqual(result.status, "INVALID")
        self.assertAlmostEqual(result.bias_zg_mm, 3.0)
        self.assertAlmostEqual(result.rmse_zg_mm, 3.0)
        self.assertIn("valid_point_count_below_minimum", result.threshold_violations)
        self.assertIn("abs_bias_exceeds_limit", result.threshold_violations)
        self.assertIn("laser_on_frame_not_after_session_calibration", result.warnings)

    def test_reference_source_and_nonfinite_points_are_invalid(self) -> None:
        points = np.asarray(
            [[float(index), 0.0, 0.0] for index in range(20)],
            dtype=np.float64,
        )
        points[3, 2] = np.nan

        result = evaluate_ground_sanity(
            points,
            ground_extrinsic_source="reference",
            frame_number=2,
            session_calibration_frame_number=1,
        )

        self.assertEqual(result.status, "INVALID")
        self.assertEqual(result.valid_point_count, 19)
        self.assertIn("ground_extrinsic_source_not_session", result.warnings)
        self.assertIn("nonfinite_ground_points", result.warnings)
        self.assertIn("valid_point_count_below_minimum", result.threshold_violations)


if __name__ == "__main__":
    unittest.main()
