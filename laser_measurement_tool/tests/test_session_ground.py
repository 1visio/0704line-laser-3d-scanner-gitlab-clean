"""Tests for the shared session-ground checkerboard PnP API."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibration.session_ground import (
    BoardConfig,
    build_camera_to_ground_transform,
    estimate_session_ground_extrinsic,
)


class SessionGroundCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.board = BoardConfig(
            pattern_cols=11,
            pattern_rows=8,
            square_size_mm=20.0,
        )
        self.K = np.array(
            [[820.0, 0.0, 320.0], [0.0, 818.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.D = np.zeros(5, dtype=np.float64)
        self.rvec = np.array([[0.045], [-0.035], [0.012]], dtype=np.float64)
        self.tvec = np.array([[35.0], [-18.0], [720.0]], dtype=np.float64)
        self.corners, _ = cv2.projectPoints(
            self.board.object_points(),
            self.rvec,
            self.tvec,
            self.K,
            self.D,
        )
        self.image = np.zeros((480, 640), dtype=np.uint8)

    def test_object_point_order_matches_existing_calibration(self) -> None:
        points = self.board.object_points()
        self.assertEqual(points.shape, (88, 3))
        np.testing.assert_allclose(
            points[[0, 1, 7, 8, 87]],
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [140.0, 0.0, 0.0],
                    [160.0, 0.0, 0.0],
                    [200.0, 140.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_estimate_returns_pnp_and_camera_to_ground_semantics(self) -> None:
        corners = self.corners.astype(np.float32)
        with patch.object(
            cv2,
            "findChessboardCornersSB",
            return_value=(True, corners),
        ):
            result = estimate_session_ground_extrinsic(
                self.image,
                {"K": self.K, "D": self.D},
                self.board,
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.detection_method, "SB")
        assert result.detected_corners is not None
        assert result.rvec is not None
        assert result.tvec is not None
        assert result.R is not None
        assert result.t is not None
        assert result.T_ground_from_camera is not None
        assert result.T_camera_from_ground is not None
        self.assertEqual(result.detected_corners.shape, (88, 2))
        self.assertLess(result.reprojection_rmse_px or 1.0, 1.0e-4)
        np.testing.assert_allclose(result.rvec, self.rvec, atol=1.0e-6)
        np.testing.assert_allclose(result.tvec, self.tvec, atol=1.0e-4)

        ground_point = np.array([15.0, -25.0, 0.0, 1.0])
        camera_point = result.T_camera_from_ground @ ground_point
        recovered = result.T_ground_from_camera @ camera_point
        np.testing.assert_allclose(recovered, ground_point, atol=1.0e-10)
        self.assertAlmostEqual(float(recovered[2]), 0.0, places=10)

    def test_detection_failure_returns_status_without_throwing(self) -> None:
        with patch.object(
            cv2,
            "findChessboardCornersSB",
            return_value=(False, None),
        ), patch.object(
            cv2,
            "findChessboardCorners",
            return_value=(False, None),
        ):
            result = estimate_session_ground_extrinsic(
                self.image,
                {"K": self.K, "D": self.D},
                self.board,
            )

        self.assertEqual(result.status, "board_not_detected")
        self.assertIsNone(result.T_ground_from_camera)
        self.assertIsNone(result.reprojection_rmse_px)

    def test_transform_builder_matches_current_ground_axis_definition(self) -> None:
        R_board_to_camera, _ = cv2.Rodrigues(self.rvec)
        R_camera_to_ground, t_camera_to_ground, T_ground, T_camera = (
            build_camera_to_ground_transform(R_board_to_camera, self.tvec)
        )
        np.testing.assert_allclose(
            R_camera_to_ground @ R_camera_to_ground.T,
            np.eye(3),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            T_ground @ T_camera,
            np.eye(4),
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            t_camera_to_ground,
            T_ground[:3, 3],
            atol=1.0e-12,
        )
        self.assertLess(float(T_camera[2, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
