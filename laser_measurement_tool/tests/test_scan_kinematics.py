"""Tests for camera-to-scan pitch kinematics."""

import unittest

import numpy as np

from scan.kinematics import transform_points_camera_to_scan


class CameraToScanKinematicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = np.eye(4, dtype=np.float64)

    def test_zero_angle_is_zero_pose_rigid_transform(self) -> None:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        transform[:3, 3] = [10.0, 20.0, 30.0]
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])

        actual = transform_points_camera_to_scan(
            points,
            angle_deg=0.0,
            axis_point_scan_mm=np.zeros(3),
            axis_direction_scan=np.array([0.0, 0.0, 1.0]),
            zero_offset_deg=0.0,
            T_scan_from_camera_zero=transform,
        )

        expected = (points @ transform[:3, :3].T) + transform[:3, 3]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_zero_angle_identity_transform_is_exactly_zero_pose(self) -> None:
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])

        actual = transform_points_camera_to_scan(
            points,
            angle_deg=0.0,
            axis_point_scan_mm=np.array([0.0, 0.0, 0.0]),
            axis_direction_scan=np.array([1.0, 0.0, 0.0]),
            zero_offset_deg=0.0,
            T_scan_from_camera_zero=np.eye(4, dtype=np.float64),
        )

        np.testing.assert_allclose(actual, points, atol=1e-12, rtol=0.0)

    def test_ninety_degree_rotation_about_origin_x_axis(self) -> None:
        points = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        actual = transform_points_camera_to_scan(
            points,
            90.0,
            np.zeros(3),
            np.array([2.0, 0.0, 0.0]),
            0.0,
            self.identity,
        )

        expected = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_rotation_about_axis_not_through_origin(self) -> None:
        point = np.array([[1.0, 1.0, 0.0]])

        actual = transform_points_camera_to_scan(
            point,
            90.0,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            0.0,
            self.identity,
        )

        np.testing.assert_allclose(actual, [[0.0, 0.0, 0.0]], atol=1e-12)

    def test_pairwise_distance_is_preserved(self) -> None:
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]])

        actual = transform_points_camera_to_scan(
            points,
            37.0,
            np.array([4.0, -3.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
            11.0,
            self.identity,
        )

        before = np.linalg.norm(points[1] - points[0])
        after = np.linalg.norm(actual[1] - actual[0])
        self.assertAlmostEqual(before, after, places=12)

    def test_non_unit_axis_is_normalized(self) -> None:
        point = np.array([[1.0, 0.0, 0.0]])

        actual_non_unit = transform_points_camera_to_scan(
            point,
            90.0,
            np.zeros(3),
            np.array([0.0, 5.0, 0.0]),
            0.0,
            self.identity,
        )
        actual_unit = transform_points_camera_to_scan(
            point,
            90.0,
            np.zeros(3),
            np.array([0.0, 1.0, 0.0]),
            0.0,
            self.identity,
        )

        np.testing.assert_allclose(actual_non_unit, [[0.0, 0.0, -1.0]], atol=1e-12)
        np.testing.assert_allclose(actual_non_unit, actual_unit, atol=1e-12, rtol=0.0)

    def test_zero_axis_direction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            transform_points_camera_to_scan(
                np.zeros((1, 3)),
                0.0,
                np.zeros(3),
                np.zeros(3),
                0.0,
                self.identity,
            )

    def test_zero_offset_is_added_to_commanded_angle(self) -> None:
        actual = transform_points_camera_to_scan(
            np.array([[0.0, 1.0, 0.0]]),
            45.0,
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            45.0,
            self.identity,
        )

        np.testing.assert_allclose(actual, [[-1.0, 0.0, 0.0]], atol=1e-12)

    def test_positive_then_negative_angle_recovers_original_points(self) -> None:
        points = np.array([[2.0, -1.0, 4.0], [-3.0, 5.0, 0.5]])
        axis_point = np.array([3.0, -2.0, 1.0])
        axis_direction = np.array([1.0, 2.0, 3.0])

        positive = transform_points_camera_to_scan(
            points,
            23.5,
            axis_point,
            axis_direction,
            0.0,
            self.identity,
        )
        recovered = transform_points_camera_to_scan(
            positive,
            -23.5,
            axis_point,
            axis_direction,
            0.0,
            self.identity,
        )

        np.testing.assert_allclose(recovered, points, atol=1e-12, rtol=0.0)

    def test_camera_to_zero_transform_precedes_scan_axis_rotation(self) -> None:
        zero_transform = np.eye(4, dtype=np.float64)
        zero_transform[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        zero_transform[:3, 3] = [10.0, 20.0, 30.0]
        point_camera = np.array([[1.0, 0.0, 0.0]])

        actual = transform_points_camera_to_scan(
            point_camera,
            90.0,
            np.array([10.0, 20.0, 30.0]),
            np.array([1.0, 0.0, 0.0]),
            0.0,
            zero_transform,
        )

        # T maps [1,0,0] to [10,21,30]; Rx(+90) about [10,20,30]
        # maps its relative [0,1,0] vector to [0,0,1].
        expected = np.array([[10.0, 20.0, 31.0]])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

    def test_rotation_about_axis_with_explicit_analytic_translation(self) -> None:
        axis_point = np.array([100.0, 0.0, 0.0])
        axis_direction = np.array([0.0, 0.0, 1.0])
        point = np.array([[101.0, 0.0, 0.0]])

        actual = transform_points_camera_to_scan(
            point,
            90.0,
            axis_point,
            axis_direction,
            0.0,
            self.identity,
        )

        # c + Rz(+90) @ (p-c) = [100,0,0] + [0,1,0].
        expected = np.array([[100.0, 1.0, 0.0]])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
