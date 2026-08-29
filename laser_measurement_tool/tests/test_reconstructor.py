"""reconstruction.reconstructor 的合成数据单元测试。"""

import unittest

import numpy as np

from reconstruction.reconstructor import (
    ReconstructionInputError,
    ReconstructionParams,
    apply_ground_u_compensation,
    build_ground_transform,
    project_ground_points_to_pixels,
    reconstruct_uv_to_ground,
)


def _synthetic_calibration() -> dict:
    """无畸变针孔相机 + Zc=500 激光平面 + 已知地面外参。"""
    K = np.array(
        [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]
    )
    D = np.zeros(5)
    plane_abcd = np.array([0.0, 0.0, 1.0, -500.0])
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    t = np.array([0.0, 0.0, 700.0])
    return {"K": K, "D": D, "plane_abcd": plane_abcd, "R": R, "t": t}


def _synthetic_cone_calibration() -> dict:
    """轴向为 Z 的圆锥，便于验证前向根筛选。"""
    return {
        "K": np.array(
            [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        "D": np.zeros(5),
        "laser_model": {
            "model_type": "circular_cone",
            "axis_unit_camera": np.array([0.0, 0.0, 1.0]),
            "apex_camera_mm": np.array([0.0, 0.0, 200.0]),
            "half_apex_angle_deg": 45.0,
        },
        "R": np.eye(3),
        "t": np.zeros(3),
    }


class ReconstructUvToGroundTest(unittest.TestCase):
    def test_principal_point_reconstructs_on_axis(self) -> None:
        calibration = _synthetic_calibration()
        result = reconstruct_uv_to_ground(
            np.array([[320.0, 240.0]]), calibration
        )
        self.assertEqual(result.point_count, 1)
        np.testing.assert_allclose(
            result.points_camera[0], [0.0, 0.0, 500.0], atol=1e-9
        )
        # 地面系: Zg = 700 - Zc = 200
        np.testing.assert_allclose(
            result.points_ground[0], [0.0, 0.0, 200.0], atol=1e-9
        )

    def test_off_axis_pixel_matches_manual_projection(self) -> None:
        calibration = _synthetic_calibration()
        result = reconstruct_uv_to_ground(
            np.array([[420.0, 140.0]]), calibration
        )
        # 归一化坐标 (0.1, -0.1)，深度 500
        np.testing.assert_allclose(
            result.points_camera[0], [50.0, -50.0, 500.0], atol=1e-9
        )
        np.testing.assert_allclose(
            result.points_ground[0], [50.0, 50.0, 200.0], atol=1e-9
        )

    def test_depth_range_filters_points(self) -> None:
        calibration = _synthetic_calibration()
        params = ReconstructionParams(
            min_camera_depth_mm=600.0, max_camera_depth_mm=1500.0
        )
        result = reconstruct_uv_to_ground(
            np.array([[320.0, 240.0]]), calibration, params
        )
        self.assertEqual(result.point_count, 0)
        self.assertEqual(result.filtered["outside_working_distance"], 1)

    def test_negative_depth_is_rejected(self) -> None:
        calibration = _synthetic_calibration()
        calibration["plane_abcd"] = np.array([0.0, 0.0, 1.0, 500.0])
        result = reconstruct_uv_to_ground(
            np.array([[320.0, 240.0]]), calibration
        )
        self.assertEqual(result.point_count, 0)
        self.assertEqual(result.filtered["negative_depth"], 1)

    def test_plane_scale_invariance(self) -> None:
        calibration = _synthetic_calibration()
        scaled = dict(calibration)
        scaled["plane_abcd"] = calibration["plane_abcd"] * 3.7
        pixels = np.array([[320.0, 240.0], [420.0, 140.0]])
        original = reconstruct_uv_to_ground(pixels, calibration)
        rescaled = reconstruct_uv_to_ground(pixels, scaled)
        np.testing.assert_allclose(
            original.points_ground, rescaled.points_ground, atol=1e-9
        )

    def test_invalid_shape_raises(self) -> None:
        calibration = _synthetic_calibration()
        with self.assertRaises(ReconstructionInputError):
            reconstruct_uv_to_ground(np.zeros((3, 3)), calibration)

    def test_empty_input_returns_empty_result(self) -> None:
        calibration = _synthetic_calibration()
        result = reconstruct_uv_to_ground(
            np.empty((0, 2)), calibration
        )
        self.assertEqual(result.point_count, 0)

    def test_image_roi_keeps_board_interior_and_counts_outside_points(self) -> None:
        calibration = _synthetic_calibration()
        params = ReconstructionParams(
            image_roi_polygon=(
                (300.0, 200.0),
                (500.0, 200.0),
                (500.0, 400.0),
                (300.0, 400.0),
            )
        )
        pixels = np.array([[320.0, 240.0], [499.999, 399.999], [250.0, 240.0]])
        result = reconstruct_uv_to_ground(pixels, calibration, params)
        self.assertEqual(result.point_count, 2)
        self.assertEqual(result.filtered["outside_image_roi"], 1)
        np.testing.assert_allclose(result.pixels_uv, pixels[:2], atol=1e-6)

    def test_degenerate_image_roi_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReconstructionInputError, "退化多边形"):
            ReconstructionParams(
                image_roi_polygon=((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
            )

    def test_reconstruction_applies_interpolated_ground_u_bias(self) -> None:
        calibration = _synthetic_calibration()
        pixels = np.array([[300.0, 240.0], [350.0, 240.0], [450.0, 240.0]])
        raw = reconstruct_uv_to_ground(pixels, calibration)
        calibration["ground_u_compensation"] = {
            "column_u_px": np.array([300.0, 400.0]),
            "bias_mm": np.array([1.0, 3.0]),
        }

        with self.assertWarnsRegex(RuntimeWarning, "nearest endpoint bias"):
            corrected = reconstruct_uv_to_ground(pixels, calibration)

        np.testing.assert_allclose(corrected.points_ground[:, :2], raw.points_ground[:, :2])
        np.testing.assert_allclose(
            raw.points_ground[:, 2] - corrected.points_ground[:, 2],
            [1.0, 2.0, 3.0],
        )

    def test_reconstruction_applies_interpolated_ground_v_bias(self) -> None:
        calibration = _synthetic_calibration()
        pixels = np.array([[320.0, 200.0], [320.0, 250.0], [320.0, 300.0]])
        raw = reconstruct_uv_to_ground(pixels, calibration)
        calibration["ground_u_compensation"] = {
            "compensation_axis": "v",
            "row_v_px": np.array([200.0, 300.0]),
            "bias_mm": np.array([1.0, 3.0]),
        }

        corrected = reconstruct_uv_to_ground(pixels, calibration)

        np.testing.assert_allclose(
            raw.points_ground[:, 2] - corrected.points_ground[:, 2],
            [1.0, 2.0, 3.0],
        )

    def test_reconstruction_applies_ground_u_z_offset(self) -> None:
        calibration = _synthetic_calibration()
        pixels = np.array([[300.0, 240.0], [400.0, 240.0]])
        raw = reconstruct_uv_to_ground(pixels, calibration)
        calibration["ground_u_compensation"] = {
            "column_u_px": np.array([300.0, 400.0]),
            "bias_mm": np.array([1.0, 3.0]),
            "z_offset_mm": 10.0,
        }

        corrected = reconstruct_uv_to_ground(pixels, calibration)

        np.testing.assert_allclose(
            raw.points_ground[:, 2] - corrected.points_ground[:, 2],
            [11.0, 13.0],
        )

    def test_circular_cone_selects_forward_intersection(self) -> None:
        result = reconstruct_uv_to_ground(
            np.array([[820.0, 240.0]]), _synthetic_cone_calibration()
        )
        self.assertEqual(result.point_count, 1)
        np.testing.assert_allclose(
            result.points_camera[0], [200.0, 0.0, 400.0],
            atol=1.0e-9,
        )

    def test_circular_cone_rejects_invalid_half_angle(self) -> None:
        calibration = _synthetic_cone_calibration()
        calibration["laser_model"]["half_apex_angle_deg"] = 90.0
        with self.assertRaisesRegex(ReconstructionInputError, "half_apex_angle_deg"):
            reconstruct_uv_to_ground(np.array([[320.0, 240.0]]), calibration)


class GroundUCompensationTest(unittest.TestCase):
    def test_none_keeps_points_unchanged(self) -> None:
        points = np.array([[1.0, 2.0, 3.0]])
        pixels = np.array([[10.0, 20.0]])
        corrected = apply_ground_u_compensation(points, pixels, None)
        np.testing.assert_array_equal(corrected, points)

    def test_malformed_table_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReconstructionInputError, "非空且等长"):
            apply_ground_u_compensation(
                np.zeros((2, 3)),
                np.zeros((2, 2)),
                {"column_u_px": [0.0, 1.0], "bias_mm": [0.0]},
            )

    def test_v_compensation_is_pointwise_for_different_roi_lengths(self) -> None:
        compensation = {
            "compensation_axis": "v",
            "row_v_px": [10.0, 20.0],
            "bias_mm": [2.0, 4.0],
        }
        for point_count in (3, 7, 19):
            with self.subTest(point_count=point_count):
                v = np.linspace(10.0, 20.0, point_count)
                pixels = np.column_stack((np.full(point_count, 123.0), v))
                points = np.column_stack(
                    (np.zeros(point_count), np.zeros(point_count), 50.0 + 0.2 * v)
                )

                corrected = apply_ground_u_compensation(
                    points, pixels, compensation
                )

                np.testing.assert_allclose(corrected[:, 2], 50.0)


class ProjectGroundPointsTest(unittest.TestCase):
    def test_round_trip_matches_original_pixels(self) -> None:
        calibration = _synthetic_calibration()
        pixels = np.array([[320.0, 240.0], [420.0, 140.0], [250.0, 300.0]])
        result = reconstruct_uv_to_ground(pixels, calibration)
        reprojected = project_ground_points_to_pixels(
            result.points_ground, calibration
        )
        np.testing.assert_allclose(reprojected, pixels, atol=1e-6)


class BuildGroundTransformTest(unittest.TestCase):
    def test_transform_layout(self) -> None:
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        transform = build_ground_transform(R, t)
        np.testing.assert_allclose(transform[:3, :3], R)
        np.testing.assert_allclose(transform[:3, 3], t)
        np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
