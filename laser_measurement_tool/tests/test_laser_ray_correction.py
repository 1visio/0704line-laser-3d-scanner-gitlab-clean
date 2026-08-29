"""Frozen Daheng C1 ray-depth correction tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from app_config import DEFAULT_CONFIG_PATH, load_app_config
from calibration.config_loader import load_calibration_files
from calibration.manifest import load_calibration_package
from reconstruction.laser_ray_correction import (
    apply_frozen_laser_ray_correction,
    evaluate_frozen_laser_ray_correction,
    load_frozen_laser_ray_correction,
)
from reconstruction.reconstructor import (
    ReconstructionInputError,
    ReconstructionParams,
    reconstruct_uv_to_ground,
)


TOOL_ROOT = Path(__file__).resolve().parents[1]
C1_PATH = TOOL_ROOT / "configs" / "calibration_daheng_0811" / "frozen_c1_4k.json"
DAHENG_CONFIG = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DAHENG_MANIFEST = TOOL_ROOT / "configs" / "calibration_daheng_0811" / "manifest.yaml"
FROZEN_C0 = TOOL_ROOT / "tests" / "data" / "daheng_frozen_c0_reference.yaml"


def _rays_for_s(values: np.ndarray, correction: object) -> np.ndarray:
    model = correction
    center = np.array([model.center_xn, model.center_yn], dtype=np.float64)
    axis = np.array([model.axis_s_xn, model.axis_s_yn], dtype=np.float64)
    xy = center + np.asarray(values, dtype=np.float64)[:, None] * axis
    return np.column_stack((xy, np.ones(len(xy), dtype=np.float64)))


def _cox_de_boor(index: int, degree: int, value: float, knots: np.ndarray) -> float:
    if degree == 0:
        is_last_endpoint = value == knots[-1] and index == len(knots) - 2
        return float((knots[index] <= value < knots[index + 1]) or is_last_endpoint)
    left_denominator = knots[index + degree] - knots[index]
    right_denominator = knots[index + degree + 1] - knots[index + 1]
    left = 0.0
    right = 0.0
    if left_denominator:
        left = (value - knots[index]) / left_denominator * _cox_de_boor(
            index, degree - 1, value, knots
        )
    if right_denominator:
        right = (knots[index + degree + 1] - value) / right_denominator * _cox_de_boor(
            index + 1, degree - 1, value, knots
        )
    return left + right


def _manual_cubic_spline(
    values: np.ndarray, knots: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            coefficients[-1]
            if value == knots[-1]
            else sum(
                coefficients[index]
                * _cox_de_boor(index, 3, float(value), knots)
                for index in range(len(coefficients))
            )
            for value in values
        ],
        dtype=np.float64,
    )


def _synthetic_plane_calibration() -> dict[str, object]:
    return {
        "K": np.array(
            [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        "D": np.zeros(5),
        "plane_abcd": np.array([0.0, 0.0, 1.0, -500.0]),
        "R": np.eye(3),
        "t": np.zeros(3),
    }


class FrozenC1MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.correction = load_frozen_laser_ray_correction(C1_PATH)

    def test_c1_frozen_reproduction_uses_full36_pca_and_exact_cubic_spline(self) -> None:
        values = np.array([-0.18, -0.08, 0.02, 0.11, 0.19], dtype=np.float64)
        rays = _rays_for_s(values, self.correction)
        result = evaluate_frozen_laser_ray_correction(rays, self.correction)
        expected_s = (
            rays[:, :2]
            - np.array(
                [self.correction.center_xn, self.correction.center_yn],
                dtype=np.float64,
            )
        ) @ np.array(
            [self.correction.axis_s_xn, self.correction.axis_s_yn],
            dtype=np.float64,
        )
        expected = _manual_cubic_spline(
            expected_s,
            self.correction.knots,
            self.correction.coefficients_mm,
        )
        np.testing.assert_allclose(result.s_raw, expected_s, rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(result.s_eval, expected_s, rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(
            result.correction_mm, expected, rtol=0.0, atol=2.0e-15
        )
        self.assertFalse(result.clamped.any())

    def test_c1_clamps_both_domain_ends_without_extrapolation(self) -> None:
        values = np.array(
            [self.correction.domain_min - 0.1, self.correction.domain_max + 0.1],
            dtype=np.float64,
        )
        result = evaluate_frozen_laser_ray_correction(
            _rays_for_s(values, self.correction), self.correction
        )
        expected_s = np.array(
            [self.correction.domain_min, self.correction.domain_max],
            dtype=np.float64,
        )
        np.testing.assert_allclose(result.s_eval, expected_s, rtol=0.0, atol=1.0e-15)
        np.testing.assert_array_equal(result.clamped, [True, True])
        np.testing.assert_allclose(
            result.correction_mm,
            _manual_cubic_spline(
                expected_s,
                self.correction.knots,
                self.correction.coefficients_mm,
            ),
            rtol=0.0,
            atol=2.0e-15,
        )

    def test_c1_off_is_exact_c0_regression(self) -> None:
        with_c1 = _synthetic_plane_calibration()
        with_c1["laser_ray_correction"] = self.correction
        without_c1 = dict(with_c1)
        without_c1.pop("laser_ray_correction")
        pixels = np.array([[320.0, 240.0], [420.0, 140.0]], dtype=np.float64)
        off = ReconstructionParams(enable_laser_ray_correction=False)
        actual = reconstruct_uv_to_ground(pixels, with_c1, off)
        baseline = reconstruct_uv_to_ground(pixels, without_c1, off)
        np.testing.assert_array_equal(actual.pixels_uv, baseline.pixels_uv)
        np.testing.assert_array_equal(actual.points_camera, baseline.points_camera)
        np.testing.assert_array_equal(actual.points_ground, baseline.points_ground)
        self.assertEqual(actual.filtered, baseline.filtered)

    def test_c1_lambda_formula_is_c0_plus_f(self) -> None:
        values = np.array([-0.1, 0.1], dtype=np.float64)
        evaluation = evaluate_frozen_laser_ray_correction(
            _rays_for_s(values, self.correction), self.correction
        )
        lambda_c0 = np.array([700.0, 701.0], dtype=np.float64)
        lambda_final = apply_frozen_laser_ray_correction(
            lambda_c0,
            _rays_for_s(values, self.correction),
            self.correction,
        )
        np.testing.assert_allclose(
            lambda_final,
            lambda_c0 + evaluation.correction_mm,
            rtol=0.0,
            atol=0.0,
        )


class FrozenC1IntegrationTests(unittest.TestCase):
    def test_enabled_c1_is_applied_after_quadratic_intersection(self) -> None:
        config = load_app_config(DAHENG_CONFIG)
        calibration = load_calibration_files(
            config.calibration.intrinsics,
            config.calibration.laser_plane,
            config.calibration.extrinsics,
            config.calibration.ground_u_compensation,
            laser_ray_correction=config.calibration.laser_ray_correction,
        )
        pixels = np.array(
            [[2092.0815280371376, 898.0115496512335], [2060.098014433376, 1094.0188303570471]],
            dtype=np.float64,
        )
        off = reconstruct_uv_to_ground(
            pixels,
            calibration,
            ReconstructionParams(
                min_camera_depth_mm=0.0,
                max_camera_depth_mm=2000.0,
                enable_laser_ray_correction=False,
            ),
        )
        on = reconstruct_uv_to_ground(
            pixels,
            calibration,
            ReconstructionParams(
                min_camera_depth_mm=0.0,
                max_camera_depth_mm=2000.0,
                enable_laser_ray_correction=True,
            ),
        )
        self.assertEqual(off.point_count, 2)
        self.assertEqual(on.point_count, 2)
        normalized = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2),
            calibration["K"],
            calibration["D"],
        ).reshape(-1, 2)
        rays = np.column_stack((normalized, np.ones(len(normalized))))
        evaluation = evaluate_frozen_laser_ray_correction(
            rays, calibration["laser_ray_correction"]
        )
        np.testing.assert_allclose(
            on.points_camera[:, 2],
            off.points_camera[:, 2] + evaluation.correction_mm,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_explicit_and_manifest_loaders_share_the_same_c1_parameters(self) -> None:
        config = load_app_config(DAHENG_CONFIG)
        explicit = load_calibration_files(
            intrinsics=config.calibration.intrinsics,
            laser_plane=config.calibration.laser_plane,
            extrinsics=config.calibration.extrinsics,
            ground_u_compensation=config.calibration.ground_u_compensation,
            laser_ray_correction=config.calibration.laser_ray_correction,
        )
        package = load_calibration_package(DAHENG_MANIFEST)
        explicit_c1 = explicit["laser_ray_correction"]
        manifest_c1 = package.calibration["laser_ray_correction"]
        self.assertIsNotNone(explicit_c1)
        self.assertIsNotNone(manifest_c1)
        self.assertEqual(explicit_c1.model_id, manifest_c1.model_id)
        np.testing.assert_array_equal(explicit_c1.knots, manifest_c1.knots)
        np.testing.assert_array_equal(
            explicit_c1.coefficients_mm, manifest_c1.coefficients_mm
        )
        self.assertTrue(config.reconstruction.enable_laser_ray_correction)

    def test_enabled_c1_requires_quadratic_graph_and_parameters(self) -> None:
        calibration = _synthetic_plane_calibration()
        with self.assertRaisesRegex(ReconstructionInputError, "quadratic_graph"):
            reconstruct_uv_to_ground(
                np.array([[320.0, 240.0]]),
                calibration,
                ReconstructionParams(enable_laser_ray_correction=True),
            )

        quadratic = dict(calibration)
        quadratic["laser_model"] = {
            "model_type": "quadratic_graph",
            "dependent_axis": "X",
            "independent_axes": ["Y", "Z"],
            "normalization": {
                "independent_center_mm": [0.0, 500.0],
                "independent_scale_mm": [100.0, 100.0],
            },
            "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        with self.assertRaisesRegex(ReconstructionInputError, "缺少有效 frozen C1"):
            reconstruct_uv_to_ground(
                np.array([[320.0, 240.0]]),
                quadratic,
                ReconstructionParams(enable_laser_ray_correction=True),
            )

    def test_default_and_non_daheng_configs_remain_c1_off(self) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        self.assertFalse(config.reconstruction.enable_laser_ray_correction)
        self.assertIsNone(config.calibration.laser_ray_correction)
        calibration = load_calibration_files(
            config.calibration.intrinsics,
            config.calibration.laser_plane,
            config.calibration.extrinsics,
            config.calibration.ground_u_compensation,
        )
        self.assertIsNone(calibration["laser_ray_correction"])

    def test_c0_semantic_match_is_parameter_level_not_hash_only(self) -> None:
        online = yaml.safe_load(
            (
                TOOL_ROOT / "configs" / "calibration_daheng_0811" / "quadratic_graph.yaml"
            ).read_text(encoding="utf-8")
        )
        frozen = yaml.safe_load(FROZEN_C0.read_text(encoding="utf-8"))
        for key in ("model_type", "dependent_axis", "independent_axes", "equation"):
            self.assertEqual(online[key], frozen[key])
        for key in ("independent_center_mm", "independent_scale_mm"):
            np.testing.assert_allclose(
                online["normalization"][key], frozen["normalization"][key],
                rtol=0.0, atol=0.0,
            )
        np.testing.assert_allclose(
            online["coefficients"], frozen["coefficients"], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            online["z_valid_range_mm"], frozen["z_valid_range_mm"],
            rtol=0.0, atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
