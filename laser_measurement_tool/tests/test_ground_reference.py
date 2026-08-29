from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app_config import DEFAULT_CONFIG_PATH, load_app_config
from measurement.ground_reference import (
    fit_session_ground_reference,
    fit_session_ground_reference_from_support,
    load_frozen_session_ground_reference,
)
from online.fake_camera import SyntheticCameraSession
from online.models import CameraConfig
from online.pipeline import FramePipeline
from online.session_calibration import (
    merge_session_ground_reference,
    save_session_ground_payload,
)


def _empty_ground_points() -> np.ndarray:
    s = np.linspace(-100.0, 100.0, 81)
    return np.column_stack(
        [
            s,
            np.zeros_like(s),
            0.02 * s + 1.5,
        ]
    )


def _bound_reference(points: np.ndarray, config, pipeline: FramePipeline):
    return fit_session_ground_reference_from_support(
        points,
        config.measurement,
        support_source="manual_ground_roi",
        active_ground_extrinsic_source=pipeline.ground_extrinsic_source,
        ground_extrinsic_generation=pipeline.ground_extrinsic_generation,
        frame_host_monotonic_ns=1,
        support_metadata={"roi_count": 1},
    )


class SessionGroundReferenceTests(unittest.TestCase):
    def test_ground5c_frozen_json_loads_without_refitting(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "ground5c_frozen_session_linear_0821"
            / "frozen_session_linear.json"
        )
        if not path.exists():
            self.skipTest("Ground-5C A-2 output is not present in this checkout")

        reference = load_frozen_session_ground_reference(
            path,
            active_ground_extrinsic_source="session",
            ground_extrinsic_generation=7,
        )
        self.assertEqual(reference.coordinate, "physical_S")
        self.assertEqual(reference.coordinate_formula, "S=(XY-origin_xy) dot direction_xy")
        self.assertEqual(reference.fit_pose_ids, ("001", "002", "003", "004", "005"))
        self.assertEqual(reference.ground_extrinsic_generation, 7)
        self.assertEqual(
            reference.frozen_json_sha256,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self.assertAlmostEqual(reference.slope, -0.000382799377854498)
        self.assertAlmostEqual(reference.intercept, -0.1969349667207237)
        self.assertEqual(
            reference.valid_s_range,
            (-139.76604886428078, 144.30211420107466),
        )
        self.assertEqual(
            reference.support_metadata["formal_bin_count"],
            39,
        )

    def test_ground5c_frozen_json_preserves_domain_boundary_behavior(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "ground5c_frozen_session_linear_0821"
            / "frozen_session_linear.json"
        )
        if not path.exists():
            self.skipTest("Ground-5C A-2 output is not present in this checkout")
        reference = load_frozen_session_ground_reference(
            path,
            active_ground_extrinsic_source="session",
            ground_extrinsic_generation=1,
        )
        origin = reference.origin_xy
        direction = reference.direction_xy
        s = np.asarray(
            [reference.valid_s_range_mm[0], 0.0, reference.valid_s_range_mm[1] + 1.0],
            dtype=np.float64,
        )
        xy = origin + s[:, None] * direction
        points = np.column_stack(
            [xy, reference.slope_z_per_mm * s + reference.intercept_z_mm + 3.0]
        )
        corrected, valid = reference.apply_to_points(points)
        np.testing.assert_array_equal(valid, np.asarray([True, True, False]))
        np.testing.assert_allclose(
            corrected[:2, 2],
            np.full(2, 3.0),
            atol=1.0e-12,
        )
        self.assertEqual(corrected[2, 2], points[2, 2])

    def test_session_fit_reuses_linear_baseline_kernel(self) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        reference = fit_session_ground_reference(
            _empty_ground_points(), config.measurement
        )

        self.assertEqual(reference.status, "VALID")
        self.assertEqual(reference.source, "session_laser_ground")
        self.assertAlmostEqual(reference.slope, 0.02, places=10)
        self.assertAlmostEqual(reference.intercept, 1.5, places=10)
        self.assertLess(reference.rmse, 1.0e-10)
        self.assertEqual(reference.valid_s_range, (-100.0, 100.0))

        corrected, valid = reference.apply_to_points(_empty_ground_points())
        self.assertTrue(valid.all())
        np.testing.assert_allclose(corrected[:, 2], 0.0, atol=1.0e-10)

    def test_ground_reference_does_not_extrapolate_outside_fitted_s_domain(
        self,
    ) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        reference = fit_session_ground_reference(
            _empty_ground_points(), config.measurement
        )
        points = np.asarray(
            [
                [0.0, 0.0, 1.5],
                [150.0, 0.0, 20.0],
            ],
            dtype=np.float64,
        )
        corrected, valid = reference.apply_to_points(points)
        np.testing.assert_array_equal(valid, np.asarray([True, False]))
        self.assertAlmostEqual(corrected[0, 2], 0.0, places=10)
        self.assertAlmostEqual(corrected[1, 2], 20.0, places=10)

    def test_reference_survives_switch_between_reference_and_session_pnp(self) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        pipeline = FramePipeline(config)
        pipeline.apply_session_ground_extrinsic(
            np.eye(3), np.zeros(3), generation=1
        )
        reference = _bound_reference(_empty_ground_points(), config, pipeline)
        pipeline.apply_session_ground_reference(reference)

        points = _empty_ground_points()
        first, first_meta = pipeline.apply_ground_reference_to_points(points)
        self.assertEqual(reference.provenance_source, "manual_ground_roi")
        self.assertEqual(reference.ground_extrinsic_generation, 1)

        # Re-applying the same active session pose must preserve the reference.
        pipeline.apply_session_ground_extrinsic(
            np.eye(3), np.zeros(3), generation=1
        )
        second, second_meta = pipeline.apply_ground_reference_to_points(points)

        np.testing.assert_allclose(first, second)
        self.assertEqual(first_meta["ground_reference_status"], "applied")
        self.assertEqual(second_meta["ground_reference_status"], "applied")
        self.assertEqual(pipeline.ground_extrinsic_source, "session")
        self.assertIs(pipeline.session_ground_reference, reference)

        # A genuinely new PnP generation invalidates the fitted reference.
        pipeline.apply_session_ground_extrinsic(
            np.eye(3), np.zeros(3), generation=2
        )
        self.assertIsNone(pipeline.session_ground_reference)
        _, stale_meta = pipeline.apply_ground_reference_to_points(points)
        self.assertEqual(stale_meta["ground_reference_status"], "inactive")

        pipeline.reset_ground_extrinsic(generation=3)
        self.assertEqual(pipeline.ground_extrinsic_source, "reference")
        self.assertIsNone(pipeline.session_ground_reference)

    def test_unbound_reference_cannot_enter_runtime_pipeline(self) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        pipeline = FramePipeline(config)
        reference = fit_session_ground_reference(
            _empty_ground_points(), config.measurement
        )
        with self.assertRaises(ValueError):
            pipeline.apply_session_ground_reference(reference)

    def test_pipeline_exposes_corrected_and_raw_ground_views(self) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        pipeline = FramePipeline(config)
        camera = SyntheticCameraSession(
            CameraConfig(width=2448, height=128, offset_y=960), target_fps=1000
        )
        camera.start()
        try:
            frame = camera.get_frame()
            raw_result = pipeline.run_frame(frame)
            reference = _bound_reference(raw_result.points_ground, config, pipeline)
            pipeline.apply_session_ground_reference(reference)
            corrected_result = pipeline.run_frame(frame)
        finally:
            camera.stop()

        np.testing.assert_array_equal(
            corrected_result.points_ground_raw, raw_result.points_ground
        )
        self.assertEqual(corrected_result.ground_reference_status, "applied")
        self.assertEqual(
            corrected_result.ground_reference_applied_count,
            len(corrected_result.points_ground),
        )
        self.assertEqual(corrected_result.ground_reference_out_of_range_count, 0)
        self.assertFalse(
            np.array_equal(corrected_result.points_ground, raw_result.points_ground)
        )

    def test_pnp_record_update_preserves_frozen_ground_reference_record(self) -> None:
        config = load_app_config(DEFAULT_CONFIG_PATH)
        reference = fit_session_ground_reference_from_support(
            _empty_ground_points(),
            config.measurement,
            support_source="manual_ground_roi",
            active_ground_extrinsic_source="reference",
            ground_extrinsic_generation=0,
            frame_host_monotonic_ns=1,
            support_metadata={"roi_count": 1},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/session_ground_calibration.json"
            save_session_ground_payload(
                path,
                {
                    "schema_version": 2,
                    "status": "VALID",
                    "runtime": {
                        "ground_extrinsic_source": "reference",
                        "ground_extrinsic_generation": 0,
                    },
                },
            )
            merge_session_ground_reference(
                path,
                reference.as_dict(),
                ground_extrinsic_source="reference",
            )
            save_session_ground_payload(
                path,
                {
                    "schema_version": 2,
                    "status": "VALID",
                    "runtime": {
                        "ground_extrinsic_source": "session",
                        "ground_extrinsic_generation": 1,
                    },
                },
            )
            saved = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(saved["session_ground_reference"]["status"], "VALID")
        self.assertEqual(saved["runtime"]["ground_extrinsic_source"], "session")
        self.assertEqual(
            saved["session_ground_reference_status"],
            "STALE_EXTRINSIC_GENERATION",
        )
        self.assertFalse(saved["session_ground_reference_runtime_valid"])


if __name__ == "__main__":
    unittest.main()
