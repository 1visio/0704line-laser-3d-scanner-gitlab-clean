"""Frozen H-B2 runtime semantics and mutually-exclusive mode contracts."""

from dataclasses import replace
import unittest

from app_config import DEFAULT_CONFIG_PATH, load_app_config
from correction.stage_a_height_scale import (
    HB2_Q2_CLAMP_DIAGNOSTIC_POLICY,
    HB2_Q2_EXTRAPOLATE_DIAGNOSTIC_POLICY,
    HB2_Q2_REJECT_POLICY,
    CorrectionConfig,
    resolve_height_correction,
)
from reconstruction.reconstructor import frozen_c0_q_coordinates


class FrozenHB2HeightCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_app_config(
            DEFAULT_CONFIG_PATH.parent / "measure_tool_daheng_0811.yaml"
        )
        cls.correction = cls.config.correction
        cls.hb2 = cls.correction.hb2_height_correction
        assert cls.hb2 is not None

    def test_frozen_parameters_are_loaded_without_refit(self) -> None:
        assert self.hb2 is not None
        self.assertEqual(self.hb2.a0_mm, -0.10068827127712787)
        self.assertEqual(self.hb2.a2_mm_per_q2, 0.053274373969597236)
        self.assertEqual(
            self.hb2.q2_domain,
            (-0.50189189917237, 1.4605125871893883),
        )
        self.assertFalse(self.hb2.production_default)

    def test_modes_are_mutually_exclusive(self) -> None:
        raw = 12.5
        h1 = resolve_height_correction(
            raw,
            q2=0.2,
            system="daheng",
            correction=self.correction,
            mode_override="h1",
        )
        hb2 = resolve_height_correction(
            raw,
            q2=0.2,
            system="daheng",
            correction=self.correction,
            mode_override="hb2",
        )
        none = resolve_height_correction(
            raw,
            q2=0.2,
            system="daheng",
            correction=self.correction,
            mode_override="none",
        )
        self.assertEqual(h1.active_height_correction, "h1")
        self.assertEqual(hb2.active_height_correction, "hb2")
        self.assertEqual(none.active_height_correction, "none")
        self.assertAlmostEqual(h1.active_height, h1.height_h1)
        self.assertAlmostEqual(hb2.active_height, hb2.height_hb2)
        self.assertEqual(none.active_height, raw)
        self.assertNotEqual(h1.active_height, hb2.active_height)

    def test_hb2_matches_frozen_formula(self) -> None:
        raw = 12.5
        q2 = 0.2
        result = resolve_height_correction(
            raw,
            q2=q2,
            system="daheng",
            correction=self.correction,
            mode_override="hb2",
        )
        expected = raw - (self.hb2.a0_mm + self.hb2.a2_mm_per_q2 * q2)
        self.assertEqual(result.hb2_q2_status, "applied")
        self.assertTrue(result.q2_in_domain)
        self.assertAlmostEqual(result.height_hb2, expected, places=14)

    def test_ood_extrapolates_with_status_and_reject_remains_available(self) -> None:
        result = resolve_height_correction(
            12.5,
            q2=2.0,
            system="daheng",
            correction=self.correction,
            mode_override="hb2",
        )
        expected_extrapolated = 12.5 - (
            self.hb2.a0_mm + self.hb2.a2_mm_per_q2 * 2.0
        )
        self.assertFalse(result.q2_in_domain)
        self.assertAlmostEqual(result.height_hb2, expected_extrapolated, places=14)
        self.assertAlmostEqual(result.active_height, expected_extrapolated, places=14)
        self.assertFalse(result.active_height_valid)
        self.assertEqual(result.active_height_status, "HB2_Q2_OOD")

        reject_config = replace(
            self.correction,
            hb2_q2_policy=HB2_Q2_REJECT_POLICY,
        )
        rejected = resolve_height_correction(
            12.5,
            q2=2.0,
            system="daheng",
            correction=reject_config,
            mode_override="hb2",
        )
        self.assertIsNone(rejected.height_hb2)
        self.assertIsNone(rejected.active_height)
        self.assertFalse(rejected.active_height_valid)
        self.assertEqual(rejected.active_height_status, "HB2_Q2_OOD")

        clamp_config = CorrectionConfig(
            mode="hb2",
            stage_a_height_scale_enabled=False,
            stage_a_height_scale_config=self.correction.stage_a_height_scale_config,
            stage_a_height_scale=self.correction.stage_a_height_scale,
            hb2_height_correction_config=self.correction.hb2_height_correction_config,
            hb2_height_correction=self.correction.hb2_height_correction,
            hb2_q2_policy=HB2_Q2_CLAMP_DIAGNOSTIC_POLICY,
        )
        clamped = resolve_height_correction(
            12.5,
            q2=2.0,
            system="daheng",
            correction=clamp_config,
            mode_override="hb2",
        )
        expected = 12.5 - (
            self.hb2.a0_mm + self.hb2.a2_mm_per_q2 * self.hb2.q2_domain[1]
        )
        self.assertEqual(clamped.active_height_status, "HB2_Q2_CLAMPED_DIAGNOSTIC")
        self.assertAlmostEqual(clamped.active_height, expected, places=14)

        self.assertEqual(
            self.correction.hb2_q2_policy,
            HB2_Q2_EXTRAPOLATE_DIAGNOSTIC_POLICY,
        )

    def test_invalid_q2_is_explicitly_flagged(self) -> None:
        result = resolve_height_correction(
            12.5,
            q2="not-a-number",
            system="daheng",
            correction=self.correction,
            mode_override="hb2",
        )
        self.assertFalse(result.q2_in_domain)
        self.assertIsNone(result.height_hb2)
        self.assertEqual(result.active_height_status, "HB2_Q2_INVALID")

    def test_q_coordinates_use_frozen_c0_axes_and_normalization(self) -> None:
        calibration = {
            "laser_model": {
                "model_type": "quadratic_graph",
                "independent_axes": ["Y", "Z"],
                "normalization": {
                    "independent_center_mm": [10.0, 500.0],
                    "independent_scale_mm": [20.0, 50.0],
                },
            }
        }
        q1, q2 = frozen_c0_q_coordinates(
            [[123.0, 14.0, 520.0]], calibration
        )
        self.assertAlmostEqual(q1[0], 0.2)
        self.assertAlmostEqual(q2[0], 0.4)


if __name__ == "__main__":
    unittest.main()
