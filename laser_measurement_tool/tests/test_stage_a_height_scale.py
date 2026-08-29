"""Stage-A height-scale integration contract tests."""

import unittest

from app_config import DEFAULT_CONFIG_PATH, load_app_config
from correction.stage_a_height_scale import (
    CorrectionConfig,
    STAGE_A_HEIGHT_SCALE_MODE,
    apply_stage_a_height_scale,
    resolve_stage_a_height_scale,
)


class StageAHeightScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.default_config = load_app_config(DEFAULT_CONFIG_PATH)
        cls.daheng_config = load_app_config(
            DEFAULT_CONFIG_PATH.parent / "measure_tool_daheng_0811.yaml"
        )
        cls.stage_a_config = cls.daheng_config.correction.stage_a_height_scale
        assert cls.stage_a_config is not None

    def test_default_switch_is_off_and_preserves_raw_height(self) -> None:
        self.assertFalse(
            self.default_config.correction.stage_a_height_scale_enabled
        )
        result = resolve_stage_a_height_scale(
            12.0,
            system=self.default_config.system,
            correction=self.default_config.correction,
        )
        self.assertEqual(result.height_raw, 12.0)
        self.assertEqual(result.height_stage_a, 12.0)
        self.assertFalse(result.stage_a_enabled)
        self.assertFalse(result.stage_a_valid)
        self.assertEqual(result.stage_a_status, "disabled")

    def test_frozen_daheng_config_metadata(self) -> None:
        assert self.stage_a_config is not None
        self.assertEqual(self.daheng_config.system, "daheng")
        self.assertEqual(self.stage_a_config.system, "daheng")
        self.assertEqual(self.stage_a_config.status, "experimental_stage_validated")
        self.assertEqual(self.stage_a_config.valid_height_mm, (1.0, 30.0))
        self.assertEqual(self.stage_a_config.scale, 1.00403395913372)
        self.assertTrue(
            self.daheng_config.correction.stage_a_height_scale_enabled
        )

    def test_daheng_only_gate(self) -> None:
        result = apply_stage_a_height_scale(
            10.0,
            system="daheng",
            enabled=True,
            correction_mode=STAGE_A_HEIGHT_SCALE_MODE,
            config=self.stage_a_config,
        )
        self.assertTrue(result.stage_a_enabled)
        self.assertTrue(result.stage_a_valid)
        self.assertEqual(result.stage_a_status, "applied")

        unsupported = apply_stage_a_height_scale(
            10.0,
            system="mvs",
            enabled=True,
            correction_mode=STAGE_A_HEIGHT_SCALE_MODE,
            config=self.stage_a_config,
        )
        self.assertEqual(unsupported.height_stage_a, 10.0)
        self.assertFalse(unsupported.stage_a_enabled)
        self.assertFalse(unsupported.stage_a_valid)
        self.assertEqual(unsupported.stage_a_status, "unsupported_system")

    def test_one_and_thirty_mm_are_inclusive_boundaries(self) -> None:
        for raw in (1.0, 30.0):
            with self.subTest(raw=raw):
                result = apply_stage_a_height_scale(
                    raw,
                    system="daheng",
                    enabled=True,
                    correction_mode=STAGE_A_HEIGHT_SCALE_MODE,
                    config=self.stage_a_config,
                )
                self.assertTrue(result.stage_a_valid)
                self.assertEqual(result.stage_a_status, "applied")
                self.assertEqual(
                    result.height_stage_a,
                    self.stage_a_config.scale * raw,
                )

    def test_out_of_domain_retains_raw_and_is_marked(self) -> None:
        for raw in (0.999, 30.000001):
            with self.subTest(raw=raw):
                result = apply_stage_a_height_scale(
                    raw,
                    system="daheng",
                    enabled=True,
                    correction_mode=STAGE_A_HEIGHT_SCALE_MODE,
                    config=self.stage_a_config,
                )
                self.assertEqual(result.height_stage_a, raw)
                self.assertTrue(result.stage_a_enabled)
                self.assertFalse(result.stage_a_valid)
                self.assertEqual(result.stage_a_status, "out_of_valid_domain")

    def test_raw_below_one_mm_is_not_rejected_but_is_not_corrected(self) -> None:
        result = apply_stage_a_height_scale(
            0.5,
            system="daheng",
            enabled=True,
            correction_mode=STAGE_A_HEIGHT_SCALE_MODE,
            config=self.stage_a_config,
        )
        self.assertEqual(result.height_raw, 0.5)
        self.assertEqual(result.height_stage_a, 0.5)
        self.assertTrue(result.stage_a_enabled)
        self.assertFalse(result.stage_a_valid)
        self.assertEqual(result.stage_a_status, "out_of_valid_domain")

    def test_frozen_numeric_reproduction(self) -> None:
        result = apply_stage_a_height_scale(
            12.5,
            system="daheng",
            enabled=True,
            correction_mode=STAGE_A_HEIGHT_SCALE_MODE,
            config=self.stage_a_config,
        )
        self.assertEqual(result.height_raw, 12.5)
        self.assertAlmostEqual(result.height_stage_a, 12.5504244891715, places=14)

    def test_surface_aware_mode_is_reserved_and_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            CorrectionConfig(
                mode="surface_aware",
                stage_a_height_scale_enabled=True,
            )
        result = apply_stage_a_height_scale(
            12.5,
            system="daheng",
            enabled=False,
            correction_mode="surface_aware",
            config=self.stage_a_config,
        )
        self.assertEqual(result.height_stage_a, 12.5)
        self.assertEqual(result.stage_a_status, "disabled")


if __name__ == "__main__":
    unittest.main()
