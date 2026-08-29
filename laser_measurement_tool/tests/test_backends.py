"""laser.backends 重心法提取的合成条纹单元测试。"""

import unittest

import numpy as np

from laser.backends import (
    AVAILABLE_METHODS,
    CentroidParams,
    _correct_segment_v,
    _load_shared_steger_module,
    centroid_backend,
    create_extraction_params,
    shared_steger_backend,
    steger_backend,
)
from laser.laser_extractor import extract_laser_center


def _horizontal_stripe_image(
    width: int = 260,
    height: int = 120,
    slope: float = 0.05,
    intercept: float = 40.0,
    sigma: float = 1.6,
    amplitude: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成接近水平的高斯条纹图像及每列中心真值。"""
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    centre_v = intercept + slope * u
    profile = amplitude * np.exp(
        -((v[:, None] - centre_v[None, :]) ** 2) / (2.0 * sigma**2)
    )
    image = np.clip(profile, 0.0, 255.0).astype(np.uint8)
    return image, centre_v


_TEST_OPTIONS = {
    "background_kernel": 31,
    "min_local_contrast_dn": 20.0,
    "centroid_window_radius": 4,
    "segment_min_columns": 20,
    "continuity_max_column_gap": 2,
    "continuity_max_vertical_jump": 5.0,
    "correction_window": 1,
    "correction_max_shift": 3.5,
    "scan_axis": "column",
}

_STEGER_OPTIONS = {
    "sigma": 3.0,
    "threshold": 30.0,
    "deriv_thresh": 0.5,
    "roi_margin": 10,
    "roi_max_height": 100,
    "scan_axis": "column",
}


class CentroidBackendTest(unittest.TestCase):
    def test_segment_correction_matches_reference_loop(self) -> None:
        values = np.array(
            [11.0, 12.5, 9.25, 15.0, 16.75, 13.0, 14.5], dtype=np.float64
        )
        contrast = np.array(
            [22.0, 45.0, 31.0, 80.0, 55.0, 19.0, 64.0], dtype=np.float64
        )
        params = CentroidParams(correction_window=5, correction_max_shift=2.25)
        weights = contrast / max(float(np.max(contrast)), 1.0e-6) + 1.0e-4
        expected = values.copy()
        radius = params.correction_window // 2
        for index in range(len(values)):
            left = max(0, index - radius)
            right = min(len(values), index + radius + 1)
            estimate = np.sum(
                weights[left:right] * values[left:right]
            ) / np.sum(weights[left:right])
            expected[index] = values[index] + np.clip(
                estimate - values[index],
                -params.correction_max_shift,
                params.correction_max_shift,
            )

        np.testing.assert_allclose(
            _correct_segment_v(values, contrast, params), expected, rtol=1e-12
        )

    def test_recovers_subpixel_centres_column_axis(self) -> None:
        image, truth = _horizontal_stripe_image()
        points = centroid_backend(image, _TEST_OPTIONS)
        self.assertGreater(len(points), 200)
        # 只核对内部列，避免边界效应
        for u, v in points:
            column = int(u)
            if 10 <= column < image.shape[1] - 10:
                self.assertAlmostEqual(v, truth[column], delta=0.3)

    def test_row_axis_handles_vertical_stripe(self) -> None:
        image, truth = _horizontal_stripe_image()
        vertical = np.ascontiguousarray(image.T)
        options = dict(_TEST_OPTIONS, scan_axis="row")
        points = centroid_backend(vertical, options)
        self.assertGreater(len(points), 200)
        # 竖直条纹: 每行 v 对应中心 u = truth[v]
        for u, v in points:
            row = int(v)
            if 10 <= row < vertical.shape[0] - 10:
                self.assertAlmostEqual(u, truth[row], delta=0.3)

    def test_low_contrast_image_returns_empty(self) -> None:
        image = np.full((80, 80), 10, dtype=np.uint8)
        points = centroid_backend(image, _TEST_OPTIONS)
        self.assertEqual(len(points), 0)

    def test_unknown_option_raises(self) -> None:
        image, _ = _horizontal_stripe_image()
        with self.assertRaises(ValueError):
            centroid_backend(image, {"no_such_option": 1})

    def test_invalid_kernel_raises(self) -> None:
        image, _ = _horizontal_stripe_image()
        with self.assertRaises(ValueError):
            centroid_backend(image, {"background_kernel": 4})


class CreateExtractionParamsTest(unittest.TestCase):
    def test_centroid_params_flow_through_extract(self) -> None:
        image, truth = _horizontal_stripe_image()
        params = create_extraction_params("centroid", _TEST_OPTIONS)
        points = extract_laser_center(image, params)
        self.assertGreater(len(points), 200)
        self.assertEqual(points.shape[1], 2)

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_extraction_params("no_such_method")

    def test_steger_is_registered_and_flows_through_extract(self) -> None:
        image, _ = _horizontal_stripe_image(slope=0.0, intercept=40.25)
        params = create_extraction_params("steger", _STEGER_OPTIONS)
        points = extract_laser_center(image, params)

        self.assertIn("steger", AVAILABLE_METHODS)
        self.assertTrue(callable(AVAILABLE_METHODS["steger"]))
        self.assertGreater(len(points), 200)

    def test_shared_steger_implementation_is_bundled(self) -> None:
        shared = _load_shared_steger_module()

        self.assertEqual(shared.__package__, "laser")
        self.assertTrue(callable(shared.extract_steger_columns))


class StegerBackendTest(unittest.TestCase):
    def test_shared_name_is_an_exact_realtime_alias(self) -> None:
        image, _ = _horizontal_stripe_image(slope=0.0, intercept=40.25)
        realtime = steger_backend(image, _STEGER_OPTIONS)
        alias = shared_steger_backend(image, _STEGER_OPTIONS)
        np.testing.assert_array_equal(alias, realtime)

    def test_recovers_exactly_horizontal_subpixel_stripe(self) -> None:
        image, truth = _horizontal_stripe_image(
            slope=0.0, intercept=40.25, sigma=2.0
        )
        points = steger_backend(image, _STEGER_OPTIONS)

        self.assertGreater(len(points), 200)
        np.testing.assert_allclose(
            points[:, 1], truth[points[:, 0].astype(int)], atol=0.08
        )

    def test_row_axis_handles_vertical_stripe(self) -> None:
        image, truth = _horizontal_stripe_image(sigma=2.0)
        vertical = np.ascontiguousarray(image.T)
        options = dict(_STEGER_OPTIONS, scan_axis="row")
        points = steger_backend(vertical, options)

        self.assertGreater(len(points), 200)
        rows = points[:, 1].astype(int)
        interior = (rows >= 10) & (rows < len(truth) - 10)
        np.testing.assert_allclose(
            points[interior, 0], truth[rows[interior]], atol=0.08
        )

    def test_configured_search_roi_forces_full_region_and_restores_local_coordinates(self) -> None:
        height, width = 260, 180
        rows = np.arange(height, dtype=np.float64)
        columns = np.arange(width, dtype=np.float64)
        image = np.zeros((height, width), dtype=np.float64)
        centres = np.where(rows < height / 2, 35.25, 140.5)
        image += 220.0 * np.exp(
            -((columns[None, :] - centres[:, None]) ** 2) / (2.0 * 1.8**2)
        )
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        options = dict(
            _STEGER_OPTIONS,
            scan_axis="row",
            roi_margin=5,
            roi_max_height=30,
            search_roi={
                "offset_x": 1000,
                "offset_y": 500,
                "width": width,
                "height": height,
            },
            _image_offset=(1000, 500),
        )

        points = steger_backend(image, options)

        self.assertGreater(len(points), 200)
        self.assertTrue(np.any(points[:, 0] < 50.0))
        self.assertTrue(np.any(points[:, 0] > 125.0))
        self.assertGreaterEqual(float(points[:, 1].min()), 0.0)
        self.assertLess(float(points[:, 1].max()), height)

    def test_search_roi_outside_frame_returns_empty(self) -> None:
        image, _ = _horizontal_stripe_image()
        options = dict(
            _STEGER_OPTIONS,
            search_roi={
                "offset_x": 1000,
                "offset_y": 1000,
                "width": 100,
                "height": 100,
            },
        )
        self.assertEqual(len(steger_backend(image, options)), 0)

    def test_configured_search_roi_skips_auto_band_seed_validation(self) -> None:
        image = np.zeros((80, 80), dtype=np.uint8)
        image[10, :] = 20
        image[50, 40] = 255
        options = dict(
            _STEGER_OPTIONS,
            search_roi={
                "offset_x": 0,
                "offset_y": 0,
                "width": image.shape[1],
                "height": image.shape[0],
            },
        )

        points = steger_backend(image, options)

        self.assertEqual(points.ndim, 2)
        self.assertEqual(points.shape[1], 2)

    def test_low_contrast_image_returns_empty(self) -> None:
        points = steger_backend(
            np.full((80, 80), 10, dtype=np.uint8), _STEGER_OPTIONS
        )
        self.assertEqual(len(points), 0)

    def test_invalid_or_unknown_options_raise(self) -> None:
        image, _ = _horizontal_stripe_image()
        with self.assertRaises(ValueError):
            steger_backend(image, {"sigma": 0.0})
        with self.assertRaises(ValueError):
            steger_backend(image, {"no_such_option": 1})


if __name__ == "__main__":
    unittest.main()
