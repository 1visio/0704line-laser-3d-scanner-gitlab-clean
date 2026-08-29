"""Artifact-backed regression for the Daheng session-ground PnP API.

The historical TIFFs are intentionally external calibration artifacts and are
not copied into the repository.  When the recorded artifact path is present,
this test runs the real image regression; otherwise it is skipped explicitly.
"""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

import cv2
import numpy as np

from calibration.config_loader import load_calibration
from calibration.session_ground import BoardConfig, estimate_session_ground_extrinsic


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_TOOL_ROOT = REPO_ROOT.parent / "calibration_tool"
CALIBRATION_DIR = (
    REPO_ROOT
    / "laser_measurement_tool"
    / "configs"
    / "calibration_daheng_0811"
)
HISTORICAL_EXTRINSICS_DIR = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "extrinsics" / "fit"
)
LEGACY_LASER_PLANE_DIR = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit"
)
LEGACY_FRAME_SUMMARY = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0814"
    / "board_coordinate_audit"
    / "frame_board_geometry_summary.csv"
)


def _historical_image_paths() -> list[Path]:
    return sorted(HISTORICAL_EXTRINSICS_DIR.glob("chess *.tif"))


def _legacy_frame_rows() -> dict[str, dict[str, str]]:
    if not LEGACY_FRAME_SUMMARY.is_file():
        return {}
    with LEGACY_FRAME_SUMMARY.open(newline="", encoding="utf-8") as stream:
        return {
            row["frame_id"]: row
            for row in csv.DictReader(stream)
            if row.get("row_type") == "frame"
        }


class DahengSessionGroundRegressionTests(unittest.TestCase):
    def test_historical_daheng_image_matches_recorded_ground_frame(self) -> None:
        image_paths = _historical_image_paths()
        if not image_paths:
            self.skipTest(
                "historical Daheng extrinsics TIFFs are external and unavailable"
            )

        calibration = load_calibration(CALIBRATION_DIR)
        expected_normal = np.asarray(calibration["R"], dtype=np.float64)[2]
        expected_origin = -calibration["R"].T @ calibration["t"]
        board = BoardConfig(
            pattern_cols=11,
            pattern_rows=8,
            square_size_mm=20.0,
            detector="sb_then_classic",
        )

        for image_path in image_paths:
            with self.subTest(image=image_path.name):
                encoded = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(image)

                result = estimate_session_ground_extrinsic(
                    image,
                    calibration,
                    board,
                )
                self.assertEqual(result.status, "success", result.message)
                self.assertLess(result.reprojection_rmse_px or np.inf, 0.5)
                assert result.ground_normal_in_camera is not None
                assert result.ground_origin_in_camera is not None

                normal_dot = float(
                    result.ground_normal_in_camera @ expected_normal
                )
                self.assertGreater(normal_dot, np.cos(np.deg2rad(0.5)))
                self.assertLess(
                    float(
                        np.linalg.norm(
                            result.ground_origin_in_camera - expected_origin
                        )
                    ),
                    2.0,
                )

    def test_legacy_pnp_summary_matches_shared_api(self) -> None:
        rows = _legacy_frame_rows()
        image_paths = {
            path.stem.split()[-1]: path
            for path in LEGACY_LASER_PLANE_DIR.glob("chess *.tif")
        }
        frame_ids = sorted(set(rows) & set(image_paths))
        if not frame_ids:
            self.skipTest(
                "legacy Daheng PnP images/summary are external and unavailable"
            )

        calibration = load_calibration(CALIBRATION_DIR)
        board = BoardConfig(
            pattern_cols=11,
            pattern_rows=8,
            square_size_mm=20.0,
            detector="sb_then_classic",
        )
        for frame_id in frame_ids:
            with self.subTest(frame=frame_id):
                row = rows[frame_id]
                image_path = image_paths[frame_id]
                image = cv2.imdecode(
                    np.fromfile(image_path, dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
                result = estimate_session_ground_extrinsic(
                    image,
                    calibration,
                    board,
                )
                self.assertEqual(result.status, "success", result.message)
                assert result.rvec is not None
                assert result.tvec is not None
                self.assertAlmostEqual(
                    result.reprojection_rmse_px or np.inf,
                    float(row["pnp_rmse_px"]),
                    delta=0.01,
                )
                expected_rvec = np.array(
                    [float(row[key]) for key in ("rvec_x", "rvec_y", "rvec_z")],
                    dtype=np.float64,
                )
                expected_tvec = np.array(
                    [
                        float(row[key])
                        for key in (
                            "board_origin_x_mm",
                            "board_origin_y_mm",
                            "board_origin_z_mm",
                        )
                    ],
                    dtype=np.float64,
                )
                np.testing.assert_allclose(
                    result.rvec.reshape(3),
                    expected_rvec,
                    atol=2.5e-4,
                )
                np.testing.assert_allclose(
                    result.tvec.reshape(3),
                    expected_tvec,
                    atol=0.02,
                )


if __name__ == "__main__":
    unittest.main()
