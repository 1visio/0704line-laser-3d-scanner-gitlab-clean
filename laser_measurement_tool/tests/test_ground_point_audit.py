"""专项测试 for point-level Session ground audit export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from calibration.manifest import sha256_file
from online.ground_point_audit import (
    GroundPointAuditValidationError,
    build_frozen_chain_provenance,
    build_session_ground_plane_provenance,
    export_ground_point_audit,
)
from online.ground_sanity import evaluate_ground_sanity


TOOL_ROOT = Path(__file__).resolve().parents[1]
DAHENG_MANIFEST = TOOL_ROOT / "configs" / "calibration_daheng_0811" / "manifest.yaml"


class GroundPointAuditTests(unittest.TestCase):
    def _fixture(self) -> dict[str, object]:
        raw = np.asarray(
            [
                [10.0, 0.0, 0.1],
                [20.0, 0.0, 9.0],
                [30.0, 0.0, 0.2],
                [40.0, 0.0, 8.0],
            ],
            dtype=np.float64,
        )
        uv = np.asarray(
            [[100.0, 200.0], [110.0, 210.0], [120.0, 220.0], [130.0, 230.0]],
            dtype=np.float64,
        )
        camera = raw + np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        indices = np.asarray([2, 0], dtype=np.int64)
        selected = raw[indices]
        sanity = evaluate_ground_sanity(
            selected,
            ground_extrinsic_source="session",
            frame_number=101,
            session_calibration_frame_number=100,
            frame_host_monotonic_ns=200,
            session_calibration_host_monotonic_ns=100,
            session_generation=4,
            thresholds={"min_valid_points": 1},
        )
        return {
            "raw": raw,
            "uv": uv,
            "camera": camera,
            "indices": indices,
            "selected": selected,
            "sanity": sanity,
            "mask": np.asarray([True, False, True, False], dtype=bool),
            "metric": raw + np.asarray([0.0, 0.0, 1.0]),
        }

    def test_export_preserves_source_indices_and_raw_residual_view(self) -> None:
        fixture = self._fixture()
        frozen = {
            "frozen_c0": {"path": "quadratic_graph.yaml", "sha256": "a" * 64},
            "frozen_c1": {"path": "frozen_c1_4k.json", "sha256": "b" * 64},
        }
        plane = {
            "coordinate_system": "ground",
            "equation": "Zg=0",
            "a": 0.0,
            "b": 0.0,
            "c": 1.0,
            "d_mm": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            exported = export_ground_point_audit(
                Path(directory) / "ground_spatial_audit",
                session_id="session01",
                session_generation=4,
                ground_extrinsic_generation=4,
                frame_id="camera_000101_host_200",
                camera_frame_number=101,
                frame_host_monotonic_ns=200,
                frame_offset=(1760, 0),
                ground_extrinsic_source="session",
                calibration_package_id="package-1",
                calibration_manifest_sha256="c" * 64,
                algorithm_config_sha256="d" * 64,
                pixels_uv=fixture["uv"],
                points_camera=fixture["camera"],
                points_ground_raw=fixture["raw"],
                points_ground_metric=fixture["metric"],
                selected_indices=fixture["indices"],
                selected_mask=fixture["mask"],
                sanity_points=fixture["selected"],
                sanity_result=fixture["sanity"],
                mask_metadata={
                    "enabled": True,
                    "status": "applied",
                    "input_point_count": 4,
                    "selected_point_count": 2,
                },
                ground_plane=plane,
                session_pnp={"R_camera_to_ground": np.eye(3)},
                frozen_provenance=frozen,
            )

            manifest = json.loads(
                exported.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["point_count"], 2)
            self.assertEqual(manifest["candidate_point_count"], 4)
            self.assertEqual(manifest["sanity_input_point_count"], 2)
            self.assertTrue(manifest["coordinate_views"]["points_ground_raw"]["used_for_residual"])
            self.assertFalse(manifest["coordinate_views"]["points_ground_metric"]["used_for_residual"])
            self.assertEqual(
                manifest["validation"]["READY_FOR_SPATIAL_RESIDUAL_AUDIT"],
                "YES",
            )

            with exported.csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["source_point_index"]) for row in rows], [2, 0])
            self.assertEqual([float(row["Zg_mm"]) for row in rows], [0.2, 0.1])
            self.assertEqual(
                [float(row["Zg_metric_mm"]) for row in rows], [1.2, 1.1]
            )
            self.assertTrue(all(row["board_mask_selected"] == "True" for row in rows))
            np.testing.assert_allclose(
                [float(row["r_G_mm"]) for row in rows], [0.2, 0.1]
            )

    def test_mismatched_sanity_slice_is_rejected_before_writing(self) -> None:
        fixture = self._fixture()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "ground_spatial_audit"
            with self.assertRaises(GroundPointAuditValidationError):
                export_ground_point_audit(
                    target,
                    session_id="session01",
                    session_generation=4,
                    ground_extrinsic_generation=4,
                    frame_id="frame",
                    camera_frame_number=101,
                    frame_host_monotonic_ns=200,
                    frame_offset=(0, 0),
                    ground_extrinsic_source="session",
                    calibration_package_id="package-1",
                    calibration_manifest_sha256="c" * 64,
                    algorithm_config_sha256="d" * 64,
                    pixels_uv=fixture["uv"],
                    points_camera=fixture["camera"],
                    points_ground_raw=fixture["raw"],
                    points_ground_metric=None,
                    selected_indices=fixture["indices"],
                    selected_mask=fixture["mask"],
                    sanity_points=fixture["raw"][np.asarray([0, 2])],
                    sanity_result=fixture["sanity"],
                    mask_metadata={},
                    ground_plane={"a": 0.0, "b": 0.0, "c": 1.0, "d_mm": 0.0},
                    session_pnp={},
                    frozen_provenance={
                        "frozen_c0": {"sha256": "a" * 64},
                        "frozen_c1": {"sha256": "b" * 64},
                    },
                )
            self.assertFalse(target.exists())

    def test_manifest_provenance_hashes_are_verified(self) -> None:
        provenance = build_frozen_chain_provenance(
            DAHENG_MANIFEST,
            calibration_package_id="package-1",
            calibration_manifest_sha256=sha256_file(DAHENG_MANIFEST),
            algorithm_config_sha256="d" * 64,
        )
        self.assertTrue(provenance["frozen_c0"]["verified"])
        self.assertTrue(provenance["frozen_c1"]["verified"])
        self.assertEqual(len(provenance["frozen_c0"]["sha256"]), 64)
        self.assertEqual(len(provenance["frozen_c1"]["sha256"]), 64)

    def test_session_plane_metadata_uses_existing_pnp_transform(self) -> None:
        result = SimpleNamespace(
            status="success",
            reprojection_rmse_px=0.12,
            R=np.eye(3, dtype=np.float64),
            t=np.asarray([0.0, 0.0, 700.0], dtype=np.float64),
            rvec=np.zeros(3, dtype=np.float64),
            tvec=np.asarray([0.0, 0.0, 700.0], dtype=np.float64),
            T_ground_from_camera=np.eye(4, dtype=np.float64),
        )
        provenance = build_session_ground_plane_provenance(result)
        self.assertEqual(provenance["ground_plane"]["z_plane_mm"], 0.0)
        self.assertEqual(provenance["session_pnp"]["camera_plane"]["d_mm"], 700.0)


if __name__ == "__main__":
    unittest.main()
