"""Tests for the in-place leave-one-frame-out Session PnP QA."""

from __future__ import annotations

import json
import unittest

import cv2
import numpy as np

from calibration.session_ground import BoardConfig, estimate_session_ground_extrinsic_from_corners
from online.session_calibration import (
    SessionGroundPnPQA,
    assess_session_pnp_qa,
    aggregate_session_ground_extrinsic,
    build_session_ground_payload,
)


class SessionPnPQATests(unittest.TestCase):
    def setUp(self) -> None:
        self.board = BoardConfig(11, 8, 20.0)
        self.K = np.asarray(
            [[820.0, 0.0, 320.0], [0.0, 818.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.D = np.zeros(5, dtype=np.float64)
        rvec = np.asarray([[0.045], [-0.035], [0.012]], dtype=np.float64)
        tvec = np.asarray([[35.0], [-18.0], [720.0]], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            self.board.object_points(), rvec, tvec, self.K, self.D
        )
        base = projected.reshape(-1, 2)
        self.results = [
            estimate_session_ground_extrinsic_from_corners(
                base + np.random.default_rng(seed).normal(0.0, 0.01, base.shape),
                {"K": self.K, "D": self.D},
                self.board,
            )
            for seed in range(5)
        ]
        self.final, self.repeatability = aggregate_session_ground_extrinsic(
            self.results,
            {"K": self.K, "D": self.D},
            self.board,
        )

    def test_five_leave_one_out_folds_report_pose_and_zg_metrics(self) -> None:
        qa = assess_session_pnp_qa(
            self.results,
            self.final,
            {"K": self.K, "D": self.D},
            self.board,
        )

        self.assertEqual(qa.method, "leave_one_frame_out")
        self.assertEqual(qa.fold_count, 5)
        self.assertEqual(qa.successful_folds, 5)
        self.assertEqual(qa.status, "PASS")
        self.assertEqual(len(qa.heldout_reprojection_rmse_px), 5)
        self.assertTrue(all(value is not None for value in qa.translation_delta_mm))
        self.assertGreater(qa.zg_propagation["full_fov"]["count"], 0)
        self.assertIsNotNone(qa.zg_propagation["rmse_mm"])
        self.assertIsNotNone(qa.zg_propagation["edge_p95_abs_mm"])
        self.assertEqual(qa.as_dict()["final_metrics"]["SESSION_PNP_JACKKNIFE"], "PASS")

    def test_payload_keeps_legacy_and_explicit_repeatability_protocols(self) -> None:
        qa = assess_session_pnp_qa(
            self.results,
            self.final,
            {"K": self.K, "D": self.D},
            self.board,
        )
        reference_R = np.eye(3, dtype=np.float64)
        reference_t = np.zeros(3, dtype=np.float64)
        payload = build_session_ground_payload(
            self.final,
            self.board,
            frame_number=5,
            frame_offset=(0, 0),
            reference_R=reference_R,
            reference_t=reference_t,
            runtime_source="session",
            repeatability=self.repeatability,
            session_pnp_qa=qa,
        )

        self.assertEqual(payload["repeatability"], payload["frame_repeatability"])
        self.assertEqual(payload["session_pnp_qa"]["method"], "leave_one_frame_out")
        json.dumps(payload, ensure_ascii=False)

    def test_failure_record_is_json_safe_and_does_not_claim_stability(self) -> None:
        qa = SessionGroundPnPQA.failure("synthetic QA failure")

        record = qa.as_dict()

        self.assertEqual(record["status"], "FAIL")
        self.assertEqual(record["stability"], "LOW")
        self.assertEqual(record["successful_fold_count"], 0)
        self.assertEqual(len(record["heldout_reprojection_rmse_px"]), 5)
        json.dumps(record, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
