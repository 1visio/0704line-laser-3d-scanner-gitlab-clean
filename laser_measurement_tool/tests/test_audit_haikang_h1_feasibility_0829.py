from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import audit_haikang_h1_feasibility_0829 as audit


def synthetic_conditions() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for height_id, truth in zip(audit.HEIGHT_IDS, audit.HEIGHTS_MM, strict=True):
        for position_id in ("p01", "p02"):
            rows.append(
                {
                    "condition": f"{height_id}_{position_id}",
                    "height_id": height_id,
                    "position_id": position_id,
                    "truth_mm": truth,
                    "raw_height_mm": truth / 1.01,
                }
            )
    return rows


class H1FitTest(unittest.TestCase):
    def test_reuses_through_origin_scale_fit(self) -> None:
        train = [
            {"raw_height_mm": 2.0, "truth_mm": 2.2},
            {"raw_height_mm": 4.0, "truth_mm": 4.4},
        ]
        self.assertAlmostEqual(audit.fit_h1_scale(train), 1.1)

    def test_uses_existing_h1_predictor(self) -> None:
        self.assertAlmostEqual(audit.apply_h1(10.0, 1.01), 10.1)

    def test_compare_reports_improvement_and_worsening(self) -> None:
        test = [
            {
                "condition": "h10_p01",
                "height_id": "h10",
                "position_id": "p01",
                "truth_mm": 10.0,
                "raw_height_mm": 10.0 / 1.01,
            },
            {
                "condition": "h10_p02",
                "height_id": "h10",
                "position_id": "p02",
                "truth_mm": 10.0,
                "raw_height_mm": 9.95,
            },
        ]
        comparison, predictions = audit.compare_metrics(test, 1.01)
        self.assertLess(comparison["h1"]["mae_mm"], comparison["raw"]["mae_mm"])
        self.assertEqual(len(predictions), 2)
        self.assertAlmostEqual(predictions[0]["h1_error_mm"], 0.0)

    def test_loho_keeps_one_condition_level_row_per_test(self) -> None:
        rows, predictions, _ = audit.run_scheme(
            synthetic_conditions(),
            scheme="LOHO",
            groups=[6.0, 10.0, 20.0],
            group_of=lambda row: row["truth_mm"],
            fold_type="INTERPOLATION",
            held_out_height=lambda group: group,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(predictions), 6)
        self.assertTrue(all(row["train_condition_count"] == 8 for row in rows))
        self.assertTrue(all(row["test_condition_count"] == 2 for row in rows))


if __name__ == "__main__":
    unittest.main()
