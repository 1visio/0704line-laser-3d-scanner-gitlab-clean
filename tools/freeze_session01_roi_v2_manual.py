#!/usr/bin/env python3
"""Materialize the user-reviewed Session01 Auto ROI V2 registry.

This file intentionally does not alter any ROI geometry.  It only records the
explicit human-review decision supplied for Task A-13B-v2.  Auto QC status and
reasons remain unchanged, so an ``UNCERTAIN`` automatic result is not silently
rewritten as an automatic pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_CONDITIONS = {
    f"{height}_p{position:02d}"
    for height in ("h10", "h20", "h30")
    for position in range(1, 11)
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def freeze_registry(draft_path: Path, output_path: Path) -> dict[str, Any]:
    draft_bytes = draft_path.read_bytes()
    draft = json.loads(draft_bytes.decode("utf-8"))
    entries = list(draft.get("entries", []))
    condition_ids = {str(entry.get("condition_id")) for entry in entries}
    if len(entries) != 30 or condition_ids != EXPECTED_CONDITIONS:
        raise RuntimeError(
            f"V2 draft must contain exactly 30 expected conditions; got {len(entries)}"
        )

    frozen_at = datetime.now(timezone.utc).isoformat()
    registry = copy.deepcopy(draft)
    registry.update(
        {
            "roi_stage": "manual_frozen_v2",
            "human_reviewed": True,
            "human_decision": "ACCEPT_ALL_V2",
            "manual_confirmed": True,
            "manual_confirmed_count": 30,
            "frozen": True,
            "frozen_at": frozen_at,
            "review_status": "FROZEN_MANUAL_ACCEPT_ALL_V2",
            "freeze_policy": (
                "Geometry-only manual review of all 30 V2 overlays. The user supplied "
                "HUMAN_REVIEW_30_OF_30=COMPLETE and HUMAN_GEOMETRY_DECISION="
                "ACCEPT_ALL_V2. No truth height, reconstruction, residual, q1/q2, "
                "or model result was used to alter ROI geometry."
            ),
            "human_review_provenance": {
                "review_source": "user_supplied_task_completion_statement",
                "reviewed_overlay_count": 30,
                "decision": "ACCEPT_ALL_V2",
                "reviewed_geometry_only": True,
                "reviewed_at_utc": frozen_at,
                "auto_qc_status_preserved": True,
                "auto_qc_reasons_preserved": True,
            },
            "source_draft": str(draft_path.resolve()),
            "source_draft_sha256": hashlib.sha256(draft_bytes).hexdigest(),
            "position_coordinate_definition": "height_roi_formal_v_median; never whole-frame v median",
        }
    )
    for entry in registry["entries"]:
        # These fields are the formal human decision.  Do not touch any V2
        # geometry or automatic QC fields, in particular UNCERTAIN statuses.
        entry["auto_candidate_generated"] = True
        entry["human_reviewed"] = True
        entry["human_decision"] = "ACCEPTED"
        entry["manual_confirmed"] = True
        entry["frozen"] = True
        entry["review_status"] = "FROZEN_MANUALLY_ACCEPTED_V2"
        entry["manual_review_source"] = "user_supplied_task_completion_statement"
        entry["manual_review_note"] = (
            "Accepted under HUMAN_REVIEW_30_OF_30=COMPLETE and "
            "HUMAN_GEOMETRY_DECISION=ACCEPT_ALL_V2; geometry unchanged."
        )

    # Add the formal-point coordinate snapshot from the already-frozen cache.
    # This is geometry/provenance only; it does not run Steger or reconstruct.
    cache_npz = draft_path.with_name("session01_steger_centers.npz")
    cache_manifest = draft_path.with_name("session01_steger_centers_manifest.json")
    if cache_npz.is_file() and cache_manifest.is_file():
        manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
        with np.load(cache_npz, allow_pickle=False) as bundle:
            centers = np.asarray(bundle["centers_full"], dtype=np.float64)
            offsets = np.asarray(bundle["frame_offsets"], dtype=np.int64)
        centers_by_condition: dict[str, list[np.ndarray]] = {}
        for index, frame in enumerate(manifest.get("frames", [])):
            start, end = int(offsets[index]), int(offsets[index + 1])
            condition = f"{frame['height_label']}_{frame['position_id']}"
            centers_by_condition.setdefault(condition, []).append(centers[start:end])
        for entry in registry["entries"]:
            condition = str(entry["condition_id"])
            condition_centers = np.concatenate(centers_by_condition[condition], axis=0)
            top, bottom = (float(value) for value in entry["height_v_range"])
            selected = condition_centers[
                (condition_centers[:, 1] >= top) & (condition_centers[:, 1] <= bottom)
            ]
            if not len(selected):
                raise RuntimeError(f"No frozen centerline points in height ROI: {condition}")
            v_values = selected[:, 1]
            entry["height_roi_formal_v_median"] = float(np.median(v_values))
            entry["height_roi_formal_v_min"] = float(np.min(v_values))
            entry["height_roi_formal_v_max"] = float(np.max(v_values))
        for height in ("h10", "h20", "h30"):
            height_entries = sorted(
                (entry for entry in registry["entries"] if entry.get("height_label") == height),
                key=lambda entry: (float(entry["height_roi_formal_v_median"]), str(entry["position_id"])),
            )
            for rank, entry in enumerate(height_entries, start=1):
                entry["v_order_rank"] = rank
        registry["formal_v_coordinate_source"] = str(cache_npz.resolve())
        registry["formal_v_coordinate_cache_reused"] = True
    else:
        registry["formal_v_coordinate_source"] = None
        registry["formal_v_coordinate_cache_reused"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, registry)
    return {
        "output": str(output_path.resolve()),
        "source_draft": str(draft_path.resolve()),
        "source_draft_sha256": registry["source_draft_sha256"],
        "entry_count": len(registry["entries"]),
        "auto_qc_status_counts": {
            status: sum(1 for entry in registry["entries"] if entry.get("auto_qc_status") == status)
            for status in ("PASS", "UNCERTAIN", "FAIL")
        },
        "human_reviewed_count": sum(bool(entry.get("human_reviewed")) for entry in registry["entries"]),
        "frozen_count": sum(bool(entry.get("frozen")) for entry in registry["entries"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_registry(args.draft, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
