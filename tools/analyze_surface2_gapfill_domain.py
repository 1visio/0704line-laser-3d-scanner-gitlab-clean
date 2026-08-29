"""Audit the 33/38/43/48 mm q2 gap-fill against the frozen Surface-2 data.

This is a post-capture entry point.  It deliberately refuses to run until the
four new datasets have passed the one-pass Steger/integrity audit and the
20-entry geometry-only ROI registry is manually frozen.  Existing
30/36/40/46/50 mm Surface-2 samples are copied from the canonical Surface-2B
artifact; they are not reconstructed a second time.

The script reuses the Surface-2B reconstruction, ground-proxy and q-coordinate
functions.  It does not refit C0/C1, redefine q1/q2, fit a correction, or use
50 mm for model selection.  Its output is a separate gap-fill artifact tree.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import analyze_surface2b as surface2b


BASE_OUTPUT = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a"
DEFAULT_REVIEW = BASE_OUTPUT / "surface2_gapfill_3348_review"
DEFAULT_SURFACE2B = BASE_OUTPUT / "surface2" / "surface2b"
DEFAULT_SURFACE1A = BASE_OUTPUT / "surface1a"
DEFAULT_OUTPUT = BASE_OUTPUT / "surface2_gapfill_3348_domain"
DEFAULT_CONFIG = REPO_ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "calibration_tool" / "projects" / "daheng" / "data"

NEW_DATASETS = ("obs_33mm", "obs_38mm", "obs_43mm", "obs_48mm")
OLD_DATASETS = ("obs_30mm", "obs_36mm", "obs_40mm", "obs_46mm", "obs_50mm")
ALL_DATASETS = OLD_DATASETS[:1] + ("obs_33mm",) + ("obs_36mm", "obs_38mm", "obs_40mm", "obs_43mm", "obs_46mm", "obs_48mm", "obs_50mm")
HEIGHTS = (30.0, 33.0, 36.0, 38.0, 40.0, 43.0, 46.0, 48.0, 50.0)
TRUTH_MM = {f"obs_{int(height)}mm": height for height in HEIGHTS}
POSE_IDS = tuple(f"{index:03d}" for index in range(1, 6))
DEVELOPMENT_HEIGHTS = HEIGHTS[:-1]
Q_TOLERANCE = 0.05


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else f"{float(value):.15g}"
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def validate_new_integrity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"new-data integrity artifact is missing: {path}")
    summary = read_json(path)
    expected = len(NEW_DATASETS) * len(POSE_IDS) * 5
    if summary.get("expected_frame_count") != expected:
        raise RuntimeError("new-data integrity expected_frame_count is not 100")
    if summary.get("extracted_frame_count") != expected:
        raise RuntimeError("new-data integrity does not contain 100 extracted frames")
    if summary.get("steger_call_count") != expected:
        raise RuntimeError("new-data integrity does not prove one Steger call per TIFF")
    if summary.get("missing_keys") or summary.get("unexpected_keys") or summary.get("extraction_errors"):
        raise RuntimeError("new-data integrity contains missing keys, unexpected keys, or extraction errors")
    for dataset in NEW_DATASETS:
        item = summary.get("dataset_summaries", {}).get(dataset, {})
        if item.get("frame_csv_count") != 25 or item.get("tiff_count") != 25:
            raise RuntimeError(f"{dataset} is not 5 positions x 5 repeats")
        if item.get("errors"):
            raise RuntimeError(f"{dataset} has structural audit errors: {item['errors']}")
    return summary


def _entry_without_manual_flags(entry: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(entry)
    for key in ("manual_confirmed", "manual_confirmation_basis", "auto_candidates"):
        value.pop(key, None)
    return json_safe(value)


def validate_registry(final_path: Path, draft_path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    if not final_path.is_file():
        raise RuntimeError(f"frozen 20-entry ROI registry is missing: {final_path}")
    if not draft_path.is_file():
        raise RuntimeError(f"ROI draft provenance is missing: {draft_path}")
    final = read_json(final_path)
    draft = read_json(draft_path)
    entries = final.get("entries")
    draft_entries = draft.get("entries")
    expected_keys = {(dataset, pose) for dataset in NEW_DATASETS for pose in POSE_IDS}
    if not isinstance(entries, list) or len(entries) != 20:
        raise RuntimeError("gap-fill requires exactly 20 frozen ROI entries")
    if not isinstance(draft_entries, list) or len(draft_entries) != 20:
        raise RuntimeError("gap-fill ROI draft must contain exactly 20 entries")
    if final.get("manual_confirmed") is not True or int(final.get("manual_confirmed_count", 0)) != 20:
        raise RuntimeError("gap-fill ROI registry is not top-level 20/20 manual_confirmed")
    if not all(entry.get("manual_confirmed") is True for entry in entries):
        raise RuntimeError("one or more gap-fill ROI entries are not manually confirmed")
    keys = {(str(entry.get("dataset")), str(entry.get("pose_id"))) for entry in entries}
    if keys != expected_keys:
        raise RuntimeError(f"gap-fill ROI keys mismatch: missing={sorted(expected_keys - keys)}")
    if draft.get("protocol", {}).get("residual_values_used") is not False:
        raise RuntimeError("ROI draft does not prove residual_values_used=false")
    if draft.get("protocol", {}).get("geometry_only") is not True:
        raise RuntimeError("ROI draft does not prove geometry_only=true")
    draft_by_key = {(str(entry.get("dataset")), str(entry.get("pose_id"))): entry for entry in draft_entries}
    final_by_key = {(str(entry.get("dataset")), str(entry.get("pose_id"))): entry for entry in entries}
    if set(draft_by_key) != expected_keys:
        raise RuntimeError("gap-fill ROI draft keys mismatch")
    if any(_entry_without_manual_flags(final_by_key[key]) != _entry_without_manual_flags(draft_by_key[key]) for key in expected_keys):
        raise RuntimeError("frozen ROI geometry differs from the reviewed draft")
    for dataset in NEW_DATASETS:
        ranks = sorted(int(final_by_key[(dataset, pose)]["position_rank"]) for pose in POSE_IDS)
        if ranks != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"{dataset} position ranks are not 1..5: {ranks}")
    return final_by_key, {
        "final_path": final_path,
        "final_sha256": sha256_file(final_path),
        "draft_path": draft_path,
        "draft_sha256": sha256_file(draft_path),
        "entry_count": 20,
        "manual_confirmed_count": 20,
        "entries_geometry_match_draft": True,
    }


def c1_parameter_sha(correction: Any) -> tuple[str, dict[str, Any]]:
    payload = {
        "model_id": correction.model_id,
        "center_xn": correction.center_xn,
        "center_yn": correction.center_yn,
        "axis_s_xn": correction.axis_s_xn,
        "axis_s_yn": correction.axis_s_yn,
        "domain_min": correction.domain_min,
        "domain_max": correction.domain_max,
        "degree": correction.degree,
        "knots": np.asarray(correction.knots, dtype=np.float64).tolist(),
        "coefficients_mm": np.asarray(correction.coefficients_mm, dtype=np.float64).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def validate_frozen_inputs(config_path: Path, surface1a_dir: Path) -> tuple[Any, Any, Any, Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    coordinate_path = surface1a_dir / "surface_coordinate_definition.json"
    summary_path = surface1a_dir / "surface1a_summary.json"
    app, calibration, correction, laser_model, provenance = surface2b.validate_frozen_provenance(
        config_path.resolve(), summary_path.resolve(), coordinate_path.resolve()
    )
    c1_path = Path(app.calibration.laser_ray_correction).resolve()
    c1_file_hash = sha256_file(c1_path)
    parameter_hash, parameter_payload = c1_parameter_sha(correction)
    legacy_hash = provenance["current"].get("frozen_c1_sha256")
    if legacy_hash != c1_file_hash:
        raise RuntimeError("Surface-1A legacy frozen_c1_sha256 is not the raw C1 file SHA")
    provenance["current"].update(
        {
            "C1_PARAMETER_SHA": parameter_hash,
            "C1_FILE_SHA256": c1_file_hash,
            "C1_PARAMETER_DEFINITION": "canonical runtime C1_4k model_id/PCA domain/knots/coefficients payload",
            "C1_PARAMETER_PAYLOAD": parameter_payload,
            "legacy_surface1a_frozen_c1_sha256": legacy_hash,
        }
    )
    return app, calibration, correction, laser_model, provenance, surface2b.read_json(coordinate_path)


def normalize_reused_row(raw: dict[str, str]) -> dict[str, Any]:
    numeric = (
        "true_height_mm", "u", "v", "xn", "yn", "C1_s", "C1_s_raw", "lambda_c0", "lambda_c1",
        "P_c0_x_mm", "P_c0_y_mm", "P_c0_z_mm", "q1", "q2", "q1_mm", "q2_mm", "Xg", "Yg", "Zg",
        "S_mm", "ground_proxy_a_mm_per_mm", "ground_proxy_b_mm", "height_value_mm", "height_residual_mm",
    )
    row: dict[str, Any] = dict(raw)
    for field in numeric:
        row[field] = float(raw[field])
    for field in ("repeat_index", "point_index", "position_rank"):
        row[field] = int(raw[field])
    for field in ("C1_s_clamped", "height_measurement_inlier", "jacobian_valid", "analysis_included"):
        row[field] = surface2b.as_bool(raw[field])
    row["schema_version"] = 1
    row["source"] = "surface2b_reused"
    row["spatial_position_key"] = f"q1_rank_{row['position_rank']}"
    row["model_selection_eligible"] = row["true_height_mm"] in DEVELOPMENT_HEIGHTS
    return row


def load_reused_surface2b(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"canonical Surface-2B samples are missing: {path}")
    rows: list[dict[str, Any]] = []
    accepted_roles = {"development_formal_repeat2_5", "surface2_formal_repeat2_5", "heldout_formal_repeat2_5"}
    for raw in surface2b.read_csv(path):
        dataset = raw.get("dataset", "")
        if dataset not in OLD_DATASETS or raw.get("split_role") not in accepted_roles:
            continue
        if not all(surface2b.finite(raw.get(field)) for field in ("q1", "q2", "height_residual_mm", "Zg")):
            continue
        rows.append(normalize_reused_row(raw))
    counts = defaultdict(int)
    for row in rows:
        counts[row["dataset"]] += 1
    missing = [dataset for dataset in OLD_DATASETS if counts[dataset] == 0]
    if missing:
        raise RuntimeError(f"canonical Surface-2B samples have no formal rows for {missing}")
    return rows


def grouped_cv_support(domain_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_height = {float(row["true_height_mm"]): row for row in domain_rows}
    result: list[dict[str, Any]] = []
    dev = list(DEVELOPMENT_HEIGHTS)
    for height in dev:
        held = by_height[height]
        train = [by_height[item] for item in dev if item != height]
        train_full_min = min(float(item["q2_min"]) for item in train)
        train_full_max = max(float(item["q2_max"]) for item in train)
        train_robust_min = min(float(item["q2_p05"]) for item in train)
        train_robust_max = max(float(item["q2_p95"]) for item in train)
        held_full_min = float(held["q2_min"])
        held_full_max = float(held["q2_max"])
        held_robust_min = float(held["q2_p05"])
        held_robust_max = float(held["q2_p95"])
        result.append(
            {
                "held_out_height_mm": height,
                "train_height_count": len(train),
                "held_q2_full_min": held_full_min,
                "held_q2_full_max": held_full_max,
                "train_q2_full_min": train_full_min,
                "train_q2_full_max": train_full_max,
                "held_q2_p05": held_robust_min,
                "held_q2_p95": held_robust_max,
                "train_q2_p05_min": train_robust_min,
                "train_q2_p95_max": train_robust_max,
                "full_hull_in_domain": held_full_min >= train_full_min and held_full_max <= train_full_max,
                "robust_hull_in_domain": held_robust_min >= train_robust_min and held_robust_max <= train_robust_max,
                "robust_band_overlap_with_train": surface2b.interval_relation(held_robust_min, held_robust_max, train_robust_min, train_robust_max)[1] > 0.0,
            }
        )
    return result


def make_plots(output: Path, formal_rows: list[dict[str, Any]], domain_rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]]) -> None:
    analysis = [row for row in formal_rows if row["analysis_included"]]
    colors = dict(zip(HEIGHTS, plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(HEIGHTS)))))
    rng = np.random.default_rng(20260820)

    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for height in HEIGHTS:
        rows = [row for row in analysis if row["true_height_mm"] == height]
        if len(rows) > 1800:
            rows = [rows[index] for index in rng.choice(len(rows), 1800, replace=False)]
        axis.scatter([row["q1"] for row in rows], [row["q2"] for row in rows], s=5, alpha=0.22, linewidths=0, color=colors[height], label=f"{height:g} mm")
    for row in condition_rows:
        axis.scatter(row["q1_median"], row["q2_median"], s=34, facecolors="none", edgecolors=colors[row["true_height_mm"]], linewidths=1.1)
    axis.set_xlabel("q1 (Frozen C0 intrinsic coordinate)")
    axis.set_ylabel("q2 (Frozen C0 intrinsic coordinate)")
    axis.set_title("Surface-2 gap-fill q1-q2 formal coverage")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    fig.savefig(output / "surface2_gapfill_q1_q2_coverage.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    rank_colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, 5))
    for rank, color in zip(range(1, 6), rank_colors):
        rows = sorted([row for row in condition_rows if row["position_rank"] == rank], key=lambda row: row["true_height_mm"])
        x = np.asarray([row["true_height_mm"] for row in rows])
        y = np.asarray([row["q2_median"] for row in rows])
        low = y - np.asarray([row["q2_p05"] for row in rows])
        high = np.asarray([row["q2_p95"] for row in rows]) - y
        axis.errorbar(x, y, yerr=np.vstack([low, high]), marker="o", linewidth=1.4, capsize=2, color=color, label=f"position_rank {rank}")
    axis.plot([row["true_height_mm"] for row in domain_rows], [row["q2_median"] for row in domain_rows], color="#212121", marker="s", linewidth=2.0, label="pooled median")
    axis.set_xlabel("true height [mm]")
    axis.set_ylabel("q2")
    axis.set_title("q2 continuity versus height (P05-P95)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.savefig(output / "surface2_gapfill_q2_vs_height.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for height in HEIGHTS:
        rows = [row for row in analysis if row["true_height_mm"] == height]
        if len(rows) > 1600:
            rows = [rows[index] for index in rng.choice(len(rows), 1600, replace=False)]
        q2 = np.asarray([row["q2"] for row in rows], dtype=np.float64)
        residual = np.asarray([row["height_residual_mm"] for row in rows], dtype=np.float64)
        axis.scatter(q2, residual, s=5, alpha=0.18, linewidths=0, color=colors[height], label=f"{height:g} mm")
        if len(q2) >= 20:
            edges = np.linspace(np.min(q2), np.max(q2), 13)
            centers = 0.5 * (edges[:-1] + edges[1:])
            medians = []
            valid_centers = []
            for left, right, center in zip(edges[:-1], edges[1:], centers):
                mask = (q2 >= left) & (q2 < right)
                if np.count_nonzero(mask) >= 5:
                    valid_centers.append(center)
                    medians.append(float(np.median(residual[mask])))
            axis.plot(valid_centers, medians, color=colors[height], linewidth=2.0)
    axis.axhline(0.0, color="#212121", linewidth=1.0)
    axis.set_xlabel("q2")
    axis.set_ylabel("raw session-linear height residual [mm]")
    axis.set_title("Raw residual versus Frozen-C0 q2")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    fig.savefig(output / "surface2_gapfill_raw_residual_vs_q2.png", dpi=180)
    plt.close(fig)


def report_text(registry_info: dict[str, Any], provenance: dict[str, Any], integrity: dict[str, Any], domain_rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]], gap: dict[str, Any], support_rows: list[dict[str, Any]], reused_count: int, new_count: int, model_allowed: str) -> str:
    domain_lines = []
    for row in domain_rows:
        domain_lines.append(
            "| {h:g} | {n} | {q1min:.4f} | {q1p05:.4f} | {q1med:.4f} | {q1p95:.4f} | {q1max:.4f} | {q2min:.4f} | {q2p05:.4f} | {q2med:.4f} | {q2p95:.4f} | {q2max:.4f} | {clamp}/{valid} ({rate:.2%}) |".format(
                h=row["true_height_mm"], n=row["q1_count"], q1min=row["q1_min"], q1p05=row["q1_p05"], q1med=row["q1_median"], q1p95=row["q1_p95"], q1max=row["q1_max"], q2min=row["q2_min"], q2p05=row["q2_p05"], q2med=row["q2_median"], q2p95=row["q2_p95"], q2max=row["q2_max"], clamp=row["c1_clamp_count"], valid=row["formal_valid_point_count"], rate=row["c1_clamp_rate"] or 0.0,
            )
        )
    gap_lines = [
        "| {a:g}→{b:g} | {delta:.5f} | {fg:.5f} | {fo:.5f} | {rg:.5f} | {ro:.5f} | {ok} |".format(a=row["height_low_mm"], b=row["height_high_mm"], delta=row["median_delta_q2_high_minus_low"], fg=row["full_gap_q2"], fo=row["full_overlap_q2"], rg=row["robust_p05_p95_gap_q2"], ro=row["robust_p05_p95_overlap_q2"], ok=row["robust_gap_within_q_tolerance"])
        for row in gap["adjacent_pairs"]
    ]
    support_lines = [
        "| {h:g} | {full} | {robust} | {overlap} |".format(h=row["held_out_height_mm"], full=row["full_hull_in_domain"], robust=row["robust_hull_in_domain"], overlap=row["robust_band_overlap_with_train"])
        for row in support_rows
    ]
    return f"""# Surface-2 gap-fill 33/38/43/48 mm：q-domain continuity audit

## 结论（独立于旧 Surface-2/2BR2）

`Q2_GAP_FILLED={gap['Q2_GAP_FILLED']}`  
`Q2_VS_Q1Q2_MODEL_SELECTION_ALLOWED={model_allowed}`

本轮只判断 q-domain 是否已具备进入 B2(q2-only) 与 S0(q1+q2) 正式 grouped-CV 比较的覆盖条件；没有拟合 S1、spline、LUT、Δh 或 Δlambda，也不是 production validation。若 `Q2_GAP_FILLED` 不是 `YES`，正式模型选择保持禁止。

## Provenance / 复用与新增计算

- 新增数据完整性：{integrity['extracted_frame_count']}/{integrity['expected_frame_count']} 帧，Steger call count={integrity['steger_call_count']}，每 TIFF 一次；ROI 20/20 geometry-only manual confirmed。
- 复用：30/36/40/46/50 mm 的 `{reused_count}` 条 formal q/residual 样本来自 canonical Surface-2B CSV，没有再次 Steger/C0/C1 重建。
- 新增：33/38/43/48 mm 的 `{new_count}` 条 formal sample；repeat1 只用于每个 height×position 的 session-linear ground proxy，repeat2–5 才进入 formal。
- Frozen C0/q 定义保持 Surface-1B：`P_c0=lambda_c0*[xn,yn,1]`，q1/q2 使用 Frozen C0 独立轴、center/scale；没有使用 C1 后坐标重定义 q。
- C1 开启状态经配置检查为 `enable_laser_ray_correction=true`；C1 clamp 只在 Frozen C1 domain 边界取值，不做 extrapolation。
- `C1_PARAMETER_SHA={provenance['current']['C1_PARAMETER_SHA']}`：canonical runtime C1_4k 参数 payload 的 SHA。
- `C1_FILE_SHA256={provenance['current']['C1_FILE_SHA256']}`：Frozen C1 JSON 原始字节 SHA；两者不能混用。Surface-1A legacy `frozen_c1_sha256` 与该 file SHA 一致。
- 50 mm 只作为 q-domain/残差审计的 strict-held-out 输入；未进入任何模型选择或参数拟合。

## Height-level q domain 与 C1 clamp

| height | q1 points | q1 min | q1 P05 | q1 median | q1 P95 | q1 max | q2 min | q2 P05 | q2 median | q2 P95 | q2 max | C1 clamp / formal valid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(domain_lines)}

## 相邻 q2 band gap / overlap

q2 median 30→50 mm 严格有序：`{gap['strictly_ordered']}`；robust P05–P95 gap tolerance=`{Q_TOLERANCE}`。

| adjacent height | median Δq2 | full gap | full overlap | P05–P95 gap | P05–P95 overlap | robust gap≤tol |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(gap_lines)}

## Grouped-CV 的 q2 support（诊断性）

此表只描述 leave-one-development-height-out 时 held-out q2 band 相对于其余 development heights 的支持，不执行 correction CV。`full_hull_in_domain` 是总体 q2 min/max 凸包支持；`robust_band_overlap_with_train` 是 P05–P95 是否与训练总体 robust band 有实际重叠，二者分开记录。

| held-out height | full hull in-domain | robust hull in-domain | robust band overlap |
|---:|---|---|---|
{chr(10).join(support_lines)}

## Position / residual 诊断

condition statistics 使用 q1 从小到大重排的 `position_rank=1..5`；原始 pose_id 仅作为采集 provenance，不作为跨高度统一位置。`surface2_gapfill_condition_statistics.csv` 及 `surface2_gapfill_samples.csv` 保留了 q1/q2、raw session-linear residual、position_rank 和 C1 clamp 字段，供后续 B2/S0 grouped-CV 使用。当前没有因 residual 大小删点，也没有 residual-driven ROI 调整。

## ROI 与异常门槛

- frozen registry SHA：`{registry_info['final_sha256']}`；draft SHA：`{registry_info['draft_sha256']}`；20/20 geometry 与 draft 一致。
- 若任一 height 的 condition count 不是 5、formal repeat 不是 repeat2–5、C1 clamp 或重建有效点异常，需先处理数据/ROI 审计，不进入模型选择；本报告不自动删点。

## 输出

- `surface2_gapfill_samples.csv` / `surface2_gapfill_frame_metrics.csv`
- `surface2_gapfill_domain_statistics.csv` / `surface2_gapfill_condition_statistics.csv`
- `surface2_gapfill_q2_gap_overlap.json` / `surface2_gapfill_grouped_cv_support.csv`
- `surface2_gapfill_q1_q2_coverage.png` / `surface2_gapfill_q2_vs_height.png` / `surface2_gapfill_raw_residual_vs_q2.png`
- `surface2_gapfill_summary.json`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--surface1a", type=Path, default=DEFAULT_SURFACE1A)
    parser.add_argument("--surface2b", type=Path, default=DEFAULT_SURFACE2B)
    parser.add_argument("--integrity", type=Path, default=DEFAULT_REVIEW / "surface2_input_integrity.json")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REVIEW / "surface2_roi_registry_manual.json")
    parser.add_argument("--draft", type=Path, default=DEFAULT_REVIEW / "surface2_roi_registry_manual_draft.json")
    parser.add_argument("--center-cache", type=Path, default=DEFAULT_REVIEW / "surface2_center_cache.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    integrity = validate_new_integrity(args.integrity.resolve())
    recorded_data_root = Path(str(integrity.get("data_root", ""))).resolve()
    if recorded_data_root != args.data_root.resolve():
        raise RuntimeError(
            f"integrity data_root mismatch: {recorded_data_root} != {args.data_root.resolve()}"
        )
    registry, registry_info = validate_registry(args.registry.resolve(), args.draft.resolve())
    app, calibration, correction, laser_model, provenance, coordinate = validate_frozen_inputs(args.config.resolve(), args.surface1a.resolve())

    # Reuse Surface-2B's tested implementation with only the new dataset and
    # nine-height mappings injected.  No existing output directory is touched.
    surface2b.NEW_DATASETS = NEW_DATASETS
    surface2b.TRUTH_MM = TRUTH_MM
    surface2b.HEIGHTS = HEIGHTS
    cache = surface2b.load_center_cache(args.center_cache.resolve())
    origin = np.asarray(coordinate["ground_origin_xy"], dtype=np.float64)
    direction = np.asarray(coordinate["ground_direction_xy"], dtype=np.float64)
    models, proxy_rows = surface2b.fit_new_proxies(cache, registry, calibration, app.reconstruction, correction, origin, direction, app.measurement)

    new_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for dataset in NEW_DATASETS:
        for pose in POSE_IDS:
            for repeat in range(1, 6):
                rows, frame = surface2b.evaluate_new_frame(dataset, pose, repeat, cache[(dataset, pose, repeat)], registry[(dataset, pose)], models[(dataset, pose)], calibration, app.reconstruction, correction, laser_model, origin, direction, app.measurement)
                for row in rows:
                    row["model_selection_eligible"] = True
                new_rows.extend(rows)
                frame_rows.append(frame)

    old_rows = load_reused_surface2b(args.surface2b.resolve())
    for row in old_rows:
        row["model_selection_eligible"] = row["true_height_mm"] in DEVELOPMENT_HEIGHTS
    frame_rows.extend(surface2b.reused_frame_metrics(old_rows))
    formal_rows = [row for row in new_rows if row["split_role"] == "surface2_formal_repeat2_5"] + old_rows
    rank_mapping = surface2b.assign_q1_position_ranks(formal_rows, frame_rows, proxy_rows)
    formal_rows.sort(key=lambda row: (float(row["true_height_mm"]), int(row["position_rank"]), int(row["repeat_index"]), int(row["point_index"])))
    frame_rows.sort(key=lambda row: (float(row["true_height_mm"]), int(row["position_rank"]), int(row["repeat_index"])))
    domain_rows, condition_rows, clamp_rows = surface2b.domain_and_condition_stats(formal_rows)
    if any(int(row["condition_count"]) != 5 or int(row["q2_count"]) == 0 for row in domain_rows):
        raise RuntimeError("one or more heights lacks five populated q2 conditions")
    gap = surface2b.q2_gap_payload(domain_rows)
    support_rows = grouped_cv_support(domain_rows)
    model_allowed = "YES" if gap["Q2_GAP_FILLED"] == "YES" and all(row["full_hull_in_domain"] for row in support_rows) else "NO"

    output.mkdir(parents=True, exist_ok=True)
    sample_fields = list(surface2b.SAMPLE_FIELDS) + ["model_selection_eligible"]
    write_csv(output / "surface2_gapfill_samples.csv", formal_rows, sample_fields)
    write_csv(output / "surface2_gapfill_frame_metrics.csv", frame_rows, ["dataset", "true_height_mm", "pose_id", "position_rank", "repeat_index", "split_role", "source", "selected_height_count", "valid_height_count", "height_inlier_count", "jacobian_valid_count", "analysis_point_count", "c1_clamp_count", "c1_clamp_rate", "C1_s_raw_min", "C1_s_raw_max", "proxy_a_mm_per_mm", "proxy_b_mm", "proxy_rmse_mm", "proxy_S_span_mm", "height_fit_status", "status"])
    write_csv(output / "surface2_gapfill_ground_proxy_metrics.csv", proxy_rows, list(proxy_rows[0].keys()))
    write_csv(output / "surface2_gapfill_domain_statistics.csv", domain_rows, list(domain_rows[0].keys()))
    write_csv(output / "surface2_gapfill_condition_statistics.csv", condition_rows, list(condition_rows[0].keys()))
    write_csv(output / "surface2_gapfill_clamp_statistics.csv", clamp_rows, list(clamp_rows[0].keys()))
    write_csv(output / "surface2_gapfill_grouped_cv_support.csv", support_rows, list(support_rows[0].keys()))
    write_json(output / "surface2_gapfill_q2_gap_overlap.json", gap)
    make_plots(output, formal_rows, domain_rows, condition_rows)
    summary = {
        "Q2_GAP_FILLED": gap["Q2_GAP_FILLED"],
        "Q2_VS_Q1Q2_MODEL_SELECTION_ALLOWED": model_allowed,
        "registry": registry_info,
        "integrity": integrity,
        "provenance": provenance,
        "q_definition": coordinate,
        "point_counts": {"reused_old_formal_rows": len(old_rows), "new_formal_rows": len(formal_rows) - len(old_rows), "formal_rows_total": len(formal_rows), "analysis_included_total": sum(bool(row["analysis_included"]) for row in formal_rows)},
        "domain_statistics": domain_rows,
        "condition_statistics": condition_rows,
        "q2_gap_overlap": gap,
        "grouped_cv_support": support_rows,
        "position_rank_definition": {"rule": "within each height, ascending formal-analysis q1 median", "pose_id_cross_height_alignment": False, "mapping": rank_mapping},
        "strict_heldout": {"height_mm": 50.0, "used_for_domain_audit": True, "used_for_model_selection": False},
        "constraints": {"c0_refit": False, "c1_refit": False, "q_redefined": False, "residual_driven_roi": False, "correction_fit": False, "random_point_split": False, "c1_extrapolation": False},
        "created_at_utc": now_utc(),
    }
    write_json(output / "surface2_gapfill_summary.json", summary)
    (output / "surface2_gapfill_report.md").write_text(report_text(registry_info, provenance, integrity, domain_rows, condition_rows, gap, support_rows, len(old_rows), len(formal_rows) - len(old_rows), model_allowed), encoding="utf-8")
    print(json.dumps({"output": str(output), "Q2_GAP_FILLED": gap["Q2_GAP_FILLED"], "Q2_VS_Q1Q2_MODEL_SELECTION_ALLOWED": model_allowed, "formal_rows": len(formal_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
