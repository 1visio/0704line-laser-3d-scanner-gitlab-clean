"""Prepare geometry-only ROI review artifacts for the Surface-2 acquisition.

This is an independent preparation stage for obs_36mm/obs_40mm/obs_46mm.
It deliberately reuses the existing gauge-block implementation for:

* one Steger call per TIFF;
* five-repeat median images and binned centerlines;
* image-space ROI candidate detection and overlay rendering.

No C0/C1 reconstruction, truth value, residual, or height result is read by
this script.  The generated registry is a draft and is never marked as
manually confirmed.  Downstream q1/q2/domain analysis must be run only after
the draft has been independently confirmed from the geometry overlays.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOT = REPO_ROOT / "laser_measurement_tool"
DATA_ROOT_DEFAULT = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data"
)
CONFIG_DEFAULT = (
    REPO_ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"
)
OUTPUT_DEFAULT = (
    REPO_ROOT
    / "outputs"
    / "daheng_c1_gauge_blocks_20260819_ground4a"
    / "surface2"
)

DATASETS = ("obs_36mm", "obs_40mm", "obs_46mm")
POSE_IDS = tuple(f"{index:03d}" for index in range(1, 6))
TRUTH_MM = {"obs_36mm": 36.0, "obs_40mm": 40.0, "obs_46mm": 46.0}


def load_local_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import local module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# These are the existing implementations, imported rather than copied.  The
# module globals are narrowed to the three new datasets below before calls.
GAUGE = load_local_module(
    "surface2_gauge_impl",
    REPO_ROOT / "tools" / "evaluate_daheng_c1_gauge_blocks.py",
)
ANNOTATE = load_local_module(
    "surface2_annotation_impl",
    REPO_ROOT / "tools" / "annotate_daheng_gauge_rois.py",
)
GAUGE.DATASETS = DATASETS
GAUGE.DATASET_ORDER = {name: index for index, name in enumerate(DATASETS)}
GAUGE.TRUTH_MM = TRUTH_MM
ANNOTATE.DATASETS = DATASETS
ANNOTATE.DATASET_ORDER = {name: index for index, name in enumerate(DATASETS)}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.15g}"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore", restval=""
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def basename_key(value: Any) -> str:
    return Path(str(value or "").replace("\\", "/")).name


def quality_text(row: dict[str, str], field: str) -> str:
    value = row.get(field, "")
    return "" if value is None else str(value)


def display_image(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1.0, 99.8])
    return np.clip(
        (image.astype(np.float32) - low) * 255.0 / max(1.0, high - low),
        0,
        255,
    ).astype(np.uint8)


def expected_keys() -> set[tuple[str, str, int]]:
    return {
        (dataset, pose_id, repeat)
        for dataset in DATASETS
        for pose_id in POSE_IDS
        for repeat in range(1, 6)
    }


def audit_and_extract(
    data_root: Path,
    config_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Audit and extract all 75 frames, with exactly one Steger call each."""
    app = GAUGE.load_app_config(config_path)
    extraction_params = GAUGE.create_extraction_params(
        app.extraction_method,
        app.extraction_options_by_method.get(app.extraction_method, {}),
    )
    offsets = ANNOTATE.read_frame_offsets(data_root)
    entries: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    sha_paths: dict[str, list[str]] = defaultdict(list)
    seen_keys: set[tuple[str, str, int]] = set()
    extraction_errors: list[str] = []

    for dataset in DATASETS:
        folder = data_root / dataset
        frames_path = folder / "frames.csv"
        manifest_path = folder / "dataset_manifest.yaml"
        csv_rows = GAUGE.read_csv_rows(frames_path) if frames_path.is_file() else []
        rows_by_name = {basename_key(row.get("filename")): row for row in csv_rows}
        tiffs = sorted(folder.rglob("*.tif"))
        dataset_errors: list[str] = []
        counts: dict[str, int] = defaultdict(int)
        if not frames_path.is_file():
            dataset_errors.append("frames.csv missing")
        if not manifest_path.is_file():
            dataset_errors.append("dataset_manifest.yaml missing")
        if len(csv_rows) != 25:
            dataset_errors.append(f"frames.csv row count {len(csv_rows)} != 25")
        if len(tiffs) != 25:
            dataset_errors.append(f"TIFF count {len(tiffs)} != 25")

        for image_path in tiffs:
            frame_error = ""
            try:
                pose_id, repeat_index = GAUGE.parse_tiff_name(image_path)
            except ValueError as error:
                frame_error = str(error)
                audit_rows.append(
                    {
                        "row_type": "frame",
                        "dataset": dataset,
                        "filename": image_path.name,
                        "audit_status": "invalid_filename",
                        "error": frame_error,
                    }
                )
                dataset_errors.append(frame_error)
                continue
            key = (dataset, pose_id, repeat_index)
            if key in seen_keys:
                frame_error = f"duplicate key {key}"
                dataset_errors.append(frame_error)
            seen_keys.add(key)
            counts[pose_id] += 1
            csv_row = rows_by_name.get(image_path.name, {})
            offset_x = parse_int(csv_row.get("offset_x"))
            offset_y = parse_int(csv_row.get("offset_y"))
            csv_width = parse_int(csv_row.get("width"))
            csv_height = parse_int(csv_row.get("height"))
            pixel_format = quality_text(csv_row, "pixel_format")
            actual_sha = sha256(image_path)
            sha_paths[actual_sha].append(str(image_path.resolve()))
            csv_sha = quality_text(csv_row, "sha256")
            sha_match = bool(csv_sha) and csv_sha == actual_sha
            if csv_sha and not sha_match:
                frame_error = f"sha256 mismatch: {image_path.name}"
                dataset_errors.append(frame_error)

            try:
                image, centers = GAUGE.load_image_and_centers(
                    image_path,
                    extraction_params,
                    offset_x,
                    offset_y,
                )
            except Exception as error:  # preserve the audit row before failing later
                extraction_errors.append(
                    f"{dataset}/{image_path.name}: {type(error).__name__}: {error}"
                )
                audit_rows.append(
                    {
                        "row_type": "frame",
                        "dataset": dataset,
                        "filename": image_path.name,
                        "pose_id": pose_id,
                        "repeat_index": repeat_index,
                        "audit_status": "steger_error",
                        "csv_row_present": bool(csv_row),
                        "sha256_match": sha_match,
                        "error": extraction_errors[-1],
                        "quality": quality_text(csv_row, "quality_passed"),
                        "quality_warnings": quality_text(csv_row, "quality_warnings"),
                        "steger_called_once": True,
                    }
                )
                continue

            image_shape = tuple(int(item) for item in image.shape)
            dtype = str(image.dtype)
            shape_match = image_shape == (csv_height, csv_width)
            dtype_match = (
                pixel_format.lower() in {"mono8", "mono12", "mono12p"}
                and (
                    pixel_format.lower() == "mono8" and image.dtype == np.uint8
                    or pixel_format.lower() != "mono8"
                    and image.dtype in {np.uint8, np.uint16}
                )
            )
            if not shape_match:
                dataset_errors.append(
                    f"shape mismatch {image_path.name}: {image_shape} vs "
                    f"({csv_height}, {csv_width})"
                )
            if not dtype_match:
                dataset_errors.append(
                    f"dtype/format mismatch {image_path.name}: {dtype}/{pixel_format}"
                )
            entries.append(
                {
                    "dataset": dataset,
                    "height_truth_mm": TRUTH_MM[dataset],
                    "path": image_path,
                    "pose_id": pose_id,
                    "repeat_index": repeat_index,
                    "frames_csv": csv_row,
                    "image_shape": image_shape,
                    "image_dtype": dtype,
                    "image_min": int(np.min(image)),
                    "image_max": int(np.max(image)),
                    "image_offset_x": offset_x,
                    "image_offset_y": offset_y,
                    "centers": np.asarray(centers, dtype=np.float64).reshape(-1, 2),
                    "center_count": int(len(centers)),
                    "sha256": actual_sha,
                }
            )
            audit_rows.append(
                {
                    "row_type": "frame",
                    "dataset": dataset,
                    "filename": image_path.name,
                    "pose_id": pose_id,
                    "repeat_index": repeat_index,
                    "audit_status": "ok" if not frame_error else "warning",
                    "error": frame_error,
                    "csv_row_present": bool(csv_row),
                    "csv_sha256": csv_sha,
                    "actual_sha256": actual_sha,
                    "sha256_match": sha_match,
                    "csv_offset_x": offset_x,
                    "csv_offset_y": offset_y,
                    "csv_width": csv_width,
                    "csv_height": csv_height,
                    "csv_pixel_format": pixel_format,
                    "actual_width": image_shape[1],
                    "actual_height": image_shape[0],
                    "actual_dtype": dtype,
                    "shape_match": shape_match,
                    "dtype_match": dtype_match,
                    "center_count": int(len(centers)),
                    "image_min": int(np.min(image)),
                    "image_max": int(np.max(image)),
                    "quality": quality_text(csv_row, "quality_passed"),
                    "quality_warnings": quality_text(csv_row, "quality_warnings"),
                    "steger_called_once": True,
                }
            )

        if set(counts) != set(POSE_IDS):
            dataset_errors.append(f"pose groups {dict(counts)} != 001..005")
        for pose_id in POSE_IDS:
            if counts.get(pose_id, 0) != 5:
                dataset_errors.append(
                    f"{pose_id} repeat count {counts.get(pose_id, 0)} != 5"
                )
        manifest_info = GAUGE.manifest_summary(manifest_path)
        dataset_summaries[dataset] = {
            "dataset": dataset,
            "frame_csv_count": len(csv_rows),
            "tiff_count": len(tiffs),
            "pose_repeat_counts": dict(counts),
            "manifest": manifest_info,
            "errors": dataset_errors,
            "quality_warning_frame_count": sum(
                bool(row.get("quality_warnings"))
                for row in audit_rows
                if row.get("dataset") == dataset and row.get("row_type") == "frame"
            ),
            "quality_failed_frame_count": sum(
                str(row.get("quality", "")).lower() == "false"
                for row in audit_rows
                if row.get("dataset") == dataset and row.get("row_type") == "frame"
            ),
        }
        audit_rows.append(
            {
                "row_type": "dataset_summary",
                "dataset": dataset,
                "filename": "__dataset_summary__",
                "audit_status": "pass" if not dataset_errors else "warning",
                "error": " | ".join(dataset_errors),
                "frame_csv_count": len(csv_rows),
                "tiff_count": len(tiffs),
                "manifest_status": manifest_info.get("status"),
                "manifest_quality_passed": manifest_info.get("quality_passed"),
                "manifest_quality_warnings": json.dumps(
                    manifest_info.get("quality_warnings"), ensure_ascii=False
                ),
                "manifest_json": json.dumps(manifest_info, ensure_ascii=False),
                "steger_called_once": True,
            }
        )

    expected = expected_keys()
    missing = sorted(expected - seen_keys)
    unexpected = sorted(seen_keys - expected)
    duplicate_sha_groups = {
        digest: paths for digest, paths in sha_paths.items() if len(paths) > 1
    }
    summary = {
        "schema_version": 1,
        "data_root": str(data_root.resolve()),
        "config_path": str(config_path.resolve()),
        "dataset_ids": list(DATASETS),
        "expected_frame_count": len(expected),
        "discovered_filename_key_count": len(seen_keys),
        "extracted_frame_count": len(entries),
        "all_expected_keys_present": not missing and not unexpected,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "steger_call_count": len(entries) + len(extraction_errors),
        "steger_call_count_matches_discovered_tiffs": len(tiffs_for_root(data_root))
        == len(entries) + len(extraction_errors),
        "extraction_errors": extraction_errors,
        "duplicate_sha256_groups": duplicate_sha_groups,
        "dataset_summaries": dataset_summaries,
        "protocol": {
            "steger": "one call per input TIFF; no residual/height value used",
            "quality_warning_policy": "retain warning frames; do not auto-delete",
            "expected_layout": f"{len(DATASETS)} heights x 5 poses x 5 repeats",
        },
    }
    return entries, audit_rows, summary


def tiffs_for_root(data_root: Path) -> list[Path]:
    return [
        path
        for dataset in DATASETS
        for path in sorted((data_root / dataset).rglob("*.tif"))
    ]


def write_center_cache(path: Path, entries: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for entry in sorted(
        entries,
        key=lambda item: (
            GAUGE.DATASET_ORDER[item["dataset"]],
            item["pose_id"],
            item["repeat_index"],
        ),
    ):
        for point_index, (u, v) in enumerate(entry["centers"]):
            rows.append(
                {
                    "dataset": entry["dataset"],
                    "pose_id": entry["pose_id"],
                    "repeat_index": entry["repeat_index"],
                    "filename": entry["path"].name,
                    "point_index": point_index,
                    "u_px": float(u),
                    "v_px": float(v),
                    "steger_called_once": True,
                    "source": "surface2_fresh_single_pass",
                }
            )
    write_csv(
        path,
        rows,
        [
            "dataset",
            "pose_id",
            "repeat_index",
            "filename",
            "point_index",
            "u_px",
            "v_px",
            "steger_called_once",
            "source",
        ],
    )


def write_display_median(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), display_image(image)):
        raise RuntimeError(f"failed to write median image {path}")


def make_review_artifacts(
    data_root: Path,
    config_path: Path,
    output_dir: Path,
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    registry, candidate_rows, registry_summary = GAUGE.build_roi_registry(entries)
    entry_map = {
        (item["dataset"], item["pose_id"], item["repeat_index"]): item
        for item in entries
    }
    app = GAUGE.load_app_config(config_path)
    image_offset_x = int(app.camera.offset_x) if app.camera is not None else 0
    centerline_rows: list[dict[str, Any]] = []
    for item in registry:
        dataset = item["dataset"]
        pose_id = item["pose_id"]
        repeat_entries = [
            entry_map[(dataset, pose_id, repeat)] for repeat in range(1, 6)
        ]
        repeat_paths = [entry["path"] for entry in repeat_entries]
        image = ANNOTATE.median_image(repeat_paths)
        repeat_centers = [entry["centers"] for entry in repeat_entries]
        center_v, center_u = ANNOTATE.binned_centerline(
            repeat_centers, image.shape[0]
        )
        for bin_v, median_u in zip(center_v, center_u):
            centerline_rows.append(
                {
                    "dataset": dataset,
                    "pose_id": pose_id,
                    "v_px": float(bin_v),
                    "median_u_px": None if not np.isfinite(median_u) else float(median_u),
                }
            )
        stem = f"{dataset}_pose{pose_id}"
        write_display_median(
            output_dir / "roi_review" / "median_images" / f"{stem}.png", image
        )
        raw_median_path = output_dir / "roi_review" / "median_images" / f"{stem}.tif"
        if not cv2.imwrite(str(raw_median_path), image):
            raise RuntimeError(f"failed to write raw median image {raw_median_path}")
        auto_entry = dict(item)
        auto_entry["auto_candidates"] = [
            row for row in candidate_rows
            if row["dataset"] == dataset and row["pose_id"] == pose_id
        ]
        ANNOTATE.render_overlay(
            output_dir / "roi_review" / "overlays" / f"{stem}.png",
            image,
            center_v,
            center_u,
            auto_entry,
            None,
            image_offset_x,
            f"Surface-2 geometry ROI review | {dataset} pose {pose_id}",
            "image geometry only; median of 5 repeats; manual confirmation pending; no truth/error shown",
            contrast_stretch=True,
        )

    write_csv(
        output_dir / "surface2_centerline_profiles.csv",
        centerline_rows,
        ["dataset", "pose_id", "v_px", "median_u_px"],
    )
    write_csv(
        output_dir / "surface2_roi_candidates.csv",
        candidate_rows,
        [
            "dataset",
            "pose_id",
            "candidate_rank_by_score",
            "v_center_px",
            "depth_px",
            "prominence_px",
            "support_width_px",
            "score",
            "selected",
        ],
    )
    draft_entries = []
    for item in registry:
        draft = json_safe(dict(item))
        draft["manual_confirmed"] = False
        draft["manual_confirmation_basis"] = "pending geometry-only image review"
        draft_entries.append(draft)
    draft = {
        "schema_version": 1,
        "created_at_utc": now_utc(),
        "protocol": {
            "roi_source": "auto_geometry_candidate_pending_manual_confirmation",
            "geometry_only": True,
            "c0_c1_values_used": False,
            "truth_values_used": False,
            "residual_values_used": False,
            "selection_rule": "physical ground plane geometry in median image and Steger overlay only",
            "repeat_policy": "repeat1 is reserved for ground proxy; repeat2-5 are formal after confirmation",
        },
        "manual_confirmed": False,
        "manual_confirmed_count": 0,
        "manual_confirmed_expected": len(draft_entries),
        "all_entries_manual_confirmed": False,
        "summary": json_safe(registry_summary),
        "entries": draft_entries,
    }
    write_json(output_dir / "surface2_roi_registry_manual_draft.json", draft)
    write_json(output_dir / "surface2_auto_registry.json", {
        "protocol": {
            "roi_source": "auto_geometry",
            "geometry_only": True,
            "c0_c1_values_used": False,
            "truth_values_used": False,
        },
        "summary": registry_summary,
        "entries": registry,
    })
    return registry, registry_summary, candidate_rows


def report_text(
    output_dir: Path,
    data_root: Path,
    config_path: Path,
    audit_summary: dict[str, Any],
    registry_summary: dict[str, Any] | None,
    error: str | None = None,
) -> str:
    dataset_lines = []
    for dataset in DATASETS:
        item = audit_summary["dataset_summaries"].get(dataset, {})
        manifest = item.get("manifest", {})
        dataset_lines.append(
            "| {dataset} | {csv} | {tiff} | {poses} | {status} | {passed} | {warnings} | {errors} |".format(
                dataset=dataset,
                csv=item.get("frame_csv_count", ""),
                tiff=item.get("tiff_count", ""),
                poses="5×5" if item.get("pose_repeat_counts") == {p: 5 for p in POSE_IDS} else item.get("pose_repeat_counts", ""),
                status=manifest.get("status", ""),
                passed=(
                    "N/A"
                    if not item.get("tiff_count", 0)
                    else item.get("quality_failed_frame_count", 0) == 0
                ),
                warnings=item.get("quality_warning_frame_count", 0),
                errors=len(item.get("errors", [])),
            )
        )
    dataset_table = "\n".join(dataset_lines)
    quality_lines = []
    for dataset in DATASETS:
        item = audit_summary["dataset_summaries"].get(dataset, {})
        warning_count = item.get("quality_warning_frame_count", 0)
        failed_count = item.get("quality_failed_frame_count", 0)
        if warning_count or failed_count:
            quality_lines.append(
                f"- {dataset}：warning={warning_count}，quality_failed={failed_count}；"
                "这些是采集质量告警，不是本轮 ROI 删除规则。"
            )
    if not quality_lines:
        if int(audit_summary.get("extracted_frame_count", 0)) == 0:
            quality_lines.append("- 当前没有可审计帧；不能据此判定采集质量通过。")
        else:
            quality_lines.append(
                "- 本次完整性审计未发现 manifest quality warning/failed frame；仍需人工复核图像几何。"
            )
    quality_summary = "\n".join(quality_lines)
    registry_state = "not generated"
    if registry_summary is not None:
        registry_state = (
            f"{len(DATASETS) * len(POSE_IDS)} geometry candidates generated; "
            f"manual_review_required={registry_summary.get('manual_review_required')}"
        )
    expected_frames = int(audit_summary.get("expected_frame_count", 0))
    extracted_frames = int(audit_summary.get("extracted_frame_count", 0))
    steger_calls = int(audit_summary.get("steger_call_count", 0))
    review_group_count = len(DATASETS) * len(POSE_IDS) if registry_summary is not None else 0
    if extracted_frames == expected_frames and expected_frames > 0:
        stage_statement = "本轮已完成新数据完整性审计、一次/帧 Steger 中心线提取、五帧 median 与几何 overlay 准备。"
    else:
        stage_statement = (
            f"当前仅完成输入目录审计：{extracted_frames}/{expected_frames} 帧已提取中心线，"
            "尚未具备生成 median、overlay 或人工 ROI draft 的数据条件。"
        )
    extra = f"\n运行错误：`{error}`\n" if error else ""
    return f"""# Surface-2A/2B 数据接入、ROI 准备与 q-domain 审计

## 当前状态

`SURFACE2_STATUS=ROI_REVIEW_PENDING`

{stage_statement} ROI 仍是 draft，尚未人工冻结；因此本轮**停止在 ROI review**，没有生成 q1/q2、height residual 或 Surface-2C 的最终结论。

`Q2_GAP_FILLED=UNDECIDED`  
`Q1Q2_STATE_CONSISTENCY=UNDECIDED`  
`SURFACE2C_ALLOWED=NO`

{extra}
## Provenance / 复用边界

- 输入根目录：`{data_root.resolve()}`
- 配置：`{config_path.resolve()}`；仅用于同一 Steger extraction 参数，未调用重建。
- 复用 `evaluate_daheng_c1_gauge_blocks.py` 的 `load_image_and_centers()`、`profile_candidates()`、`build_roi_registry()`。
- 复用 `annotate_daheng_gauge_rois.py` 的 `median_image()`、`binned_centerline()`、`render_overlay()`。
- 实际新增计算：{extracted_frames}/{expected_frames} 帧完整性/manifest/尺寸/hash 审计、{steger_calls} 次 Steger（每 TIFF 一次）、{review_group_count} 组 median/centerline/geometry overlay 和 draft registry。
- 未执行：Frozen C0、Frozen C1、q1/q2 计算、残差驱动筛点、residual/height 计算、任何补偿拟合。
- warning 帧保留在审计和 Steger 输入中，没有因 `dynamic_range_low` 自动删除。

## 数据完整性

| dataset | frames.csv | TIFF | pose×repeat | manifest status | quality_failed_count==0 | warning frame count | structural errors |
|---|---:|---:|---|---|---|---:|---:|
{dataset_table}

整体：`{audit_summary.get('extracted_frame_count')}/{audit_summary.get('expected_frame_count')}` 帧成功完成中心线提取；Steger call count=`{audit_summary.get('steger_call_count')}`；缺失 key=`{len(audit_summary.get('missing_keys', []))}`；提取异常=`{len(audit_summary.get('extraction_errors', []))}`。

### 质量告警

{quality_summary}

是否可用于 q-domain 只能在几何 ROI 人工确认、中心线质量复核后再决定。

## ROI review 输出

`{registry_state}`

- median image：`roi_review/median_images/`
- median Steger / candidate overlay：`roi_review/overlays/`
- geometry candidate：`surface2_roi_candidates.csv`
- centerline profile：`surface2_centerline_profiles.csv`
- 未冻结 draft：`surface2_roi_registry_manual_draft.json`

人工确认时，只允许依据上述 median 图像、五帧 Steger 点和 physical ground plane geometry 调整 `height_v_range` 与 `baseline_v_ranges`；不要查看或使用 height error/residual。确认后应写出完整的 {len(DATASETS) * len(POSE_IDS)}-entry manual registry，并将每个 entry 与顶层 `manual_confirmed` 明确置为 true，再进入 Surface-2B。

## Surface-2B 尚未执行的项目

ROI 冻结后，下一轮才可在同一 frozen Ground-1 q 坐标定义下：复用 Frozen C0/C1 计算 q1/q2、用 repeat1 ground proxy 和 repeat2–5 formal 生成 residual，合并已有 30/50 mm，并报告 q2 gap/overlap、q1/q2 coverage、residual continuity 与相近 `(q1,q2)` 的跨 height/position 一致性。本报告不提前推断这些结论。

## 下一步

`Surface-2C` 当前不允许进入。先完成 {len(DATASETS) * len(POSE_IDS)} 组 ROI 的人工 geometry confirmation；若任一 pose 的物理 ground plane、包边/突起边界或 centerline 质量无法确认，应在 registry/report 中标记该项并重新采集或人工处理，不用 residual 阈值补救。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    audit_summary: dict[str, Any]
    registry_summary: dict[str, Any] | None = None
    error: str | None = None
    try:
        entries, audit_rows, audit_summary = audit_and_extract(data_root, config_path)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        audit_summary = {
            "schema_version": 1,
            "data_root": str(data_root),
            "config_path": str(config_path),
            "dataset_ids": list(DATASETS),
            "expected_frame_count": len(expected_keys()),
            "extracted_frame_count": len(entries),
            "steger_call_count": 0,
            "missing_keys": [],
            "unexpected_keys": [],
            "extraction_errors": [error],
            "dataset_summaries": {},
        }

    write_csv(
        output_dir / "surface2_input_integrity.csv",
        audit_rows,
        [
            "row_type",
            "dataset",
            "filename",
            "pose_id",
            "repeat_index",
            "audit_status",
            "error",
            "csv_row_present",
            "csv_sha256",
            "actual_sha256",
            "sha256_match",
            "csv_offset_x",
            "csv_offset_y",
            "csv_width",
            "csv_height",
            "csv_pixel_format",
            "actual_width",
            "actual_height",
            "actual_dtype",
            "shape_match",
            "dtype_match",
            "center_count",
            "image_min",
            "image_max",
            "quality",
            "quality_warnings",
            "frame_csv_count",
            "tiff_count",
            "manifest_status",
            "manifest_quality_passed",
            "manifest_quality_warnings",
            "manifest_json",
            "steger_called_once",
        ],
    )
    write_json(output_dir / "surface2_input_integrity.json", audit_summary)
    write_json(
        output_dir / "surface2_provenance.json",
        {
            "created_at_utc": now_utc(),
            "data_root": str(data_root),
            "config_path": str(config_path),
            "config_sha256": sha256(config_path) if config_path.is_file() else None,
            "reused_scripts": {
                "gauge_impl": str((REPO_ROOT / "tools" / "evaluate_daheng_c1_gauge_blocks.py").resolve()),
                "gauge_impl_sha256": sha256(REPO_ROOT / "tools" / "evaluate_daheng_c1_gauge_blocks.py"),
                "annotation_impl": str((REPO_ROOT / "tools" / "annotate_daheng_gauge_rois.py").resolve()),
                "annotation_impl_sha256": sha256(REPO_ROOT / "tools" / "annotate_daheng_gauge_rois.py"),
            },
            "steger_rerun": bool(audit_summary.get("steger_call_count", 0)),
            "steger_call_count": int(audit_summary.get("steger_call_count", 0)),
            "expected_frame_count": int(audit_summary.get("expected_frame_count", 0)),
            "steger_calls_per_tiff": 1,
            "c0_c1_reconstruction": False,
            "q_domain_analysis": False,
            "residual_driven_roi": False,
            "roi_manual_confirmed": False,
        },
    )
    if not error and len(entries) == len(expected_keys()) and not audit_summary.get("missing_keys"):
        write_center_cache(output_dir / "surface2_center_cache.csv", entries)
        try:
            _, registry_summary, _ = make_review_artifacts(
                data_root, config_path, output_dir, entries
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    report = report_text(
        output_dir, data_root, config_path, audit_summary, registry_summary, error
    )
    (output_dir / "surface2a2b_report.md").write_text(report, encoding="utf-8")
    status = {
        "surface2_status": "ROI_REVIEW_PENDING" if error is None else "PREPARATION_ERROR",
        "q2_gap_filled": "UNDECIDED",
        "q1q2_state_consistency": "UNDECIDED",
        "surface2c_allowed": False,
        "roi_manual_confirmed": False,
        "error": error,
    }
    write_json(output_dir / "surface2a2b_summary.json", status)
    print(json.dumps({"output": str(output_dir), **status}, ensure_ascii=False, indent=2))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
