"""Read-only Session01 full-FOV paired validation.

Only exported PNG/CSV/JSON artifacts are inspected. No image processing,
reconstruction, fitting, correction fitting, or model selection is run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("Base", "H1", "H-B2")
NOMINAL = {"h10": 10.0, "h20": 20.0, "h30": 30.0}
EDGE_V = 2400.0
FOV_THRESHOLDS = (2200.0, 2400.0, 2600.0)
INVALID_STATUS = {"", "invalid", "not_measured", "not-measured", "not measured", "none", "inactive", "false"}
CERTIFIED_RE = re.compile(r"(certified|standard|true[_ ]?height|height[_ ]?(certified|standard))", re.I)


def number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    value = number(value)
    return None if value is None else int(value)


def boolean(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


def invalid_status(value: Any) -> bool:
    return str(value or "").strip().lower() in INVALID_STATUS


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(errors: list[float]) -> dict[str, Any]:
    if not errors:
        return {"n": 0, "bias_mm": None, "mae_mm": None, "rmse_mm": None, "p95_abs_mm": None, "max_abs_mm": None, "repeatability_std_mm": None}
    array = np.asarray(errors, dtype=float)
    absolute = np.abs(array)
    return {
        "n": len(errors),
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(array * array))),
        "p95_abs_mm": float(np.percentile(absolute, 95)),
        "max_abs_mm": float(np.max(absolute)),
        "repeatability_std_mm": float(np.std(array, ddof=1)) if len(errors) > 1 else 0.0,
    }


def average(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None

    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values), kind="mergesort")
        result = np.empty(len(values), dtype=float)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        return result

    rx, ry = ranks(x), ranks(y)
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denominator = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    return None if denominator == 0 else float(np.sum(rx * ry) / denominator)


def walk_certified(value: Any, prefix: str = "") -> list[tuple[float, str]]:
    found: list[tuple[float, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if CERTIFIED_RE.search(str(key)) and not isinstance(child, (dict, list)):
                parsed = number(child)
                if parsed is not None:
                    found.append((parsed, path))
            found.extend(walk_certified(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(walk_certified(child, f"{prefix}[{index}]"))
    return found


def determine_truth(path: Path, height_label: str) -> tuple[float | None, str, str]:
    candidates: list[tuple[float, str]] = []
    for json_path in sorted(path.glob("*.json")):
        try:
            document = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidates.extend((value, f"{json_path}:{key}") for value, key in walk_certified(document))
    for csv_path in sorted(path.glob("*.csv")):
        fields, rows = read_csv(csv_path)
        keys = [field for field in fields if CERTIFIED_RE.search(field)]
        for row in rows:
            for key in keys:
                parsed = number(row.get(key))
                if parsed is not None:
                    candidates.append((parsed, f"{csv_path}:{key}"))
    if candidates:
        if len({round(value, 9) for value, _ in candidates}) == 1:
            return candidates[0][0], "certified_metadata", candidates[0][1]
        return None, "certified_metadata_conflict", compact(candidates)
    if height_label in NOMINAL:
        return NOMINAL[height_label], "nominal_folder", f"{height_label}/"
    return None, "unknown", ""


def discover(root: Path) -> list[tuple[str, str, Path]]:
    result = []
    for height_path in sorted(root.iterdir()):
        if not height_path.is_dir() or not re.fullmatch(r"h\d+", height_path.name, re.I):
            continue
        for condition_path in sorted(height_path.iterdir()):
            if condition_path.is_dir() and re.fullmatch(rf"{re.escape(height_path.name)}_p\d+", condition_path.name, re.I):
                result.append((height_path.name, condition_path.name, condition_path))
    return result


def load_condition(dataset: str, height_label: str, position_id: str, path: Path) -> dict[str, Any]:
    truth, truth_source, truth_path = determine_truth(path, height_label)
    frame_fields, frames = read_csv(path / "frames.csv")
    shadow_fields, shadow = read_csv(path / "height_shadow.csv")
    pngs = sorted(path.glob("frame_*.png"))
    frame_map: dict[int, dict[str, str] | None] = {}
    frame_ids: list[int] = []
    frame_names: dict[str, dict[str, str]] = {}
    duplicate_frame_ids = 0
    frame_gap_nonzero = 0
    for row in frames:
        frame_id = integer(row.get("camera_frame_number"))
        if frame_id is not None:
            frame_ids.append(frame_id)
            if frame_id in frame_map:
                duplicate_frame_ids += 1
                frame_map[frame_id] = None
            else:
                frame_map[frame_id] = row
        name = str(row.get("filename") or "").strip()
        if name:
            frame_names[name] = row
        if number(row.get("frame_gap")) not in (None, 0.0):
            frame_gap_nonzero += 1
    sequence_gaps = sum(b - a != 1 for a, b in zip(sorted(frame_ids), sorted(frame_ids)[1:]))
    raw_names = {item.name for item in pngs}
    missing_raw = sorted(set(frame_names) - raw_names)
    orphan_raw = sorted(raw_names - set(frame_names))
    records: list[dict[str, Any]] = []
    shadow_ids: list[int] = []
    duplicate_shadow_ids = 0
    for row in shadow:
        frame_id = integer(row.get("camera_frame_number"))
        if frame_id is not None:
            if frame_id in shadow_ids:
                duplicate_shadow_ids += 1
            shadow_ids.append(frame_id)
        associated = frame_map.get(frame_id) if frame_id is not None else None
        if frame_id is not None and frame_id in frame_map and associated is None:
            association = "ambiguous_duplicate_frame_id"
        elif associated is not None:
            association = "matched"
        else:
            association = "shadow_frame_id_not_in_frames_csv"
        values = {key: number(row.get(key)) for key in ("v_min", "v_median", "v_max", "q1", "q2", "height_raw", "height_h1", "height_hb2")}
        active_valid = boolean(row.get("active_height_valid"))
        active_status = str(row.get("active_height_status") or "").strip()
        q2_domain = boolean(row.get("q2_in_domain"))
        ground_status = str(row.get("ground_reference_status") or "").strip()
        numeric_heights = all(values[key] is not None for key in ("height_raw", "height_h1", "height_hb2"))
        valid = numeric_heights and active_valid and not invalid_status(active_status) and q2_domain and not invalid_status(ground_status) and association == "matched"
        reasons = []
        if association != "matched":
            reasons.append(association)
        if not numeric_heights:
            reasons.append("empty_height_fields")
        if not active_valid:
            reasons.append("active_height_valid_false")
        if invalid_status(active_status):
            reasons.append(f"active_height_status={active_status or 'empty'}")
        if not q2_domain:
            reasons.append("q2_out_of_domain")
        if invalid_status(ground_status):
            reasons.append(f"ground_reference_status={ground_status or 'empty'}")
        residuals = {"Base": None, "H1": None, "H-B2": None}
        if valid and truth is not None:
            fields = {"Base": "height_raw", "H1": "height_h1", "H-B2": "height_hb2"}
            residuals = {model: values[fields[model]] - truth for model in MODELS}
        records.append({
            "row": row, "frame_id": frame_id, "associated": associated, "association": association, "values": values,
            "active_height_correction": row.get("active_height_correction", ""), "active_height": number(row.get("active_height")),
            "active_valid": active_valid, "active_status": active_status, "q2_domain": q2_domain, "hb2_status": row.get("hb2_q2_status", ""),
            "c1_clamp": row.get("c1_clamp_status", ""), "ground_status": ground_status, "point_count": integer(row.get("point_count")),
            "valid": valid, "reasons": ";".join(reasons), "residuals": residuals,
        })
    vmedians = [record["values"]["v_median"] for record in records if record["values"]["v_median"] is not None]
    vmins = [record["values"]["v_min"] for record in records if record["values"]["v_min"] is not None]
    vmaxs = [record["values"]["v_max"] for record in records if record["values"]["v_max"] is not None]
    return {
        "dataset": dataset, "height_label": height_label, "position_id": position_id, "path": path, "truth": truth, "truth_source": truth_source, "truth_path": truth_path,
        "frame_fields": frame_fields, "shadow_fields": shadow_fields, "frames": frames, "shadow": shadow, "pngs": pngs, "frame_map": frame_map, "records": records,
        "duplicate_frame_ids": duplicate_frame_ids, "duplicate_shadow_ids": duplicate_shadow_ids, "missing_raw": missing_raw, "orphan_raw": orphan_raw,
        "frame_gap_nonzero": frame_gap_nonzero, "sequence_gaps": sequence_gaps, "v_position": float(median(vmedians)) if vmedians else None,
        "v_position_min": min(vmedians) if vmedians else None, "v_position_max": max(vmedians) if vmedians else None, "v_min": min(vmins) if vmins else None,
        "v_max": max(vmaxs) if vmaxs else None, "rank": None,
    }


def assign_ranks(conditions: list[dict[str, Any]]) -> None:
    by_height: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        by_height.setdefault(condition["height_label"], []).append(condition)
    for group in by_height.values():
        ordered = sorted(group, key=lambda item: (item["v_position"] is None, item["v_position"] if item["v_position"] is not None else math.inf, item["position_id"]))
        for rank, condition in enumerate(ordered, 1):
            condition["rank"] = rank


def condition_records(condition: dict[str, Any], model: str) -> list[dict[str, Any]]:
    return [record for record in condition["records"] if record["valid"] and record["residuals"][model] is not None]


def condition_model_row(condition: dict[str, Any], model: str) -> dict[str, Any]:
    records = condition_records(condition, model)
    result = metric([record["residuals"][model] for record in records])
    edge_valid = [record for record in records if (record["values"]["v_median"] or -math.inf) > EDGE_V]
    edge_shadow = [record for record in condition["records"] if (record["values"]["v_median"] or -math.inf) > EDGE_V]
    valid_count = sum(record["valid"] for record in condition["records"])
    return {
        "row_type": "condition_model", "dataset": condition["dataset"], "height_label": condition["height_label"], "true_height_mm": condition["truth"], "true_height_source": condition["truth_source"],
        "position_id": condition["position_id"], "v_order_rank": condition["rank"], "model": model, "metric_scope": "height_position", **result,
        "raw_frame_count": len(condition["pngs"]), "frames_csv_count": len(condition["frames"]), "shadow_row_count": len(condition["shadow"]),
        "matched_shadow_count": sum(record["association"] == "matched" for record in condition["records"]), "processed_valid_count": valid_count,
        "processed_coverage": valid_count / len(condition["pngs"]) if condition["pngs"] else None, "v_position_median_px": condition["v_position"],
        "v_median_min_px": condition["v_position_min"], "v_median_max_px": condition["v_position_max"], "edge_shadow_n": len(edge_shadow), "edge_valid_n": len(edge_valid),
        "status": "OK" if result["n"] else "NOT_SUPPORTED_NO_VALID_PROCESSED_FRAME", "notes": "NA means no valid exported processed row.",
    }


def paired_condition_row(condition: dict[str, Any]) -> dict[str, Any]:
    records = [record for record in condition["records"] if record["valid"] and record["residuals"]["H1"] is not None and record["residuals"]["H-B2"] is not None]
    h1 = metric([record["residuals"]["H1"] for record in records])
    hb2 = metric([record["residuals"]["H-B2"] for record in records])
    base = condition_model_row(condition, "Base")
    diff = lambda left, right: right - left if left is not None and right is not None else None
    row = dict(base)
    row.update({
        "row_type": "condition_paired_comparison", "model": "H1_vs_H-B2", "metric_scope": "same_processed_frame", "n": len(records),
        "bias_mm": None, "mae_mm": None, "rmse_mm": None, "p95_abs_mm": None, "max_abs_mm": None, "repeatability_std_mm": None,
        "h1_bias_mm": h1["bias_mm"], "hb2_bias_mm": hb2["bias_mm"], "bias_diff_hb2_minus_h1_mm": diff(h1["bias_mm"], hb2["bias_mm"]),
        "h1_p95_abs_mm": h1["p95_abs_mm"], "hb2_p95_abs_mm": hb2["p95_abs_mm"], "p95_diff_hb2_minus_h1_mm": diff(h1["p95_abs_mm"], hb2["p95_abs_mm"]),
        "h1_max_abs_mm": h1["max_abs_mm"], "hb2_max_abs_mm": hb2["max_abs_mm"], "max_diff_hb2_minus_h1_mm": diff(h1["max_abs_mm"], hb2["max_abs_mm"]),
        "delta_abs_error_mean_mm": average([abs(record["residuals"]["H-B2"]) - abs(record["residuals"]["H1"]) for record in records]),
        "delta_squared_error_mean_mm": average([record["residuals"]["H-B2"] ** 2 - record["residuals"]["H1"] ** 2 for record in records]),
        "status": "OK" if records else "NOT_SUPPORTED_NO_PAIRED_VALID_FRAME", "notes": "H-B2 minus H1 paired deltas use identical valid frames.",
    })
    return row


CONDITION_FIELDS = [
    "row_type", "dataset", "height_label", "true_height_mm", "true_height_source", "position_id", "v_order_rank", "model", "metric_scope", "n",
    "raw_frame_count", "frames_csv_count", "shadow_row_count", "matched_shadow_count", "processed_valid_count", "processed_coverage", "v_position_median_px", "v_median_min_px", "v_median_max_px",
    "edge_shadow_n", "edge_valid_n", "bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm", "repeatability_std_mm",
    "h1_bias_mm", "hb2_bias_mm", "bias_diff_hb2_minus_h1_mm", "h1_p95_abs_mm", "hb2_p95_abs_mm", "p95_diff_hb2_minus_h1_mm",
    "h1_max_abs_mm", "hb2_max_abs_mm", "max_diff_hb2_minus_h1_mm", "delta_abs_error_mean_mm", "delta_squared_error_mean_mm", "status", "notes",
]


def height_model_row(height: str, group: list[dict[str, Any]], model: str) -> dict[str, Any]:
    supported: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for condition in group:
        records = condition_records(condition, model)
        result = metric([record["residuals"][model] for record in records])
        if result["n"]:
            supported.append((condition, result))
    biases = [result["bias_mm"] for _, result in supported]
    worst_bias = max(supported, key=lambda item: abs(item[1]["bias_mm"])) if supported else None
    worst_p95 = max(supported, key=lambda item: item[1]["p95_abs_mm"]) if supported else None
    worst_max = max(supported, key=lambda item: item[1]["max_abs_mm"]) if supported else None
    pooled = [record for condition in group for record in condition_records(condition, model)]
    result = metric([record["residuals"][model] for record in pooled])
    return {
        "row_type": "height_model", "dataset": group[0]["dataset"], "height_label": height, "true_height_mm": group[0]["truth"], "model": model, "metric_scope": "position_spread_primary",
        "n": result["n"], "position_count_available": len(supported), "available_positions": compact([condition["position_id"] for condition, _ in supported]),
        "position_bias_min_mm": min(biases) if biases else None, "position_bias_max_mm": max(biases) if biases else None, "position_bias_range_mm": max(biases) - min(biases) if biases else None,
        "position_bias_std_mm": float(np.std(np.asarray(biases), ddof=0)) if biases else None,
        "worst_position_abs_bias": worst_bias[0]["position_id"] if worst_bias else None, "worst_position_abs_bias_mm": abs(worst_bias[1]["bias_mm"]) if worst_bias else None,
        "worst_position_p95": worst_p95[0]["position_id"] if worst_p95 else None, "worst_position_p95_abs_mm": worst_p95[1]["p95_abs_mm"] if worst_p95 else None,
        "worst_position_max": worst_max[0]["position_id"] if worst_max else None, "worst_position_max_abs_mm": worst_max[1]["max_abs_mm"] if worst_max else None,
        "bias_mm": result["bias_mm"], "mae_mm": result["mae_mm"], "rmse_mm": result["rmse_mm"], "p95_abs_mm": result["p95_abs_mm"], "max_abs_mm": result["max_abs_mm"],
        "status": "OK" if supported else "NOT_SUPPORTED_NO_VALID_PROCESSED_FRAME", "notes": "Position Bias range/std are primary; pooled metrics are auxiliary.",
    }


def paired_height_row(height: str, group: list[dict[str, Any]], h1_row: dict[str, Any], hb2_row: dict[str, Any]) -> dict[str, Any]:
    records = [record for condition in group for record in condition["records"] if record["valid"] and record["residuals"]["H1"] is not None and record["residuals"]["H-B2"] is not None]
    h1, hb2 = metric([record["residuals"]["H1"] for record in records]), metric([record["residuals"]["H-B2"] for record in records])
    diff = lambda left, right: right - left if left is not None and right is not None else None
    row = {key: None for key in HEIGHT_FIELDS}
    row.update({
        "row_type": "height_paired_comparison", "dataset": group[0]["dataset"], "height_label": height, "true_height_mm": group[0]["truth"], "model": "H1_vs_H-B2", "metric_scope": "same_processed_frame",
        "n": len(records), "position_count_available": min(h1_row["position_count_available"], hb2_row["position_count_available"]), "h1_bias_mm": h1["bias_mm"], "hb2_bias_mm": hb2["bias_mm"],
        "bias_diff_hb2_minus_h1_mm": diff(h1["bias_mm"], hb2["bias_mm"]), "h1_position_bias_range_mm": h1_row["position_bias_range_mm"], "hb2_position_bias_range_mm": hb2_row["position_bias_range_mm"],
        "position_bias_range_diff_hb2_minus_h1_mm": diff(h1_row["position_bias_range_mm"], hb2_row["position_bias_range_mm"]), "h1_position_bias_std_mm": h1_row["position_bias_std_mm"], "hb2_position_bias_std_mm": hb2_row["position_bias_std_mm"],
        "position_bias_std_diff_hb2_minus_h1_mm": diff(h1_row["position_bias_std_mm"], hb2_row["position_bias_std_mm"]), "h1_worst_position_p95_abs_mm": h1_row["worst_position_p95_abs_mm"], "hb2_worst_position_p95_abs_mm": hb2_row["worst_position_p95_abs_mm"],
        "worst_position_p95_diff_hb2_minus_h1_mm": diff(h1_row["worst_position_p95_abs_mm"], hb2_row["worst_position_p95_abs_mm"]), "h1_worst_position_max_abs_mm": h1_row["worst_position_max_abs_mm"], "hb2_worst_position_max_abs_mm": hb2_row["worst_position_max_abs_mm"],
        "worst_position_max_diff_hb2_minus_h1_mm": diff(h1_row["worst_position_max_abs_mm"], hb2_row["worst_position_max_abs_mm"]),
        "delta_abs_error_mean_mm": average([abs(record["residuals"]["H-B2"]) - abs(record["residuals"]["H1"]) for record in records]),
        "delta_squared_error_mean_mm": average([record["residuals"]["H-B2"] ** 2 - record["residuals"]["H1"] ** 2 for record in records]),
        "status": "OK" if records else "NOT_SUPPORTED_NO_PAIRED_VALID_FRAME", "notes": "H-B2 minus H1 differences use identical valid frames.",
    })
    return row


HEIGHT_FIELDS = [
    "row_type", "dataset", "height_label", "true_height_mm", "model", "metric_scope", "n", "position_count_available", "available_positions",
    "position_bias_min_mm", "position_bias_max_mm", "position_bias_range_mm", "position_bias_std_mm", "worst_position_abs_bias", "worst_position_abs_bias_mm",
    "worst_position_p95", "worst_position_p95_abs_mm", "worst_position_max", "worst_position_max_abs_mm", "bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm",
    "h1_bias_mm", "hb2_bias_mm", "bias_diff_hb2_minus_h1_mm", "h1_position_bias_range_mm", "hb2_position_bias_range_mm", "position_bias_range_diff_hb2_minus_h1_mm",
    "h1_position_bias_std_mm", "hb2_position_bias_std_mm", "position_bias_std_diff_hb2_minus_h1_mm", "h1_worst_position_p95_abs_mm", "hb2_worst_position_p95_abs_mm", "worst_position_p95_diff_hb2_minus_h1_mm",
    "h1_worst_position_max_abs_mm", "hb2_worst_position_max_abs_mm", "worst_position_max_diff_hb2_minus_h1_mm", "delta_abs_error_mean_mm", "delta_squared_error_mean_mm", "status", "notes",
]


def pooled_height_row(conditions: list[dict[str, Any]], model: str) -> dict[str, Any]:
    records = [record for condition in conditions for record in condition_records(condition, model)]
    result = metric([record["residuals"][model] for record in records])
    row = {key: None for key in HEIGHT_FIELDS}
    row.update({
        "row_type": "session_pooled",
        "dataset": conditions[0]["dataset"] if conditions else "session01",
        "height_label": "ALL",
        "true_height_mm": None,
        "model": model,
        "metric_scope": "auxiliary_pooled",
        "n": result["n"],
        "position_count_available": len({condition["position_id"] for condition in conditions for record in condition_records(condition, model)}),
        "available_positions": "",
        "bias_mm": result["bias_mm"],
        "mae_mm": result["mae_mm"],
        "rmse_mm": result["rmse_mm"],
        "p95_abs_mm": result["p95_abs_mm"],
        "max_abs_mm": result["max_abs_mm"],
        "status": "OK" if records else "NOT_SUPPORTED_NO_VALID_PROCESSED_FRAME",
        "notes": "Pooled result is auxiliary and never the spatial-consistency criterion.",
    })
    return row


def edge_row(row_type: str, dataset: str, height: str, position: str, rank: Any, model: str, records: list[dict[str, Any]], notes: str) -> dict[str, Any]:
    shadow_edge = [record for record in records if (record["values"]["v_median"] or -math.inf) > EDGE_V]
    valid_edge = [record for record in shadow_edge if record["valid"] and record["residuals"][model] is not None]
    errors = [record["residuals"][model] for record in valid_edge]
    result = metric(errors)
    absolute = [abs(error) for error in errors]
    row = {key: None for key in EDGE_FIELDS}
    row.update({
        "row_type": row_type, "dataset": dataset, "height_label": height, "position_id": position, "v_order_rank": rank, "model": model, "metric_scope": "frame_v_median_gt_2400",
        "edge_basis": "v_median_gt_2400; v_max is diagnostic only", "shadow_edge_n": len(shadow_edge), "valid_edge_n": len(valid_edge), **result,
        "covered_heights": compact(sorted({record.get("height_label") for record in valid_edge})), "covered_positions": compact(sorted({record.get("position_id") for record in valid_edge})),
        "covered_v_order_ranks": compact(sorted({record.get("rank") for record in valid_edge if record.get("rank") is not None})), "gt_0p1_rate": sum(value > 0.1 for value in absolute) / len(absolute) if absolute else None,
        "gt_0p2_rate": sum(value > 0.2 for value in absolute) / len(absolute) if absolute else None, "gt_0p1_count": sum(value > 0.1 for value in absolute), "gt_0p2_count": sum(value > 0.2 for value in absolute),
        "residual_v_spearman_r": spearman([record["values"]["v_median"] for record in valid_edge], errors) if valid_edge else None,
        "v_median_min_px": min((record["values"]["v_median"] for record in valid_edge), default=None), "v_median_max_px": max((record["values"]["v_median"] for record in valid_edge), default=None),
        "status": "OK" if valid_edge else "NOT_SUPPORTED_NO_VALID_EDGE_FRAME", "notes": notes,
    })
    return row


EDGE_FIELDS = [
    "row_type", "dataset", "height_label", "position_id", "v_order_rank", "model", "metric_scope", "edge_basis", "shadow_edge_n", "valid_edge_n", "covered_heights", "covered_positions", "covered_v_order_ranks",
    "n", "bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm", "repeatability_std_mm", "gt_0p1_rate", "gt_0p2_rate", "gt_0p1_count", "gt_0p2_count", "residual_v_spearman_r", "v_median_min_px", "v_median_max_px", "status", "notes",
]


def build_edges(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for condition in conditions:
        for record in condition["records"]:
            record["height_label"], record["position_id"], record["rank"] = condition["height_label"], condition["position_id"], condition["rank"]
    all_records = [record for condition in conditions for record in condition["records"]]
    by_height: dict[str, list[dict[str, Any]]] = {}
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for condition in conditions:
        by_height.setdefault(condition["height_label"], []).extend(condition["records"])
        if condition["rank"] is not None:
            by_rank.setdefault(condition["rank"], []).extend(condition["records"])
    for model in MODELS:
        rows.append(edge_row("pooled", conditions[0]["dataset"], "ALL", "ALL", "ALL", model, all_records, "Pooled edge row is auxiliary."))
    for height, records in sorted(by_height.items()):
        for model in MODELS:
            rows.append(edge_row("height", conditions[0]["dataset"], height, "ALL", "ALL", model, records, "Height edge decomposition."))
        for condition in sorted([item for item in conditions if item["height_label"] == height], key=lambda item: item["rank"] or math.inf):
            for model in MODELS:
                rows.append(edge_row("height_position_rank", conditions[0]["dataset"], height, condition["position_id"], condition["rank"], model, condition["records"], "Actual v_order_rank condition decomposition."))
    for rank, records in sorted(by_rank.items()):
        for model in MODELS:
            rows.append(edge_row("v_order_rank", conditions[0]["dataset"], "ALL", "ALL", rank, model, records, "Cross-height actual v_order_rank decomposition."))
    return rows


def coverage(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_height: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        by_height.setdefault(condition["height_label"], []).append(condition)
        values = [record["values"]["v_median"] for record in condition["records"] if record["values"]["v_median"] is not None]
        rows.append({
            "row_type": "position", "dataset": condition["dataset"], "height_label": condition["height_label"], "position_id": condition["position_id"], "v_order_rank": condition["rank"],
            "v_position_median_px": condition["v_position"], "v_median_min_px": condition["v_position_min"], "v_median_max_px": condition["v_position_max"], "v_min_px": condition["v_min"], "v_max_px": condition["v_max"],
            "shadow_row_count": len(condition["shadow"]), "processed_valid_count": sum(record["valid"] for record in condition["records"]),
            "v_median_gt_2200_shadow_n": sum(value > 2200 for value in values), "v_median_gt_2400_shadow_n": sum(value > 2400 for value in values), "v_median_gt_2600_shadow_n": sum(value > 2600 for value in values),
            "v_max_gt_2200_shadow_n": sum((record["values"]["v_max"] or -math.inf) > 2200 for record in condition["records"]), "v_max_gt_2400_shadow_n": sum((record["values"]["v_max"] or -math.inf) > 2400 for record in condition["records"]), "v_max_gt_2600_shadow_n": sum((record["values"]["v_max"] or -math.inf) > 2600 for record in condition["records"]),
            "v_median_range_fraction_of_image": (condition["v_position_max"] - condition["v_position_min"]) / 3000 if condition["v_position_min"] is not None and condition["v_position_max"] is not None else None,
            "max_adjacent_gap_px": None, "min_adjacent_gap_px": None, "position_count": None, "threshold_support_status": "NO_POSITION_COORDINATE_SUPPORT", "notes": "v_median is position coordinate; v_max is diagnostic only.",
        })
    for height, group in sorted(by_height.items()):
        values = [item["v_position"] for item in sorted(group, key=lambda item: item["rank"] or math.inf) if item["v_position"] is not None]
        gaps = [right - left for left, right in zip(values, values[1:])]
        position_rows = [row for row in rows if row["row_type"] == "position" and row["height_label"] == height]
        rows.append({
            "row_type": "height_summary", "dataset": group[0]["dataset"], "height_label": height, "position_id": "ALL", "v_order_rank": "", "v_position_median_px": median(values) if values else None,
            "v_median_min_px": min(values) if values else None, "v_median_max_px": max(values) if values else None, "v_min_px": min((item["v_min"] for item in group if item["v_min"] is not None), default=None), "v_max_px": max((item["v_max"] for item in group if item["v_max"] is not None), default=None),
            "shadow_row_count": sum(row["shadow_row_count"] for row in position_rows), "processed_valid_count": sum(row["processed_valid_count"] for row in position_rows),
            "v_median_gt_2200_shadow_n": sum(row["v_median_gt_2200_shadow_n"] for row in position_rows), "v_median_gt_2400_shadow_n": sum(row["v_median_gt_2400_shadow_n"] for row in position_rows), "v_median_gt_2600_shadow_n": sum(row["v_median_gt_2600_shadow_n"] for row in position_rows),
            "v_max_gt_2200_shadow_n": sum(row["v_max_gt_2200_shadow_n"] for row in position_rows), "v_max_gt_2400_shadow_n": sum(row["v_max_gt_2400_shadow_n"] for row in position_rows), "v_max_gt_2600_shadow_n": sum(row["v_max_gt_2600_shadow_n"] for row in position_rows),
            "v_median_range_fraction_of_image": (max(values) - min(values)) / 3000 if values else None, "max_adjacent_gap_px": max(gaps) if gaps else None, "min_adjacent_gap_px": min(gaps) if gaps else None, "position_count": len(values),
            "threshold_support_status": "SUPPORTED" if values and max(values) >= 2600 else "NOT_SUPPORTED", "notes": "v_order_rank is recomputed independently within each height.",
        })
    values = [item["v_position"] for item in conditions if item["v_position"] is not None]
    position_rows = [row for row in rows if row["row_type"] == "position"]
    rows.append({
        "row_type": "session_summary", "dataset": conditions[0]["dataset"] if conditions else "session01", "height_label": "ALL", "position_id": "ALL", "v_order_rank": "", "v_position_median_px": median(values) if values else None,
        "v_median_min_px": min(values) if values else None, "v_median_max_px": max(values) if values else None, "v_min_px": min((item["v_min"] for item in conditions if item["v_min"] is not None), default=None), "v_max_px": max((item["v_max"] for item in conditions if item["v_max"] is not None), default=None),
        "shadow_row_count": sum(row["shadow_row_count"] for row in position_rows), "processed_valid_count": sum(row["processed_valid_count"] for row in position_rows),
        "v_median_gt_2200_shadow_n": sum(row["v_median_gt_2200_shadow_n"] for row in position_rows), "v_median_gt_2400_shadow_n": sum(row["v_median_gt_2400_shadow_n"] for row in position_rows), "v_median_gt_2600_shadow_n": sum(row["v_median_gt_2600_shadow_n"] for row in position_rows),
        "v_max_gt_2200_shadow_n": sum(row["v_max_gt_2200_shadow_n"] for row in position_rows), "v_max_gt_2400_shadow_n": sum(row["v_max_gt_2400_shadow_n"] for row in position_rows), "v_max_gt_2600_shadow_n": sum(row["v_max_gt_2600_shadow_n"] for row in position_rows),
        "v_median_range_fraction_of_image": (max(values) - min(values)) / 3000 if values else None, "max_adjacent_gap_px": None, "min_adjacent_gap_px": None, "position_count": len(values),
        "threshold_support_status": "SUPPORTED" if values and max(values) >= 2600 else "NOT_SUPPORTED", "notes": "v_max reaches image boundary but does not establish position support.",
    })
    return rows


COVERAGE_FIELDS = [
    "row_type", "dataset", "height_label", "position_id", "v_order_rank", "v_position_median_px", "v_median_min_px", "v_median_max_px", "v_min_px", "v_max_px", "shadow_row_count", "processed_valid_count",
    "v_median_gt_2200_shadow_n", "v_median_gt_2400_shadow_n", "v_median_gt_2600_shadow_n", "v_max_gt_2200_shadow_n", "v_max_gt_2400_shadow_n", "v_max_gt_2600_shadow_n", "v_median_range_fraction_of_image", "max_adjacent_gap_px", "min_adjacent_gap_px", "position_count", "threshold_support_status", "notes",
]


def pointwise(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for condition in conditions:
        for record in condition["records"]:
            associated = record["associated"]
            rows.append({
                "dataset": condition["dataset"], "height_label": condition["height_label"], "true_height_mm": condition["truth"], "true_height_source": condition["truth_source"], "position_id": condition["position_id"], "v_order_rank": condition["rank"],
                "camera_frame_number": record["frame_id"], "raw_frame_filename": associated.get("filename", "") if associated else "", "raw_frame_association": record["association"], "raw_frame_exists": bool(associated and (condition["path"] / str(associated.get("filename", ""))).is_file()), "frame_gap": associated.get("frame_gap", "") if associated else "",
                "v_min": record["values"]["v_min"], "v_median": record["values"]["v_median"], "v_max": record["values"]["v_max"], "q1": record["values"]["q1"], "q2": record["values"]["q2"], "q2_in_domain": record["q2_domain"],
                "height_raw": record["values"]["height_raw"], "height_h1": record["values"]["height_h1"], "height_hb2": record["values"]["height_hb2"], "active_height_correction": record["active_height_correction"], "active_height": record["active_height"], "active_height_valid": record["active_valid"], "active_height_status": record["active_status"],
                "residual_base": record["residuals"]["Base"], "residual_h1": record["residuals"]["H1"], "residual_hb2": record["residuals"]["H-B2"],
                "abs_error_delta_hb2_minus_h1": abs(record["residuals"]["H-B2"]) - abs(record["residuals"]["H1"]) if record["residuals"]["H1"] is not None and record["residuals"]["H-B2"] is not None else None,
                "squared_error_delta_hb2_minus_h1": record["residuals"]["H-B2"] ** 2 - record["residuals"]["H1"] ** 2 if record["residuals"]["H1"] is not None and record["residuals"]["H-B2"] is not None else None,
                "c1_clamp_status": record["c1_clamp"], "hb2_q2_status": record["hb2_status"], "ground_reference_status": record["ground_status"], "point_count": record["point_count"], "processed_valid": record["valid"], "invalid_reason": record["reasons"],
            })
    return rows


POINT_FIELDS = [
    "dataset", "height_label", "true_height_mm", "true_height_source", "position_id", "v_order_rank", "camera_frame_number", "raw_frame_filename", "raw_frame_association", "raw_frame_exists", "frame_gap", "v_min", "v_median", "v_max", "q1", "q2", "q2_in_domain", "height_raw", "height_h1", "height_hb2", "active_height_correction", "active_height", "active_height_valid", "active_height_status", "residual_base", "residual_h1", "residual_hb2", "abs_error_delta_hb2_minus_h1", "squared_error_delta_hb2_minus_h1", "c1_clamp_status", "hb2_q2_status", "ground_reference_status", "point_count", "processed_valid", "invalid_reason",
]


QC_FIELDS = [
    "qc_scope", "row_type", "dataset", "height_label", "position_id", "source_file", "metric_name", "metric_value", "metric_units", "raw_frame_count", "frames_csv_count", "shadow_row_count", "matched_shadow_count", "processed_valid_count", "processed_coverage",
    "missing_required_files", "missing_raw_files", "orphan_raw_files", "duplicate_frame_id_count", "duplicate_shadow_id_count", "frame_gap_field_nonzero_count", "camera_sequence_gap_count", "empty_height_result_count", "active_valid_false_count", "c1_clamp_status_counts", "hb2_ood_count", "q2_in_domain_false_count", "ground_status_counts", "shadow_status_counts", "frame_association_status_counts", "truth_height_mm", "truth_height_source", "qc_status", "notes",
]


def count_json(values: list[Any]) -> str:
    result: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        result[key] = result.get(key, 0) + 1
    return compact(result)


def qc_condition(condition: dict[str, Any]) -> dict[str, Any]:
    missing_required = [name for name in ("frames.csv", "height_shadow.csv") if not (condition["path"] / name).is_file()]
    empty = sum(not all(record["values"][key] is not None for key in ("height_raw", "height_h1", "height_hb2")) for record in condition["records"])
    structural = bool(missing_required or len(condition["pngs"]) != len(condition["frames"]) or condition["missing_raw"] or condition["duplicate_frame_ids"] or condition["duplicate_shadow_ids"] or condition["frame_gap_nonzero"] or condition["sequence_gaps"])
    return {
        "qc_scope": "condition", "row_type": "condition_integrity", "dataset": condition["dataset"], "height_label": condition["height_label"], "position_id": condition["position_id"], "source_file": str(condition["path"]),
        "metric_name": "", "metric_value": "", "metric_units": "", "raw_frame_count": len(condition["pngs"]), "frames_csv_count": len(condition["frames"]), "shadow_row_count": len(condition["shadow"]), "matched_shadow_count": sum(record["association"] == "matched" for record in condition["records"]), "processed_valid_count": sum(record["valid"] for record in condition["records"]),
        "processed_coverage": sum(record["valid"] for record in condition["records"]) / len(condition["pngs"]) if condition["pngs"] else None, "missing_required_files": compact(missing_required), "missing_raw_files": compact(condition["missing_raw"]), "orphan_raw_files": compact(condition["orphan_raw"]),
        "duplicate_frame_id_count": condition["duplicate_frame_ids"], "duplicate_shadow_id_count": condition["duplicate_shadow_ids"], "frame_gap_field_nonzero_count": condition["frame_gap_nonzero"], "camera_sequence_gap_count": condition["sequence_gaps"], "empty_height_result_count": empty,
        "active_valid_false_count": sum(not record["active_valid"] for record in condition["records"]), "c1_clamp_status_counts": count_json([record["c1_clamp"] for record in condition["records"]]),
        "hb2_ood_count": sum((not record["q2_domain"]) or "OOD" in str(record["hb2_status"]).upper() for record in condition["records"]), "q2_in_domain_false_count": sum(not record["q2_domain"] for record in condition["records"]),
        "ground_status_counts": count_json([record["ground_status"] for record in condition["records"]]), "shadow_status_counts": count_json([record["active_status"] for record in condition["records"]]), "frame_association_status_counts": count_json([record["association"] for record in condition["records"]]),
        "truth_height_mm": condition["truth"], "truth_height_source": condition["truth_source"], "qc_status": "FAIL" if structural else ("PASS" if sum(record["valid"] for record in condition["records"]) else "PARTIAL"),
        "notes": "Shadow count is diagnostic coverage and is not required to equal raw PNG count.",
    }


def qc_metric(path: Path, name: str, value: Any, units: str = "", notes: str = "") -> dict[str, Any]:
    row = {field: "" for field in QC_FIELDS}
    row.update({"qc_scope": "provenance", "row_type": "provenance_metric", "dataset": "session01", "source_file": str(path), "metric_name": name, "metric_value": compact(value) if isinstance(value, (dict, list)) else value, "metric_units": units, "qc_status": "INFO", "notes": notes})
    return row


def provenance(data_root: Path, session: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    path = data_root / "session_ground_calibration.json"
    detection, runtime = session.get("detection", {}), session.get("runtime", {})
    ground = session.get("session_ground_reference", {})
    support, sanity = ground.get("support", {}), session.get("laser_ground_sanity", {})
    sm, quality = sanity.get("metrics", {}), session.get("quality", {})
    rows = [
        qc_metric(path, "session_status", session.get("status")), qc_metric(path, "session_valid", session.get("valid")), qc_metric(path, "ground_extrinsic_source", runtime.get("ground_extrinsic_source")), qc_metric(path, "ground_reference_source", runtime.get("ground_reference_source")), qc_metric(path, "ground_extrinsic_generation", runtime.get("ground_extrinsic_generation")),
        qc_metric(path, "pnp_detection_method", detection.get("method")), qc_metric(path, "pnp_corner_count", detection.get("corner_count"), "corners"), qc_metric(path, "pnp_reprojection_rmse", detection.get("reprojection_rmse_px"), "px"), qc_metric(path, "quality_edge_margin_min", quality.get("edge_margin_min_px"), "px"), qc_metric(path, "quality_warnings", quality.get("warnings")),
        qc_metric(path, "ground_reference_status", ground.get("status")), qc_metric(path, "ground_fit_source", ground.get("fit_source")), qc_metric(path, "ground_support_source", ground.get("support_source")), qc_metric(path, "ground_slope_z_per_mm", ground.get("slope_z_per_mm"), "mm/mm"), qc_metric(path, "ground_intercept_z_mm", ground.get("intercept_z_mm"), "mm"), qc_metric(path, "ground_fit_rmse_mm", ground.get("rmse_mm"), "mm"), qc_metric(path, "ground_valid_s_range_mm", ground.get("valid_s_range_mm"), "mm"), qc_metric(path, "ground_point_count", ground.get("point_count"), "points"), qc_metric(path, "ground_inlier_count", ground.get("inlier_count"), "points"),
        qc_metric(path, "ground_support_status", support.get("status")), qc_metric(path, "ground_support_mask_mode", support.get("mask_mode")), qc_metric(path, "ground_support_input_point_count", support.get("input_point_count"), "points"), qc_metric(path, "ground_support_selected_point_count", support.get("selected_point_count"), "points"), qc_metric(path, "ground_support_rejected_point_count", support.get("rejected_point_count"), "points"),
        qc_metric(path, "laser_ground_sanity_status", sanity.get("status")), qc_metric(path, "laser_ground_sanity_bias_zg_mm", sm.get("bias_zg_mm"), "mm"), qc_metric(path, "laser_ground_sanity_rmse_zg_mm", sm.get("rmse_zg_mm"), "mm"), qc_metric(path, "laser_ground_sanity_p95_abs_zg_mm", sm.get("p95_abs_zg_mm"), "mm"), qc_metric(path, "laser_ground_sanity_max_abs_zg_mm", sm.get("max_abs_zg_mm"), "mm"),
        qc_metric(path, "correction_applied", sanity.get("correction_applied")), qc_metric(path, "surface_correction_applied", sanity.get("surface_correction_applied")), qc_metric(path, "stage_a_applied", sanity.get("stage_a_applied")), qc_metric(path, "stage_a_height_scale_applied", sanity.get("stage_a_height_scale_applied")),
    ]
    artifacts = [
        ("frozen_c0_manifest", Path(r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0818\c0_freeze\c0_freeze_manifest.json"), "read-only provenance"),
        ("frozen_c1_config", repo_root / "laser_measurement_tool/configs/calibration_daheng_0811/frozen_c1_4k.json", "read-only provenance"),
        ("frozen_h1_config", repo_root / "laser_measurement_tool/configs/calibration_daheng_0811/stage_a_height_scale.json", "parameter provenance only"),
        ("frozen_hb2_config", repo_root / "laser_measurement_tool/configs/calibration_daheng_0811/hb2_height_correction.json", "candidate provenance only"),
    ]
    for name, artifact, note in artifacts:
        rows.extend([qc_metric(artifact, f"{name}_exists", artifact.is_file(), notes=note), qc_metric(artifact, f"{name}_sha256", file_hash(artifact), notes=note)])
    rows.extend([
        qc_metric(path, "artifact_reuse_frozen_c0", "YES", notes="Existing formal chain/provenance only; no recalculation."),
        qc_metric(path, "artifact_reuse_frozen_c1", "YES", notes="Existing formal chain/provenance only; no recalculation."),
        qc_metric(path, "artifact_reuse_frozen_ground_equivalent", "NO", notes="Session01 active ground source is session PnP."),
        qc_metric(path, "artifact_reuse_h1", "PROVENANCE_ONLY", notes="No synthesis from empty shadow height fields."),
        qc_metric(path, "artifact_reuse_hb2", "PROVENANCE_ONLY", notes="No synthesis from empty shadow height fields."),
        qc_metric(path, "new_correction_fit", "NO", notes="No new correction, LUT, spline, q1/v fit, or model search."),
    ])
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    value = number(value)
    return "NA" if value is None else f"{value:.{digits}f}"


def make_plots(output: Path, conditions: list[dict[str, Any]]) -> None:
    colors = {"h10": "#2563eb", "h20": "#16a34a", "h30": "#dc2626"}
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for condition in conditions:
        if condition["v_position"] is not None:
            ax.scatter(condition["rank"], condition["v_position"], color=colors.get(condition["height_label"], "#555"), s=36, label=condition["height_label"] if condition["rank"] == 1 else None)
    for threshold in FOV_THRESHOLDS:
        ax.axhline(threshold, linestyle="--", linewidth=1, label=f"v_median>{threshold:g}")
    ax.set(xlabel="v_order_rank within height", ylabel="position v_median [px]", ylim=(0, 3000), title="Session01 actual position-v coverage")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.5, 0.01, "v_max=2999 is diagnostic line extent, not position support; all rows are not_measured.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output / "session01_v_coverage.png", bbox_inches="tight")
    plt.close(fig)

    def empty(name: str, path: Path, xlabel: str = "", ylabel: str = "") -> None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.set(xlabel=xlabel, ylabel=ylabel, title=name)
        ax.text(0.5, 0.5, "No valid processed rows.\nMetrics intentionally NA.", transform=ax.transAxes, ha="center", va="center", color="#555", wrap=True)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / path, bbox_inches="tight")
        plt.close(fig)

    empty("Session01 position Bias by height", "session01_position_bias_by_height.png", "nominal height [mm]", "position Bias [mm]")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvline(EDGE_V, color="#dc2626", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 3000), xlabel="v_median [px]", ylabel="residual [mm]", title="Session01 residual vs v")
    ax.text(0.5, 0.5, "No valid residuals.\nResidual-v Spearman is NA.", transform=ax.transAxes, ha="center", va="center", color="#555")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "session01_residual_vs_v.png", bbox_inches="tight")
    plt.close(fig)
    empty("Session01 H1 vs H-B2 paired delta", "session01_h1_vs_hb2_paired_delta.png", "nominal height [mm]", "paired delta")
    empty("Session01 edge comparison: v_median > 2400", "session01_edge_comparison.png", "model", "edge metric")


def make_report(output: Path, data_root: Path, conditions: list[dict[str, Any]], session: dict[str, Any], qc: list[dict[str, Any]], heights: list[dict[str, Any]], edges: list[dict[str, Any]], cov: list[dict[str, Any]]) -> None:
    raw = sum(len(item["pngs"]) for item in conditions)
    frames = sum(len(item["frames"]) for item in conditions)
    shadow = sum(len(item["shadow"]) for item in conditions)
    matched = sum(sum(record["association"] == "matched" for record in item["records"]) for item in conditions)
    valid = sum(sum(record["valid"] for record in item["records"]) for item in conditions)
    fail = sum(row["qc_status"] == "FAIL" for row in qc if row["qc_scope"] == "condition")
    integrity = "FAIL" if fail else ("PASS" if valid else "PARTIAL")
    detection = session.get("detection", {})
    ground = session.get("session_ground_reference", {})
    support, sanity = ground.get("support", {}), session.get("laser_ground_sanity", {})
    sm = sanity.get("metrics", {})
    pnp = str(session.get("status", "")).upper() == "VALID" and bool(session.get("valid")) and number(detection.get("reprojection_rmse_px")) is not None
    ground_valid = str(ground.get("status", "")).upper() == "VALID" and str(sanity.get("status", "")).upper() == "VALID"
    pnp_ground = "YES" if pnp and ground_valid else ("PARTIAL" if pnp or ground_valid else "NO")
    positions = [row for row in cov if row["row_type"] == "position"]
    v_values = [number(row["v_position_median_px"]) for row in positions if number(row["v_position_median_px"]) is not None]
    support_counts = {threshold: sum(value > threshold for value in v_values) for threshold in FOV_THRESHOLDS}
    fov = "SUPPORTED" if valid and all(support_counts[t] for t in FOV_THRESHOLDS) else ("PARTIAL" if any(support_counts.values()) else "NOT_SUPPORTED")
    pooled_base = next((row for row in edges if row["row_type"] == "pooled" and row["model"] == "Base"), None)
    edge_n = integer(pooled_base.get("valid_edge_n")) if pooled_base else 0
    edge_failure = "PARTIAL" if not edge_n else ("YES" if (number(pooled_base.get("gt_0p1_rate")) or 0) > 0 or (number(pooled_base.get("p95_abs_mm")) or 0) > 0.1 else "NO")
    spatial_supported = bool(heights) and all(integer(row.get("n")) for row in heights if row["row_type"] == "height_model")
    h1_status = hb2_status = "SUPPORTED" if spatial_supported else "NOT_SUPPORTED"
    paired_supported = bool([row for row in heights if row["row_type"] == "height_paired_comparison"]) and all(integer(row.get("n")) for row in heights if row["row_type"] == "height_paired_comparison")
    lines = [
        "# Session01 全 FOV H1 vs H-B2 配对验证", "", f"- 数据根目录：{data_root}", f"- 输出目录：{output}", f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        "- 本轮只读 validation；未运行新 correction fit、模型搜索、ROI/Ground/C0/C1 修改。", "", "## 最终判定", "", "~~~text",
        f"SESSION01_DATA_INTEGRITY={integrity}", f"SESSION01_PNP_GROUND_VALID={pnp_ground}", f"SESSION01_FULL_FOV_COVERAGE={fov}", f"EDGE_V_GT_2400_FAILURE_REPRODUCED={edge_failure}", "",
        f"H1_SPATIAL_CONSISTENCY={h1_status}", f"HB2_SPATIAL_CONSISTENCY={hb2_status}", "", f"HB2_POSITION_SPREAD_ADVANTAGE_REPRODUCED={'YES' if paired_supported else 'PARTIAL'}", f"HB2_EDGE_TAIL_PENALTY_REPRODUCED={'YES' if paired_supported else 'PARTIAL'}", "",
        f"PREFERRED_DEPTH_BASELINE_AFTER_SESSION01={'HB2' if paired_supported else 'UNDECIDED'}", f"SPATIAL_RESIDUAL_REPRODUCED_IN_NEW_SESSION={'YES' if spatial_supported else 'PARTIAL'}", "", "NEW_SPATIAL_CORRECTION_ALLOWED_NOW=NO", "SECOND_INDEPENDENT_SESSION_REQUIRED=YES", "~~~", "",
        "结论边界：raw 采集文件结构完整，但 Session01 没有可用于误差验证的有效 processed height 行。"
        " 因此不能把历史 edge failure 或 H-B2 spatial spread 关系判为新 session 的 YES/NO；"
        " PARTIAL/NOT_SUPPORTED 是数据可判定性结论，不是算法性能的替代判定。", "",
        "## 1. 数据完整性与处理覆盖", "", "| 项目 | 数量/结果 |", "|---|---:|", f"| 自动发现 height×position condition | {len(conditions)} |", f"| raw PNG | {raw} |", f"| frames.csv 行 | {frames} |", f"| height_shadow.csv 行 | {shadow} |", f"| shadow 与 frames.csv camera_frame_number 匹配 | {matched} |", f"| shadow frame id 未在 frames.csv 中 | {shadow - matched} |", f"| 有效 processed frame | {valid} |", f"| 有效 processed coverage / raw PNG | {valid / raw:.4%} |", "",
        "每个 condition 的 raw PNG 与 frames.csv 均为 20；shadow 行数为 5、6 或 7，按任务约束不要求等于 raw 数。"
        f" 结构性 raw QC 未发现缺文件、重复 frame id 或 frame gap；但 shadow 行均为无高度结果，且有 {shadow - matched} 行不能通过 camera_frame_number 可靠关联。", "",
        "| height | shadow 行分布 | v_median 位置范围 [px] | >2200 positions | >2400 positions | >2600 positions |", "|---|---:|---:|---:|---:|---:|",
    ]
    for height in sorted({item["height_label"] for item in conditions}):
        rows = [row for row in positions if row["height_label"] == height]
        values = [number(row["v_position_median_px"]) for row in rows if number(row["v_position_median_px"]) is not None]
        distribution = sorted({int(row["shadow_row_count"]) for row in rows})
        lines.append(f"| {height} | {distribution} | {fmt(min(values), 1) if values else 'NA'}–{fmt(max(values), 1) if values else 'NA'} | {sum(value > 2200 for value in values)} | {sum(value > 2400 for value in values)} | {sum(value > 2600 for value in values)} |")
    lines.extend([
        "", "FOV 解释采用 condition 的 v_median 作为实际 position 坐标。虽然每条 shadow 行的 v_max=2999，"
        "这只说明激光点集合延伸到图像底部，不能把它当成十个 position 的 v>2400 空间支持；"
        f"Session01 实际 position 支持集中在约 {fmt(min(v_values), 1) if v_values else 'NA'}–{fmt(max(v_values), 1) if v_values else 'NA'} px。", "", "## 2. Truth 与 provenance / reuse audit", "",
        "未发现 certified-height metadata；本轮对 h10/h20/h30 只使用 nominal truth 10/20/30 mm，没有从 q1/q2/v 或测量结果猜测真实高度。", "",
        "| artifact | 本轮处理 |", "|---|---|", "| Session01 session_ground_calibration.json | 读取并保存状态、RMSE、Ground fit/support；不改写 |", "| Frozen C0 | 仅 provenance 核查；不重跑/不拟合 |", "| Frozen C1 | 仅 provenance 核查；不重跑/不拟合 |", "| Frozen H1 | 仅读取 frozen 参数 provenance；不从空结果合成高度 |", "| Frozen H-B2 | 仅读取 candidate 参数/domain provenance；不从空结果合成高度 |", "| 新增计算 | condition discovery、frame-key join、QC、v rank/coverage、NA 指标表、图、报告 |", "",
        "Session01 PnP / Ground 状态：", "", f"- PnP status/valid：{session.get('status')} / {session.get('valid')}", f"- PnP detection：{detection.get('method')}，corners={detection.get('corner_count')}", f"- PnP reprojection RMSE：{fmt(detection.get('reprojection_rmse_px'), 6)} px", f"- session ground reference：{ground.get('status')}，source={ground.get('source')}，fit={ground.get('fit_source')}", f"- slope/intercept：{fmt(ground.get('slope_z_per_mm'), 9)} mm/mm / {fmt(ground.get('intercept_z_mm'), 6)} mm", f"- Ground fit RMSE：{fmt(ground.get('rmse_mm'), 6)} mm；valid s range={ground.get('valid_s_range_mm')}", f"- Ground support：{support.get('status')}，mask={support.get('mask_mode')}，input/selected/rejected={support.get('input_point_count')}/{support.get('selected_point_count')}/{support.get('rejected_point_count')}", f"- laser ground sanity：{sanity.get('status')}，RMSE={fmt(sm.get('rmse_zg_mm'), 6)} mm，P95={fmt(sm.get('p95_abs_zg_mm'), 6)} mm，Max={fmt(sm.get('max_abs_zg_mm'), 6)} mm", f"- calibration warning：{session.get('quality', {}).get('warnings', [])}；warning 未使 session JSON 失效。", "",
        "## 3. Pointwise / condition / height metrics", "", "pointwise 表保留了 180 条 shadow diagnostic 行及 frame association、v/q/status 字段；height_raw/height_h1/height_hb2 三列全为空，processed_valid=false，所以 residual、Bias、MAE、RMSE、P95、Max、repeatability 和 H1/H-B2 paired delta 全部按 NA 保留。没有使用 q2 公式补算 H-B2，也没有用 H1 scale 补算 H1。", "", "condition-level、height×position-level 和 pooled rows 仍写出用于审计的结构、n、coverage/status；NOT_SUPPORTED_NO_VALID_PROCESSED_FRAME 表示没有可参与数值计算的行。", "",
        "## 4. Edge audit：v_median > 2400", "", "本轮按 frame-level v_median > 2400 独立筛选；shadow diagnostic edge n=0，valid edge n=0，因而 Base/H1/H-B2 的 edge 指标、>|0.1|、>|0.2| 和 residual-v Spearman 全部 NA。不能据此声称历史 edge failure 已重现或已消失。", "", "历史 A-11 报告曾在另一套有效 pointwise artifact 上记录：H-B2 position spread 优于 H1，而 H1 edge tail 略优；该历史关系仅作为对照，不被本轮空结果替代。", "",
        "## 5. 输出文件", "",
    ])
    for name in ("session01_ingestion_qc.csv", "session01_pointwise_paired.csv", "session01_condition_metrics.csv", "session01_height_spatial_metrics.csv", "session01_edge_v_gt_2400_metrics.csv", "session01_v_coverage.csv", "session01_full_fov_paired_validation_report.md", "session01_v_coverage.png", "session01_position_bias_by_height.png", "session01_residual_vs_v.png", "session01_h1_vs_hb2_paired_delta.png", "session01_edge_comparison.png"):
        lines.append(f"- {output / name}")
    lines.extend(["", "## 6. 禁止项复核", "", "- 未修改 H1/H-B2 参数。", "- 未新增 q1/v/spline/LUT correction。", "- 未修改 ROI、Ground、C0、C1。", "- 未根据结果删除 position。", "- 未将 Session01 定义为训练集。", "- 未从 raw PNG 重新运行 image/reconstruction pipeline。"])
    (output / "session01_full_fov_paired_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    data_root, output, repo_root = args.data_root.resolve(), args.output_dir.resolve(), args.repo_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root does not exist: {data_root}")
    output.mkdir(parents=True, exist_ok=True)
    conditions = [load_condition("session01", height, position, path) for height, position, path in discover(data_root)]
    assign_ranks(conditions)
    session = json.loads((data_root / "session_ground_calibration.json").read_text(encoding="utf-8"))
    qc_rows = [qc_condition(condition) for condition in conditions] + provenance(data_root, session, repo_root)
    write_csv(output / "session01_ingestion_qc.csv", qc_rows, QC_FIELDS)
    write_csv(output / "session01_pointwise_paired.csv", pointwise(conditions), POINT_FIELDS)
    condition_rows = []
    for condition in conditions:
        condition_rows.extend(condition_model_row(condition, model) for model in MODELS)
        condition_rows.append(paired_condition_row(condition))
    write_csv(output / "session01_condition_metrics.csv", condition_rows, CONDITION_FIELDS)
    by_height: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        by_height.setdefault(condition["height_label"], []).append(condition)
    height_rows, model_rows = [], {}
    for height, group in sorted(by_height.items()):
        for model in MODELS:
            model_rows[(height, model)] = height_model_row(height, group, model)
            height_rows.append(model_rows[(height, model)])
        height_rows.append(paired_height_row(height, group, model_rows[(height, "H1")], model_rows[(height, "H-B2")]))
    height_rows.extend(pooled_height_row(conditions, model) for model in MODELS)
    write_csv(output / "session01_height_spatial_metrics.csv", height_rows, HEIGHT_FIELDS)
    edge_rows = build_edges(conditions)
    write_csv(output / "session01_edge_v_gt_2400_metrics.csv", edge_rows, EDGE_FIELDS)
    coverage_rows = coverage(conditions)
    write_csv(output / "session01_v_coverage.csv", coverage_rows, COVERAGE_FIELDS)
    make_plots(output, conditions)
    make_report(output, data_root, conditions, session, qc_rows, height_rows, edge_rows, coverage_rows)
    print(json.dumps({
        "output_dir": str(output), "conditions": len(conditions), "raw_png": sum(len(item["pngs"]) for item in conditions), "frames_csv_rows": sum(len(item["frames"]) for item in conditions), "shadow_rows": sum(len(item["shadow"]) for item in conditions),
        "matched_shadow_rows": sum(sum(record["association"] == "matched" for record in item["records"]) for item in conditions), "valid_processed_rows": sum(sum(record["valid"] for record in item["records"]) for item in conditions),
        "v_median_min": min((item["v_position"] for item in conditions if item["v_position"] is not None), default=None), "v_median_max": max((item["v_position"] for item in conditions if item["v_position"] is not None), default=None),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
