#!/usr/bin/env python3
"""Select a Session-level Haikang measurement search polygon.

The polygon is a manual, Session-wide search-domain prior only.  It is stored
in full-sensor ``(u, v)`` coordinates and is never used as a per-condition
height ROI.  The interactive mode is intended for a workstation with a
Matplotlib GUI; ``--vertices`` provides a reproducible non-interactive manual
selection for offline audits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import PolygonSelector


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import generate_haikang_c0_h_raw_0829 as h0  # noqa: E402


DATA_ROOT_DEFAULT = h0.DATA_ROOT_DEFAULT
OUTPUT_DIR_DEFAULT = DATA_ROOT_DEFAULT / "c0_height_audit" / "roi_v2_manual_search"
DEFAULT_CONDITION = "h30_p05"
DEFAULT_FRAME_INDEX = 9


class SelectionError(RuntimeError):
    """Raised when a manual selection cannot satisfy the artifact contract."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=DEFAULT_FRAME_INDEX,
        help="zero-based source-frame index in the condition frames.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="override the output session_measurement_search_roi.json path",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="override the saved manual selection preview PNG path",
    )
    parser.add_argument(
        "--vertices",
        default=None,
        help="manual vertices as u,v;u,v;... in full-sensor coordinates",
    )
    return parser.parse_args(argv)


def parse_vertices(text: str) -> list[list[float]]:
    vertices: list[list[float]] = []
    for token in text.split(";"):
        values = [part.strip() for part in token.split(",")]
        if len(values) != 2:
            raise SelectionError(f"invalid vertex token: {token!r}")
        try:
            point = [float(values[0]), float(values[1])]
        except ValueError as error:
            raise SelectionError(f"non-numeric vertex: {token!r}") from error
        if not all(math.isfinite(value) for value in point):
            raise SelectionError(f"non-finite vertex: {token!r}")
        vertices.append(point)
    validate_polygon(vertices)
    return vertices


def validate_polygon(vertices: list[list[float]]) -> None:
    if len(vertices) < 3:
        raise SelectionError("a manual search polygon needs at least three vertices")
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (len(vertices), 2) or not np.isfinite(points).all():
        raise SelectionError("manual search polygon must contain finite (u, v) pairs")
    closed = np.vstack([points, points[0]])
    area = 0.5 * float(
        np.sum(closed[:-1, 0] * closed[1:, 1] - closed[1:, 0] * closed[:-1, 1])
    )
    if abs(area) < 1.0:
        raise SelectionError("manual search polygon has near-zero area")
    # Concave polygons are allowed; the audit uses the same general
    # point-in-polygon containment rule.  Self-intersection is left visible in
    # the preview so a human can correct it before the audit is run.


def load_source_frame(
    input_dir: Path, condition_id: str, frame_index: int
) -> tuple[h0.Condition, dict[str, str], Any]:
    conditions, _ = h0.discover_conditions(input_dir)
    condition = next(
        (item for item in conditions if item.condition_id == condition_id), None
    )
    if condition is None:
        raise SelectionError(f"unknown target condition: {condition_id}")
    rows = h0.read_csv_rows(condition.path / "frames.csv", h0.FRAME_FIELDS)
    if frame_index < 0 or frame_index >= len(rows):
        raise SelectionError(
            f"frame index {frame_index} outside {condition_id} frames.csv ({len(rows)} rows)"
        )
    row = rows[frame_index]
    frame = h0.frame_from_row(condition, row)
    return condition, row, frame


def image_extent(row: dict[str, str], image: np.ndarray) -> tuple[float, float, float, float]:
    x0 = float(h0.parse_int(row["offset_x"], "offset_x"))
    y0 = float(h0.parse_int(row["offset_y"], "offset_y"))
    width = float(h0.parse_int(row["width"], "width"))
    height = float(h0.parse_int(row["height"], "height"))
    if image.shape != (int(height), int(width)):
        raise SelectionError(f"source image shape mismatch: {image.shape}")
    return x0, x0 + width, y0 + height, y0


def render_preview(
    image: np.ndarray,
    row: dict[str, str],
    polygon: list[list[float]],
    output_path: Path,
    title: str,
) -> None:
    x0, x1, y1, y0 = image_extent(row, image)
    low = float(np.nanmin(image)) if image.size else 0.0
    high = float(np.nanmax(image)) if image.size else 1.0
    if high <= low:
        high = low + 1.0
    points = np.asarray(polygon, dtype=np.float64)
    closed = np.vstack([points, points[0]])
    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    ax.imshow(
        image,
        cmap="gray",
        vmin=low,
        vmax=high,
        extent=(x0, x1, y1, y0),
        aspect="auto",
        interpolation="nearest",
    )
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color="#00e5a0",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="manual Session search ROI",
    )
    ax.fill(closed[:, 0], closed[:, 1], color="#00e5a0", alpha=0.10)
    ax.set_xlabel("full-sensor u (px)")
    ax.set_ylabel("full-sensor v (px)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def interactive_vertices(
    image: np.ndarray, row: dict[str, str], title: str
) -> list[list[float]]:
    x0, x1, y1, y0 = image_extent(row, image)
    low = float(np.nanmin(image)) if image.size else 0.0
    high = float(np.nanmax(image)) if image.size else 1.0
    if high <= low:
        high = low + 1.0
    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    ax.imshow(
        image,
        cmap="gray",
        vmin=low,
        vmax=high,
        extent=(x0, x1, y1, y0),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel("full-sensor u (px)")
    ax.set_ylabel("full-sensor v (px)")
    ax.set_title(
        title
        + " | click polygon vertices, press Enter to accept, Esc to cancel"
    )
    selector = PolygonSelector(ax, lambda _vertices: None, useblit=True)
    plt.show(block=True)
    vertices = getattr(selector, "verts", None)
    selector.disconnect_events()
    plt.close(fig)
    if not vertices:
        raise SelectionError("interactive selection was cancelled or empty")
    parsed = [[float(u), float(v)] for u, v in vertices]
    validate_polygon(parsed)
    return parsed


def build_document(
    condition: h0.Condition,
    row: dict[str, str],
    frame: Any,
    polygon: list[list[float]],
    output_json: Path,
    selection_method: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "coordinate_system": "full_sensor_uv",
        "polygon_full_uv": polygon,
        "source_frame": {
            "condition_id": condition.condition_id,
            "path": str((condition.path / row["filename"]).resolve()),
            "filename": row["filename"],
            "frame_index_zero_based": int(row.get("frame_index", 0) or 0),
            "camera_frame_number": h0.parse_int(
                row["camera_frame_number"], "camera_frame_number"
            ),
            "offset_x": h0.parse_int(row["offset_x"], "offset_x"),
            "offset_y": h0.parse_int(row["offset_y"], "offset_y"),
            "width": h0.parse_int(row["width"], "width"),
            "height": h0.parse_int(row["height"], "height"),
            "exposure_us": h0.parse_float(row["exposure_us"], "exposure_us"),
            "gain_db": h0.parse_float(row["gain_db"], "gain_db"),
        },
        "created_mode": "manual",
        "selection_method": selection_method,
        "purpose": "target_search_only",
        "selection_contract": {
            "session_wide": True,
            "per_condition_final_height_roi": False,
            "directory_height_truth_used": False,
            "board_polygon_used_as_search_domain": False,
            "c0_or_ground_used_for_selection": False,
            "height_compensation_used": False,
        },
        "artifact_path": str(output_json.resolve()),
        "source_frame_note": (
            "Representative raw PNG shown in full-sensor coordinates; source frame "
            "is visual context only and does not supply height truth."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = args.output_json or args.output_dir / "session_measurement_search_roi.json"
    preview = args.preview or args.output_dir / "session_measurement_search_roi_overlay.png"
    try:
        condition, row, frame = load_source_frame(
            args.input_dir, args.condition, args.frame_index
        )
        row = dict(row)
        row["frame_index"] = str(args.frame_index)
        title = (
            f"{condition.condition_id} raw frame {args.frame_index} | "
            "Session-level target search only"
        )
        if args.vertices:
            polygon = parse_vertices(args.vertices)
            selection_method = "manual_vertices_noninteractive"
        else:
            polygon = interactive_vertices(frame.image, row, title)
            selection_method = "matplotlib_polygon_selector"
        document = build_document(
            condition, row, frame, polygon, output_json, selection_method
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        render_preview(frame.image, row, polygon, preview, title)
        print(json.dumps({"output_json": str(output_json.resolve()), "preview": str(preview.resolve()), "polygon_full_uv": polygon}, ensure_ascii=False))
        return 0
    except (SelectionError, h0.AuditError, OSError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
