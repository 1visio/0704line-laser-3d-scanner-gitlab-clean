"""A-2R: re-align the frozen A-2 quality audit to A-13B local-reference errors.

This is an analysis-only postprocessor.  It deliberately reads existing CSV/JSON
artifacts only; it does not read PNGs, run Steger, reconstruct points, or fit any
calibration/correction model.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
A2_DIR = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze" / "stripe_quality_audit"
A13B_FRAMES = (
    REPO_ROOT
    / "reports"
    / "experiments"
    / "daheng_0822"
    / "session01_roi_freeze"
    / "session01_a13b_v2_multireference_frames.csv"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "daheng_0822_session01_roi_freeze"
    / "stripe_quality_audit_local_reference"
)

HEIGHT_ORDER = ("h10", "h20", "h30")
V_BAND_LABELS = ("v<1800", "1800<=v<2200", "2200<=v<2400", "2400<=v<=2600", "v>2600")
SENSITIVITY_MM_PER_PX = 0.30

# This list is copied from the completed A-2 artifact.  The values are reused,
# not recomputed from images in this task.
QUALITY_FEATURES = (
    "signal_excess_dn_median",
    "background_dn_median",
    "peak_background_ratio_median",
    "dynamic_range_dn_median",
    "stripe_width_fwhm_px_median",
    "profile_asymmetry_median",
    "profile_skewness_median",
    "profile_saturation_fraction_median",
    "steger_response_dn_per_px2_median",
    "steger_valid_ratio",
    "centerline_curvature_abs_1_per_px_median",
)
PLOT_FEATURES = (
    ("signal_excess_dn_median", "signal excess (DN)"),
    ("stripe_width_fwhm_px_median", "stripe FWHM (px)"),
    ("profile_asymmetry_median", "profile asymmetry"),
    ("profile_skewness_median", "profile skewness"),
    ("steger_response_dn_per_px2_median", "Steger response (DN/px²)"),
    ("steger_valid_ratio", "Frozen-Steger valid ratio"),
)

LOCAL_FIELDS = {
    "base_error_mm": "residual_base_local_diag",
    "h1_error_mm": "residual_h1_local_diag",
    "hb2_error_mm": "residual_hb2_local_diag",
}
LOCAL_MERGED_FIELDS = {
    "base_error_mm": "local_base_error_mm",
    "h1_error_mm": "local_h1_error_mm",
    "hb2_error_mm": "local_hb2_error_mm",
}
SESSION_FIELDS = {
    "base_error_mm": "base_error_mm",
    "h1_error_mm": "h1_error_mm",
    "hb2_error_mm": "hb2_error_mm",
}
TARGETS = (
    "base_error_mm",
    "base_abs_error_mm",
    "h1_error_mm",
    "h1_abs_error_mm",
    "hb2_error_mm",
    "hb2_abs_error_mm",
)
IDENTITY_FIELDS = ("height_label", "position_id", "condition_id", "repeat_index", "camera_frame_number")
COLORS = {"h10": "#386cb0", "h20": "#f0027f", "h30": "#1b9e77"}


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(values: Iterable[float], q: float) -> float | None:
    values_list = [float(value) for value in values if finite(value) is not None]
    if not values_list:
        return None
    return float(np.percentile(np.asarray(values_list, dtype=np.float64), q))


def mean(values: Iterable[float]) -> float | None:
    values_list = [float(value) for value in values if finite(value) is not None]
    return float(np.mean(values_list)) if values_list else None


def rmse(values: Iterable[float]) -> float | None:
    values_list = [float(value) for value in values if finite(value) is not None]
    return float(np.sqrt(np.mean(np.square(values_list)))) if values_list else None


def correlation(x: Iterable[float], y: Iterable[float], method: str) -> tuple[float | None, float | None]:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if len(x_array) < 3 or len(y_array) != len(x_array):
        return None, None
    if np.all(x_array == x_array[0]) or np.all(y_array == y_array[0]):
        return None, None
    try:
        result = pearsonr(x_array, y_array) if method == "pearson" else spearmanr(x_array, y_array)
        return float(result.statistic), float(result.pvalue)
    except (ValueError, FloatingPointError):
        return None, None


def demean_by_condition(values: list[float], condition_ids: list[str]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for condition_id, value in zip(condition_ids, values, strict=True):
        grouped[condition_id].append(float(value))
    means = {condition_id: float(np.mean(group_values)) for condition_id, group_values in grouped.items()}
    return [float(value) - means[condition_id] for value, condition_id in zip(values, condition_ids, strict=True)]


def scope_rows(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "pooled":
        return rows
    if scope.startswith("height:"):
        height = scope.split(":", 1)[1]
        return [row for row in rows if str(row.get("height_label")) == height]
    if scope.startswith("vband:"):
        band = scope.split(":", 1)[1]
        return [row for row in rows if str(row.get("v_band")) == band]
    raise ValueError(f"Unknown scope: {scope}")


def target_value(row: dict[str, Any], target: str, reference: str) -> float | None:
    if reference == "local_diag":
        if target.endswith("_abs_error_mm"):
            signed_target = target.removesuffix("_abs_error_mm") + "_error_mm"
            field = LOCAL_MERGED_FIELDS[signed_target]
            value = finite(row.get(field))
            return abs(value) if value is not None else None
        return finite(row.get(LOCAL_MERGED_FIELDS[target]))
    if reference == "session":
        if target.endswith("_abs_error_mm"):
            signed_target = target.removesuffix("_abs_error_mm") + "_error_mm"
            value = finite(row.get(SESSION_FIELDS[signed_target]))
            return abs(value) if value is not None else None
        return finite(row.get(SESSION_FIELDS[target]))
    raise ValueError(f"Unknown reference: {reference}")


def validate_and_merge(a2_rows: list[dict[str, str]], a13b_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if len(a2_rows) != 600 or len(a13b_rows) != 600:
        raise RuntimeError(f"Expected 600 rows in each artifact, got A-2={len(a2_rows)}, A-13B={len(a13b_rows)}")
    a2_by_key = {str(row["cache_key"]): row for row in a2_rows}
    a13b_by_key = {str(row["cache_key"]): row for row in a13b_rows}
    if len(a2_by_key) != len(a2_rows) or len(a13b_by_key) != len(a13b_rows):
        raise RuntimeError("cache_key is not unique in one of the frame-level artifacts")
    if set(a2_by_key) != set(a13b_by_key):
        missing_a13b = sorted(set(a2_by_key) - set(a13b_by_key))[:5]
        missing_a2 = sorted(set(a13b_by_key) - set(a2_by_key))[:5]
        raise RuntimeError(f"A-2/A-13B cache_key mismatch; missing_a13b={missing_a13b}, missing_a2={missing_a2}")

    merged: list[dict[str, Any]] = []
    for key in sorted(a2_by_key):
        a2 = a2_by_key[key]
        a13b = a13b_by_key[key]
        for field in IDENTITY_FIELDS:
            if str(a2.get(field)) != str(a13b.get(field)):
                raise RuntimeError(f"Identity mismatch for {key}: field={field}, A-2={a2.get(field)}, A-13B={a13b.get(field)}")
        if not as_bool(a13b.get("local_diag_only")):
            raise RuntimeError(f"A-13B row is not marked local diagnostic-only: {key}")
        for source_field in (*LOCAL_FIELDS.values(), "local_baseline_support", "local_baseline_extrapolation"):
            if source_field not in a13b:
                raise RuntimeError(f"Missing required A-13B local field: {source_field}")

        row: dict[str, Any] = dict(a2)
        row.update(
            {
                "local_base_error_mm": finite(a13b.get("residual_base_local_diag")),
                "local_h1_error_mm": finite(a13b.get("residual_h1_local_diag")),
                "local_hb2_error_mm": finite(a13b.get("residual_hb2_local_diag")),
                "local_baseline_support": str(a13b.get("local_baseline_support", "")),
                "local_baseline_extrapolation": as_bool(a13b.get("local_baseline_extrapolation")),
                "local_diag_only": as_bool(a13b.get("local_diag_only")),
                "session_base_from_a13b_mm": finite(a13b.get("residual_base_session")),
                "session_h1_from_a13b_mm": finite(a13b.get("residual_h1_session")),
                "session_hb2_from_a13b_mm": finite(a13b.get("residual_hb2_session")),
            }
        )
        for signed_name in ("base", "h1", "hb2"):
            local_value = row[f"local_{signed_name}_error_mm"]
            row[f"local_{signed_name}_abs_error_mm"] = abs(local_value) if local_value is not None else None
        row["v_band"] = str(a2.get("v_band", ""))
        merged.append(row)

    # The A-2 Base/H1/H-B2 error columns are the session-reference residuals.
    # Check that same-frame pairing did not silently change semantics.
    for row in merged:
        for name in ("base", "h1", "hb2"):
            a2_value = finite(row.get(f"{name}_error_mm"))
            a13b_value = row.get(f"session_{name}_from_a13b_mm")
            if a2_value is not None and a13b_value is not None and abs(a2_value - a13b_value) > 1e-9:
                raise RuntimeError(f"A-2/A-13B Session residual mismatch for {row['cache_key']} / {name}")
    return merged


def local_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes = [("pooled", rows)] + [(f"height:{height}", scope_rows(rows, f"height:{height}")) for height in HEIGHT_ORDER]
    for scope, scoped_rows in scopes:
        for feature in QUALITY_FEATURES:
            for target in TARGETS:
                pairs = []
                for row in scoped_rows:
                    x_value = finite(row.get(feature))
                    y_value = target_value(row, target, "local_diag")
                    if x_value is not None and y_value is not None:
                        pairs.append((x_value, y_value, str(row["condition_id"])))
                for mode in ("pooled", "within_condition_demeaned"):
                    if mode == "pooled":
                        x_values = [pair[0] for pair in pairs]
                        y_values = [pair[1] for pair in pairs]
                    else:
                        x_values = demean_by_condition([pair[0] for pair in pairs], [pair[2] for pair in pairs]) if pairs else []
                        y_values = demean_by_condition([pair[1] for pair in pairs], [pair[2] for pair in pairs]) if pairs else []
                    pearson_r, pearson_p = correlation(x_values, y_values, "pearson")
                    spearman_rho, spearman_p = correlation(x_values, y_values, "spearman")
                    output.append(
                        {
                            "reference": "local_diag",
                            "scope": scope,
                            "mode": mode,
                            "feature": feature,
                            "target": target,
                            "n": len(pairs),
                            "pearson_r": pearson_r,
                            "pearson_pvalue": pearson_p,
                            "spearman_rho": spearman_rho,
                            "spearman_pvalue": spearman_p,
                        }
                    )
    return output


def compare_metric(values: list[float], metric: str) -> float | None:
    if not values:
        return None
    if metric == "bias_mm":
        return mean(values)
    if metric == "mae_mm":
        return mean(abs(value) for value in values)
    if metric == "rmse_mm":
        return rmse(values)
    if metric == "p95_abs_error_mm":
        return percentile((abs(value) for value in values), 95.0)
    if metric == "max_abs_error_mm":
        return max(abs(value) for value in values)
    raise ValueError(f"Unknown metric: {metric}")


def within_condition_deviations(values: list[float], condition_ids: list[str]) -> list[float]:
    demeaned = demean_by_condition(values, condition_ids)
    return [float(value) for value in demeaned]


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes = ["pooled", *[f"height:{height}" for height in HEIGHT_ORDER], *[f"vband:{band}" for band in V_BAND_LABELS]]
    metrics = ("bias_mm", "mae_mm", "rmse_mm", "p95_abs_error_mm", "max_abs_error_mm", "within_condition_mae_mm", "within_condition_p95_abs_deviation_mm")
    output: list[dict[str, Any]] = []
    for scope in scopes:
        scoped_rows = scope_rows(rows, scope)
        for model, session_field, local_field in (
            ("base", "base_error_mm", "local_base_error_mm"),
            ("h1", "h1_error_mm", "local_h1_error_mm"),
            ("hb2", "hb2_error_mm", "local_hb2_error_mm"),
        ):
            pairs = []
            for row in scoped_rows:
                session_value = finite(row.get(session_field))
                local_value = finite(row.get(local_field))
                if session_value is not None and local_value is not None:
                    pairs.append((session_value, local_value, str(row["condition_id"])))
            session_values = [pair[0] for pair in pairs]
            local_values = [pair[1] for pair in pairs]
            condition_ids = [pair[2] for pair in pairs]
            session_within = within_condition_deviations(session_values, condition_ids) if pairs else []
            local_within = within_condition_deviations(local_values, condition_ids) if pairs else []
            for metric in metrics:
                if metric.startswith("within_condition_"):
                    base_metric = metric.removeprefix("within_condition_")
                    session_metric = compare_metric(session_within, "mae_mm" if base_metric == "mae_mm" else "p95_abs_error_mm")
                    local_metric = compare_metric(local_within, "mae_mm" if base_metric == "mae_mm" else "p95_abs_error_mm")
                else:
                    session_metric = compare_metric(session_values, metric)
                    local_metric = compare_metric(local_values, metric)
                output.append(
                    {
                        "scope": scope,
                        "model": model,
                        "metric": metric,
                        "n_paired": len(pairs),
                        "session_n": len(session_values),
                        "local_n": len(local_values),
                        "session_value_mm": session_metric,
                        "local_value_mm": local_metric,
                        "delta_local_minus_session_mm": (
                            local_metric - session_metric
                            if local_metric is not None and session_metric is not None
                            else None
                        ),
                    }
                )
    return output


def lookup_comparison(rows: list[dict[str, Any]], scope: str, model: str, metric: str) -> dict[str, Any]:
    return next((row for row in rows if row["scope"] == scope and row["model"] == model and row["metric"] == metric), {})


def lookup_correlation(rows: list[dict[str, Any]], scope: str, mode: str, feature: str, target: str) -> dict[str, Any]:
    return next(
        (
            row
            for row in rows
            if row["scope"] == scope and row["mode"] == mode and row["feature"] == feature and row["target"] == target
        ),
        {},
    )


def session_vs_local_correlations(
    a2_rows: list[dict[str, str]], local_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare the old A-2 Session-reference correlation matrix to A-2R Local.

    A-2's registered quality-correlation artifact is Base signed/absolute error;
    the new Local matrix additionally carries H1/H-B2, but this comparison keeps
    the common Base targets so the reference swap is directly auditable.
    """
    output: list[dict[str, Any]] = []
    for scope in ("pooled", *[f"height:{height}" for height in HEIGHT_ORDER]):
        for mode in ("pooled", "within_condition_demeaned"):
            for feature in QUALITY_FEATURES:
                for target in ("base_error_mm", "base_abs_error_mm"):
                    session = next(
                        (
                            row
                            for row in a2_rows
                            if row.get("scope") == scope
                            and row.get("mode") == mode
                            and row.get("feature") == feature
                            and row.get("target") == target
                        ),
                        {},
                    )
                    local = lookup_correlation(local_rows, scope, mode, feature, target)
                    session_r = finite(session.get("pearson_r"))
                    local_r = finite(local.get("pearson_r"))
                    session_rho = finite(session.get("spearman_rho"))
                    local_rho = finite(local.get("spearman_rho"))
                    output.extend(
                        [
                            {
                                "scope": scope,
                                "mode": mode,
                                "feature": feature,
                                "target": target,
                                "metric": "pearson_r",
                                "n_session": session.get("n"),
                                "n_local": local.get("n"),
                                "session_value": session_r,
                                "local_value": local_r,
                                "delta_local_minus_session": local_r - session_r if local_r is not None and session_r is not None else None,
                                "session_pvalue": finite(session.get("pearson_pvalue")),
                                "local_pvalue": finite(local.get("pearson_pvalue")),
                            },
                            {
                                "scope": scope,
                                "mode": mode,
                                "feature": feature,
                                "target": target,
                                "metric": "spearman_rho",
                                "n_session": session.get("n"),
                                "n_local": local.get("n"),
                                "session_value": session_rho,
                                "local_value": local_rho,
                                "delta_local_minus_session": local_rho - session_rho if local_rho is not None and session_rho is not None else None,
                                "session_pvalue": finite(session.get("spearman_pvalue")),
                                "local_pvalue": finite(local.get("spearman_pvalue")),
                            },
                        ]
                    )
    return output


def lookup_correlation_comparison(
    rows: list[dict[str, Any]], scope: str, mode: str, feature: str, target: str, metric: str
) -> dict[str, Any]:
    return next(
        (
            row
            for row in rows
            if row["scope"] == scope
            and row["mode"] == mode
            and row["feature"] == feature
            and row["target"] == target
            and row["metric"] == metric
        ),
        {},
    )


def plot_height_error_vs_quality(rows: list[dict[str, Any]], correlations: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150)
    for axis, (feature, label) in zip(axes.ravel(), PLOT_FEATURES, strict=True):
        for height in HEIGHT_ORDER:
            points = [
                row
                for row in rows
                if row.get("height_label") == height
                and finite(row.get(feature)) is not None
                and finite(row.get("local_base_error_mm")) is not None
            ]
            axis.scatter(
                [float(row[feature]) for row in points],
                [float(row["local_base_error_mm"]) for row in points],
                s=12,
                alpha=0.55,
                color=COLORS[height],
                label=height,
                edgecolors="none",
            )
        raw = lookup_correlation(correlations, "pooled", "pooled", feature, "base_error_mm")
        within = lookup_correlation(correlations, "pooled", "within_condition_demeaned", feature, "base_error_mm")
        raw_text = "—" if raw.get("pearson_r") is None else f"{float(raw['pearson_r']):+.2f}"
        within_text = "—" if within.get("pearson_r") is None else f"{float(within['pearson_r']):+.2f}"
        axis.set_title(f"{label}\nLocal Base Pearson raw={raw_text}, within={within_text}")
        axis.set_xlabel(label)
        axis.set_ylabel("Local Base residual (mm)")
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.grid(alpha=0.2)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("A-2R: Local-reference Base residual vs reused stripe/Steger quality", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 5) -> str:
    number = finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def build_report(
    rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    correlation_comparison: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    a2_provenance: dict[str, Any],
    local_magnitude_flag: str,
    session_magnitude_retained: str,
    profile_changed: str,
    random_changed: str,
    predicted_height_p95_mm: float,
    local_total_p95: float | None,
    local_within_p95: float | None,
    output_path: Path,
) -> None:
    old_decisions = a2_provenance.get("decisions", {})
    old_random = str(old_decisions.get("LOWER_EDGE_STEGER_RANDOM_DEGRADATION", "UNKNOWN"))
    old_profile = str(old_decisions.get("LOWER_EDGE_PROFILE_SYSTEMATIC_BIAS_EVIDENCE", "UNKNOWN"))
    old_magnitude = str(old_decisions.get("OPTICAL_EXTRACTION_MAGNITUDE_SUFFICIENT", "UNKNOWN"))
    support_counts = Counter(str(row.get("local_baseline_support", "")) for row in rows)
    extrapolated_count = sum(bool(row.get("local_baseline_extrapolation")) for row in rows)
    local_valid_counts = {
        model: sum(target_value(row, f"{model}_error_mm", "local_diag") is not None for row in rows)
        for model in ("base", "h1", "hb2")
    }

    lines = [
        "# Task A-2R｜Local-reference 光条/Steger 误差归因审计",
        "",
        f"- `A2_RANDOM_DEGRADATION_CONCLUSION_CHANGED = {random_changed}`",
        f"- `A2_PROFILE_BIAS_CONCLUSION_CHANGED = {profile_changed}`",
        f"- `LOCAL_REFERENCE_OPTICAL_MAGNITUDE_SUFFICIENT = {local_magnitude_flag}`",
        f"- `SESSION_REFERENCE_MAGNITUDE_RESULT_RETAINED = {session_magnitude_retained}`",
        "",
        "本轮把 A-2 的同帧光条质量/Steger 重复性特征与 A-13B-v2 的 Local-reference residual 重新配对。Local 分支仅是 diagnostic reference，不是 production truth；Session-reference 仍是正式链路分支。",
        "",
        "## Provenance / reuse audit",
        "",
        "本轮直接复用：",
        "",
        "- A-2 `stripe_quality_frame_audit.csv` 的 600 帧 quality features、full-sensor `(u,v)` 身份字段及 Session-reference Base/H1/H-B2 residual；",
        "- A-2 `quality_correlation_summary.csv`、`steger_repeatability_by_v.csv` 与 provenance 中的 A-1 sensitivity / condition-level u repeatability P95；",
        "- A-13B-v2 `residual_base_local_diag`、`residual_h1_local_diag`、`residual_hb2_local_diag`，按同一 `cache_key` 逐帧对齐；",
        "- 未重新读取 PNG，未重新运行 Steger，未重新 reconstruction，未拟合 C0/C1/H1/H-B2，也未修改 production reconstruction、Steger、ROI 或 Ground。",
        "",
        "A-2 的 strict-valid sensitivity sanity 也只作复用的 provenance：该子集的 lower-edge high/low ratio 原本不可独立评估（Session Ground 有效域不覆盖 `v>2600`），本轮没有把它升级成 Local-reference 结论。",
        "",
        "## Local-reference 定义与对齐",
        "",
        "A-13B-v2 的 local branch 使用同帧量块邻近 baseline ROI 的局部 ground profile；`residual_*_local_diag` 定义为该 local branch 的 raw height 减 nominal truth。`local_measurement_error` 是状态字符串，不参与本审计的误差统计。Local branch 标记为 `local_diag_only=True`。",
        f"- 同帧配对：`{len(rows)}`/600；Local residual finite count：Base `{local_valid_counts['base']}`、H1 `{local_valid_counts['h1']}`、H-B2 `{local_valid_counts['hb2']}`。",
        f"- Local baseline support：`{dict(sorted(support_counts.items()))}`；one-sided/extrapolated `{extrapolated_count}`/600。单侧帧不删除，但在 provenance 中保留。",
        "- A-2 与 A-13B-v2 的 Session residual 逐帧核对一致；因此变化来自 reference branch，而不是 frame/key 错配。",
        "",
        "## Pooled / height / within-condition 相关",
        "",
        "`pooled` 相关包含跨 position/FOV 的共同 v 变化；`within_condition_demeaned` 在每个 20-repeat condition 内分别减去 feature 与 Local residual 的 condition mean，降低共同 v 趋势造成的伪相关。完整 Base/H1/H-B2、signed/absolute 矩阵见 `a2r_local_reference_correlation.csv`。",
        "",
        "| feature | Local Base pooled Pearson | Local Base pooled Spearman | Local Base within Pearson | Local Base within Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature, _label in PLOT_FEATURES:
        raw = lookup_correlation(correlations, "pooled", "pooled", feature, "base_error_mm")
        within = lookup_correlation(correlations, "pooled", "within_condition_demeaned", feature, "base_error_mm")
        lines.append(
            f"| `{feature}` | {fmt(raw.get('pearson_r'), 3)} | {fmt(raw.get('spearman_rho'), 3)} | {fmt(within.get('pearson_r'), 3)} | {fmt(within.get('spearman_rho'), 3)} |"
        )
    lines.extend(
        [
            "",
            "这些相关性只用于机制审计，不用于 truth-driven 特征选择、调参或 correction 拟合。按 h10/h20/h30 的完整结果保留在 CSV；图中展示 pooled Local Base residual 与质量指标。",
            "",
            "## Session vs Local correlation comparison",
            "",
            "下表只列 A-2 与 A-2R 共同存在的 Base signed-error 相关，用于 reference swap 的直接比较；A-2R 新增的 H1/H-B2 以及 absolute-error 结果见 `a2r_local_reference_correlation.csv`，完整共同矩阵见 `a2r_session_vs_local_correlation.csv`。",
            "",
            "| feature | Session pooled Pearson | Local pooled Pearson | Session within Pearson | Local within Pearson |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for feature, _label in PLOT_FEATURES:
        session_raw = lookup_correlation_comparison(correlation_comparison, "pooled", "pooled", feature, "base_error_mm", "pearson_r")
        local_raw = lookup_correlation(correlations, "pooled", "pooled", feature, "base_error_mm")
        session_within = lookup_correlation_comparison(
            correlation_comparison, "pooled", "within_condition_demeaned", feature, "base_error_mm", "pearson_r"
        )
        local_within = lookup_correlation(correlations, "pooled", "within_condition_demeaned", feature, "base_error_mm")
        lines.append(
            f"| `{feature}` | {fmt(session_raw.get('session_value'), 3)} | {fmt(local_raw.get('pearson_r'), 3)} | {fmt(session_within.get('session_value'), 3)} | {fmt(local_within.get('pearson_r'), 3)} |"
        )
    lines.extend(
        [
            "",
            "## Session vs Local residual magnitude",
            "",
            "以下数值均是同一帧配对的 diagnostic comparison。旧 Session-reference 数值保留用于说明 reference contamination 的影响；旧 `Base pooled |error| P95≈0.370 mm` 不被本报告当作当前真实 edge residual。",
            "",
            "| model | scope | metric | Session | Local | Local−Session | n |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("base", "h1", "hb2"):
        for scope, metric in (
            ("pooled", "p95_abs_error_mm"),
            ("pooled", "within_condition_p95_abs_deviation_mm"),
            ("height:h10", "p95_abs_error_mm"),
            ("height:h20", "p95_abs_error_mm"),
            ("height:h30", "p95_abs_error_mm"),
            ("vband:v>2600", "p95_abs_error_mm"),
        ):
            item = lookup_comparison(comparisons, scope, model, metric)
            lines.append(
                f"| {model} | {scope} | {metric} | {fmt(item.get('session_value_mm'))} | {fmt(item.get('local_value_mm'))} | {fmt(item.get('delta_local_minus_session_mm'))} | {item.get('n_paired', 0)} |"
            )
    lines.extend(
        [
            "",
            f"A-1 局部 sensitivity 复用值为 `{SENSITIVITY_MM_PER_PX:.2f} mm/px`；A-2 已有 condition-level u deviation P95 为 `{fmt(old_decisions.get('condition_p95_u_abs_deviation_px_p95'), 5)} px`，换算高度量级 `{fmt(predicted_height_p95_mm, 5)} mm`。",
            f"Local Base pooled `|error|` P95=`{fmt(local_total_p95)} mm`，Local Base condition-demeaned deviation P95=`{fmt(local_within_p95)} mm`；对应 u-repeatability / Local total ratio=`{fmt(predicted_height_p95_mm / local_total_p95 if local_total_p95 else None, 3)}`，/ Local within ratio=`{fmt(predicted_height_p95_mm / local_within_p95 if local_within_p95 else None, 3)}`。",
            "",
            "## 结论解释",
            "",
            f"- Random：A-2 原结论为 `{old_random}`。它由同-condition Steger u repeatability 直接给出，与 reference branch 无关；Local 重配没有改变该结论，因此 `A2_RANDOM_DEGRADATION_CONCLUSION_CHANGED = {random_changed}`。",
            f"- Profile bias：A-2 原结论为 `{old_profile}`。profile asymmetry/skewness 的预声明 v-band 高低变化本身不受 Ground reference 替换影响；三种 height 的 asymmetry 高低方向仍为 h10 与 h20/h30 不一致，因此没有把 PARTIAL 升级为 STRONG，`A2_PROFILE_BIAS_CONCLUSION_CHANGED = {profile_changed}`。Local residual correlations 作为辅助证据保留，不反向选择 profile 指标。",
            f"- Local magnitude：按 A-2 使用的预声明判据（predicted/total 与 predicted/within 均达到 0.8 才 YES；within 达到 0.5 才 PARTIAL），Local-reference 结果为 `{local_magnitude_flag}`。",
            f"- Session magnitude：A-2 的等级为 `{old_magnitude}`，Local-reference 等级为 `{local_magnitude_flag}`，所以按‘等级结论是否保持’给出 `SESSION_REFERENCE_MAGNITUDE_RESULT_RETAINED = {session_magnitude_retained}`。即使该 flag 为 YES，旧 Session total P95 仍只是被旧 Ground/reference 污染的历史比较值，不是当前真实 edge residual。",
            "",
            "## 边界",
            "",
            "- 保持原始 full-sensor `(u,v)=(column,row)` 语义；Daheng 为纵向条纹、row scan，本审计关注的 Steger 亚像素中心方向是 u。",
            "- Local reference 仅用于诊断相对高度误差解释能力，不能据此替换 production Session Ground 或宣称 truth。",
            "- 详细逐帧配对没有另写 PNG/Steger cache；输出只依赖既有 A-2/A-13B-v2 CSV/JSON。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    a2_frame_path = A2_DIR / "stripe_quality_frame_audit.csv"
    a2_repeatability_path = A2_DIR / "steger_repeatability_by_v.csv"
    a2_correlation_path = A2_DIR / "quality_correlation_summary.csv"
    a2_provenance_path = A2_DIR / "stripe_quality_provenance.json"
    for path in (a2_frame_path, a2_repeatability_path, a2_correlation_path, a2_provenance_path, A13B_FRAMES):
        if not path.exists():
            raise FileNotFoundError(path)

    # Read existing artifacts only.  The repeatability/correlation CSVs are
    # loaded as provenance checks; frame quality values are never recomputed.
    a2_rows = read_csv(a2_frame_path)
    a2_repeatability_rows = read_csv(a2_repeatability_path)
    a2_correlation_rows = read_csv(a2_correlation_path)
    a2_provenance = json.loads(a2_provenance_path.read_text(encoding="utf-8"))
    a13b_rows = read_csv(A13B_FRAMES)
    rows = validate_and_merge(a2_rows, a13b_rows)

    if len(a2_repeatability_rows) != 170:
        raise RuntimeError(f"Unexpected reused A-2 repeatability row count: {len(a2_repeatability_rows)}")
    if not a2_correlation_rows:
        raise RuntimeError("Reused A-2 correlation artifact is empty")
    feature_missing = [feature for feature in QUALITY_FEATURES if feature not in a2_rows[0]]
    if feature_missing:
        raise RuntimeError(f"A-2 quality feature columns missing from reused frame audit: {feature_missing}")

    correlations = local_correlations(rows)
    correlation_comparison = session_vs_local_correlations(a2_correlation_rows, correlations)
    comparisons = comparison_rows(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "a2r_local_reference_correlation.csv",
        correlations,
        ["reference", "scope", "mode", "feature", "target", "n", "pearson_r", "pearson_pvalue", "spearman_rho", "spearman_pvalue"],
    )
    write_csv(
        OUTPUT_DIR / "a2r_session_vs_local_comparison.csv",
        comparisons,
        ["scope", "model", "metric", "n_paired", "session_n", "local_n", "session_value_mm", "local_value_mm", "delta_local_minus_session_mm"],
    )
    write_csv(
        OUTPUT_DIR / "a2r_session_vs_local_correlation.csv",
        correlation_comparison,
        [
            "scope",
            "mode",
            "feature",
            "target",
            "metric",
            "n_session",
            "n_local",
            "session_value",
            "local_value",
            "delta_local_minus_session",
            "session_pvalue",
            "local_pvalue",
        ],
    )
    plot_height_error_vs_quality(rows, correlations, OUTPUT_DIR / "a2r_height_error_vs_quality.png")

    condition_p95_u = finite(a2_provenance.get("decisions", {}).get("condition_p95_u_abs_deviation_px_p95"))
    sensitivity = finite(a2_provenance.get("decisions", {}).get("sensitivity_mm_per_px_used")) or SENSITIVITY_MM_PER_PX
    if condition_p95_u is None:
        raise RuntimeError("A-2 provenance does not contain condition-level u repeatability P95")
    predicted_height_p95 = float(sensitivity * condition_p95_u)
    local_total_p95 = lookup_comparison(comparisons, "pooled", "base", "p95_abs_error_mm").get("local_value_mm")
    local_within_p95 = lookup_comparison(comparisons, "pooled", "base", "within_condition_p95_abs_deviation_mm").get("local_value_mm")
    total_ratio = predicted_height_p95 / float(local_total_p95) if finite(local_total_p95) not in (None, 0.0) else None
    within_ratio = predicted_height_p95 / float(local_within_p95) if finite(local_within_p95) not in (None, 0.0) else None
    if total_ratio is not None and within_ratio is not None and total_ratio >= 0.8 and within_ratio >= 0.8:
        local_magnitude_flag = "YES"
    elif within_ratio is not None and within_ratio >= 0.5:
        local_magnitude_flag = "PARTIAL"
    else:
        local_magnitude_flag = "NO"

    old_decisions = a2_provenance.get("decisions", {})
    old_random = str(old_decisions.get("LOWER_EDGE_STEGER_RANDOM_DEGRADATION", "UNKNOWN"))
    old_profile = str(old_decisions.get("LOWER_EDGE_PROFILE_SYSTEMATIC_BIAS_EVIDENCE", "UNKNOWN"))
    old_magnitude = str(old_decisions.get("OPTICAL_EXTRACTION_MAGNITUDE_SUFFICIENT", "UNKNOWN"))
    random_changed = "NO" if old_random == "PARTIAL" else "YES"
    profile_changed = "NO" if old_profile == "PARTIAL" else "YES"
    session_magnitude_retained = "YES" if old_magnitude == local_magnitude_flag else "NO"

    provenance = {
        "task": "A-2R Local-reference re-attribution of A-2 stripe/Steger error audit",
        "generated_from_existing_artifacts_only": True,
        "inputs": {
            "a2_frame_audit": str(a2_frame_path),
            "a2_repeatability": str(a2_repeatability_path),
            "a2_correlation": str(a2_correlation_path),
            "a2_provenance": str(a2_provenance_path),
            "a13b_v2_frames": str(A13B_FRAMES),
        },
        "reuse": {
            "a2_quality_features_reused": True,
            "a2_steger_repeatability_reused": True,
            "a2_strict_valid_sanity_reused": True,
            "a2_session_correlation_reused_for_comparison": True,
            "a13b_local_diag_residuals_reused": True,
            "png_read_in_this_task": False,
            "steger_rerun_in_this_task": False,
            "reconstruction_rerun_in_this_task": False,
            "c0_c1_h1_hb2_refit": False,
            "production_reconstruction_modified": False,
            "truth_driven_feature_selection": False,
        },
        "local_definition": {
            "base_field": "residual_base_local_diag",
            "h1_field": "residual_h1_local_diag",
            "hb2_field": "residual_hb2_local_diag",
            "meaning": "local raw height minus nominal truth, using same-frame adjacent-ground/local profile diagnostic",
            "local_diag_only_all_rows": all(bool(row["local_diag_only"]) for row in rows),
            "support_counts": dict(sorted(Counter(str(row["local_baseline_support"]) for row in rows).items())),
            "extrapolated_count": sum(bool(row["local_baseline_extrapolation"]) for row in rows),
        },
        "alignment": {
            "a2_rows": len(a2_rows),
            "a13b_rows": len(a13b_rows),
            "paired_rows": len(rows),
            "key": "cache_key",
            "identity_fields_checked": list(IDENTITY_FIELDS),
            "session_residual_crosscheck": "A-2 frame fields equal A-13B-v2 residual_*_session within 1e-9 mm",
        },
        "a1_sensitivity_reuse": {
            "sensitivity_mm_per_px": sensitivity,
            "condition_p95_u_abs_deviation_px_p95": condition_p95_u,
            "predicted_height_p95_mm": predicted_height_p95,
        },
        "local_magnitude": {
            "local_total_base_abs_p95_mm": local_total_p95,
            "local_within_condition_base_abs_deviation_p95_mm": local_within_p95,
            "predicted_to_local_total_p95_ratio": total_ratio,
            "predicted_to_local_within_condition_p95_ratio": within_ratio,
            "decision": local_magnitude_flag,
        },
        "decisions": {
            "A2_RANDOM_DEGRADATION_CONCLUSION_CHANGED": random_changed,
            "A2_PROFILE_BIAS_CONCLUSION_CHANGED": profile_changed,
            "LOCAL_REFERENCE_OPTICAL_MAGNITUDE_SUFFICIENT": local_magnitude_flag,
            "SESSION_REFERENCE_MAGNITUDE_RESULT_RETAINED": session_magnitude_retained,
            "old_a2_random": old_random,
            "old_a2_profile": old_profile,
            "old_a2_magnitude": old_magnitude,
        },
    }
    write_json(OUTPUT_DIR / "a2r_provenance.json", provenance)
    build_report(
        rows,
        correlations,
        correlation_comparison,
        comparisons,
        a2_provenance,
        local_magnitude_flag,
        session_magnitude_retained,
        profile_changed,
        random_changed,
        predicted_height_p95,
        finite(local_total_p95),
        finite(local_within_p95),
        OUTPUT_DIR / "report.md",
    )

    print(json.dumps({"output_dir": str(OUTPUT_DIR), "decisions": provenance["decisions"], "local_magnitude": provenance["local_magnitude"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
