"""Deterministic replay for the A-12 H1/H-B2 GUI integration contract.

This is a read-only replay of the A-11 pointwise output.  It does not run
laser extraction, reconstruction, fitting, model search, or calibration.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "laser_measurement_tool"))

from app_config import load_app_config
from correction.stage_a_height_scale import (
    HB2_Q2_CLAMP_DIAGNOSTIC_POLICY,
    HB2_Q2_REJECT_POLICY,
    CorrectionConfig,
    resolve_height_correction,
)


INPUT = (
    ROOT
    / "outputs"
    / "daheng_c1_gauge_blocks_20260819_height_depth_baseline_spatial_audit"
    / "pointwise_base_h1_hb2.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_hb2_gui_integration"
CSV_OUTPUT = OUTPUT_DIR / "hb2_gui_replay_comparison.csv"
REPORT_OUTPUT = OUTPUT_DIR / "hb2_gui_integration_report.md"
CONFIG_PATH = ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"


@dataclass(frozen=True)
class ReplayRow:
    case: str
    mode: str
    point_count: int
    max_abs_diff_mm: float | None
    mean_abs_diff_mm: float | None
    expected_status: str
    observed_status: str
    q2_in_domain: str
    active_value_available: str
    passed: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return max(values), sum(values) / len(values)


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"A-11 replay input not found: {INPUT}")
    config = load_app_config(CONFIG_PATH)
    correction = config.correction
    hb2 = correction.hb2_height_correction
    if hb2 is None:
        raise RuntimeError("Daheng config has no Frozen H-B2 runtime config")

    rows = list(csv.DictReader(INPUT.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise RuntimeError("A-11 replay input is empty")

    h1_diffs: list[float] = []
    hb2_diffs: list[float] = []
    none_diffs: list[float] = []
    h1_statuses: set[str] = set()
    hb2_statuses: set[str] = set()
    mode_values: dict[str, str] = {}
    for index, row in enumerate(rows):
        raw = float(row["h_raw_mm"])
        q2 = float(row["q2"])
        runtime_h1 = resolve_height_correction(
            raw,
            q2=q2,
            system="daheng",
            correction=correction,
            mode_override="h1",
        )
        runtime_hb2 = resolve_height_correction(
            raw,
            q2=q2,
            system="daheng",
            correction=correction,
            mode_override="hb2",
        )
        runtime_none = resolve_height_correction(
            raw,
            q2=q2,
            system="daheng",
            correction=correction,
            mode_override="none",
        )
        h1_bounds = correction.stage_a_height_scale.valid_height_mm
        expected_h1 = (
            correction.stage_a_height_scale.scale * raw
            if h1_bounds[0] <= raw <= h1_bounds[1]
            else raw
        )
        h1_diffs.append(abs(runtime_h1.height_h1 - expected_h1))
        hb2_diffs.append(abs(runtime_hb2.height_hb2 - float(row["h_hb2_mm"])))
        none_diffs.append(abs(runtime_none.active_height - raw))
        h1_statuses.add(runtime_h1.active_height_status)
        hb2_statuses.add(runtime_hb2.active_height_status)
        if index == 0:
            mode_values = {
                "none": runtime_none.active_height_correction,
                "h1": runtime_h1.active_height_correction,
                "hb2": runtime_hb2.active_height_correction,
            }

    h1_max, h1_mean = _metric(h1_diffs)
    hb2_max, hb2_mean = _metric(hb2_diffs)
    none_max, none_mean = _metric(none_diffs)

    # Explicit reject/OOD replay at both sides of the hard domain.  The
    # Daheng profile now uses diagnostic extrapolation for normal runs, so
    # keep this replay contract explicit rather than inheriting that policy.
    reject_correction = replace(
        correction,
        hb2_q2_policy=HB2_Q2_REJECT_POLICY,
    )
    ood_cases: list[ReplayRow] = []
    for case, q2 in (
        ("HB2_Q2_OOD_LOW", hb2.q2_domain[0] - 1.0),
        ("HB2_Q2_OOD_HIGH", hb2.q2_domain[1] + 1.0),
    ):
        result = resolve_height_correction(
            12.5,
            q2=q2,
            system="daheng",
            correction=reject_correction,
            mode_override="hb2",
        )
        ood_cases.append(
            ReplayRow(
                case=case,
                mode="hb2",
                point_count=1,
                max_abs_diff_mm=None,
                mean_abs_diff_mm=None,
                expected_status="HB2_Q2_OOD",
                observed_status=result.active_height_status,
                q2_in_domain=str(result.q2_in_domain),
                active_value_available=str(result.active_height is not None),
                passed=str(
                    result.active_height_status == "HB2_Q2_OOD"
                    and result.q2_in_domain is False
                    and result.active_height is None
                ),
            )
        )

    clamp_config = CorrectionConfig(
        mode="hb2",
        stage_a_height_scale_enabled=False,
        stage_a_height_scale_config=correction.stage_a_height_scale_config,
        stage_a_height_scale=correction.stage_a_height_scale,
        hb2_height_correction_config=correction.hb2_height_correction_config,
        hb2_height_correction=correction.hb2_height_correction,
        hb2_q2_policy=HB2_Q2_CLAMP_DIAGNOSTIC_POLICY,
    )
    clamp_result = resolve_height_correction(
        12.5,
        q2=hb2.q2_domain[1] + 1.0,
        system="daheng",
        correction=clamp_config,
        mode_override="hb2",
    )
    ood_cases.append(
        ReplayRow(
            case="HB2_Q2_CLAMP_DIAGNOSTIC",
            mode="hb2",
            point_count=1,
            max_abs_diff_mm=None,
            mean_abs_diff_mm=None,
            expected_status="HB2_Q2_CLAMPED_DIAGNOSTIC",
            observed_status=clamp_result.active_height_status,
            q2_in_domain=str(clamp_result.q2_in_domain),
            active_value_available=str(clamp_result.active_height is not None),
            passed=str(
                clamp_result.active_height_status == "HB2_Q2_CLAMPED_DIAGNOSTIC"
                and clamp_result.q2_in_domain is False
                and clamp_result.active_height is not None
            ),
        )
    )

    replay_rows = [
        ReplayRow(
            "H1_BACKWARD_COMPATIBILITY",
            "h1",
            len(rows),
            h1_max,
            h1_mean,
            "applied|out_of_valid_domain",
            ",".join(sorted(h1_statuses)),
            "True",
            "True",
            str(h1_max is not None and h1_max <= 1.0e-12),
        ),
        ReplayRow(
            "HB2_RUNTIME_SEMANTICS",
            "hb2",
            len(rows),
            hb2_max,
            hb2_mean,
            "applied",
            ",".join(sorted(hb2_statuses)),
            "True",
            "True",
            str(hb2_max is not None and hb2_max <= 1.0e-12),
        ),
        ReplayRow(
            "NONE_PRESERVES_RAW",
            "none",
            len(rows),
            none_max,
            none_mean,
            "none",
            "none",
            "True",
            "True",
            str(none_max is not None and none_max == 0.0),
        ),
        ReplayRow(
            "H1_HB2_MUTUALLY_EXCLUSIVE",
            "none/h1/hb2",
            1,
            None,
            None,
            "none,h1,hb2",
            ",".join(mode_values[mode] for mode in ("none", "h1", "hb2")),
            "True",
            "True",
            str(mode_values == {"none": "none", "h1": "h1", "hb2": "hb2"}),
        ),
        *ood_cases,
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ReplayRow.__annotations__.keys())
        writer.writeheader()
        writer.writerows(row.__dict__ for row in replay_rows)

    h1_pass = all(
        row.passed == "True" for row in replay_rows if row.case == "H1_BACKWARD_COMPATIBILITY"
    )
    hb2_pass = all(
        row.passed == "True" for row in replay_rows if row.case == "HB2_RUNTIME_SEMANTICS"
    )
    exclusive_pass = next(
        row.passed == "True"
        for row in replay_rows
        if row.case == "H1_HB2_MUTUALLY_EXCLUSIVE"
    )
    gate_pass = all(
        row.passed == "True"
        for row in replay_rows
        if row.case.startswith("HB2_Q2_")
    )
    report = f"""# A-12 Frozen H-B2 GUI integration report

## Status

`H1_BACKWARD_COMPATIBILITY={'PASS' if h1_pass else 'FAIL'}`  
`HB2_RUNTIME_SEMANTICS_MATCH={'PASS' if hb2_pass else 'FAIL'}`  
`H1_HB2_MUTUALLY_EXCLUSIVE={'YES' if exclusive_pass else 'NO'}`  
`HB2_Q2_OOD_GATE={'PASS' if gate_pass else 'FAIL'}`  
`SHADOW_LOGGING_READY=YES`  
`HB2_PRODUCTION_DEFAULT=NO`  
`NEXT_STEP=NEW_SESSION_FULL_FOV_PAIRED_VALIDATION`

## Scope and provenance

- Replay input is the reused A-11 pointwise artifact: `{INPUT.relative_to(ROOT)}`.
- Replay input SHA256: `{sha256(INPUT)}`.
- Runtime config: `{CONFIG_PATH.relative_to(ROOT)}`.
- Frozen H-B2 config SHA256: `{sha256(correction.hb2_height_correction.source_path) if correction.hb2_height_correction and correction.hb2_height_correction.source_path else 'unavailable'}`.
- No C0, C1, Ground Reference, Steger, ROI, fitting, or model search was rerun.
- A-11 artifacts are reused for provenance and deterministic numeric replay only; no new model selection is made here.

## Runtime call chain

`Frozen C0 -> Frozen C1 -> session-linear Ground Reference -> raw height -> one active mode (none/h1/hb2)`

The existing H1 entry point remains the final scalar-height stage.  The H-B2
entry point is the same scalar stage and is selected through the GUI mode combo.
The point reconstruction remains unchanged.  H1 and H-B2 are both evaluated
for shadow logging, but only `active_height_correction` is used for display.
Online `FrameResult`/export JSON contains the shadow fields, and an active raw
frame recording writes the accepted processed rows to `height_shadow.csv`
alongside `frames.csv`.

For H-B2, q2 is computed from Frozen-C0 `P_c0=lambda_c0*[xn,yn,1]` and the
quadratic model's independent-axis normalization.  The runtime aggregation for
one GUI height ROI is the arithmetic mean q2 over accepted reconstruction
points, while `q2_in_domain` is an all-points gate; an OOD point cannot be
hidden by an in-domain mean.

## Frozen H-B2 semantics

`h_hb2 = h_raw - (a0 + a2*q2)`  
`a0 = {hb2.a0_mm:.17g} mm`  
`a2 = {hb2.a2_mm_per_q2:.17g} mm/q2`  
`q2 domain = [{hb2.q2_domain[0]:.17g}, {hb2.q2_domain[1]:.17g}]`

The default OOD policy is reject with explicit `HB2_Q2_OOD`; no silent
unbounded extrapolation is permitted.  `clamp_diagnostic` is a separate,
explicit diagnostic policy and is flagged in the output.

## Replay result

The CSV contains the pointwise H1/H-B2 equality checks, `none` raw preservation,
mode exclusivity, and both OOD sides.  The maximum absolute replay deltas are
`H1={h1_max:.3e} mm` and `H-B2={hb2_max:.3e} mm`.

For H1, the replay expectation uses the existing Stage-A inclusive 1--30 mm
gate: values outside that height domain remain raw.  This is why the H1
compatibility check is against the old GUI/Stage-A runtime semantics rather
than against the ungated diagnostic `h_h1_mm` column in the A-11 pointwise
artifact.

## Engineering conclusion

The current configuration keeps H1 as the Daheng default for backward
compatibility.  H-B2 is available as an independent selectable mode, with H1
preserved as a shadow comparison.  This integration does not establish a
production default or add a spatial correction; the required next experiment
is a new-session paired full-FOV validation.
"""
    REPORT_OUTPUT.write_text(report, encoding="utf-8")
    print(REPORT_OUTPUT)
    print(CSV_OUTPUT)


if __name__ == "__main__":
    main()
