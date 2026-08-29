"""Experimental Daheng Stage-A height-scale correction.

Stage-A is deliberately a post-measurement height correction.  It never
changes C0/C1, lambda, reconstructed point coordinates, or Ground G(S).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


H1_CORRECTION_MODE = "h1"
HB2_CORRECTION_MODE = "hb2"
NO_CORRECTION_MODE = "none"
SURFACE_AWARE_MODE = "surface_aware"
# Kept as a source-compatible alias for the pre-A-12 Stage-A API.  New
# configuration and UI values use ``h1``.
LEGACY_STAGE_A_HEIGHT_SCALE_MODE = "stage_a_height_scale"
STAGE_A_HEIGHT_SCALE_MODE = LEGACY_STAGE_A_HEIGHT_SCALE_MODE
VALID_CORRECTION_MODES = frozenset(
    {
        NO_CORRECTION_MODE,
        H1_CORRECTION_MODE,
        HB2_CORRECTION_MODE,
        LEGACY_STAGE_A_HEIGHT_SCALE_MODE,
    }
)
CANONICAL_CORRECTION_MODES = frozenset(
    {NO_CORRECTION_MODE, H1_CORRECTION_MODE, HB2_CORRECTION_MODE}
)
HB2_Q2_REJECT_POLICY = "reject"
HB2_Q2_CLAMP_DIAGNOSTIC_POLICY = "clamp_diagnostic"
HB2_Q2_EXTRAPOLATE_DIAGNOSTIC_POLICY = "extrapolate_diagnostic"
HB2_Q2_POLICIES = frozenset(
    {
        HB2_Q2_REJECT_POLICY,
        HB2_Q2_CLAMP_DIAGNOSTIC_POLICY,
        HB2_Q2_EXTRAPOLATE_DIAGNOSTIC_POLICY,
    }
)


class StageAConfigError(ValueError):
    """Stage-A configuration is missing, malformed, or unsafe to use."""


class HB2ConfigError(ValueError):
    """Frozen H-B2 configuration is missing, malformed, or unsafe to use."""


@dataclass(frozen=True, slots=True)
class StageAHeightScaleConfig:
    """Frozen Stage-A metadata and scale factor."""

    system: str
    status: str
    valid_height_mm: tuple[float, float]
    scale: float
    source_path: Path | None = None

    def __post_init__(self) -> None:
        system = self.system.strip().lower()
        if system != "daheng":
            raise StageAConfigError(
                "Stage-A height-scale config 仅允许 system: daheng"
            )
        if self.status != "experimental_stage_validated":
            raise StageAConfigError(
                "Stage-A height-scale config 必须保持 experimental_stage_validated"
            )
        try:
            bounds = tuple(float(value) for value in self.valid_height_mm)
        except (TypeError, ValueError) as error:
            raise StageAConfigError("valid_height_mm 必须包含两个有限数") from error
        if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds):
            raise StageAConfigError("valid_height_mm 必须包含两个有限数")
        if bounds[0] > bounds[1] or bounds[0] < 0.0:
            raise StageAConfigError("valid_height_mm 必须是非负的升序范围")
        scale = float(self.scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise StageAConfigError("scale 必须是有限正数")
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "valid_height_mm", bounds)
        object.__setattr__(self, "scale", scale)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).resolve())


@dataclass(frozen=True, slots=True)
class HB2HeightCorrectionConfig:
    """Frozen H-B2 scalar height correction and its hard q2 domain."""

    system: str
    model_id: str
    status: str
    a0_mm: float
    a2_mm_per_q2: float
    q2_domain: tuple[float, float]
    out_of_domain_policy: str = HB2_Q2_REJECT_POLICY
    clamp_policy: str = "explicit_diagnostic_only"
    production_default: bool = False
    source_path: Path | None = None

    def __post_init__(self) -> None:
        system = self.system.strip().lower()
        if system != "daheng":
            raise HB2ConfigError("H-B2 config 仅允许 system: daheng")
        if self.model_id != "H-B2":
            raise HB2ConfigError("H-B2 config 的 model_id 必须为 H-B2")
        if not self.status:
            raise HB2ConfigError("H-B2 config 必须包含非空 status")
        try:
            a0 = float(self.a0_mm)
            a2 = float(self.a2_mm_per_q2)
            domain = tuple(float(value) for value in self.q2_domain)
        except (TypeError, ValueError) as error:
            raise HB2ConfigError("H-B2 参数必须是有限数值") from error
        if (
            not math.isfinite(a0)
            or not math.isfinite(a2)
            or len(domain) != 2
            or not all(math.isfinite(value) for value in domain)
            or not domain[0] < domain[1]
        ):
            raise HB2ConfigError("H-B2 a0/a2/q2_domain 参数非法")
        if self.out_of_domain_policy != HB2_Q2_REJECT_POLICY:
            raise HB2ConfigError(
                "Frozen H-B2 的 out_of_domain_policy 必须为 reject"
            )
        if self.clamp_policy != "explicit_diagnostic_only":
            raise HB2ConfigError(
                "Frozen H-B2 的 clamp_policy 必须为 explicit_diagnostic_only"
            )
        if not isinstance(self.production_default, bool):
            raise HB2ConfigError("H-B2 production_default 必须是布尔值")
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "a0_mm", a0)
        object.__setattr__(self, "a2_mm_per_q2", a2)
        object.__setattr__(self, "q2_domain", domain)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).resolve())


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    """Mutually exclusive height correction mode and frozen configurations.

    The runtime modes are ``none``, ``h1``, and ``hb2``.  The old
    ``stage_a_height_scale`` spelling is accepted only for compatibility and
    normalized to ``h1``.
    """

    mode: str = NO_CORRECTION_MODE
    stage_a_height_scale_enabled: bool = False
    stage_a_height_scale_config: Path | None = None
    stage_a_height_scale: StageAHeightScaleConfig | None = None
    hb2_height_correction_config: Path | None = None
    hb2_height_correction: HB2HeightCorrectionConfig | None = None
    hb2_q2_policy: str = HB2_Q2_REJECT_POLICY

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise ValueError("correction.mode 必须是字符串")
        mode = self.mode.strip().lower()
        if mode not in VALID_CORRECTION_MODES:
            raise ValueError(
                "correction.mode 必须是 none、h1 或 hb2"
            )
        if not isinstance(self.stage_a_height_scale_enabled, bool):
            raise ValueError("stage_a_height_scale_enabled 必须是布尔值")
        if (
            self.stage_a_height_scale_enabled
            and mode
            not in {H1_CORRECTION_MODE, LEGACY_STAGE_A_HEIGHT_SCALE_MODE}
        ):
            raise ValueError(
                "stage_a_height_scale_enabled 与 correction.mode 互斥"
            )
        if (
            mode in {H1_CORRECTION_MODE, LEGACY_STAGE_A_HEIGHT_SCALE_MODE}
            and self.stage_a_height_scale is None
        ):
            raise ValueError(
                "correction.mode=h1 必须提供 Stage-A 配置"
            )
        if mode == HB2_CORRECTION_MODE and self.hb2_height_correction is None:
            raise ValueError("correction.mode=hb2 必须提供 Frozen H-B2 配置")
        policy = str(self.hb2_q2_policy).strip().lower()
        if policy not in HB2_Q2_POLICIES:
            raise ValueError(
                "hb2_q2_policy 必须是 reject、clamp_diagnostic 或 extrapolate_diagnostic"
            )
        object.__setattr__(self, "mode", mode)
        if mode == LEGACY_STAGE_A_HEIGHT_SCALE_MODE:
            object.__setattr__(self, "mode", H1_CORRECTION_MODE)
        object.__setattr__(
            self,
            "stage_a_height_scale_enabled",
            mode in {H1_CORRECTION_MODE, LEGACY_STAGE_A_HEIGHT_SCALE_MODE},
        )
        object.__setattr__(self, "hb2_q2_policy", policy)
        if self.stage_a_height_scale_config is not None:
            object.__setattr__(
                self,
                "stage_a_height_scale_config",
                Path(self.stage_a_height_scale_config).resolve(),
            )
        if self.hb2_height_correction_config is not None:
            object.__setattr__(
                self,
                "hb2_height_correction_config",
                Path(self.hb2_height_correction_config).resolve(),
            )


@dataclass(frozen=True, slots=True)
class StageAHeightResult:
    """Raw and post-Stage-A height values for one measured height."""

    height_raw: float | None
    height_stage_a: float | None
    stage_a_enabled: bool
    stage_a_valid: bool
    stage_a_status: str

    def as_dict(self) -> dict[str, Any]:
        """Return the stable JSON fields required by online and single-frame output."""
        return {
            "height_raw": self.height_raw,
            "height_stage_a": self.height_stage_a,
            "stage_a_enabled": self.stage_a_enabled,
            "stage_a_valid": self.stage_a_valid,
            "stage_a_status": self.stage_a_status,
        }


@dataclass(frozen=True, slots=True)
class HeightCorrectionResult:
    """Raw height plus mutually exclusive active and shadow corrections."""

    height_raw: float | None
    height_h1: float | None
    height_hb2: float | None
    active_height_correction: str
    active_height: float | None
    active_height_valid: bool
    active_height_status: str
    q1: float | None
    q2: float | None
    q2_in_domain: bool | None
    hb2_q2_status: str
    h1_status: str
    h1_valid: bool

    @property
    def height_stage_a(self) -> float | None:
        """Compatibility view of the old Stage-A output."""
        return self.height_h1

    @property
    def stage_a_enabled(self) -> bool:
        return self.active_height_correction == H1_CORRECTION_MODE

    @property
    def stage_a_valid(self) -> bool:
        return self.stage_a_enabled and self.h1_valid

    @property
    def stage_a_status(self) -> str:
        return self.h1_status

    def as_dict(self) -> dict[str, Any]:
        """Return new shadow fields and the old Stage-A compatibility fields."""
        return {
            "height_raw": self.height_raw,
            "height_h1": self.height_h1,
            "height_hb2": self.height_hb2,
            "active_height_correction": self.active_height_correction,
            "active_height": self.active_height,
            "active_height_valid": self.active_height_valid,
            "active_height_status": self.active_height_status,
            "q1": self.q1,
            "q2": self.q2,
            "q2_in_domain": self.q2_in_domain,
            "hb2_q2_status": self.hb2_q2_status,
            # Legacy Stage-A output remains available to existing consumers.
            "height_stage_a": self.height_stage_a,
            "stage_a_enabled": self.stage_a_enabled,
            "stage_a_valid": self.stage_a_valid,
            "stage_a_status": self.stage_a_status,
        }


def load_stage_a_height_scale(path: str | Path) -> StageAHeightScaleConfig:
    """Load the independent frozen Stage-A parameter JSON."""
    config_path = Path(path)
    if not config_path.is_file():
        raise StageAConfigError(f"Stage-A 配置文件不存在: {config_path}")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageAConfigError(
            f"无法读取 Stage-A 配置文件 {config_path}: {error}"
        ) from error
    if not isinstance(document, Mapping):
        raise StageAConfigError("Stage-A 配置根节点必须是 JSON 映射")
    if document.get("schema_version", 1) != 1:
        raise StageAConfigError("Stage-A 配置 schema_version 必须为 1")
    required = {"system", "status", "valid_height_mm", "scale"}
    missing = required - set(document)
    if missing:
        raise StageAConfigError(
            f"Stage-A 配置缺少字段: {sorted(missing)}"
        )
    try:
        return StageAHeightScaleConfig(
            system=document["system"],
            status=document["status"],
            valid_height_mm=tuple(document["valid_height_mm"]),
            scale=document["scale"],
            source_path=config_path,
        )
    except (TypeError, ValueError, KeyError) as error:
        if isinstance(error, StageAConfigError):
            raise
        raise StageAConfigError(
            f"Stage-A 配置字段非法: {config_path}: {error}"
        ) from error


def load_hb2_height_correction(path: str | Path) -> HB2HeightCorrectionConfig:
    """Load the frozen H-B2 runtime parameter JSON without refitting."""
    config_path = Path(path)
    if not config_path.is_file():
        raise HB2ConfigError(f"Frozen H-B2 配置文件不存在: {config_path}")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HB2ConfigError(
            f"无法读取 Frozen H-B2 配置文件 {config_path}: {error}"
        ) from error
    if not isinstance(document, Mapping):
        raise HB2ConfigError("Frozen H-B2 配置根节点必须是 JSON 映射")
    if document.get("schema_version", 1) != 1:
        raise HB2ConfigError("Frozen H-B2 配置 schema_version 必须为 1")
    if document.get("frozen") is not True:
        raise HB2ConfigError("Frozen H-B2 配置必须标记 frozen=true")
    required = {
        "system",
        "model_id",
        "status",
        "a0_mm",
        "a2_mm_per_q2",
        "q2_domain",
    }
    missing = required - set(document)
    if missing:
        raise HB2ConfigError(f"Frozen H-B2 配置缺少字段: {sorted(missing)}")
    try:
        return HB2HeightCorrectionConfig(
            system=document["system"],
            model_id=document["model_id"],
            status=document["status"],
            a0_mm=document["a0_mm"],
            a2_mm_per_q2=document["a2_mm_per_q2"],
            q2_domain=tuple(document["q2_domain"]),
            out_of_domain_policy=document.get(
                "out_of_domain_policy", HB2_Q2_REJECT_POLICY
            ),
            clamp_policy=document.get("clamp_policy", "explicit_diagnostic_only"),
            production_default=document.get("production_default", False),
            source_path=config_path,
        )
    except (TypeError, ValueError, KeyError) as error:
        if isinstance(error, HB2ConfigError):
            raise
        raise HB2ConfigError(
            f"Frozen H-B2 配置字段非法: {config_path}: {error}"
        ) from error


def apply_stage_a_height_scale(
    height_raw: float | None,
    *,
    system: str,
    enabled: bool,
    correction_mode: str,
    config: StageAHeightScaleConfig | None,
) -> StageAHeightResult:
    """Apply Stage-A only to a final scalar height inside its valid domain.

    The valid interval is inclusive.  Disabled, unsupported, out-of-domain,
    or unmeasured values are returned unchanged (or as ``None``) and are
    explicitly labelled in ``stage_a_status``.
    """
    raw = None if height_raw is None else float(height_raw)
    mode = correction_mode.strip().lower()
    normalized_system = system.strip().lower()

    if not enabled:
        return StageAHeightResult(raw, raw, False, False, "disabled")
    if config is None:
        return StageAHeightResult(raw, raw, False, False, "not_configured")
    if mode not in {H1_CORRECTION_MODE, LEGACY_STAGE_A_HEIGHT_SCALE_MODE}:
        return StageAHeightResult(raw, raw, False, False, "mode_not_stage_a")
    if normalized_system != config.system:
        return StageAHeightResult(raw, raw, False, False, "unsupported_system")
    if raw is None:
        return StageAHeightResult(None, None, True, False, "not_measured")
    if not math.isfinite(raw):
        return StageAHeightResult(raw, raw, True, False, "invalid_height")

    lower, upper = config.valid_height_mm
    if not lower <= raw <= upper:
        return StageAHeightResult(
            raw, raw, True, False, "out_of_valid_domain"
        )
    return StageAHeightResult(
        raw,
        config.scale * raw,
        True,
        True,
        "applied",
    )


def resolve_stage_a_height_scale(
    height_raw: float | None,
    *,
    system: str,
    correction: CorrectionConfig | None,
) -> StageAHeightResult:
    """Resolve a loaded app correction config into one height result."""
    if correction is None:
        return apply_stage_a_height_scale(
            height_raw,
            system=system,
            enabled=False,
            correction_mode=NO_CORRECTION_MODE,
            config=None,
        )
    return apply_stage_a_height_scale(
        height_raw,
        system=system,
        enabled=correction.stage_a_height_scale_enabled,
        correction_mode=correction.mode,
        config=correction.stage_a_height_scale,
    )


def normalize_correction_mode(mode: str) -> str:
    """Normalize UI/config mode names while retaining the legacy alias."""
    if not isinstance(mode, str):
        raise ValueError("height correction mode 必须是字符串")
    normalized = mode.strip().lower()
    if normalized == LEGACY_STAGE_A_HEIGHT_SCALE_MODE:
        return H1_CORRECTION_MODE
    if normalized not in CANONICAL_CORRECTION_MODES:
        raise ValueError("height correction mode 必须是 none、h1 或 hb2")
    return normalized


def resolve_height_correction(
    height_raw: float | None,
    *,
    q1: float | None = None,
    q2: float | None = None,
    q2_in_domain: bool | None = None,
    system: str,
    correction: CorrectionConfig | None,
    mode_override: str | None = None,
) -> HeightCorrectionResult:
    """Resolve none/H1/H-B2 with H1 and H-B2 shadow values.

    ``q2_in_domain`` may be supplied by a pointwise geometry gate.  This is
    important for a scalar height: an in-domain mean must not hide any OOD
    formal point.  OOD values are rejected by ``reject``, clipped only by
    ``clamp_diagnostic``, or evaluated with the actual q2 by
    ``extrapolate_diagnostic``.  Diagnostic extrapolation keeps the OOD
    status and is never reported as an active valid height.
    """
    active_mode = normalize_correction_mode(
        mode_override
        if mode_override is not None
        else correction.mode if correction is not None else NO_CORRECTION_MODE
    )
    normalized_system = system.strip().lower()
    stage_config = correction.stage_a_height_scale if correction is not None else None
    h1_result = apply_stage_a_height_scale(
        height_raw,
        system=normalized_system,
        enabled=stage_config is not None,
        correction_mode=H1_CORRECTION_MODE,
        config=stage_config,
    )

    hb2_config = correction.hb2_height_correction if correction is not None else None
    hb2_policy = (
        correction.hb2_q2_policy
        if correction is not None
        else HB2_Q2_REJECT_POLICY
    )
    hb2_value: float | None = None
    hb2_status = "not_configured"
    domain_flag = q2_in_domain
    if hb2_config is not None and normalized_system != hb2_config.system:
        hb2_status = "unsupported_system"
    elif hb2_config is not None and q2 is None:
        if q2_in_domain is False:
            hb2_status = "HB2_Q2_INVALID"
            domain_flag = False
        else:
            hb2_status = "not_measured" if height_raw is None else "HB2_Q2_MISSING"
            domain_flag = None
    elif hb2_config is not None:
        try:
            q2_value = float(q2)
        except (TypeError, ValueError, OverflowError):
            q2_value = math.nan
        if not math.isfinite(q2_value):
            hb2_status = "HB2_Q2_INVALID"
            domain_flag = False
        else:
            lower, upper = hb2_config.q2_domain
            scalar_in_domain = lower <= q2_value <= upper
            if domain_flag is None:
                domain_flag = scalar_in_domain
            else:
                domain_flag = bool(domain_flag and scalar_in_domain)
            if not domain_flag:
                if (
                    hb2_policy == HB2_Q2_CLAMP_DIAGNOSTIC_POLICY
                    and height_raw is not None
                    and math.isfinite(float(height_raw))
                ):
                    q2_eval = min(max(q2_value, lower), upper)
                    raw = float(height_raw)
                    hb2_value = raw - (
                        hb2_config.a0_mm + hb2_config.a2_mm_per_q2 * q2_eval
                    )
                    hb2_status = "HB2_Q2_CLAMPED_DIAGNOSTIC"
                elif (
                    hb2_policy == HB2_Q2_EXTRAPOLATE_DIAGNOSTIC_POLICY
                    and height_raw is not None
                    and math.isfinite(float(height_raw))
                ):
                    raw = float(height_raw)
                    hb2_value = raw - (
                        hb2_config.a0_mm + hb2_config.a2_mm_per_q2 * q2_value
                    )
                    hb2_status = "HB2_Q2_OOD"
                else:
                    hb2_status = "HB2_Q2_OOD"
            elif height_raw is None:
                hb2_status = "not_measured"
            else:
                raw = float(height_raw)
                if not math.isfinite(raw):
                    hb2_status = "invalid_height"
                else:
                    hb2_value = raw - (
                        hb2_config.a0_mm + hb2_config.a2_mm_per_q2 * q2_value
                    )
                    hb2_status = "applied"

    if active_mode == NO_CORRECTION_MODE:
        active_height = height_raw
        active_valid = False
        active_status = "none"
    elif active_mode == H1_CORRECTION_MODE:
        active_height = h1_result.height_stage_a
        active_valid = h1_result.stage_a_valid
        active_status = h1_result.stage_a_status
    elif active_mode == HB2_CORRECTION_MODE:
        active_height = hb2_value
        active_valid = hb2_value is not None and hb2_status in {
            "applied",
            "HB2_Q2_CLAMPED_DIAGNOSTIC",
        }
        active_status = hb2_status
    else:
        active_height = height_raw
        active_valid = False
        active_status = "reserved_mode"

    try:
        q1_output = None if q1 is None else float(q1)
    except (TypeError, ValueError, OverflowError):
        q1_output = None
    try:
        q2_output = None if q2 is None else float(q2)
    except (TypeError, ValueError, OverflowError):
        q2_output = None
    return HeightCorrectionResult(
        height_raw=None if height_raw is None else float(height_raw),
        height_h1=h1_result.height_stage_a,
        height_hb2=hb2_value,
        active_height_correction=active_mode,
        active_height=active_height,
        active_height_valid=active_valid,
        active_height_status=active_status,
        q1=q1_output,
        q2=q2_output,
        q2_in_domain=domain_flag,
        hb2_q2_status=hb2_status,
        h1_status=h1_result.stage_a_status,
        h1_valid=h1_result.stage_a_valid,
    )
