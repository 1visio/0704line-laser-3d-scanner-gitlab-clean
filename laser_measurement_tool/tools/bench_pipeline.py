"""线激光提取—重建流水线性能基准 + AOI 等价性诊断。

回答三个问题：
1. 这台机器上"输入图像 → 点云"每帧要多久，各段占比多少；
2. 把处理范围限制到条纹附近（AOI）能提速多少；
3. AOI 结果与全幅结果差在哪里、谁更可信。

第 3 点不是形式检查。``laser/backends.py`` 的逐列提取用的是**整列
``np.argmax``**，视场里任何比条纹更亮的特征都会抢走峰值，那一列要么被
丢弃、要么给出错误的中心。因此本脚本会显式统计"全幅峰落在条纹带之外"
的列数与最长连续段——连续段一旦超过 ``segment_min_columns``，全幅提取
会产出一整段位置错误的伪点云，而下游没有任何环节能发现。

放置位置：``laser_measurement_tool/tools/bench_pipeline.py``
用法与参数说明见同目录 ``BENCH_PIPELINE.md``。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

# 允许以 `python tools/bench_pipeline.py` 直接运行：把 laser_measurement_tool 加入 sys.path
_TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOL_ROOT))

from laser.backends import (  # noqa: E402
    AVAILABLE_METHODS,
    CentroidParams,
    SharedStegerParams,
    StegerParams,
)
from reconstruction.reconstructor import (  # noqa: E402
    ReconstructionParams,
    reconstruct_uv_to_ground,
)

try:  # 标定为可选项：不给标定文件时用合成标定，只测耗时不测数值
    from calibration.config_loader import load_calibration_files
    from calibration.manifest import load_calibration_package
except Exception:  # pragma: no cover - 仅在包结构异常时触发
    load_calibration_files = None  # type: ignore[assignment]
    load_calibration_package = None  # type: ignore[assignment]


SYNTHETIC_CALIBRATION: dict[str, Any] = {
    # 数量级取自 outputs/calib03_opencv 与 calib03 laser_plane v4，仅用于耗时测量
    "K": np.array(
        [[4677.204, 0.0, 1211.091], [0.0, 4676.074, 996.227], [0.0, 0.0, 1.0]]
    ),
    "D": np.array([-0.05915, 0.23071, -0.000810, -0.001624, 0.0]),
    "plane_abcd": np.array(
        [-0.006877750551, 0.963551764997, 0.267433529534, -207.619290957327]
    ),
    "R": np.eye(3),
    "t": np.zeros(3),
}


@dataclass
class Timing:
    label: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    fps: float
    note: str = ""


@dataclass
class AxisSpec:
    """按 scan_axis 确定：哪一列是整数索引轴、哪一列是亚像素轴、裁哪个方向。"""

    scan_axis: str
    key_col: int  # 输出数组中作为"整数索引"的列（column 扫描时是 u）
    value_col: int  # 输出数组中作为"亚像素中心"的列（column 扫描时是 v）
    crop_axis: int  # 图像裁剪轴：0=裁行，1=裁列
    value_name: str
    key_name: str

    @classmethod
    def of(cls, scan_axis: str) -> "AxisSpec":
        if scan_axis == "column":
            # 逐列找峰 → 每列一个 v；条纹接近水平 → AOI 是行带
            return cls("column", 0, 1, 0, "v", "u")
        # 逐行找峰 → 每行一个 u；条纹接近竖直 → AOI 是列带
        return cls("row", 1, 0, 1, "u", "v")


@dataclass
class Consistency:
    count_full: int
    count_aoi: int
    common: int
    only_full: int
    only_aoi: int
    max_abs_delta: float
    differing_keys: int
    verdict: str
    examples: list[tuple[int, float, float]] = field(default_factory=list)


@dataclass
class Interference:
    """全幅 argmax 落在条纹带之外的统计（backends 整列 argmax 的固有风险）。"""

    band: tuple[int, int]
    outside_columns: int
    outside_with_contrast: int
    longest_run: int
    longest_run_range: tuple[int, int] | None
    segment_min_columns: int
    fake_segment_risk: bool


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("不能为负数")
    return parsed


def read_image_unicode(path: Path) -> np.ndarray:
    """支持中文路径的灰度读图（与仓库其他脚本一致的 fromfile+imdecode 方式）。"""
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"无法解码图像: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(image)


def make_synthetic_frame(width: int, height: int, bit_depth: int) -> np.ndarray:
    """合成一条接近水平、带弯曲的高斯亮条纹，形态接近 laser_obs 数据集。"""
    rng = np.random.default_rng(20260726)
    peak = 200.0 if bit_depth == 8 else 3200.0
    noise_mean = 8.0 if bit_depth == 8 else 128.0
    noise_std = 3.0 if bit_depth == 8 else 48.0
    ceiling = 255 if bit_depth == 8 else 4095

    columns = np.arange(width, dtype=np.float64)
    stripe_v = (
        height * 0.5
        + height * 0.03 * np.sin(columns / width * np.pi)
        + height * 0.015 * (columns / width)
    )
    rows = np.arange(height, dtype=np.float64)[:, None]
    image = peak * np.exp(-((rows - stripe_v[None, :]) ** 2) / (2.0 * 2.2**2))
    image += rng.normal(noise_mean, noise_std, size=(height, width))
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    return np.clip(image, 0, ceiling).astype(dtype)


def time_call(
    label: str,
    call: Callable[[], Any],
    repeat: int,
    note: str = "",
) -> tuple[Timing, Any]:
    call()  # 预热，排除首次分配与 OpenCV 内部初始化
    samples: list[float] = []
    result: Any = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - start) * 1000.0)
    mean_ms = statistics.fmean(samples)
    timing = Timing(
        label=label,
        mean_ms=mean_ms,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        fps=(1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
        note=note,
    )
    return timing, result


def stripe_band(
    centers: np.ndarray, axis: AxisSpec, extent: int, margin: int
) -> tuple[int, int] | None:
    """由已有提取结果推出 AOI 范围，模拟实时模式按上一帧跟踪条纹。"""
    if centers.size == 0:
        return None
    values = centers[:, axis.value_col]
    low = int(np.floor(values.min())) - margin
    high = int(np.ceil(values.max())) + margin
    return max(0, low), min(extent, high)


def crop_band(image: np.ndarray, low: int, high: int, crop_axis: int) -> np.ndarray:
    band = image[low:high, :] if crop_axis == 0 else image[:, low:high]
    return np.ascontiguousarray(band)


def compare_by_key(
    reference: np.ndarray,
    candidate: np.ndarray,
    axis: AxisSpec,
) -> Consistency:
    """按整数索引轴（列号/行号）配对比较，而不是按数组下标。

    按下标比较是错的：一旦两边点数不同，后续全部错位，会在台阶跳变处
    报出上百像素的假差异。
    """
    ref = {int(round(row[axis.key_col])): float(row[axis.value_col]) for row in reference}
    cand = {int(round(row[axis.key_col])): float(row[axis.value_col]) for row in candidate}
    ref_keys, cand_keys = set(ref), set(cand)
    common_keys = sorted(ref_keys & cand_keys)

    deltas = [(key, ref[key], cand[key]) for key in common_keys]
    differing = [item for item in deltas if abs(item[1] - item[2]) > 1.0e-6]
    max_abs = max((abs(a - b) for _, a, b in deltas), default=0.0)

    only_full = len(ref_keys - cand_keys)
    only_aoi = len(cand_keys - ref_keys)
    if max_abs <= 1.0e-6 and not only_full and not only_aoi:
        verdict = "identical"
    elif only_aoi and not only_full:
        verdict = "aoi_recovers_points"
    elif only_full and not only_aoi:
        verdict = "aoi_loses_points"
    else:
        verdict = "differs"

    return Consistency(
        count_full=len(reference),
        count_aoi=len(candidate),
        common=len(common_keys),
        only_full=only_full,
        only_aoi=only_aoi,
        max_abs_delta=max_abs,
        differing_keys=len(differing),
        verdict=verdict,
        examples=[(k, a, b) for k, a, b in differing[:5]],
    )


def diagnose_interference(
    image: np.ndarray,
    params: CentroidParams,
    axis: AxisSpec,
    band: tuple[int, int],
) -> Interference:
    """统计整列 argmax 被条纹带外的更亮特征抢走的情况。

    复刻 ``_extract_columnwise`` 的前两步（背景抑制 + argmax），因此结论
    与真实提取一致，而不是近似。
    """
    gray = image if axis.scan_axis == "column" else np.ascontiguousarray(image.T)
    background = cv2.GaussianBlur(
        gray, (params.background_kernel, params.background_kernel), 0
    )
    signal = cv2.subtract(gray, background).astype(np.float32)
    scan_count = gray.shape[1]
    indexes = np.arange(scan_count)
    peaks = np.argmax(signal, axis=0)
    contrast = signal[peaks, indexes]

    low, high = band
    outside = (peaks < low) | (peaks >= high)
    effective = outside & (contrast >= params.min_local_contrast_dn)

    longest_run = 0
    longest_range: tuple[int, int] | None = None
    hits = np.where(effective)[0]
    if hits.size:
        runs = np.split(hits, np.where(np.diff(hits) > params.continuity_max_column_gap)[0] + 1)
        best = max(runs, key=len)
        longest_run = len(best)
        longest_range = (int(best[0]), int(best[-1]))

    return Interference(
        band=band,
        outside_columns=int(outside.sum()),
        outside_with_contrast=int(effective.sum()),
        longest_run=longest_run,
        longest_run_range=longest_range,
        segment_min_columns=params.segment_min_columns,
        fake_segment_risk=longest_run >= params.segment_min_columns,
    )


def load_calibration(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.manifest:
        if load_calibration_package is None:
            raise SystemExit("无法导入 calibration.manifest，请检查运行目录")
        package = load_calibration_package(args.manifest)
        return package.calibration, f"manifest:{package.package_id}"
    if not (args.intrinsics and args.laser_plane and args.extrinsics):
        return SYNTHETIC_CALIBRATION, "synthetic"
    if load_calibration_files is None:
        raise SystemExit("无法导入 calibration.config_loader，请检查运行目录")
    calibration = load_calibration_files(
        intrinsics=args.intrinsics,
        laser_plane=args.laser_plane,
        extrinsics=args.extrinsics,
    )
    return calibration, "files"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="线激光提取—重建流水线性能基准 + AOI 等价性诊断",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_argument_group("输入图像（二选一）")
    source.add_argument("--image", type=Path, help="真实图像路径；不给则使用合成条纹")
    source.add_argument("--width", type=positive_int, default=2448, help="合成图宽")
    source.add_argument("--height", type=positive_int, default=2048, help="合成图高")
    source.add_argument(
        "--bit-depth", type=int, choices=(8, 12), default=8, help="合成图位深"
    )

    extraction = parser.add_argument_group("提取参数")
    extraction.add_argument(
        "--method",
        default="steger",
        choices=sorted(k for k, v in AVAILABLE_METHODS.items() if v is not None),
        help="提取算法",
    )
    extraction.add_argument(
        "--background-kernel", type=positive_int, default=51, help="背景抑制核（奇数）"
    )
    extraction.add_argument(
        "--min-local-contrast-dn", type=float, default=20.0, help="峰值最低局部对比度"
    )
    extraction.add_argument(
        "--steger-sigma", type=float, default=1.5, help="Steger 高斯导数尺度"
    )
    extraction.add_argument(
        "--steger-threshold", type=float, default=30.0, help="Steger 原始灰度阈值"
    )
    extraction.add_argument(
        "--steger-deriv-thresh", type=float, default=0.5, help="Steger 二阶导数阈值"
    )
    extraction.add_argument(
        "--steger-roi-margin", type=int, default=120, help="Steger 自动条纹带扩展"
    )
    extraction.add_argument(
        "--steger-roi-max-height", type=positive_int, default=512,
        help="Steger Hessian 计算带最大宽度",
    )
    extraction.add_argument(
        "--scan-axis", choices=("column", "row"), default="column", help="扫描轴"
    )
    extraction.add_argument(
        "--aoi-margin",
        type=positive_int,
        default=150,
        help="AOI 相对条纹范围的外扩余量（像素）",
    )

    calib = parser.add_argument_group("标定（可选；不给则用合成标定，只影响数值不影响耗时）")
    calib.add_argument("--manifest", type=Path, help="运行标定包 manifest.yaml")
    calib.add_argument("--intrinsics", type=Path, help="相机内参 YAML")
    calib.add_argument("--laser-plane", type=Path, help="激光平面 YAML")
    calib.add_argument("--extrinsics", type=Path, help="地面外参 YAML")

    parser.add_argument("--repeat", type=positive_int, default=10, help="每项重复次数")
    parser.add_argument(
        "--skip-variants",
        action="store_true",
        help="只测默认全幅与 AOI，跳过参数变体与背景抑制对比",
    )
    parser.add_argument("--json", type=Path, help="把结果写入 JSON 文件")
    return parser


def print_consistency(result: Consistency, axis: AxisSpec, band: tuple[int, int]) -> None:
    print(
        f"AOI 等价性（按 {axis.key_name} 配对比较，"
        f"AOI 范围 [{band[0]}, {band[1]})）"
    )
    print(
        f"  点数：全幅 {result.count_full}　AOI {result.count_aoi}　"
        f"共同 {axis.key_name} {result.common}　"
        f"仅全幅有 {result.only_full}　仅 AOI 有 {result.only_aoi}"
    )
    print(
        f"  共同 {axis.key_name} 上 max|Δ{axis.value_name}| = "
        f"{result.max_abs_delta:.6f} px（超过 1e-6 的 {axis.key_name} 数：{result.differing_keys}）"
    )
    if result.examples:
        shown = "，".join(
            f"{axis.key_name}={k}: 全幅 {a:.4f} / AOI {b:.4f}"
            for k, a, b in result.examples
        )
        print(f"  差异样例：{shown}")

    if result.verdict == "identical":
        print("  → 逐点一致，AOI 可直接用于精测模式。")
    elif result.verdict == "aoi_recovers_points":
        print(
            "  → AOI 比全幅多出点，且没有丢点。这通常意味着**全幅的整列 argmax 被条纹带外"
            "更亮的特征抢走**，那些列被全幅丢弃了。此时 AOI 结果更可信，不是精度妥协。"
        )
    elif result.verdict == "aoi_loses_points":
        print(
            "  → AOI 比全幅少点。检查 --aoi-margin 是否过小把条纹截断，"
            "或条纹在本帧跨度超出了 AOI。"
        )
    else:
        print(
            "  → 两边各有独有点。逐条核对差异样例所在区域的图像，"
            "确认是干扰被排除（AOI 更好）还是条纹被截断（AOI 更差）。"
        )


def print_interference(result: Interference, axis: AxisSpec) -> None:
    label = "列" if axis.scan_axis == "column" else "行"
    print(f"整列 argmax 干扰诊断（backends 的固有风险）")
    print(
        f"  全幅峰落在条纹带 [{result.band[0]}, {result.band[1]}) 之外的{label}数："
        f"{result.outside_columns}　其中对比度达标（会被当作有效峰）："
        f"{result.outside_with_contrast}"
    )
    if result.longest_run:
        span = result.longest_run_range
        print(
            f"  这些{label}的最长连续段：{result.longest_run} {label}"
            f"（{axis.key_name} {span[0]}–{span[1]}），"
            f"segment_min_columns = {result.segment_min_columns}"
        )
    if result.fake_segment_risk:
        print(
            "  ⚠ 最长连续段已达到 segment_min_columns —— 全幅提取会输出一整段"
            "位置错误的伪点云，且下游没有任何环节能发现。必须限制峰值搜索范围。"
        )
    elif result.outside_with_contrast:
        print(
            "  ⚠ 目前连续段短于 segment_min_columns，伪段被侥幸滤掉，但这些"
            f"{label}的点被静默丢失了。干扰一旦变宽就会产生伪点云。"
        )
    else:
        print("  本帧无带外干扰。")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.image is not None:
        image = read_image_unicode(args.image)
        source_desc = f"{args.image}（{image.shape[1]}×{image.shape[0]}, {image.dtype}）"
    else:
        image = make_synthetic_frame(args.width, args.height, args.bit_depth)
        source_desc = (
            f"合成条纹（{image.shape[1]}×{image.shape[0]}, {image.dtype}, "
            f"Mono{args.bit_depth}）"
        )

    height, width = image.shape[:2]
    backend = AVAILABLE_METHODS[args.method]
    if backend is None:
        raise SystemExit(f"提取算法 {args.method!r} 尚未接入")

    centroid_params: CentroidParams | None = None
    if args.method == "centroid":
        base_options: dict[str, Any] = {
            "background_kernel": args.background_kernel,
            "min_local_contrast_dn": args.min_local_contrast_dn,
            "scan_axis": args.scan_axis,
        }
        centroid_params = CentroidParams(**base_options)
    elif args.method == "steger":
        base_options = {
            "sigma": args.steger_sigma,
            "threshold": args.steger_threshold,
            "deriv_thresh": args.steger_deriv_thresh,
            "roi_margin": args.steger_roi_margin,
            "roi_max_height": args.steger_roi_max_height,
            "scan_axis": args.scan_axis,
        }
        StegerParams(**base_options)
    else:
        shared = SharedStegerParams(scan_axis=args.scan_axis)
        base_options = asdict(shared)
    axis = AxisSpec.of(args.scan_axis)
    extent = height if axis.crop_axis == 0 else width

    calibration, calib_source = load_calibration(args)
    recon_params = ReconstructionParams()

    print("=" * 82)
    print("线激光流水线性能基准")
    print("=" * 82)
    print(f"平台      : {platform.platform()}")
    print(f"Python    : {sys.version.split()[0]}　OpenCV: {cv2.__version__}")
    print(f"输入      : {source_desc}")
    print(f"提取算法  : {args.method}　参数: {base_options}")
    print(f"标定来源  : {calib_source}　重复次数: {args.repeat}")
    print("-" * 82)

    timings: list[Timing] = []

    full_timing, centers_full = time_call(
        "提取 · 全幅（默认参数）",
        lambda: backend(image, base_options),
        args.repeat,
        note=f"{len(backend(image, base_options))} 点",
    )
    timings.append(full_timing)

    band = stripe_band(centers_full, axis, extent, args.aoi_margin)
    centers_aoi = np.empty((0, 2))
    aoi_timing: Timing | None = None
    if band is None:
        print("警告：全幅未提取到中心点，AOI 相关测量跳过（检查 --scan-axis 与阈值）")
    else:
        low, high = band
        sub = crop_band(image, low, high, axis.crop_axis)
        shape_text = (
            f"{high - low}×{width}" if axis.crop_axis == 0 else f"{height}×{high - low}"
        )
        aoi_timing, centers_aoi_local = time_call(
            f"提取 · AOI {shape_text}（参数不变）",
            lambda: backend(sub, base_options),
            args.repeat,
            note=f"范围 [{low}, {high})",
        )
        timings.append(aoi_timing)
        centers_aoi = np.asarray(centers_aoi_local, dtype=np.float64).copy()
        if centers_aoi.size:
            centers_aoi[:, axis.value_col] += low

    recon_timing, _ = time_call(
        f"重建 · {len(centers_full)} 点 → 地面系",
        lambda: reconstruct_uv_to_ground(centers_full, calibration, recon_params),
        max(args.repeat, 20),
    )
    timings.append(recon_timing)

    if not args.skip_variants and args.method == "centroid":
        timings.append(
            time_call(
                "提取 · 全幅 + 关闭段内平滑（correction_window=1）",
                lambda: backend(image, {**base_options, "correction_window": 1}),
                args.repeat,
                note="数值会与默认不同",
            )[0]
        )
        kernel_small = max(3, (args.background_kernel // 2) | 1)
        timings.append(
            time_call(
                f"提取 · 全幅 + background_kernel={kernel_small} + 关平滑",
                lambda: backend(
                    image,
                    {
                        **base_options,
                        "background_kernel": kernel_small,
                        "correction_window": 1,
                    },
                ),
                args.repeat,
                note="数值会与默认不同",
            )[0]
        )
        k = args.background_kernel
        timings.append(
            time_call(
                f"  └ 分解：GaussianBlur({k},{k}) 单独",
                lambda: cv2.GaussianBlur(image, (k, k), 0),
                args.repeat,
            )[0]
        )
        timings.append(
            time_call(
                f"  └ 分解：boxFilter({k},{k}) 替代方案",
                lambda: cv2.blur(image, (k, k)),
                args.repeat,
            )[0]
        )
        timings.append(
            time_call(
                "  └ 分解：1/4 下采样 + Gauss13 + 上采样 替代方案",
                lambda: cv2.resize(
                    cv2.GaussianBlur(
                        cv2.resize(
                            image,
                            (width // 4, height // 4),
                            interpolation=cv2.INTER_AREA,
                        ),
                        (13, 13),
                        0,
                    ),
                    (width, height),
                ),
                args.repeat,
            )[0]
        )

    print(f"{'项目':<52}{'均值ms':>9}{'中位ms':>9}{'FPS':>11}")
    print("-" * 82)
    for item in timings:
        fps_text = f"{item.fps:.1f}" if item.fps < 1.0e5 else ">1e5"
        print(
            f"{item.label:<52}{item.mean_ms:>9.1f}{item.median_ms:>9.1f}"
            f"{fps_text:>11}" + (f"　{item.note}" if item.note else "")
        )
    print("-" * 82)

    consistency: Consistency | None = None
    interference: Interference | None = None
    if band is not None and centers_full.size:
        consistency = compare_by_key(centers_full, centers_aoi, axis)
        print_consistency(consistency, axis, band)
        print("-" * 82)
        if centroid_params is not None:
            interference = diagnose_interference(
                image, centroid_params, axis, band
            )
            print_interference(interference, axis)
            print("-" * 82)

    total_precision = full_timing.mean_ms + recon_timing.mean_ms
    print(
        f"精测链路（全幅提取+重建）：{total_precision:.1f} ms → "
        f"{1000.0 / total_precision:.1f} fps"
    )
    if aoi_timing is not None:
        aoi_total = aoi_timing.mean_ms + recon_timing.mean_ms
        print(
            f"AOI 链路（AOI 提取+重建）：{aoi_total:.1f} ms → "
            f"{1000.0 / aoi_total:.1f} fps"
            f"　（提速 {total_precision / aoi_total:.2f}×）"
        )
    print("=" * 82)

    if args.json:
        payload = {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "source": source_desc,
            "image_size": [width, height],
            "method": args.method,
            "options": base_options,
            "aoi_margin": args.aoi_margin,
            "calibration_source": calib_source,
            "repeat": args.repeat,
            "point_count_full": int(len(centers_full)),
            "aoi_band": None if band is None else list(band),
            "timings": [vars(item) for item in timings],
            "consistency": None if consistency is None else vars(consistency),
            "interference": None if interference is None else vars(interference),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"结果已写入 {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
