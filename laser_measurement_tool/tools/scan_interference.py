"""批量扫描线激光图像的"带外干扰"风险。

背景
----
``laser/backends.py`` 的 ``_extract_columnwise()`` 用**整列 ``np.argmax``**
定位条纹峰：

    peak_rows = np.argmax(signal, axis=0)   # 在整列里找最亮

视场里任何比条纹更亮的特征（二次反射、工件高光、另一处饱和亮区）都会抢走
峰值。后果分两级：

* 抢走峰的列若不足 ``segment_min_columns``，该列被静默丢弃 —— 点云少点，
  无任何日志；
* 抢走峰的列若形成 ≥ ``segment_min_columns`` 的连续段，就会输出**一整段
  位置错误的伪点云**。伪点自身内部连续一致，因此射线求交、地面变换、异常
  过滤、障碍物拟合全都不会报错。

本脚本对整个目录逐帧检测这两种情况，并给出可追溯的 CSV 与汇总报告。

检测方法（自包含，不依赖预设 AOI）
--------------------------------
直接复刻 ``_extract_columnwise`` 的前两步（背景抑制 + 逐列 argmax + 对比度
筛选），然后把候选峰按位置聚类：

1. 取所有对比度达标的列的峰位置；
2. 沿峰位置排序，按间隔 > ``--cluster-gap``（默认 200 px）切分成簇；
3. **列数最多的簇视为真实条纹**，其余簇为"带外候选"；
4. 对每个带外簇，统计它在扫描轴上的最长连续段；连续段 ≥
   ``segment_min_columns`` 即判定 ``fake_segment_risk``；
5. 另外跑一次真实提取，检查**输出点云本身**是否已经分裂成多个位置簇 ——
   这是"伪段已经进入结果"的直接证据。

这样判定不需要事先知道条纹在哪，因此对已被污染的帧同样有效。

放置位置：``laser_measurement_tool/tools/scan_interference.py``
用法与参数说明见同目录 ``SCAN_INTERFERENCE.md``。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

_TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOL_ROOT))

from laser.backends import AVAILABLE_METHODS, CentroidParams  # noqa: E402


IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")

CSV_FIELDS = (
    "file",
    "width",
    "height",
    "dtype",
    "extracted_points",
    "candidate_columns",
    "cluster_count",
    "main_cluster_columns",
    "main_cluster_range",
    "outside_columns",
    "outside_longest_run",
    "outside_longest_run_range",
    "outside_longest_run_cluster_range",
    "fake_segment_risk",
    "output_cluster_count",
    "output_split_detected",
    "output_cluster_ranges",
    "status",
    "error",
)


@dataclass
class Cluster:
    """一簇位置相近的候选峰。"""

    low: int
    high: int
    columns: np.ndarray = field(repr=False)

    @property
    def size(self) -> int:
        return int(self.columns.size)

    def longest_run(self, max_gap: int) -> tuple[int, tuple[int, int] | None]:
        if self.columns.size == 0:
            return 0, None
        ordered = np.sort(self.columns)
        runs = np.split(ordered, np.where(np.diff(ordered) > max_gap)[0] + 1)
        best = max(runs, key=len)
        return len(best), (int(best[0]), int(best[-1]))


@dataclass
class FrameReport:
    file: str
    width: int = 0
    height: int = 0
    dtype: str = ""
    extracted_points: int = 0
    candidate_columns: int = 0
    cluster_count: int = 0
    main_cluster_columns: int = 0
    main_cluster_range: str = ""
    outside_columns: int = 0
    outside_longest_run: int = 0
    outside_longest_run_range: str = ""
    outside_longest_run_cluster_range: str = ""
    fake_segment_risk: bool = False
    output_cluster_count: int = 0
    output_split_detected: bool = False
    output_cluster_ranges: str = ""
    status: str = "ok"
    error: str = ""

    def as_row(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in CSV_FIELDS}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def read_image_unicode(path: Path) -> np.ndarray:
    """支持中文路径的灰度读图（与仓库其他脚本一致的 fromfile+imdecode 方式）。"""
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("无法解码为图像")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(image)


def collect_images(inputs: Iterable[Path], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        if item.is_file():
            found.append(item)
        elif item.is_dir():
            pattern = "**/*" if recursive else "*"
            found.extend(
                child
                for child in sorted(item.glob(pattern))
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            raise SystemExit(f"输入不存在: {item}")
    # 去重并保持顺序
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def cluster_positions(
    positions: np.ndarray, columns: np.ndarray, cluster_gap: int
) -> list[Cluster]:
    """按位置间隔把候选峰切分成簇（列数最多的簇即真实条纹）。"""
    if positions.size == 0:
        return []
    order = np.argsort(positions)
    sorted_positions = positions[order]
    sorted_columns = columns[order]
    breaks = np.where(np.diff(sorted_positions) > cluster_gap)[0] + 1
    clusters: list[Cluster] = []
    for chunk_positions, chunk_columns in zip(
        np.split(sorted_positions, breaks), np.split(sorted_columns, breaks)
    ):
        if chunk_positions.size == 0:
            continue
        clusters.append(
            Cluster(
                low=int(chunk_positions.min()),
                high=int(chunk_positions.max()),
                columns=chunk_columns,
            )
        )
    return clusters


def analyse_frame(
    path: Path,
    params: CentroidParams,
    backend: Any,
    options: dict[str, Any],
    cluster_gap: int,
) -> FrameReport:
    report = FrameReport(file=str(path))
    try:
        image = read_image_unicode(path)
    except (OSError, ValueError) as error:
        report.status = "read_error"
        report.error = str(error)
        return report

    report.height, report.width = image.shape[:2]
    report.dtype = str(image.dtype)

    # ---- 复刻 _extract_columnwise 的前两步 ----
    gray = image if params.scan_axis == "column" else np.ascontiguousarray(image.T)
    background = cv2.GaussianBlur(
        gray, (params.background_kernel, params.background_kernel), 0
    )
    signal = cv2.subtract(gray, background).astype(np.float32)
    scan_count = gray.shape[1]
    scan_index = np.arange(scan_count)
    peaks = np.argmax(signal, axis=0)
    contrast = signal[peaks, scan_index]

    valid = contrast >= params.min_local_contrast_dn
    report.candidate_columns = int(valid.sum())
    clusters = cluster_positions(peaks[valid], scan_index[valid], cluster_gap)
    report.cluster_count = len(clusters)

    if clusters:
        main = max(clusters, key=lambda item: item.size)
        report.main_cluster_columns = main.size
        report.main_cluster_range = f"[{main.low},{main.high}]"
        outside = [item for item in clusters if item is not main]
        report.outside_columns = sum(item.size for item in outside)
        for item in outside:
            run_length, run_range = item.longest_run(params.continuity_max_column_gap)
            if run_length > report.outside_longest_run:
                report.outside_longest_run = run_length
                report.outside_longest_run_range = (
                    "" if run_range is None else f"[{run_range[0]},{run_range[1]}]"
                )
                report.outside_longest_run_cluster_range = f"[{item.low},{item.high}]"
        report.fake_segment_risk = (
            report.outside_longest_run >= params.segment_min_columns
        )

    # ---- 真实提取结果本身是否已经分裂成多簇 ----
    try:
        centers = backend(image, options)
    except Exception as error:  # noqa: BLE001 - 单帧失败不应中断批量
        report.status = "extract_error"
        report.error = f"{type(error).__name__}: {error}"
        return report

    report.extracted_points = int(len(centers))
    if centers.size:
        value_col = 1 if params.scan_axis == "column" else 0
        key_col = 1 - value_col
        output_clusters = cluster_positions(
            centers[:, value_col].astype(np.int64),
            centers[:, key_col].astype(np.int64),
            cluster_gap,
        )
        report.output_cluster_count = len(output_clusters)
        report.output_split_detected = len(output_clusters) > 1
        report.output_cluster_ranges = " ".join(
            f"[{item.low},{item.high}]×{item.size}" for item in output_clusters
        )

    return report


def write_csv(path: Path, reports: list[FrameReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for report in reports:
            writer.writerow(report.as_row())


def write_markdown(path: Path, reports: list[FrameReport], params: CentroidParams,
                   cluster_gap: int) -> None:
    ok = [item for item in reports if item.status == "ok"]
    failed = [item for item in reports if item.status != "ok"]
    risky = [item for item in ok if item.fake_segment_risk]
    split = [item for item in ok if item.output_split_detected]
    lossy = [item for item in ok if item.outside_columns and not item.fake_segment_risk]

    lines: list[str] = []
    lines.append("# 带外干扰批量扫描报告")
    lines.append("")
    lines.append(f"- 扫描帧数：**{len(reports)}**（成功 {len(ok)}，失败 {len(failed)}）")
    lines.append(f"- 判定为 `fake_segment_risk`（会产出伪点云）：**{len(risky)}**")
    lines.append(f"- 提取输出已分裂成多个位置簇：**{len(split)}**")
    lines.append(f"- 有带外干扰但连续段不足、表现为静默丢点：**{len(lossy)}**")
    lines.append("")
    lines.append("扫描参数：")
    lines.append("")
    lines.append(f"- `background_kernel` = {params.background_kernel}")
    lines.append(f"- `min_local_contrast_dn` = {params.min_local_contrast_dn}")
    lines.append(f"- `segment_min_columns` = {params.segment_min_columns}（伪段判定阈值）")
    lines.append(f"- `continuity_max_column_gap` = {params.continuity_max_column_gap}")
    lines.append(f"- `scan_axis` = {params.scan_axis}")
    lines.append(f"- `cluster_gap` = {cluster_gap} px（簇切分间隔）")
    lines.append("")

    if risky:
        lines.append("## ⚠ 必须处理：会产出伪点云的帧")
        lines.append("")
        lines.append("| 文件 | 带外最长连续段 | 段范围 | 干扰簇位置 | 主簇位置 | 输出点数 | 输出已分裂 |")
        lines.append("|---|---:|---|---|---|---:|---|")
        for item in risky:
            lines.append(
                f"| `{Path(item.file).name}` | {item.outside_longest_run} | "
                f"{item.outside_longest_run_range} | "
                f"{item.outside_longest_run_cluster_range} | "
                f"{item.main_cluster_range} | {item.extracted_points} | "
                f"{'是' if item.output_split_detected else '否'} |"
            )
        lines.append("")

    if split:
        lines.append("## ⚠ 输出点云已分裂成多个位置簇的帧")
        lines.append("")
        lines.append("多簇不一定是干扰 —— 高台阶本身会让条纹跳变。请对照 `cluster_gap` 与实际台阶高度判断。")
        lines.append("")
        lines.append("| 文件 | 簇数 | 各簇位置×点数 |")
        lines.append("|---|---:|---|")
        for item in split:
            lines.append(
                f"| `{Path(item.file).name}` | {item.output_cluster_count} | "
                f"{item.output_cluster_ranges} |"
            )
        lines.append("")

    if lossy:
        lines.append("## 有带外干扰、当前表现为静默丢点的帧")
        lines.append("")
        lines.append("| 文件 | 带外列数 | 最长连续段 | 段范围 | 输出点数 |")
        lines.append("|---|---:|---:|---|---:|")
        for item in lossy:
            lines.append(
                f"| `{Path(item.file).name}` | {item.outside_columns} | "
                f"{item.outside_longest_run} | {item.outside_longest_run_range} | "
                f"{item.extracted_points} |"
            )
        lines.append("")

    if failed:
        lines.append("## 处理失败的帧")
        lines.append("")
        lines.append("| 文件 | 状态 | 错误 |")
        lines.append("|---|---|---|")
        for item in failed:
            lines.append(f"| `{Path(item.file).name}` | {item.status} | {item.error} |")
        lines.append("")

    if not risky and not lossy and not failed:
        lines.append("## 结论")
        lines.append("")
        lines.append("本批数据未检出带外干扰。整列 `argmax` 的缺陷在这批数据上未被触发，")
        lines.append("但缺陷本身仍然存在 —— 换工件、换姿态、换曝光都可能触发，修复仍应执行。")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量扫描线激光图像的带外干扰风险",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs", type=Path, nargs="+", help="图像文件或目录（可给多个）"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="目录递归查找图像"
    )
    parser.add_argument(
        "--method",
        default="centroid",
        choices=sorted(k for k, v in AVAILABLE_METHODS.items() if v is not None),
        help="提取算法",
    )
    parser.add_argument(
        "--background-kernel", type=positive_int, default=51, help="背景抑制核（奇数）"
    )
    parser.add_argument(
        "--min-local-contrast-dn", type=float, default=20.0, help="峰值最低局部对比度"
    )
    parser.add_argument(
        "--segment-min-columns",
        type=positive_int,
        default=42,
        help="伪段判定阈值；应与实际使用的提取配置一致",
    )
    parser.add_argument(
        "--continuity-max-column-gap", type=positive_int, default=2, help="连续段允许的最大间隔"
    )
    parser.add_argument(
        "--scan-axis", choices=("column", "row"), default="column", help="扫描轴"
    )
    parser.add_argument(
        "--cluster-gap",
        type=positive_int,
        default=200,
        help="簇切分间隔（像素）。应大于条纹自身的合法跨度（含台阶跳变），小于干扰与条纹的距离",
    )
    parser.add_argument("--csv", type=Path, help="逐帧结果 CSV 输出路径")
    parser.add_argument("--report", type=Path, help="汇总 Markdown 报告输出路径")
    parser.add_argument("--json", type=Path, help="完整结果 JSON 输出路径")
    parser.add_argument("--quiet", action="store_true", help="不打印逐帧进度")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    images = collect_images(args.inputs, args.recursive)
    if not images:
        raise SystemExit("未找到任何图像")

    options: dict[str, Any] = {
        "background_kernel": args.background_kernel,
        "min_local_contrast_dn": args.min_local_contrast_dn,
        "segment_min_columns": args.segment_min_columns,
        "continuity_max_column_gap": args.continuity_max_column_gap,
        "scan_axis": args.scan_axis,
    }
    params = CentroidParams(**options)
    backend = AVAILABLE_METHODS[args.method]
    if backend is None:
        raise SystemExit(f"提取算法 {args.method!r} 尚未接入")

    print(f"扫描 {len(images)} 帧，cluster_gap={args.cluster_gap} px …")
    reports: list[FrameReport] = []
    for index, path in enumerate(images, start=1):
        report = analyse_frame(path, params, backend, options, args.cluster_gap)
        reports.append(report)
        if not args.quiet:
            flag = (
                "⚠ 伪段风险"
                if report.fake_segment_risk
                else ("· 静默丢点" if report.outside_columns else "  正常")
            )
            print(
                f"[{index:>4}/{len(images)}] {flag}  {path.name}  "
                f"点数={report.extracted_points}  带外列={report.outside_columns}  "
                f"最长段={report.outside_longest_run}  簇数={report.cluster_count}"
                + (f"  [{report.status}: {report.error}]" if report.status != "ok" else "")
            )

    ok = [item for item in reports if item.status == "ok"]
    risky = [item for item in ok if item.fake_segment_risk]
    lossy = [item for item in ok if item.outside_columns and not item.fake_segment_risk]
    split = [item for item in ok if item.output_split_detected]
    failed = [item for item in reports if item.status != "ok"]

    print("-" * 78)
    print(f"总计 {len(reports)} 帧：成功 {len(ok)}，失败 {len(failed)}")
    print(f"  ⚠ fake_segment_risk（会产出伪点云）：{len(risky)}")
    print(f"  · 有带外干扰但表现为静默丢点：       {len(lossy)}")
    print(f"    输出点云已分裂成多簇：             {len(split)}")
    if risky:
        print("  伪段风险帧：")
        for item in risky:
            print(
                f"    {Path(item.file).name}：干扰簇 {item.outside_longest_run_cluster_range}"
                f"，最长连续段 {item.outside_longest_run} 列 "
                f"{item.outside_longest_run_range}"
            )
    print("-" * 78)

    if args.csv:
        write_csv(args.csv, reports)
        print(f"逐帧 CSV → {args.csv}")
    if args.report:
        write_markdown(args.report, reports, params, args.cluster_gap)
        print(f"汇总报告 → {args.report}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "options": options,
                    "cluster_gap": args.cluster_gap,
                    "frames": [item.as_row() for item in reports],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"完整 JSON → {args.json}")

    return 1 if risky else 0


if __name__ == "__main__":
    raise SystemExit(main())
