"""Standalone Stage-1 offline scan command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app_config import (
    AppConfigError,
    DEFAULT_CONFIG_PATH,
    load_app_config,
)
from scan.config import ScanConfigError, load_scan_config
from scan.session import ScanSessionError, write_scan_session


DEFAULT_SCAN_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "scan_stage1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-1 线激光离线俯仰扫描程序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="现有单帧测量配置 YAML",
    )
    parser.add_argument(
        "--scan-config",
        type=Path,
        default=DEFAULT_SCAN_CONFIG_PATH,
        help="Stage-1 扫描配置 YAML",
    )
    parser.add_argument(
        "--mode",
        choices=("repeat-one", "sequence"),
        required=True,
        help="repeat-one 使用单图重复角度；sequence 使用图像目录和 pose CSV",
    )
    parser.add_argument("--image", type=Path, help="repeat-one 模式的单张激光图像")
    parser.add_argument("--frames", type=Path, help="sequence 模式的图像目录")
    parser.add_argument("--poses", type=Path, help="sequence 模式的 frame-angle CSV")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_mode_arguments(parser, args)
    return args


def main(argv: list[str] | None = None) -> int:
    """Run one offline scan and return a process exit code."""
    args = parse_args(argv)
    try:
        app_config = load_app_config(args.config)
        scan_config = load_scan_config(args.scan_config)

        from online.pipeline import FramePipeline
        from scan.offline_scan import run_repeat_one, run_sequence

        pipeline = FramePipeline(app_config)
        common_kwargs = {
            "axis_point_scan_mm": scan_config.kinematics.axis_point_scan_mm,
            "axis_direction_scan": scan_config.kinematics.axis_direction_scan,
            "zero_offset_deg": scan_config.kinematics.zero_offset_deg,
            "T_scan_from_camera_zero": scan_config.kinematics.T_scan_from_camera_zero,
        }
        if args.mode == "repeat-one":
            result = run_repeat_one(
                args.image,
                scan_config.trajectory.angles_deg,
                pipeline,
                **common_kwargs,
            )
        else:
            result = run_sequence(
                args.frames,
                args.poses,
                pipeline,
                **common_kwargs,
            )

        output_directory = write_scan_session(
            result,
            scan_config.output.directory,
            scan_config=scan_config,
            scan_config_path=args.scan_config,
            measure_tool_config_path=args.config,
            mode="repeat_one" if args.mode == "repeat-one" else "sequence",
            source_image=args.image if args.mode == "repeat-one" else None,
            frames_directory=args.frames if args.mode == "sequence" else None,
            pipeline=pipeline,
        )
        print(f"扫描结果已写入: {output_directory}")
        _print_scan_summary(result, output_directory)
        return 0
    except (
        AppConfigError,
        ScanConfigError,
        ScanSessionError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"scan_offline: {error}", file=sys.stderr)
        return 1


def _validate_mode_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.mode == "repeat-one":
        if args.image is None:
            parser.error("repeat-one 模式必须提供 --image")
        if args.frames is not None or args.poses is not None:
            parser.error("repeat-one 模式只接受 --image，不应提供 --frames/--poses")
        return
    if args.image is not None:
        parser.error("sequence 模式不应提供 --image")
    if args.frames is None or args.poses is None:
        parser.error("sequence 模式必须同时提供 --frames 和 --poses")


def _print_scan_summary(result: object, output_directory: Path) -> None:
    """打印可供首次真实单帧验证直接核对的扫描摘要。"""
    metadata = getattr(result, "metadata", {})
    profiles = tuple(getattr(result, "profiles", ()))
    points_scan = getattr(result, "points_scan", ())
    mode = str(metadata.get("mode", ""))
    display_mode = "repeat-one" if mode == "repeat_one" else mode
    print(f"Mode: {display_mode}")
    print(f"Frames: {len(profiles)}")
    if profiles:
        print(
            f"Angles: {profiles[0].angle_deg:+.1f} ... "
            f"{profiles[-1].angle_deg:+.1f} deg"
        )
    else:
        print("Angles: — ... — deg")
    print(f"Points: {len(points_scan)}")
    print("Coordinate system: scan")
    print("Units: mm")
    demo_only = bool(metadata.get("kinematic_demo_only", mode == "repeat_one"))
    print(f"Kinematic demo only: {'YES' if demo_only else 'NO'}")
    pcd_path = output_directory / "cloud_scan.pcd"
    ply_path = output_directory / "cloud_scan.ply"
    print(f"PCD: {pcd_path if pcd_path.is_file() else 'not generated'}")
    print(f"PLY: {ply_path if ply_path.is_file() else 'not generated'}")

    warnings = metadata.get("warnings", ())
    if isinstance(warnings, (list, tuple)):
        for warning in warnings:
            if isinstance(warning, dict):
                frame_index = warning.get("frame_index", "?")
                message = warning.get("message", "扫描帧存在问题")
                print(f"WARNING frame_index={frame_index}: {message}")


if __name__ == "__main__":
    raise SystemExit(main())
