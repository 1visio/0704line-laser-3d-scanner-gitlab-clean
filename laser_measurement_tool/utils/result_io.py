"""测量结果文件的路径生成与保存。"""

import csv
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .pointcloud_colors import map_zg_to_rgb


def next_laser_center_csv_path(
    image_path: str | Path,
    output_directory: str | Path,
) -> Path:
    """生成不会覆盖已有结果的激光中心 CSV 路径。"""
    image_stem = Path(image_path).stem
    output_path = Path(output_directory)
    base_name = f"{image_stem}_laser_center"

    candidate = output_path / f"{base_name}.csv"
    index = 1
    while candidate.exists():
        candidate = output_path / f"{base_name}_{index:03d}.csv"
        index += 1
    return candidate


def save_laser_centers_csv(
    file_path: str | Path,
    centers: ArrayLike,
) -> Path:
    """以 ``u,v`` 列保存亚像素激光中心点，并拒绝覆盖已有文件。"""
    points = np.asarray(centers, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("中心点必须是形状为 (N, 2) 的数组")
    if not np.isfinite(points).all():
        raise ValueError("中心点包含 NaN 或无穷值")

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("u", "v"))
        writer.writerows((f"{u:.6f}", f"{v:.6f}") for u, v in points)
    return path


def next_measurement_dir(
    image_path: str | Path,
    output_directory: str | Path,
) -> Path:
    """生成不会覆盖已有结果的测量输出目录（不实际创建）。"""
    image_stem = Path(image_path).stem
    output_path = Path(output_directory)
    base_name = f"{image_stem}_measure"

    candidate = output_path / base_name
    index = 1
    while candidate.exists():
        candidate = output_path / f"{base_name}_{index:03d}"
        index += 1
    return candidate


def save_measurement_json(file_path: str | Path, payload: dict) -> Path:
    """把测量结果字典保存为 UTF-8 JSON，拒绝覆盖已有文件。"""
    import json

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")
    return path


def save_reconstructed_points_csv(
    file_path: str | Path,
    pixels_uv: ArrayLike,
    points_camera: ArrayLike,
    points_ground: ArrayLike,
) -> Path:
    """保存重建点云：u,v,Xc,Yc,Zc,Xg,Yg,Zg（mm），拒绝覆盖已有文件。"""
    uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    camera = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    ground = np.asarray(points_ground, dtype=np.float64).reshape(-1, 3)
    if not (len(uv) == len(camera) == len(ground)):
        raise ValueError("像素、相机系与地面系点数量必须一致")

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([uv, camera, ground])
    with path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ("u", "v", "Xc_mm", "Yc_mm", "Zc_mm", "Xg_mm", "Yg_mm", "Zg_mm")
        )
        writer.writerows(tuple(f"{value:.6f}" for value in row) for row in rows)
    return path


def save_ground_pointcloud_ply(
    file_path: str | Path,
    points_ground: ArrayLike,
) -> Path:
    """保存地面系 XYZ 与由 Zg 映射得到的 RGB 颜色，拒绝覆盖。"""
    points = np.asarray(points_ground, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("地面系点云必须是形状为 (N, 3) 的数组")
    if not np.isfinite(points).all():
        raise ValueError("地面系点云包含 NaN 或无穷值")

    rgb, zg_min, zg_max = map_zg_to_rgb(points[:, 2])

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("comment coordinate_system ground\n")
        ply_file.write("comment units millimeter\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property double x\n")
        ply_file.write("property double y\n")
        ply_file.write("property double z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("comment color_mapping zg_high_contrast_from_zg\n")
        ply_file.write(f"comment color_zg_range_mm {zg_min:.9f} {zg_max:.9f}\n")
        ply_file.write("end_header\n")
        for (xg, yg, zg_value), (red, green, blue) in zip(points, rgb):
            ply_file.write(
                f"{xg:.9f} {yg:.9f} {zg_value:.9f} "
                f"{red:d} {green:d} {blue:d}\n"
            )
    return path


def save_image_png(file_path: str | Path, image_bgr: np.ndarray) -> Path:
    """把 BGR 图像保存为 PNG；用 imencode 以兼容含中文的路径。"""
    import cv2

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("PNG 编码失败")
    if path.exists():
        raise FileExistsError(f"文件已存在: {path}")
    encoded.tofile(str(path))
    return path
