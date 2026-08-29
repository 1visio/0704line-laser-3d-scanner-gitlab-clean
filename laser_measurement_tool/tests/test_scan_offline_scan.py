"""Tests for the offline scan orchestration."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from laser.laser_extractor import LaserExtractionError
from scan.offline_scan import OfflineScanRunner, run_offline_scan


class _FakeFrameResult:
    def __init__(self, frame_index: int) -> None:
        self.centers_uv_full = np.array([[10.0, 20.0]])
        self.points_camera = np.array([[1.0, 0.0, 0.0]])
        self._frame_index = frame_index

    @property
    def points_ground(self) -> np.ndarray:
        raise AssertionError("offline scan 不应读取 points_ground")


class _FakePipeline:
    def __init__(self) -> None:
        self.frames = []

    def run_frame(self, frame):
        self.frames.append(frame)
        return _FakeFrameResult(frame.camera_frame_number)


class _SizedFakePipeline(_FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.package = SimpleNamespace(image_width=10, image_height=8)


class _EmptyPipeline(_FakePipeline):
    def run_frame(self, frame):
        self.frames.append(frame)
        result = _FakeFrameResult(frame.camera_frame_number)
        result.centers_uv_full = np.empty((0, 2), dtype=np.float64)
        result.points_camera = np.empty((0, 3), dtype=np.float64)
        return result


class _FailingPipeline(_FakePipeline):
    def run_frame(self, frame):
        self.frames.append(frame)
        raise LaserExtractionError("test backend failure")


def _write_png(path: Path, value: int) -> None:
    image = np.full((4, 5), value, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise AssertionError("test image encoding failed")
    path.write_bytes(encoded.tobytes())


class OfflineScanRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.axis_point = np.zeros(3, dtype=np.float64)
        self.axis_direction = np.array([0.0, 0.0, 1.0])
        self.zero_offset = 0.0
        self.transform = np.eye(4, dtype=np.float64)

    def test_repeat_one_reuses_image_and_marks_demo_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            image_path = Path(directory_name) / "laser.png"
            _write_png(image_path, 80)
            pipeline = _FakePipeline()
            runner = OfflineScanRunner(
                pipeline,
                self.axis_point,
                self.axis_direction,
                self.zero_offset,
                self.transform,
            )

            output = runner.run_repeat_one(image_path, [0.0, 90.0])

        self.assertTrue(output.metadata["kinematic_demo_only"])
        self.assertEqual(output.metadata["mode"], "repeat_one")
        self.assertEqual(output.metadata["frame_count"], 2)
        self.assertEqual(output.profiles[0].frame_index, 0)
        self.assertEqual(output.profiles[1].frame_index, 1)
        np.testing.assert_allclose(
            output.points_scan,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            atol=1e-12,
        )
        self.assertEqual([frame.camera_frame_number for frame in pipeline.frames], [0, 1])
        self.assertEqual([int(frame.image[0, 0]) for frame in pipeline.frames], [80, 80])

    def test_sequence_binds_sorted_images_to_csv_rows_and_uses_camera_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            image_dir = directory / "images"
            image_dir.mkdir()
            _write_png(image_dir / "b.png", 22)
            _write_png(image_dir / "a.png", 11)
            csv_path = directory / "angles.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(("frame_index", "angle_deg"))
                writer.writerow((10, 0.0))
                writer.writerow((11, 90.0))

            pipeline = _FakePipeline()
            output = run_offline_scan(
                "sequence",
                pipeline,
                axis_point_scan_mm=self.axis_point,
                axis_direction_scan=self.axis_direction,
                zero_offset_deg=self.zero_offset,
                T_scan_from_camera_zero=self.transform,
                image_dir=image_dir,
                frame_angle_csv=csv_path,
            )

        self.assertFalse(output.metadata["kinematic_demo_only"])
        self.assertEqual(output.profiles[0].frame_index, 10)
        self.assertEqual(output.profiles[1].frame_index, 11)
        self.assertEqual([int(frame.image[0, 0]) for frame in pipeline.frames], [11, 22])
        np.testing.assert_allclose(
            output.points_scan,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            atol=1e-12,
        )

    def test_sequence_accepts_explicit_image_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            image_dir = directory / "images"
            image_dir.mkdir()
            _write_png(image_dir / "first.png", 31)
            _write_png(image_dir / "second.png", 32)
            csv_path = directory / "angles.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(("frame_index", "image", "angle_deg"))
                writer.writerow((5, "second.png", 0.0))
                writer.writerow((6, "first.png", 0.0))

            pipeline = _FakePipeline()
            runner = OfflineScanRunner(
                pipeline,
                self.axis_point,
                self.axis_direction,
                self.zero_offset,
                self.transform,
            )
            runner.run_sequence(image_dir, csv_path)

        self.assertEqual([int(frame.image[0, 0]) for frame in pipeline.frames], [32, 31])

    def test_unknown_mode_is_rejected(self) -> None:
        pipeline = _FakePipeline()
        with self.assertRaises(ValueError):
            run_offline_scan(
                "other",
                pipeline,
                axis_point_scan_mm=self.axis_point,
                axis_direction_scan=self.axis_direction,
                zero_offset_deg=self.zero_offset,
                T_scan_from_camera_zero=self.transform,
            )

    def test_repeat_one_uses_recorded_hardware_roi_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            image_path = directory / "laser.png"
            _write_png(image_path, 80)
            (directory / "frames.csv").write_text(
                "filename,offset_x,offset_y\nlaser.png,2,3\n",
                encoding="utf-8",
            )
            pipeline = _SizedFakePipeline()
            runner = OfflineScanRunner(
                pipeline,
                self.axis_point,
                self.axis_direction,
                self.zero_offset,
                self.transform,
            )
            output = runner.run_repeat_one(image_path, [0.0, 1.0])

        self.assertEqual(
            [(frame.offset_x, frame.offset_y) for frame in pipeline.frames],
            [(2, 3), (2, 3)],
        )
        self.assertEqual(output.metadata["image_offset"], {"offset_x": 2, "offset_y": 3})

    def test_zero_points_are_kept_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            image_path = Path(directory_name) / "laser.png"
            _write_png(image_path, 80)
            output = OfflineScanRunner(
                _EmptyPipeline(),
                self.axis_point,
                self.axis_direction,
                self.zero_offset,
                self.transform,
            ).run_repeat_one(image_path, [0.0])

        self.assertEqual(output.metadata["frame_stats"][0]["valid_laser_points"], 0)
        self.assertEqual(output.metadata["frame_stats"][0]["points_camera_count"], 0)
        self.assertEqual(output.metadata["frame_stats"][0]["points_scan_count"], 0)
        self.assertEqual(output.metadata["warnings"][0]["frame_index"], 0)

    def test_extraction_failure_keeps_frame_and_reports_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            image_path = Path(directory_name) / "laser.png"
            _write_png(image_path, 80)
            output = OfflineScanRunner(
                _FailingPipeline(),
                self.axis_point,
                self.axis_direction,
                self.zero_offset,
                self.transform,
            ).run_repeat_one(image_path, [0.0])

        self.assertEqual(len(output.profiles), 1)
        self.assertEqual(output.profiles[0].points_scan.shape, (0, 3))
        self.assertIn("中心提取失败", output.metadata["warnings"][0]["message"])


if __name__ == "__main__":
    unittest.main()
