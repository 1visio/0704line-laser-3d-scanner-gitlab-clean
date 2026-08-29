"""Tests for GUI-independent hardware ROI metadata parsing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.image_metadata import read_image_offset_metadata


class ImageMetadataTests(unittest.TestCase):
    def test_sidecar_and_result_json_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            image_path = directory / "frame.tiff"
            image_path.write_bytes(b"placeholder")
            image_path.with_suffix(".json").write_text(
                json.dumps({"image_offset": {"u": 4, "v": 9}}),
                encoding="utf-8",
            )
            self.assertEqual(read_image_offset_metadata(image_path), (4, 9))

    def test_frames_csv_offset_matches_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            image_path = directory / "frame.tiff"
            image_path.write_bytes(b"placeholder")
            (directory / "frames.csv").write_text(
                "filename,offset_x,offset_y\nother.tiff,1,2\nframe.tiff,3,8\n",
                encoding="utf-8",
            )
            self.assertEqual(read_image_offset_metadata(image_path), (3, 8))


if __name__ == "__main__":
    unittest.main()
