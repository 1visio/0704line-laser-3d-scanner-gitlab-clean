from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from calibration.manifest import CalibrationManifestError, load_calibration_package


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "configs" / "calibration"


class CalibrationManifestTests(unittest.TestCase):
    def test_runtime_package_loads_and_identifies_camera(self) -> None:
        package = load_calibration_package(PACKAGE_DIR / "manifest.yaml")
        self.assertEqual(package.camera_model, "MV-CS050-60GM")
        self.assertEqual((package.image_width, package.image_height), (2448, 2048))
        self.assertEqual(package.algorithm, "shared_steger")
        self.assertEqual(package.calibration["K"].shape, (3, 3))
        self.assertEqual(len(package.manifest_sha256), 64)

    def test_modified_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "calibration"
            shutil.copytree(PACKAGE_DIR, destination)
            with (destination / "circular_cone.yaml").open("a", encoding="utf-8") as stream:
                stream.write("\n# modified\n")
            with self.assertRaisesRegex(CalibrationManifestError, "哈希不匹配"):
                load_calibration_package(destination / "manifest.yaml")

    def test_crlf_checkout_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "calibration"
            shutil.copytree(PACKAGE_DIR, destination)
            manifest = yaml.safe_load(
                (destination / "manifest.yaml").read_text(encoding="utf-8")
            )
            for entry in manifest["files"].values():
                if entry is None:
                    continue
                path = destination / entry["path"]
                lf_data = path.read_bytes().replace(b"\r\n", b"\n")
                path.write_bytes(lf_data.replace(b"\n", b"\r\n"))

            package = load_calibration_package(destination / "manifest.yaml")
            self.assertEqual(package.camera_model, "MV-CS050-60GM")

    def test_realtime_steger_algorithm_name_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "calibration"
            shutil.copytree(PACKAGE_DIR, destination)
            path = destination / "manifest.yaml"
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            document["extractor"]["algorithm"] = "steger"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            package = load_calibration_package(path)
            self.assertEqual(package.algorithm, "steger")

    def test_null_ground_compensation_is_accepted_for_smoke_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "calibration"
            shutil.copytree(PACKAGE_DIR, destination)
            path = destination / "manifest.yaml"
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            document["files"]["ground_u_compensation"] = None
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            package = load_calibration_package(path)
            self.assertIsNone(package.calibration["ground_u_compensation"])

    def test_parent_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "calibration"
            shutil.copytree(PACKAGE_DIR, destination)
            path = destination / "manifest.yaml"
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            document["files"]["intrinsics"]["path"] = "../outside.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            with self.assertRaisesRegex(CalibrationManifestError, "包内相对路径"):
                load_calibration_package(path)


if __name__ == "__main__":
    unittest.main()
