"""Tests for the standalone offline scan CLI parser."""

import unittest
from pathlib import Path

from scan_offline import build_parser, parse_args


class ScanCliParserTests(unittest.TestCase):
    def test_repeat_one_arguments_are_parsed(self) -> None:
        args = parse_args(
            [
                "--config",
                "configs/measure_tool.yaml",
                "--scan-config",
                "configs/scan_stage1.yaml",
                "--mode",
                "repeat-one",
                "--image",
                "laser.png",
            ]
        )

        self.assertEqual(args.mode, "repeat-one")
        self.assertEqual(args.image, Path("laser.png"))
        self.assertEqual(args.frames, None)
        self.assertEqual(args.poses, None)

    def test_sequence_arguments_are_parsed(self) -> None:
        args = parse_args(
            [
                "--mode",
                "sequence",
                "--frames",
                "frames",
                "--poses",
                "poses.csv",
            ]
        )

        self.assertEqual(args.mode, "sequence")
        self.assertEqual(args.frames, Path("frames"))
        self.assertEqual(args.poses, Path("poses.csv"))
        self.assertIsNone(args.image)

    def test_repeat_one_requires_image(self) -> None:
        with self.assertRaises(SystemExit) as context:
            parse_args(["--mode", "repeat-one"])

        self.assertEqual(context.exception.code, 2)

    def test_sequence_requires_frames_and_poses(self) -> None:
        with self.assertRaises(SystemExit) as context:
            parse_args(["--mode", "sequence", "--frames", "frames"])

        self.assertEqual(context.exception.code, 2)

    def test_help_can_build_parser_without_a_camera(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit) as context:
            parser.parse_args(["--help"])

        self.assertEqual(context.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
