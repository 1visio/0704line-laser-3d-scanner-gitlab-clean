"""Standalone online line-laser acquisition and reconstruction app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from app_config import DEFAULT_CONFIG_PATH, AppConfigError, load_app_config
from laser.backends import AVAILABLE_METHODS
from online.camera_backend import available_camera_backends


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在线线激光三维截面程序")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--camera-backend",
        choices=available_camera_backends(),
        default="mvs",
        help="相机 backend；默认 mvs，也可选择 daheng（Galaxy USB3）",
    )
    parser.add_argument(
        "--method",
        choices=tuple(AVAILABLE_METHODS),
        help="覆盖配置文件中的在线激光提取算法",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="使用合成相机验证界面、算法和录制，不加载真实相机 SDK",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication(sys.argv)
    try:
        config = load_app_config(args.config)
        from online.window import OnlineCameraWindow

        window = OnlineCameraWindow(
            config,
            simulate=args.simulate,
            camera_backend=args.camera_backend,
            extraction_method=args.method,
        )
    except (AppConfigError, RuntimeError, ValueError, OSError) as error:
        QMessageBox.critical(None, "在线程序启动失败", str(error))
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
