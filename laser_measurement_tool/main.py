"""单帧线激光三维截面测量工具的应用入口。

用法：
    python main.py                          # 使用默认配置 configs/measure_tool.yaml
    python main.py --config <某个.yaml>     # 使用指定配置（换标定/换参数）

配置字段说明见 docs/USAGE_CONFIG.md。
"""

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from app_config import DEFAULT_CONFIG_PATH, AppConfigError, load_app_config
from gui.main_window import MainWindow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单帧线激光高度测量工具")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"统一配置 YAML（默认：{DEFAULT_CONFIG_PATH}）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """创建并启动桌面应用。"""
    args = parse_args(argv)
    app = QApplication(sys.argv)

    config = None
    config_error: str | None = None
    try:
        config = load_app_config(args.config)
    except AppConfigError as error:
        config_error = str(error)

    window = MainWindow(config)
    window.show()
    if config_error is not None:
        QMessageBox.warning(
            window,
            "配置加载失败",
            f"{config_error}\n\n工具仍可加载图像与提取激光线，"
            "但三维恢复需要有效配置。",
        )
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
