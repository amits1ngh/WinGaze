#!/usr/bin/env python3
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from PyQt5.QtWidgets import QApplication
except ImportError as exc:
    raise ImportError(
        "PyQt5 module is not installed. Please install it using 'pip install PyQt5'."
    ) from exc

from config.settings import AppConfig
from ui.main_window import ELANVideoPlayer
from vis.rerun_logger import RerunLogger


def main() -> None:
    config = AppConfig()
    rerun_logger = RerunLogger(
        app_name=config.rerun.app_name,
        spawn=config.rerun.spawn,
        base_path=config.rerun.base_path,
        rolling_window_seconds=config.rerun.rolling_window_seconds,
        history_window_seconds=config.rerun.history_window_seconds,
        history_clear_interval_seconds=config.rerun.history_clear_interval_seconds,
        history_max_points=config.rerun.history_max_points,
        metric_sample_rate_hz=config.rerun.metric_sample_rate_hz,
        velocity_range=config.rerun.velocity_range,
        mean_x_range=config.rerun.mean_x_range,
        mean_y_range=config.rerun.mean_y_range,
        range_padding=config.rerun.range_padding,
        range_percentiles=config.rerun.range_percentiles,
    )


    app = QApplication(sys.argv)
    window = ELANVideoPlayer(config=config, rerun_logger=rerun_logger)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
