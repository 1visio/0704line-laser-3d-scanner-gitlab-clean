"""Interactive 3D point-cloud view for reconstructed ground coordinates."""

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from utils.pointcloud_colors import ZG_HIGH_CONTRAST_CMAP


class PointCloudView(QWidget):
    """Show ground-coordinate points with Zg mapped to color."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._colorbar = None
        self._axes = self._figure.add_subplot(111, projection="3d")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)
        self.clear()

    def clear(self) -> None:
        self._figure.clear()
        self._colorbar = None
        self._axes = self._figure.add_subplot(111, projection="3d")
        self._format_axes()
        self._canvas.draw_idle()

    def set_points(self, points_ground: np.ndarray) -> None:
        points = np.asarray(points_ground, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("3D point cloud must have shape (N, 3)")
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if len(points) == 0:
            raise ValueError("3D point cloud is empty")

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        self._figure.clear()
        self._colorbar = None
        self._axes = self._figure.add_subplot(111, projection="3d")
        scatter = self._axes.scatter(
            x,
            y,
            z,
            c=z,
            cmap=ZG_HIGH_CONTRAST_CMAP,
            s=6,
            linewidths=0.0,
            depthshade=False,
        )
        self._axes.set_title(f"Ground point cloud ({len(points)} points)")
        self._format_axes()
        self._set_data_limits(x, y, z)
        self._colorbar = self._figure.colorbar(
            scatter,
            ax=self._axes,
            shrink=0.72,
            pad=0.08,
        )
        self._colorbar.set_label("Zg (mm)")
        self._canvas.draw_idle()

    def _format_axes(self) -> None:
        self._axes.set_xlabel("Xg (mm)")
        self._axes.set_ylabel("Yg (mm)")
        self._axes.set_zlabel("Zg (mm)")
        self._axes.grid(True)
        self._axes.view_init(elev=24.0, azim=-62.0)
        self._axes.set_box_aspect((1.0, 1.0, 1.0))

    def _set_data_limits(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> None:
        x_centre, x_span = self._axis_centre_span(x, minimum_span=1.0)
        y_centre, y_span = self._axis_centre_span(y, minimum_span=1.0)
        z_centre, z_span = self._axis_centre_span(z, minimum_span=1.0)
        span = max(x_span, y_span, z_span)
        half = span * 0.5
        self._axes.set_xlim(x_centre - half, x_centre + half)
        self._axes.set_ylim(y_centre - half, y_centre + half)
        self._axes.set_zlim(z_centre - half, z_centre + half)

    @staticmethod
    def _axis_centre_span(
        values: np.ndarray, *, minimum_span: float
    ) -> tuple[float, float]:
        minimum = float(values.min())
        maximum = float(values.max())
        centre = (minimum + maximum) * 0.5
        span = max(maximum - minimum, minimum_span)
        padding = span * 0.08
        return centre, span + padding * 2.0
