"""2D section view derived from reconstructed ground-coordinate points."""

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class SectionView(QWidget):
    """Show distance along the laser line versus ground height Zg."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._axes = self._figure.add_subplot(111)
        self._colorbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)
        self.clear()

    def clear(self) -> None:
        self._figure.clear()
        self._colorbar = None
        self._axes = self._figure.add_subplot(111)
        self._format_axes()
        self._canvas.draw_idle()

    def set_points(self, points_ground: np.ndarray) -> None:
        points = np.asarray(points_ground, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Section points must have shape (N, 3)")
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if len(points) < 2:
            raise ValueError("Section view needs at least two finite points")

        s = _section_distance(points[:, :2])
        z = points[:, 2]
        order = np.argsort(s)
        s = s[order]
        z = z[order]

        self._figure.clear()
        self._colorbar = None
        self._axes = self._figure.add_subplot(111)
        scatter = self._axes.scatter(
            s,
            z,
            c=z,
            cmap="turbo",
            s=8,
            linewidths=0.0,
        )
        self._axes.axhline(0.0, color="0.2", linewidth=1.0, linestyle="--")
        self._axes.set_title(f"Section view ({len(points)} points)")
        self._format_axes()
        self._set_limits(s, z)
        self._colorbar = self._figure.colorbar(scatter, ax=self._axes, pad=0.02)
        self._colorbar.set_label("Zg (mm)")
        self._canvas.draw_idle()

    def _format_axes(self) -> None:
        self._axes.set_xlabel("S along laser line (mm)")
        self._axes.set_ylabel("Zg (mm)")
        self._axes.grid(True, alpha=0.35)
        self._axes.set_aspect("equal", adjustable="datalim")

    def _set_limits(self, s: np.ndarray, z: np.ndarray) -> None:
        self._set_axis_limits(self._axes.set_xlim, s, minimum_span=1.0)
        self._set_axis_limits(self._axes.set_ylim, z, minimum_span=0.1)

    @staticmethod
    def _set_axis_limits(setter, values: np.ndarray, *, minimum_span: float) -> None:
        minimum = float(values.min())
        maximum = float(values.max())
        centre = (minimum + maximum) * 0.5
        span = max(maximum - minimum, minimum_span)
        padding = span * 0.08
        half = span * 0.5 + padding
        setter(centre - half, centre + half)


def _section_distance(points_xy: np.ndarray) -> np.ndarray:
    centre = np.mean(points_xy, axis=0)
    centred = points_xy - centre
    _, singular_values, right_vectors = np.linalg.svd(
        centred,
        full_matrices=False,
    )
    if singular_values[0] <= np.finfo(np.float64).eps:
        raise ValueError("Section direction is degenerate")
    direction = right_vectors[0]
    if direction[0] < 0.0 or (direction[0] == 0.0 and direction[1] < 0.0):
        direction = -direction
    distance = centred @ direction
    return np.ascontiguousarray(distance - float(distance.min()))
