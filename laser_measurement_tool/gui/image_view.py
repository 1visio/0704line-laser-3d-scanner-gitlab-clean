"""支持缩放、平移与坐标转换的灰度图像视图。"""

from collections.abc import Sequence
from math import floor
from typing import Protocol

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainterPath,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)


class RoiDisplayRegion(Protocol):
    """图像视图绘制 ROI 所需的最小数据接口。"""

    kind: object
    left: float
    top: float
    right: float
    bottom: float


class ImageView(QGraphicsView):
    """显示灰度图像，并保持场景坐标与图像坐标一一对应。"""

    image_coordinates_changed = Signal(float, float)
    image_coordinates_cleared = Signal()
    roi_selected = Signal(str, QRectF)

    _ZOOM_BASE = 1.25
    _MIN_ZOOM_LEVEL = -8.0
    _MAX_ZOOM_LEVEL = 30.0
    _ROI_KINDS = frozenset({"baseline", "obstacle"})

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._graphics_scene = QGraphicsScene(self)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._laser_center_item: QGraphicsPathItem | None = None
        self._laser_center_cross_item: QGraphicsPathItem | None = None
        self._laser_center_count = 0
        self._measurement_items: list[QGraphicsPathItem] = []
        self._roi_items: list[QGraphicsRectItem] = []
        self._roi_label_items: list[QGraphicsSimpleTextItem] = []
        self._draft_roi_item: QGraphicsRectItem | None = None
        self._roi_start: QPointF | None = None
        self._image_width = 0
        self._image_height = 0
        self._zoom_level = 0.0
        self._pending_roi_kind: str | None = None

        self.setScene(self._graphics_scene)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    @property
    def has_image(self) -> bool:
        """返回当前是否已经设置图像。"""
        return self._pixmap_item is not None

    @property
    def pending_roi_kind(self) -> str | None:
        """返回等待后续 ROI 交互实现处理的区域类型。"""
        return self._pending_roi_kind

    @property
    def laser_center_count(self) -> int:
        """返回当前叠加显示的激光中心点数量。"""
        return self._laser_center_count

    @property
    def roi_region_count(self) -> int:
        """返回当前显示的永久 ROI 数量。"""
        return len(self._roi_items)

    def set_image(self, image: np.ndarray) -> None:
        """显示二维灰度数组，场景中的一个单位对应一个图像像素。"""
        if image.ndim != 2:
            raise ValueError("ImageView 只支持二维灰度图像")

        display_image = _to_uint8_display(image)
        height, width = display_image.shape
        bytes_per_line = display_image.strides[0]
        qimage = QImage(
            display_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale8,
        ).copy()

        self._laser_center_item = None
        self._laser_center_cross_item = None
        self._laser_center_count = 0
        self._measurement_items = []
        self._roi_items = []
        self._roi_label_items = []
        self._draft_roi_item = None
        self._roi_start = None
        self._graphics_scene.clear()
        self._pixmap_item = self._graphics_scene.addPixmap(QPixmap.fromImage(qimage))
        self._image_width = width
        self._image_height = height
        self._graphics_scene.setSceneRect(self._pixmap_item.boundingRect())
        self._pending_roi_kind = None
        self.viewport().unsetCursor()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.fit_image()

    def set_laser_centers(self, centers: np.ndarray) -> None:
        """叠加 ``(u, v)`` 中心点；显示位置采用 OpenCV 像素中心约定。"""
        if self._pixmap_item is None:
            raise RuntimeError("请先设置图像")

        points = np.asarray(centers, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("中心点必须是形状为 (N, 2) 的数组")
        if not np.isfinite(points).all():
            raise ValueError("中心点包含 NaN 或无穷值")

        self.clear_laser_centers()
        if len(points) == 0:
            return

        path = QPainterPath()
        cross_path = QPainterPath()
        for u, v in points:
            center = QPointF(float(u) + 0.5, float(v) + 0.5)
            path.addEllipse(center, 0.5, 0.5)
            cross_path.moveTo(center.x() - 0.35, center.y())
            cross_path.lineTo(center.x() + 0.35, center.y())
            cross_path.moveTo(center.x(), center.y() - 0.35)
            cross_path.lineTo(center.x(), center.y() + 0.35)

        pen = QPen()
        pen.setStyle(Qt.PenStyle.NoPen)
        self._laser_center_item = self._graphics_scene.addPath(
            path,
            pen,
            QBrush(QColor(0, 255, 80, 220)),
        )
        self._laser_center_item.setZValue(1.0)
        cross_pen = QPen(QColor(255, 0, 0))
        cross_pen.setWidthF(1.0)
        cross_pen.setCosmetic(True)
        self._laser_center_cross_item = self._graphics_scene.addPath(
            cross_path,
            cross_pen,
        )
        self._laser_center_cross_item.setZValue(1.1)
        self._laser_center_count = len(points)

    def clear_laser_centers(self) -> None:
        """移除当前激光中心点叠加层。"""
        if self._laser_center_item is not None:
            self._graphics_scene.removeItem(self._laser_center_item)
            self._laser_center_item = None
        if self._laser_center_cross_item is not None:
            self._graphics_scene.removeItem(self._laser_center_cross_item)
            self._laser_center_cross_item = None
        self._laser_center_count = 0

    def set_measurement_overlay(
        self,
        segments: Sequence[tuple[str, np.ndarray]],
    ) -> None:
        """叠加拟合线段；``segments`` 为 ``(kind, (2, 2) 像素端点)`` 序列。

        ``kind`` 为 ``"baseline"``（青色）或 ``"height"``（橙黄色）。
        端点采用 OpenCV 亚像素坐标，显示时加 0.5 与中心点叠加一致。
        """
        if self._pixmap_item is None:
            raise RuntimeError("请先设置图像")
        self.clear_measurement_overlay()

        colors = {
            "baseline": QColor(0, 210, 255),
            "height": QColor(255, 170, 0),
        }
        for kind, endpoints in segments:
            points = np.asarray(endpoints, dtype=np.float64)
            if points.shape != (2, 2) or not np.isfinite(points).all():
                raise ValueError("线段端点必须是有限的 (2, 2) 数组")
            path = QPainterPath(
                QPointF(points[0, 0] + 0.5, points[0, 1] + 0.5)
            )
            path.lineTo(QPointF(points[1, 0] + 0.5, points[1, 1] + 0.5))
            pen = QPen(colors.get(str(kind), QColor(255, 255, 255)))
            pen.setWidthF(2.5)
            pen.setCosmetic(True)
            item = self._graphics_scene.addPath(path, pen)
            item.setZValue(4.0)
            self._measurement_items.append(item)

    def clear_measurement_overlay(self) -> None:
        """移除全部测量拟合线叠加。"""
        for item in self._measurement_items:
            self._graphics_scene.removeItem(item)
        self._measurement_items.clear()

    @property
    def measurement_overlay_count(self) -> int:
        """返回当前叠加的拟合线段数量。"""
        return len(self._measurement_items)

    def set_roi_regions(self, regions: Sequence[RoiDisplayRegion]) -> None:
        """刷新蓝色基准 ROI 和红色障碍物 ROI。"""
        self.clear_roi_regions()
        if self._pixmap_item is None:
            return

        obstacle_index = 0
        for region in regions:
            kind = str(getattr(region.kind, "value", region.kind))
            if kind == "obstacle":
                obstacle_index += 1
            rectangle = QRectF(
                region.left,
                region.top,
                region.right - region.left,
                region.bottom - region.top,
            )
            item = self._add_roi_graphics_item(rectangle, kind, dashed=False)
            item.setZValue(2.0)
            self._roi_items.append(item)
            label_text = (
                f"障碍物 {obstacle_index}" if kind == "obstacle" else "基准"
            )
            label = self._graphics_scene.addSimpleText(label_text)
            label.setBrush(
                QBrush(
                    QColor(235, 60, 60)
                    if kind == "obstacle"
                    else QColor(40, 110, 255)
                )
            )
            font = label.font()
            font.setPointSize(10)
            font.setBold(True)
            label.setFont(font)
            label.setPos(rectangle.left(), rectangle.top())
            label.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                True,
            )
            label.setZValue(3.0)
            self._roi_label_items.append(label)

    def clear_roi_regions(self) -> None:
        """移除全部永久 ROI 图形。"""
        for item in self._roi_items:
            self._graphics_scene.removeItem(item)
        self._roi_items.clear()
        for label in self._roi_label_items:
            self._graphics_scene.removeItem(label)
        self._roi_label_items.clear()

    def fit_image(self) -> None:
        """将完整图像按比例适配到当前视口。"""
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 0.0

    def map_view_to_image(self, view_position: QPoint) -> QPointF | None:
        """把视口鼠标坐标转换为连续图像坐标。"""
        if self._pixmap_item is None:
            return None

        scene_position = self.mapToScene(view_position)
        image_position = self._pixmap_item.mapFromScene(scene_position)
        if not self._contains_image_position(image_position):
            return None
        return image_position

    def map_image_to_view(self, image_position: QPointF) -> QPoint | None:
        """把有效的连续图像坐标转换为视口坐标。"""
        if self._pixmap_item is None or not self._contains_image_position(image_position):
            return None
        scene_position = self._pixmap_item.mapToScene(image_position)
        return self.mapFromScene(scene_position)

    def image_pixel_at(self, view_position: QPoint) -> tuple[int, int] | None:
        """返回鼠标位置对应的整数像素索引 ``(x, y)``。"""
        image_position = self.map_view_to_image(view_position)
        if image_position is None:
            return None
        return floor(image_position.x()), floor(image_position.y())

    def begin_roi_selection(self, roi_kind: str) -> None:
        """进入一次性矩形 ROI 框选模式。"""
        if self._pixmap_item is None:
            raise RuntimeError("请先加载图像")
        if roi_kind not in self._ROI_KINDS:
            raise ValueError(f"未知 ROI 类型: {roi_kind}")
        self._remove_draft_roi()
        self._pending_roi_kind = roi_kind
        self._roi_start = None
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def cancel_roi_selection(self) -> None:
        """取消等待中的 ROI 选择。"""
        self._remove_draft_roi()
        self._pending_roi_kind = None
        self._roi_start = None
        self.viewport().unsetCursor()
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if self._pixmap_item is not None
            else QGraphicsView.DragMode.NoDrag
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """在 ROI 模式下记录矩形起点。"""
        if (
            self._pending_roi_kind is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            start = self.map_view_to_image(event.position().toPoint())
            if start is not None:
                self._roi_start = start
                rectangle = QRectF(start, start)
                self._draft_roi_item = self._add_roi_graphics_item(
                    rectangle,
                    self._pending_roi_kind,
                    dashed=True,
                )
                self._draft_roi_item.setZValue(3.0)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """完成矩形框选并发出图像坐标 ROI。"""
        if (
            self._pending_roi_kind is not None
            and self._roi_start is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            end = self._map_view_to_image_clamped(event.position().toPoint())
            rectangle = QRectF(self._roi_start, end).normalized()
            roi_kind = self._pending_roi_kind
            self.cancel_roi_selection()
            if rectangle.width() >= 1.0 and rectangle.height() >= 1.0:
                self.roi_selected.emit(roi_kind, rectangle)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """以鼠标位置为锚点缩放图像。"""
        if self._pixmap_item is None or event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return

        steps = event.angleDelta().y() / 120.0
        target_level = min(
            self._MAX_ZOOM_LEVEL,
            max(self._MIN_ZOOM_LEVEL, self._zoom_level + steps),
        )
        applied_steps = target_level - self._zoom_level
        if applied_steps == 0:
            event.accept()
            return

        anchor_before = self.mapToScene(event.position().toPoint())
        factor = self._ZOOM_BASE**applied_steps
        self.scale(factor, factor)
        anchor_after = self.mapToScene(event.position().toPoint())
        anchor_delta = anchor_after - anchor_before
        self.translate(anchor_delta.x(), anchor_delta.y())
        self._zoom_level = target_level
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """平移后报告光标对应的图像坐标。"""
        if self._pending_roi_kind is not None and self._roi_start is not None:
            current = self._map_view_to_image_clamped(event.position().toPoint())
            if self._draft_roi_item is not None:
                self._draft_roi_item.setRect(
                    QRectF(self._roi_start, current).normalized()
                )
            self.image_coordinates_changed.emit(current.x(), current.y())
            event.accept()
            return

        super().mouseMoveEvent(event)
        image_position = self.map_view_to_image(event.position().toPoint())
        if image_position is None:
            self.image_coordinates_cleared.emit()
            return
        self.image_coordinates_changed.emit(image_position.x(), image_position.y())

    def leaveEvent(self, event: QEvent) -> None:
        """光标离开视口时清除坐标显示。"""
        self.image_coordinates_cleared.emit()
        super().leaveEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """尚未手动缩放时持续保持整图适配。"""
        super().resizeEvent(event)
        if self._pixmap_item is not None and self._zoom_level == 0.0:
            self.fit_image()

    def _contains_image_position(self, position: QPointF) -> bool:
        return (
            0.0 <= position.x() < self._image_width
            and 0.0 <= position.y() < self._image_height
        )

    def _map_view_to_image_clamped(self, view_position: QPoint) -> QPointF:
        if self._pixmap_item is None:
            raise RuntimeError("请先设置图像")
        scene_position = self.mapToScene(view_position)
        image_position = self._pixmap_item.mapFromScene(scene_position)
        return QPointF(
            min(max(image_position.x(), 0.0), float(self._image_width)),
            min(max(image_position.y(), 0.0), float(self._image_height)),
        )

    def _add_roi_graphics_item(
        self,
        rectangle: QRectF,
        roi_kind: str,
        *,
        dashed: bool,
    ) -> QGraphicsRectItem:
        color = (
            QColor(40, 110, 255)
            if roi_kind == "baseline"
            else QColor(235, 60, 60)
        )
        pen = QPen(color)
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        fill = QColor(color)
        fill.setAlpha(35)
        return self._graphics_scene.addRect(rectangle, pen, QBrush(fill))

    def _remove_draft_roi(self) -> None:
        if self._draft_roi_item is not None:
            self._graphics_scene.removeItem(self._draft_roi_item)
            self._draft_roi_item = None


def _to_uint8_display(image: np.ndarray) -> np.ndarray:
    """将任意数值位深映射为仅用于显示的 uint8 图像。"""
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    values = image.astype(np.float64, copy=False)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros(image.shape, dtype=np.uint8)

    minimum = float(values[finite_mask].min())
    maximum = float(values[finite_mask].max())
    if maximum == minimum:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (values - minimum) * (255.0 / (maximum - minimum))
    scaled[~finite_mask] = 0.0
    return np.ascontiguousarray(np.clip(scaled, 0.0, 255.0).astype(np.uint8))
