"""Standalone online camera window; reusable from the offline main tool."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import replace
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QTransform, QVector3D
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except ImportError as error:  # pragma: no cover - depends on deployment
    raise RuntimeError(
        "在线界面需要 pyqtgraph 和 PyOpenGL；请运行 requirements.txt 中的依赖安装。"
    ) from error

from app_config import AppConfig
from calibration.session_ground import (
    SessionGroundExtrinsic,
    estimate_session_ground_extrinsic,
)
from correction.stage_a_height_scale import (
    HB2_CORRECTION_MODE,
    H1_CORRECTION_MODE,
    NO_CORRECTION_MODE,
    HeightCorrectionResult,
    normalize_correction_mode,
    resolve_height_correction,
)
from laser.backends import AVAILABLE_METHODS
from measurement.ground_reference import (
    GROUND_SUPPORT_MANUAL_ROI,
    GROUND_SUPPORT_PNP_BOARD_MASK,
    MeasurementError,
    SessionGroundReference,
    fit_session_ground_reference_from_support,
    load_frozen_session_ground_reference,
)
from measurement.board_mask import (
    select_board_ground_points_with_mask,
    select_manual_ground_roi_points,
)

from .controller import OnlineController
from .camera_backend import CameraBackend, get_camera_backend
from .fake_camera import SyntheticCameraSession
from .models import CameraConfig, CameraDeviceInfo, CameraSession, CapturedFrame, FrameResult
from .pipeline import FramePipeline
from .recording import FrameRecorder
from .ground_sanity import (
    GroundSanityResult,
    evaluate_ground_sanity,
)
from .ground_point_audit import (
    GroundPointAuditValidationError,
    build_frozen_chain_provenance,
    build_session_ground_plane_provenance,
    export_ground_point_audit,
)
from .session_calibration import (
    SessionGroundPnPQA,
    SessionGroundRepeatability,
    aggregate_session_ground_extrinsic,
    assess_session_pnp_qa,
    assess_checkerboard_image_quality,
    build_session_ground_payload,
    compare_ground_extrinsics,
    merge_session_ground_sanity,
    merge_session_ground_reference,
    save_session_ground_payload,
)
from reconstruction.reconstructor import reconstruct_uv_to_ground
from utils.result_io import (
    next_measurement_dir,
    save_ground_pointcloud_ply,
    save_image_png,
    save_laser_centers_csv,
    save_measurement_json,
    save_reconstructed_points_csv,
)


pg.setConfigOptions(imageAxisOrder="row-major")
DISPLAY_FPS_WINDOW_S = 1.0


def _configure_wrapping_label(
    label: QLabel, *, selectable: bool = True, flexible: bool = True
) -> QLabel:
    """Configure a compact sidebar label that wraps only when needed."""
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    label.setMinimumWidth(0)
    label.setSizePolicy(
        QSizePolicy.Policy.Ignored if flexible else QSizePolicy.Policy.Preferred,
        QSizePolicy.Policy.Minimum,
    )
    if not flexible:
        # Let ordinary titles keep their natural width, but prevent a long
        # diagnostic title from consuming the entire value column.
        label.setMinimumWidth(
            min(label.fontMetrics().horizontalAdvance(label.text()), 170)
        )
        label.setMaximumWidth(170)
    if selectable:
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
    return label


class PointCloudGLViewWidget(gl.GLViewWidget):
    """GL view that reports camera changes for the orientation compass."""

    camera_changed = Signal()

    def setCameraPosition(self, *args, **kwargs) -> None:  # noqa: N802
        super().setCameraPosition(*args, **kwargs)
        self.camera_changed.emit()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().mouseMoveEvent(event)
        self.camera_changed.emit()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().wheelEvent(event)
        self.camera_changed.emit()


class ConstrainedImageViewBox(pg.ViewBox):
    """Keep a zoomed image reachable while allowing bounded inspection."""

    def __init__(self) -> None:
        super().__init__()
        self._image_size: tuple[float, float] | None = None
        self._home_span: tuple[float, float] | None = None
        self.reset_callback = None

    def clear_image_constraints(self) -> None:
        self._image_size = None
        self._home_span = None
        self.setLimits(
            xMin=None,
            xMax=None,
            yMin=None,
            yMax=None,
            minXRange=None,
            maxXRange=None,
            minYRange=None,
            maxYRange=None,
        )

    def set_image_constraints(self, width: int, height: int) -> None:
        x_range, y_range = self.viewRange()
        self._image_size = (float(width), float(height))
        self._home_span = (
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
        )
        self.setLimits(
            minXRange=1.0,
            maxXRange=self._home_span[0],
            minYRange=1.0,
            maxYRange=self._home_span[1],
        )
        self._constrain_to_image()

    def translateBy(self, t=None, x=None, y=None) -> None:  # noqa: N802
        if self._image_size is None:
            super().translateBy(t=t, x=x, y=y)
            return
        if t is not None:
            x = t.x() if hasattr(t, "x") else t[0]
            y = t.y() if hasattr(t, "y") else t[1]
        x_range, y_range = self.viewRange()
        translated_x = (
            x_range[0] + float(x or 0.0),
            x_range[1] + float(x or 0.0),
        )
        translated_y = (
            y_range[0] + float(y or 0.0),
            y_range[1] + float(y or 0.0),
        )
        self._set_bounded_range(translated_x, translated_y)

    def scaleBy(self, s=None, center=None, x=None, y=None) -> None:  # noqa: N802
        if self._image_size is None or self._home_span is None:
            super().scaleBy(s=s, center=center, x=x, y=y)
            return
        if s is not None:
            x, y = s[0], s[1]
        if x is None and y is None:
            return
        if self.state["aspectLocked"] is not False:
            scale = float(y if y is not None else x)
            x = y = scale
        x_scale = float(x if x is not None else 1.0)
        y_scale = float(y if y is not None else 1.0)
        x_range, y_range = self.viewRange()
        current_span = (
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
        )
        if x_scale > 1.0 or y_scale > 1.0:
            maximum_scale = min(
                self._home_span[0] / current_span[0],
                self._home_span[1] / current_span[1],
            )
            x_scale = min(x_scale, maximum_scale)
            y_scale = min(y_scale, maximum_scale)
        if center is None:
            centre_x = (x_range[0] + x_range[1]) * 0.5
            centre_y = (y_range[0] + y_range[1]) * 0.5
        else:
            centre_x = center.x() if hasattr(center, "x") else center[0]
            centre_y = center.y() if hasattr(center, "y") else center[1]
        scaled_x = (
            centre_x + (x_range[0] - centre_x) * x_scale,
            centre_x + (x_range[1] - centre_x) * x_scale,
        )
        scaled_y = (
            centre_y + (y_range[0] - centre_y) * y_scale,
            centre_y + (y_range[1] - centre_y) * y_scale,
        )
        self._set_bounded_range(scaled_x, scaled_y)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.reset_callback is not None:
            self.reset_callback()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _constrain_to_image(self) -> None:
        if self._image_size is None:
            return
        x_range, y_range = self.viewRange()
        self._set_bounded_range(x_range, y_range)

    def _set_bounded_range(self, x_range, y_range) -> None:
        if self._image_size is None:
            return
        width, height = self._image_size
        bounded_x = _bounded_view_range(x_range, width)
        bounded_y = _bounded_view_range(y_range, height)
        current_x, current_y = self.viewRange()
        if bounded_x != tuple(current_x) or bounded_y != tuple(current_y):
            self.setRange(xRange=bounded_x, yRange=bounded_y, padding=0)


class OnlineState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    STARTING = auto()
    STREAMING = auto()
    STOPPING = auto()
    ERROR = auto()


class OnlineCameraWindow(QMainWindow):
    """Connect, acquire, process, inspect, snapshot, and record camera frames."""

    # Heavy checkerboard detection and camera reconfiguration must not run in
    # the Qt GUI thread.  The worker payloads are deliberately plain Python
    # objects so stale results can be discarded by a generation token.
    _session_calibration_frame_ready = Signal(object)
    _camera_config_ready = Signal(object)

    def __init__(
        self,
        config: AppConfig,
        *,
        simulate: bool = False,
        camera_backend: str = "mvs",
        extraction_method: str | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._height_correction_mode = normalize_correction_mode(
            config.correction.mode
        )
        self._simulate = simulate
        self._camera_backend: CameraBackend = get_camera_backend(camera_backend)
        startup_camera = config.camera
        self._initial_camera_config = (
            CameraConfig(
                exposure_us=startup_camera.exposure_us,
                gain_db=startup_camera.gain_db,
                pixel_format=startup_camera.pixel_format,
                offset_x=startup_camera.offset_x,
                offset_y=startup_camera.offset_y,
                width=startup_camera.width,
                height=startup_camera.height,
                timeout_ms=startup_camera.timeout_ms,
            )
            if startup_camera is not None
            else CameraConfig()
        )
        self._initial_extraction_method = (
            extraction_method or config.extraction_method
        )
        self._pipeline = FramePipeline(
            config,
            self._initial_extraction_method,
            system=camera_backend,
        )
        self._controller = OnlineController(self)
        self._recorder = FrameRecorder()
        # ``_poll_recording`` runs from a timer while an error dialog may still
        # be visible.  Remember that the recorder failure has already been
        # reported so the same error cannot reopen the dialog repeatedly.
        self._recording_error_reported = False
        self._error_message_box: QMessageBox | None = None
        self._session: CameraSession | SyntheticCameraSession | None = None
        self._last_result: FrameResult | None = None
        self._session_ground_mode = config.session_ground_calibration.mode
        self._active_session_ground_result: SessionGroundExtrinsic | None = None
        self._last_session_ground_result: SessionGroundExtrinsic | None = None
        self._session_ground_calibration_frame_number: int | None = None
        self._session_ground_calibration_host_monotonic_ns: int | None = None
        self._session_ground_generation = 0
        self._session_ground_calibration_offset: tuple[int, int] | None = None
        self._last_ground_sanity: GroundSanityResult | None = None
        self._last_ground_reference: SessionGroundReference | None = None
        self._ground_reference_invalid_reason: str | None = None
        self._session_calibration_active = False
        self._session_calibration_restore_config: CameraConfig | None = None
        self._session_calibration_was_streaming = False
        self._session_calibration_frames: list[tuple[SessionGroundExtrinsic, CapturedFrame, dict[str, object]]] = []
        self._session_calibration_attempts = 0
        self._session_calibration_repeatability: SessionGroundRepeatability | None = None
        self._session_calibration_qa: SessionGroundPnPQA | None = None
        self._session_calibration_quality: dict[str, object] | None = None
        self._session_calibration_capturing = False
        self._session_calibration_capture_after_reconfigure = False
        self._session_calibration_capture_generation = 0
        self._session_calibration_capture_started_host_monotonic_ns: int | None = None
        self._session_calibration_worker_busy = False
        self._session_calibration_preview_image_initialized = False
        self._camera_reconfigure_in_progress = False
        self._camera_config_worker_busy = False
        self._camera_reconfigure_token = 0
        self._pending_camera_reconfigure: tuple[CameraConfig, bool, object | None] | None = None
        self._suppress_stream_stopped_once = False
        self._camera_config_syncing = False
        self._analysis_window: QMainWindow | None = None
        self._trail: deque[tuple[float, np.ndarray]] = deque(maxlen=30)
        self._displayed_frames = 0
        self._display_started = time.monotonic()
        self._display_rate_history: deque[tuple[float, int]] = deque(
            [(self._display_started, 0)]
        )
        self._last_render_at = 0.0
        self._raw_view_shape: tuple[int, int] | None = None
        self._extracted_view_shape: tuple[int, int] | None = None
        self._last_raw_preview_at = 0.0
        self._image_preview_rotated = False
        self._section_x_bounds: tuple[float, float] | None = None
        self._section_points = np.empty((0, 2), dtype=np.float64)
        self._section_points_ground = np.empty((0, 3), dtype=np.float64)
        self._online_state = OnlineState.DISCONNECTED
        self._device_count = 0
        self._pending_disconnect = False
        self._stop_due_to_error = False
        self._shutdown_thread: threading.Thread | None = None
        self._closing = False
        self._build_ui()
        self._connect_signals()
        self._session_calibration_debounce_timer = QTimer(self)
        self._session_calibration_debounce_timer.setSingleShot(True)
        self._session_calibration_debounce_timer.setInterval(250)
        self._session_calibration_debounce_timer.timeout.connect(
            self._apply_session_calibration_camera_controls
        )
        self._set_online_state(OnlineState.DISCONNECTED)
        self._record_timer = QTimer(self)
        self._record_timer.setInterval(250)
        self._record_timer.timeout.connect(self._poll_recording)
        self._record_timer.start()
        self.setWindowTitle(f"在线线激光三维截面 · {self._camera_backend.display_name}")
        self.resize(1500, 900)
        self.refresh_devices()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "control_scroll_area"):
            self._refresh_adaptive_control_layout()

    def _refresh_adaptive_control_layout(self) -> None:
        """Re-activate wrapped sidebar rows after resize or status updates."""
        panel = self.control_scroll_area.widget()
        if panel is None:
            return
        for label in panel.findChildren(QLabel):
            label.updateGeometry()
        panel_layout = panel.layout()
        if panel_layout is not None:
            panel_layout.invalidate()
            panel_layout.activate()
        panel.updateGeometry()
        self.control_scroll_area.updateGeometry()

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QHBoxLayout(central)
        self.tabs = QTabWidget(central)
        image_tab = QWidget(self.tabs)
        image_layout = QVBoxLayout(image_tab)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)
        image_toolbar = QFrame(image_tab)
        image_toolbar.setObjectName("imageToolbar")
        image_toolbar_layout = QHBoxLayout(image_toolbar)
        image_toolbar_layout.setContentsMargins(12, 7, 12, 7)
        image_toolbar_layout.setSpacing(6)
        image_title = QLabel("图像预览", image_toolbar)
        image_title.setObjectName("imageTitle")
        image_toolbar_layout.addWidget(image_title)
        image_toolbar_layout.addSpacing(12)
        image_toolbar_layout.addWidget(QLabel("视野", image_toolbar))
        self.image_view_mode_buttons = QButtonGroup(image_tab)
        self.image_view_mode_buttons.setExclusive(True)
        self.image_width_mode_button = QPushButton("铺满宽度", image_toolbar)
        self.image_fit_mode_button = QPushButton("整图适配", image_toolbar)
        for button, mode in (
            (self.image_width_mode_button, "width"),
            (self.image_fit_mode_button, "fit"),
        ):
            button.setCheckable(True)
            button.setProperty("imageViewMode", True)
            button.setFixedHeight(28)
            button.clicked.connect(
                lambda _checked=False, name=mode: self._set_image_view_mode(name)
            )
            self.image_view_mode_buttons.addButton(button)
            image_toolbar_layout.addWidget(button)
        # Daheng's 480x3000 ROI is much taller than it is wide.  Starting in
        # width-fill mode makes most of that image fall outside the viewport;
        # keep the historical width-fill default for ordinary wide frames but
        # fit tall frames to the whole preview instead.
        default_image_view_mode = (
            "fit"
            if self._initial_camera_config.height > self._initial_camera_config.width
            else "width"
        )
        self.image_width_mode_button.setChecked(
            default_image_view_mode == "width"
        )
        self.image_fit_mode_button.setChecked(default_image_view_mode == "fit")
        self._image_view_mode = default_image_view_mode
        image_toolbar_layout.addSpacing(12)
        self.image_rotate_button = QPushButton("旋转预览 90°", image_toolbar)
        self.image_rotate_button.setCheckable(True)
        self.image_rotate_button.setProperty("imagePreviewAction", True)
        self.image_rotate_button.setFixedHeight(28)
        self.image_rotate_button.setToolTip(
            "仅将两个预览顺时针旋转 90°；不改变图像数据、像素坐标或导出结果"
        )
        self.image_rotate_button.toggled.connect(
            self._set_image_preview_rotated
        )
        image_toolbar_layout.addWidget(self.image_rotate_button)
        image_toolbar_layout.addStretch(1)
        self.image_reset_button = QPushButton("复位视野", image_toolbar)
        self.image_reset_button.setFixedHeight(28)
        self.image_reset_button.setToolTip("恢复两个预览框的初始视野；也可双击图像")
        self.image_reset_button.clicked.connect(self._reset_image_views)
        image_toolbar_layout.addWidget(self.image_reset_button)
        image_layout.addWidget(image_toolbar)

        image_splitter = QSplitter(Qt.Orientation.Vertical, image_tab)
        raw_view_box = ConstrainedImageViewBox()
        raw_view_box.reset_callback = self._reset_image_views
        self.raw_image_view = pg.PlotWidget(
            image_splitter, viewBox=raw_view_box
        )
        self.raw_image_view.setTitle("原始图像")
        self.raw_image_view.setBackground("#25282d")
        self.raw_image_view.hideAxis("left")
        self.raw_image_view.hideAxis("bottom")
        self.raw_image_view.setAspectLocked(True)
        self.raw_image_view.getViewBox().invertY(True)
        self.raw_image_view.getViewBox().setMenuEnabled(False)
        self.raw_image_item = pg.ImageItem()
        self.raw_image_view.addItem(self.raw_image_item)
        self.raw_image_boundary = pg.PlotDataItem(
            pen=pg.mkPen("#aeb4bc", width=1)
        )
        self.raw_image_boundary.setZValue(10)
        self.raw_image_view.addItem(self.raw_image_boundary)
        extracted_view_box = ConstrainedImageViewBox()
        extracted_view_box.reset_callback = self._reset_image_views
        self.extracted_image_view = pg.PlotWidget(
            image_splitter, viewBox=extracted_view_box
        )
        self.extracted_image_view.setTitle("激光中心提取")
        self.extracted_image_view.setBackground("#25282d")
        self.extracted_image_view.hideAxis("left")
        self.extracted_image_view.hideAxis("bottom")
        self.extracted_image_view.setAspectLocked(True)
        self.extracted_image_view.getViewBox().invertY(True)
        self.extracted_image_view.getViewBox().setMenuEnabled(False)
        self.extracted_image_item = pg.ImageItem()
        self.extracted_image_view.addItem(self.extracted_image_item)
        self.extracted_image_boundary = pg.PlotDataItem(
            pen=pg.mkPen("#aeb4bc", width=1)
        )
        self.extracted_image_boundary.setZValue(10)
        self.extracted_image_view.addItem(self.extracted_image_boundary)
        self.extracted_corner_boundary = pg.PlotDataItem(
            pen=pg.mkPen("#24d7ff", width=2.0)
        )
        self.extracted_corner_boundary.setZValue(20)
        self.extracted_image_view.addItem(self.extracted_corner_boundary)
        self.extracted_corner_scatter = pg.ScatterPlotItem(
            size=7.0,
            pen=pg.mkPen("#ffffff", width=1.0),
            brush=pg.mkBrush("#ffb000"),
            pxMode=True,
        )
        self.extracted_corner_scatter.setZValue(21)
        self.extracted_image_view.addItem(self.extracted_corner_scatter)
        image_splitter.addWidget(self.raw_image_view)
        image_splitter.addWidget(self.extracted_image_view)
        image_splitter.setStretchFactor(0, 1)
        image_splitter.setStretchFactor(1, 1)
        image_splitter.setChildrenCollapsible(False)
        image_layout.addWidget(image_splitter, 1)
        image_tab.setStyleSheet(
            """
            QFrame#imageToolbar {
                background: #f8fafc;
                border-bottom: 1px solid #cbd3dc;
            }
            QLabel#imageTitle { font-weight: 600; color: #20262d; }
            QPushButton[imageViewMode="true"] {
                min-width: 72px;
                padding: 2px 9px;
                border: 1px solid #b8c1cb;
                border-radius: 4px;
                background: #ffffff;
            }
            QPushButton[imageViewMode="true"]:checked {
                color: #174ea6;
                border-color: #79a7e3;
                background: #e7f0fc;
            }
            QPushButton[imagePreviewAction="true"] {
                min-width: 96px;
                padding: 2px 9px;
                border: 1px solid #b8c1cb;
                border-radius: 4px;
                background: #ffffff;
            }
            QPushButton[imagePreviewAction="true"]:checked {
                color: #174ea6;
                border-color: #79a7e3;
                background: #e7f0fc;
            }
            """
        )
        self.tabs.addTab(image_tab, "图像与条纹")
        point_tab = QWidget(self.tabs)
        point_layout = QVBoxLayout(point_tab)
        point_layout.setContentsMargins(0, 0, 0, 0)
        point_layout.setSpacing(0)

        point_toolbar = QFrame(point_tab)
        point_toolbar.setObjectName("pointToolbar")
        toolbar_layout = QHBoxLayout(point_toolbar)
        toolbar_layout.setContentsMargins(12, 7, 12, 7)
        toolbar_layout.setSpacing(6)
        point_title = QLabel("三维分析", point_toolbar)
        point_title.setObjectName("pointTitle")
        toolbar_layout.addWidget(point_title)
        toolbar_layout.addSpacing(12)
        toolbar_layout.addWidget(QLabel("视角", point_toolbar))
        self.point_view_buttons = QButtonGroup(point_tab)
        self.point_view_buttons.setExclusive(True)
        for preset, title in (
            ("perspective", "透视"),
            ("top", "俯视"),
            ("front", "前视"),
            ("side", "侧视"),
        ):
            button = QPushButton(title, point_toolbar)
            button.setCheckable(True)
            button.setProperty("viewPreset", True)
            button.setFixedHeight(28)
            button.clicked.connect(
                lambda _checked=False, name=preset: self._set_point_view_preset(
                    name
                )
            )
            self.point_view_buttons.addButton(button)
            toolbar_layout.addWidget(button)
            if preset == "perspective":
                button.setChecked(True)
        toolbar_layout.addStretch(1)
        self.grid_checkbox = QCheckBox("地面网格", point_toolbar)
        self.grid_checkbox.setChecked(True)
        self.origin_checkbox = QCheckBox("原点标记", point_toolbar)
        self.origin_checkbox.setChecked(True)
        self.compass_checkbox = QCheckBox("方向罗盘", point_toolbar)
        self.compass_checkbox.setChecked(True)
        self.trail_checkbox = QCheckBox("时间轨迹", point_toolbar)
        self.trail_checkbox.setChecked(True)
        self.fit_points_button = QPushButton("适配点云", point_toolbar)
        self.fit_points_button.setFixedHeight(28)
        toolbar_layout.addWidget(self.grid_checkbox)
        toolbar_layout.addWidget(self.origin_checkbox)
        toolbar_layout.addWidget(self.compass_checkbox)
        toolbar_layout.addWidget(self.trail_checkbox)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self.fit_points_button)
        point_layout.addWidget(point_toolbar)

        point_scene = QWidget(point_tab)
        point_stack = QStackedLayout(point_scene)
        point_stack.setContentsMargins(0, 0, 0, 0)
        point_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self.point_view = PointCloudGLViewWidget(point_scene)
        self.point_view.setBackgroundColor("#eef1f4")
        self.point_view.setCameraPosition(
            distance=560,
            elevation=24,
            azimuth=-60,
        )
        self._add_ground_reference()
        self.trail_point_item = gl.GLScatterPlotItem(
            size=2.0,
            pxMode=True,
            glOptions="translucent",
        )
        self.current_point_item = gl.GLScatterPlotItem(
            size=4.0,
            pxMode=True,
            glOptions="translucent",
        )
        self.point_view.addItem(self.trail_point_item)
        self.point_view.addItem(self.current_point_item)
        self._height_colormap = pg.colormap.get("viridis")
        point_stack.addWidget(self.point_view)

        compass_overlay = QWidget(point_scene)
        compass_overlay.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        compass_layout = QVBoxLayout(compass_overlay)
        compass_layout.setContentsMargins(0, 0, 14, 14)
        compass_layout.addStretch(1)
        compass_row = QHBoxLayout()
        compass_row.addStretch(1)
        compass_frame = QFrame(compass_overlay)
        self.orientation_compass = compass_frame
        compass_frame.setObjectName("orientationCompass")
        compass_frame.setFixedSize(138, 138)
        compass_frame_layout = QVBoxLayout(compass_frame)
        compass_frame_layout.setContentsMargins(1, 1, 1, 1)
        self.orientation_view = gl.GLViewWidget(compass_frame)
        self.orientation_view.setBackgroundColor("#eef1f4")
        self.orientation_view.setCameraPosition(
            distance=3.8,
            elevation=24,
            azimuth=-60,
        )
        compass_frame_layout.addWidget(self.orientation_view)
        self._add_orientation_gizmo()
        compass_row.addWidget(compass_frame)
        compass_layout.addLayout(compass_row)
        point_stack.addWidget(compass_overlay)
        point_stack.setCurrentWidget(compass_overlay)
        point_layout.addWidget(point_scene, 1)

        point_footer = QFrame(point_tab)
        point_footer.setObjectName("pointFooter")
        footer_layout = QHBoxLayout(point_footer)
        footer_layout.setContentsMargins(12, 7, 12, 7)
        footer_layout.setSpacing(14)
        self.point_count_label = QLabel("当前截面 0 点", point_footer)
        self.point_range_label = QLabel(
            "Xg --  |  Yg --  |  Zg --", point_footer
        )
        self.height_min_label = QLabel("--", point_footer)
        self.height_max_label = QLabel("--", point_footer)
        height_gradient = QFrame(point_footer)
        height_gradient.setObjectName("heightGradient")
        height_gradient.setFixedSize(110, 9)
        self.point_compensation_label = QLabel(point_footer)
        self._set_compensation_status()
        footer_layout.addWidget(self.point_count_label)
        footer_layout.addWidget(self.point_range_label, 1)
        footer_layout.addWidget(QLabel("Zg 高度", point_footer))
        footer_layout.addWidget(self.height_min_label)
        footer_layout.addWidget(height_gradient)
        footer_layout.addWidget(self.height_max_label)
        trail_legend = QLabel(
            '<span style="color:#1266b3; font-size:16px;">&#8226;</span> '
            "1 s 轨迹",
            point_footer,
        )
        footer_layout.addWidget(trail_legend)
        footer_layout.addWidget(self.point_compensation_label)
        point_layout.addWidget(point_footer)

        point_tab.setStyleSheet(
            """
            QFrame#pointToolbar, QFrame#pointFooter {
                background: #f8fafc;
                border-color: #cbd3dc;
            }
            QFrame#pointToolbar { border-bottom: 1px solid #cbd3dc; }
            QFrame#pointFooter { border-top: 1px solid #cbd3dc; }
            QFrame#orientationCompass {
                background: #eef1f4;
                border: none;
            }
            QLabel#pointTitle { font-weight: 600; color: #20262d; }
            QPushButton[viewPreset="true"] {
                min-width: 48px;
                padding: 2px 9px;
                border: 1px solid #b8c1cb;
                border-radius: 4px;
                background: #ffffff;
            }
            QPushButton[viewPreset="true"]:checked {
                color: #174ea6;
                border-color: #79a7e3;
                background: #e7f0fc;
            }
            QFrame#heightGradient {
                border: 1px solid #9da7b2;
                border-radius: 2px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #440154, stop:0.25 #3b528b,
                    stop:0.5 #21918c, stop:0.75 #5ec962,
                    stop:1 #fde725
                );
            }
            """
        )
        self.grid_checkbox.toggled.connect(self.ground_grid.setVisible)
        self.origin_checkbox.toggled.connect(self._set_origin_visible)
        self.compass_checkbox.toggled.connect(
            self.orientation_compass.setVisible
        )
        self.trail_checkbox.toggled.connect(self.trail_point_item.setVisible)
        self.fit_points_button.clicked.connect(self._fit_point_cloud)
        self.point_view.camera_changed.connect(self._sync_orientation_gizmo)
        self.tabs.addTab(point_tab, "三维截面与时间轨迹")

        section_tab = QWidget(self.tabs)
        section_layout = QVBoxLayout(section_tab)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(0)
        section_toolbar = QFrame(section_tab)
        section_toolbar.setObjectName("sectionToolbar")
        section_toolbar_layout = QHBoxLayout(section_toolbar)
        section_toolbar_layout.setContentsMargins(12, 7, 12, 7)
        section_toolbar_layout.setSpacing(10)
        section_title = QLabel("截面分析", section_toolbar)
        section_title.setObjectName("sectionTitle")
        section_toolbar_layout.addWidget(section_title)
        self.section_grid_checkbox = QCheckBox("网格", section_toolbar)
        self.section_grid_checkbox.setChecked(True)
        self.section_zero_checkbox = QCheckBox("零基准", section_toolbar)
        self.section_zero_checkbox.setChecked(True)
        self.section_crosshair_checkbox = QCheckBox("十字游标", section_toolbar)
        self.section_crosshair_checkbox.setChecked(True)
        self.section_auto_height_checkbox = QCheckBox("自动高度", section_toolbar)
        self.section_auto_height_checkbox.setChecked(True)
        self.section_auto_height_checkbox.setToolTip(
            "按截面主体高度自动调整视野，少量孤立点不参与视野范围计算"
        )
        self.section_max_ds = _double_spin(
            section_toolbar, 0.1, 100.0, 2.0, " mm"
        )
        self.section_max_dz = _double_spin(
            section_toolbar, 0.1, 100.0, 3.0, " mm"
        )
        self.section_max_distance = _double_spin(
            section_toolbar, 0.1, 100.0, 4.0, " mm"
        )
        for control in (
            self.section_max_ds,
            self.section_max_dz,
            self.section_max_distance,
        ):
            control.setDecimals(1)
            control.setSingleStep(0.5)
            control.setFixedWidth(82)
        self.section_fit_button = QPushButton("适配截面", section_toolbar)
        self.section_fit_button.setFixedHeight(28)
        self.section_fit_button.setToolTip("显示包括孤立点在内的完整截面")
        section_toolbar_layout.addWidget(QLabel("断线阈值", section_toolbar))
        section_toolbar_layout.addWidget(QLabel("ΔS", section_toolbar))
        section_toolbar_layout.addWidget(self.section_max_ds)
        section_toolbar_layout.addWidget(QLabel("ΔZg", section_toolbar))
        section_toolbar_layout.addWidget(self.section_max_dz)
        section_toolbar_layout.addWidget(QLabel("3D", section_toolbar))
        section_toolbar_layout.addWidget(self.section_max_distance)
        section_toolbar_layout.addStretch(1)
        section_toolbar_layout.addWidget(self.section_grid_checkbox)
        section_toolbar_layout.addWidget(self.section_zero_checkbox)
        section_toolbar_layout.addWidget(self.section_crosshair_checkbox)
        section_toolbar_layout.addWidget(self.section_auto_height_checkbox)
        section_toolbar_layout.addSpacing(8)
        section_toolbar_layout.addWidget(self.section_fit_button)
        section_layout.addWidget(section_toolbar)

        self.section_view = pg.PlotWidget(section_tab)
        self.section_view.setBackground("#eef1f4")
        self.section_view.setLabel("bottom", "S（沿激光线）", units="mm")
        self.section_view.setLabel("left", "Zg", units="mm")
        self.section_view.showGrid(x=True, y=True, alpha=0.22)
        axis_pen = pg.mkPen("#687482", width=1)
        text_pen = pg.mkPen("#28313a")
        for axis_name in ("bottom", "left"):
            axis = self.section_view.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(text_pen)
            axis.setTickFont(QFont("Segoe UI", 10))
        self.section_curve = self.section_view.plot(
            pen=pg.mkPen((0, 127, 123, 155), width=1.4),
        )
        self.section_curve.setZValue(1)
        self.section_scatter = pg.ScatterPlotItem(
            size=4.0,
            pen=pg.mkPen((0, 91, 88, 210), width=0.7),
            brush=pg.mkBrush(0, 154, 148, 205),
            pxMode=True,
        )
        self.section_scatter.setZValue(3)
        self.section_view.addItem(self.section_scatter)
        self.section_zero_line = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=pg.mkPen(
                "#657585", width=1.3, style=Qt.PenStyle.DashLine
            ),
        )
        self.section_zero_line.setZValue(2)
        self.section_view.addItem(self.section_zero_line)
        crosshair_pen = pg.mkPen(
            "#31475e", width=1, style=Qt.PenStyle.DashLine
        )
        self.section_crosshair_x = pg.InfiniteLine(
            angle=90, movable=False, pen=crosshair_pen
        )
        self.section_crosshair_z = pg.InfiniteLine(
            angle=0, movable=False, pen=crosshair_pen
        )
        self.section_crosshair_x.setZValue(10)
        self.section_crosshair_z.setZValue(10)
        self.section_crosshair_x.hide()
        self.section_crosshair_z.hide()
        self.section_view.addItem(self.section_crosshair_x, ignoreBounds=True)
        self.section_view.addItem(self.section_crosshair_z, ignoreBounds=True)
        section_layout.addWidget(self.section_view, 1)

        section_footer = QFrame(section_tab)
        section_footer.setObjectName("sectionFooter")
        section_footer_layout = QHBoxLayout(section_footer)
        section_footer_layout.setContentsMargins(12, 7, 12, 7)
        section_footer_layout.setSpacing(16)
        self.section_count_label = QLabel("截面 0 点", section_footer)
        self.section_range_label = QLabel(
            "S --  |  Zg --", section_footer
        )
        self.section_extrema_label = QLabel(
            "最低 --  |  最高 --", section_footer
        )
        self.section_cursor_label = QLabel(
            "游标 S --  Zg --", section_footer
        )
        self.section_cursor_label.setObjectName("sectionCursor")
        section_footer_layout.addWidget(self.section_count_label)
        section_footer_layout.addWidget(self.section_range_label, 1)
        section_footer_layout.addWidget(self.section_extrema_label)
        section_footer_layout.addWidget(self.section_cursor_label)
        section_layout.addWidget(section_footer)

        section_tab.setStyleSheet(
            """
            QFrame#sectionToolbar, QFrame#sectionFooter {
                background: #f8fafc;
                border-color: #cbd3dc;
            }
            QFrame#sectionToolbar { border-bottom: 1px solid #cbd3dc; }
            QFrame#sectionFooter { border-top: 1px solid #cbd3dc; }
            QLabel#sectionTitle { font-weight: 600; color: #20262d; }
            QLabel#sectionCursor { color: #174ea6; font-weight: 600; }
            """
        )
        self.section_grid_checkbox.toggled.connect(
            self._set_section_grid_visible
        )
        self.section_zero_checkbox.toggled.connect(
            self.section_zero_line.setVisible
        )
        self.section_crosshair_checkbox.toggled.connect(
            self._set_section_crosshair_enabled
        )
        self.section_auto_height_checkbox.toggled.connect(
            self._set_section_auto_height
        )
        self.section_max_ds.valueChanged.connect(
            self._refresh_section_connections
        )
        self.section_max_dz.valueChanged.connect(
            self._refresh_section_connections
        )
        self.section_max_distance.valueChanged.connect(
            self._refresh_section_connections
        )
        self.section_fit_button.clicked.connect(self._fit_section_view)
        self.section_view.getViewBox().sigRangeChangedManually.connect(
            self._on_section_range_changed_manually
        )
        self._section_mouse_proxy = pg.SignalProxy(
            self.section_view.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._on_section_mouse_moved,
        )
        self.tabs.addTab(section_tab, "二维截面")
        layout.addWidget(self.tabs, 1)
        control_panel = self._control_panel()
        control_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        control_scroll = QScrollArea(central)
        control_scroll.setObjectName("controlScrollArea")
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        control_scroll.setMinimumWidth(300)
        control_scroll.setMaximumWidth(350)
        control_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        control_scroll.setWidget(control_panel)
        self.control_scroll_area = control_scroll
        layout.addWidget(control_scroll)
        self.setCentralWidget(central)

    def _add_ground_reference(self) -> None:
        self.ground_grid = gl.GLGridItem(
            color=(88, 100, 112, 72),
            antialias=True,
            glOptions="translucent",
        )
        self.ground_grid.setSize(x=500, y=300)
        self.ground_grid.setSpacing(50, 50)
        self.ground_grid.setDepthValue(-10)
        self.point_view.addItem(self.ground_grid)

        self.ground_origin_cross = gl.GLLinePlotItem(
            pos=np.asarray(
                [
                    (-9.0, 0.0, 0.15),
                    (9.0, 0.0, 0.15),
                    (0.0, -9.0, 0.15),
                    (0.0, 9.0, 0.15),
                ],
                dtype=np.float32,
            ),
            color=(0.18, 0.21, 0.24, 0.9),
            width=2.0,
            antialias=True,
            mode="lines",
            glOptions="translucent",
        )
        self.ground_origin = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.asarray([[0.08, 0.09, 0.11, 1.0]], dtype=np.float32),
            size=7.0,
            pxMode=True,
            glOptions="opaque",
        )
        self.ground_origin_items = (
            self.ground_origin_cross,
            self.ground_origin,
        )
        self.point_view.addItem(self.ground_origin_cross)
        self.point_view.addItem(self.ground_origin)

    def _add_orientation_gizmo(self) -> None:
        axis_specs = (
            (
                "Xg",
                (1.2, 0.0, 0.0),
                ((-0.18, 0.08, 0.0), (-0.18, -0.08, 0.0)),
                (0.82, 0.12, 0.12, 1.0),
            ),
            (
                "Yg",
                (0.0, 1.2, 0.0),
                ((0.08, -0.18, 0.0), (-0.08, -0.18, 0.0)),
                (0.08, 0.55, 0.22, 1.0),
            ),
            (
                "Zg",
                (0.0, 0.0, 1.2),
                ((0.08, 0.0, -0.18), (-0.08, 0.0, -0.18)),
                (0.08, 0.30, 0.82, 1.0),
            ),
        )
        label_font = QFont("Segoe UI", 9)
        label_font.setBold(True)
        self.orientation_axes: list[gl.GLLinePlotItem] = []
        self.orientation_labels: list[gl.GLTextItem] = []
        for label, endpoint, arrow_offsets, color in axis_specs:
            endpoint_array = np.asarray(endpoint, dtype=np.float32)
            vertices = np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    endpoint_array,
                    endpoint_array,
                    endpoint_array + arrow_offsets[0],
                    endpoint_array,
                    endpoint_array + arrow_offsets[1],
                ],
                dtype=np.float32,
            )
            axis = gl.GLLinePlotItem(
                pos=vertices,
                color=color,
                width=3.0,
                antialias=True,
                mode="lines",
                glOptions="opaque",
            )
            text = gl.GLTextItem(
                pos=endpoint_array * 1.05,
                color=tuple(int(channel * 255) for channel in color),
                text=label,
                font=label_font,
                glOptions="translucent",
            )
            self.orientation_axes.append(axis)
            self.orientation_labels.append(text)
            self.orientation_view.addItem(axis)
            self.orientation_view.addItem(text)
        origin = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.asarray([[0.12, 0.14, 0.17, 1.0]], dtype=np.float32),
            size=6.0,
            pxMode=True,
            glOptions="opaque",
        )
        self.orientation_view.addItem(origin)

    def _set_origin_visible(self, visible: bool) -> None:
        for item in self.ground_origin_items:
            item.setVisible(visible)

    def _sync_orientation_gizmo(self) -> None:
        camera = self.point_view.cameraParams()
        self.orientation_view.setCameraPosition(
            pos=QVector3D(0.0, 0.0, 0.0),
            distance=3.8,
            elevation=float(camera["elevation"]),
            azimuth=float(camera["azimuth"]),
        )

    def _set_compensation_status(self) -> None:
        compensation = self._pipeline.package.calibration.get(
            "ground_u_compensation"
        )
        if compensation is None:
            self.point_compensation_label.setText("Zg 补偿 未启用")
            self.point_compensation_label.setStyleSheet("color: #6b7280;")
            self.point_compensation_label.setToolTip(
                "当前标定包没有 ground_u_compensation"
            )
            return
        bias = np.asarray(compensation["bias_mm"], dtype=np.float64)
        z_offset = float(compensation.get("z_offset_mm", 0.0))
        self.point_compensation_label.setText("Zg 补偿 已应用")
        self.point_compensation_label.setStyleSheet(
            "color: #167344; font-weight: 600;"
        )
        self.point_compensation_label.setToolTip(
            "Zg = Zg_raw - bias(u) - z_offset\n"
            f"bias 范围：{bias.min():.3f} ~ {bias.max():.3f} mm\n"
            f"z_offset：{z_offset:.3f} mm"
        )

    def _set_point_view_preset(self, preset: str) -> None:
        camera = {
            "perspective": (24.0, -60.0),
            "top": (89.9, -90.0),
            "front": (0.0, -90.0),
            "side": (0.0, 0.0),
        }
        elevation, azimuth = camera[preset]
        self.point_view.setCameraPosition(
            pos=QVector3D(0.0, 0.0, 20.0),
            distance=560.0,
            elevation=elevation,
            azimuth=azimuth,
        )

    def _fit_point_cloud(self) -> None:
        if self._last_result is None or not len(
            self._last_result.points_ground
        ):
            return
        points = np.asarray(
            self._last_result.points_ground, dtype=np.float64
        )
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        centre = (minimum + maximum) * 0.5
        span = max(float(np.max(maximum - minimum)), 160.0)
        self.point_view.setCameraPosition(
            pos=QVector3D(*(float(value) for value in centre)),
            distance=min(max(span * 1.45, 280.0), 1800.0),
        )

    def _update_point_summary(self, points_ground: np.ndarray) -> None:
        points = np.asarray(points_ground, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            points = np.empty((0, 3), dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            self.point_count_label.setText("当前截面 0 点")
            self.point_range_label.setText("Xg --  |  Yg --  |  Zg --")
            self.height_min_label.setText("--")
            self.height_max_label.setText("--")
            return
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        self.point_count_label.setText(f"当前截面 {len(points)} 点")
        self.point_range_label.setText(
            f"Xg {minimum[0]:.1f} ~ {maximum[0]:.1f} mm  |  "
            f"Yg {minimum[1]:.1f} ~ {maximum[1]:.1f} mm  |  "
            f"Zg {minimum[2]:.1f} ~ {maximum[2]:.1f} mm"
        )
        self.height_min_label.setText(f"{minimum[2]:.1f}")
        self.height_max_label.setText(f"{maximum[2]:.1f} mm")

    def _invalidate_live_reconstruction_after_ground_update(self) -> None:
        """Discard the displayed reconstruction from the previous ground pose.

        Applying a Session pose does not retroactively change a
        ``FrameResult`` that was already reconstructed with the reference
        pose. Keep the raw frame/Session metadata, but require one new
        pipeline result before showing or exporting a live point cloud.
        """
        self._last_result = None
        self._trail.clear()
        self._update_point_summary(np.empty((0, 3), dtype=np.float64))
        self.current_point_item.setData(
            pos=np.empty((0, 3), dtype=np.float32)
        )
        self.trail_point_item.setData(
            pos=np.empty((0, 3), dtype=np.float32)
        )
        self._reset_section_view()
        self._last_render_at = 0.0

    def _height_colors(self, points_ground: np.ndarray) -> np.ndarray:
        zg = np.asarray(points_ground[:, 2], dtype=np.float64)
        minimum = float(zg.min())
        maximum = float(zg.max())
        if maximum > minimum:
            normalized = (zg - minimum) / (maximum - minimum)
        else:
            normalized = np.full(len(zg), 0.5, dtype=np.float64)
        return np.ascontiguousarray(
            self._height_colormap.map(normalized, mode="float"),
            dtype=np.float32,
        )

    def _set_section_grid_visible(self, visible: bool) -> None:
        self.section_view.showGrid(x=visible, y=visible, alpha=0.22)

    def _set_section_crosshair_enabled(self, enabled: bool) -> None:
        if enabled:
            return
        self.section_crosshair_x.hide()
        self.section_crosshair_z.hide()
        self.section_cursor_label.setText("游标 S --  Zg --")

    def _set_section_auto_height(self, enabled: bool) -> None:
        view_box = self.section_view.getViewBox()
        view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        if enabled and len(self._section_points):
            lower, upper = _section_height_range(
                self._section_points[:, 1], robust=True
            )
            view_box.setYRange(lower, upper, padding=0)

    def _fit_section_view(self) -> None:
        if self._section_x_bounds is not None:
            lower, upper = self._section_x_bounds
            self.section_view.getViewBox().setXRange(
                lower, upper, padding=0
            )
        if not len(self._section_points):
            return
        lower, upper = _section_height_range(
            self._section_points[:, 1], robust=False
        )
        self.section_auto_height_checkbox.setChecked(False)
        self.section_view.getViewBox().setYRange(lower, upper, padding=0)

    def _on_section_range_changed_manually(
        self, changed_axes: tuple[bool, bool]
    ) -> None:
        if changed_axes[1] and self.section_auto_height_checkbox.isChecked():
            self.section_auto_height_checkbox.setChecked(False)

    def _reset_section_view(self) -> None:
        self._section_x_bounds = None
        self._section_points = np.empty((0, 2), dtype=np.float64)
        self._section_points_ground = np.empty((0, 3), dtype=np.float64)
        self.section_curve.setData([], [])
        self.section_scatter.setData([], [])
        view_box = self.section_view.getViewBox()
        view_box.setLimits(xMin=None, xMax=None, maxXRange=None)
        view_box.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        self.section_auto_height_checkbox.setChecked(True)
        self.section_count_label.setText("截面 0 点")
        self.section_range_label.setText("S --  |  Zg --")
        self.section_extrema_label.setText("最低 --  |  最高 --")
        self._set_section_crosshair_enabled(False)

    def _update_section_view(self, points_ground: np.ndarray) -> None:
        ground = np.asarray(points_ground, dtype=np.float64)
        if ground.ndim != 2 or ground.shape[1] != 3:
            ground = np.empty((0, 3), dtype=np.float64)
        ground = ground[np.isfinite(ground).all(axis=1)]
        self._section_points_ground = np.ascontiguousarray(ground)
        distance = _section_distance(ground[:, :2])
        points = np.column_stack((distance, ground[:, 2]))
        self._section_points = np.ascontiguousarray(points)
        if not len(points):
            self.section_curve.setData([], [])
            self.section_scatter.setData([], [])
            self.section_count_label.setText("截面 0 点")
            self.section_range_label.setText("S --  |  Zg --")
            self.section_extrema_label.setText("最低 --  |  最高 --")
            return

        self.section_scatter.setData(x=points[:, 0], y=points[:, 1])
        self._refresh_section_connections()
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        self.section_range_label.setText(
            f"S {minimum[0]:.1f} ~ {maximum[0]:.1f} mm  |  "
            f"Zg {minimum[1]:.1f} ~ {maximum[1]:.1f} mm"
        )
        self.section_extrema_label.setText(
            f"最低 {minimum[1]:.1f} mm  |  最高 {maximum[1]:.1f} mm"
        )
        if self._section_x_bounds is None:
            self._set_section_x_limits(minimum[0], maximum[0])
        if self.section_auto_height_checkbox.isChecked():
            self._set_section_auto_height(True)

    def _refresh_section_connections(self) -> None:
        points = self._section_points
        if not len(points):
            self.section_curve.setData([], [])
            self.section_count_label.setText("截面 0 点")
            return
        connections = _section_connection_mask(
            self._section_points_ground,
            self._section_points[:, 0],
            max_ds=self.section_max_ds.value(),
            max_dz=self.section_max_dz.value(),
            max_distance=self.section_max_distance.value(),
        )
        self.section_curve.setData(
            points[:, 0],
            points[:, 1],
            connect=connections,
            skipFiniteCheck=True,
        )
        segment_count = 1 + int(np.count_nonzero(connections[:-1] == 0))
        self.section_count_label.setText(
            f"截面 {len(points)} 点 · {segment_count} 段"
        )

    def _set_section_x_limits(self, minimum: float, maximum: float) -> None:
        interval = 10.0
        lower = float(np.floor(minimum / interval) * interval)
        upper = float(np.ceil(maximum / interval) * interval)
        if upper <= lower:
            centre = (minimum + maximum) * 0.5
            lower = centre - interval * 0.5
            upper = centre + interval * 0.5
        self._section_x_bounds = (lower, upper)
        view_box = self.section_view.getViewBox()
        view_box.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        view_box.setLimits(
            xMin=lower,
            xMax=upper,
            maxXRange=upper - lower,
        )
        view_box.setXRange(lower, upper, padding=0)

    def _on_section_mouse_moved(self, event: tuple[object, ...]) -> None:
        if (
            not self.section_crosshair_checkbox.isChecked()
            or not len(self._section_points)
        ):
            return
        scene_position = event[0]
        if not self.section_view.sceneBoundingRect().contains(scene_position):
            self.section_crosshair_x.hide()
            self.section_crosshair_z.hide()
            return
        mouse_point = self.section_view.getViewBox().mapSceneToView(
            scene_position
        )
        index = int(
            np.argmin(np.abs(self._section_points[:, 0] - mouse_point.x()))
        )
        distance, zg = self._section_points[index]
        self.section_crosshair_x.setPos(float(distance))
        self.section_crosshair_z.setPos(float(zg))
        self.section_crosshair_x.show()
        self.section_crosshair_z.show()
        self.section_cursor_label.setText(
            f"游标 S {distance:.2f} mm  Zg {zg:.2f} mm"
        )

    def _control_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setMinimumWidth(0)
        panel.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        device_group = QGroupBox("相机", panel)
        device_layout = QVBoxLayout(device_group)
        device_layout.setContentsMargins(6, 6, 6, 6)
        device_layout.setSpacing(5)
        self.device_combo = QComboBox(device_group)
        row = QHBoxLayout()
        self.refresh_button = QPushButton("刷新", device_group)
        self.connect_button = QPushButton("连接", device_group)
        self.disconnect_button = QPushButton("断开", device_group)
        row.addWidget(self.refresh_button)
        row.addWidget(self.connect_button)
        row.addWidget(self.disconnect_button)
        row.setSpacing(5)
        device_layout.addWidget(self.device_combo)
        device_layout.addLayout(row)
        layout.addWidget(device_group)

        self.camera_settings_group = QGroupBox("采集参数（停流后应用）", panel)
        form = QFormLayout(self.camera_settings_group)
        self._configure_result_form(form)
        self.pixel_format = QComboBox(self.camera_settings_group)
        self.pixel_format.addItems(["Mono8", "Mono12"])
        self.pixel_format.setCurrentText(self._initial_camera_config.pixel_format)
        self.exposure = QDoubleSpinBox(self.camera_settings_group)
        self.exposure.setRange(1.0, 1_000_000.0)
        self.exposure.setValue(self._initial_camera_config.exposure_us)
        self.exposure.setSuffix(" μs")
        self.gain = QDoubleSpinBox(self.camera_settings_group)
        self.gain.setRange(-20.0, 40.0)
        self.gain.setValue(self._initial_camera_config.gain_db)
        # The SDK adapter performs the authoritative node-range/increment
        # validation.  These controls must also allow the 4096x3000 sensor
        # used by the Daheng ME2P-1230 profile.
        self.offset_x = _spin(
            self.camera_settings_group, 0, 65535, self._initial_camera_config.offset_x
        )
        self.offset_y = _spin(
            self.camera_settings_group, 0, 65535, self._initial_camera_config.offset_y
        )
        self.roi_width = _spin(
            self.camera_settings_group, 1, 65535, self._initial_camera_config.width
        )
        self.roi_height = _spin(
            self.camera_settings_group, 1, 65535, self._initial_camera_config.height
        )
        form.addRow("像素格式", self.pixel_format)
        form.addRow("曝光", self.exposure)
        form.addRow("增益", self.gain)
        form.addRow("Offset X", self.offset_x)
        form.addRow("Offset Y", self.offset_y)
        form.addRow("宽度", self.roi_width)
        form.addRow("高度", self.roi_height)
        layout.addWidget(self.camera_settings_group)

        self.processing_group = QGroupBox("处理参数（停流后应用）", panel)
        processing_form = QFormLayout(self.processing_group)
        self._configure_result_form(processing_form)
        self.extraction_method_combo = QComboBox(self.processing_group)
        self.extraction_method_combo.addItems(list(AVAILABLE_METHODS))
        method_index = self.extraction_method_combo.findText(
            self._initial_extraction_method
        )
        if method_index >= 0:
            self.extraction_method_combo.setCurrentIndex(method_index)
        processing_form.addRow("提取算法", self.extraction_method_combo)
        self.height_correction_combo = QComboBox(self.processing_group)
        for mode, label in (
            (NO_CORRECTION_MODE, "none"),
            (H1_CORRECTION_MODE, "h1 (保留对照)"),
            (HB2_CORRECTION_MODE, "hb2 (Frozen H-B2)"),
        ):
            self.height_correction_combo.addItem(label, mode)
        correction_index = self.height_correction_combo.findData(
            self._height_correction_mode
        )
        if correction_index >= 0:
            self.height_correction_combo.setCurrentIndex(correction_index)
        processing_form.addRow("高度修正模式", self.height_correction_combo)
        layout.addWidget(self.processing_group)

        stream_group = QGroupBox("在线运行", panel)
        stream_layout = QVBoxLayout(stream_group)
        stream_layout.setContentsMargins(6, 6, 6, 6)
        stream_layout.setSpacing(5)
        row = QHBoxLayout()
        self.start_button = QPushButton("开始", stream_group)
        self.stop_button = QPushButton("停止", stream_group)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.setSpacing(5)
        stream_layout.addLayout(row)
        self.snapshot_button = QPushButton("保存当前帧", stream_group)
        stream_layout.addWidget(self.snapshot_button)
        self.session_ground_button = QPushButton("Session 基准标定", stream_group)
        self.session_ground_button.setToolTip(
            "将当前无障碍棋盘格帧用于 Session-1 PnP；成功后仅更新本次运行时外参"
        )
        stream_layout.addWidget(self.session_ground_button)
        self.session_capture_button = QPushButton(
            "采集 PnP 棋盘格（5 帧）", stream_group
        )
        self.session_capture_button.setToolTip(
            "先在上方全幅预览中调好曝光/增益；点击后才开始连续采集有效棋盘帧"
        )
        stream_layout.addWidget(self.session_capture_button)
        support_row = QHBoxLayout()
        support_row.addWidget(QLabel("地面基准支撑", stream_group))
        self.ground_reference_source_combo = QComboBox(stream_group)
        self.ground_reference_source_combo.addItem(
            "PnP 棋盘物理 mask", GROUND_SUPPORT_PNP_BOARD_MASK
        )
        self.ground_reference_source_combo.addItem(
            "手工 ground ROI", GROUND_SUPPORT_MANUAL_ROI
        )
        configured_support_source = (
            self._config.session_ground_calibration.ground_reference.support_source
        )
        configured_index = self.ground_reference_source_combo.findData(
            configured_support_source
        )
        if configured_index >= 0:
            self.ground_reference_source_combo.setCurrentIndex(configured_index)
        support_row.addWidget(self.ground_reference_source_combo, 1)
        stream_layout.addLayout(support_row)
        self.ground_reference_button = QPushButton(
            "Session 激光地面基准", stream_group
        )
        self.ground_reference_button.setToolTip(
            "先选择明确的 ground 支撑源。PnP 模式使用完整棋盘物理 mask；"
            "手工模式使用单帧工具中已确认的基准 ROI。仅拟合当前 active extrinsic，"
            "不覆盖 reference 标定文件。"
        )
        stream_layout.addWidget(self.ground_reference_button)
        self.frozen_session_ground_button = QPushButton(
            "加载 Frozen Ground", stream_group
        )
        self.frozen_session_ground_button.setToolTip(
            "仅在有效 Session PnP 外参下加载已验证的 Ground-5C physical_S JSON；"
            "不重新拟合、不修改 C0/C1/H1，PnP 更新后自动失效。"
        )
        stream_layout.addWidget(self.frozen_session_ground_button)
        self.ground_sanity_button = QPushButton(
            "激光地面一致性检查", stream_group
        )
        self.ground_sanity_button.setToolTip(
            "PnP 成功后保持棋盘不动并打开激光；使用新的激光帧检查基准面。"
            "自动使用完整棋盘物理 mask（0 mm inset）；只报警，不自动减 Bias 或拟合 a*S+b。"
        )
        stream_layout.addWidget(self.ground_sanity_button)
        self.export_button = QPushButton("导出当前点云/CSV", stream_group)
        self.export_button.setToolTip(
            "保存当前帧的激光中心 CSV、重建点 CSV、地面系 PLY 和叠加图"
        )
        stream_layout.addWidget(self.export_button)
        self.analysis_button = QPushButton("单帧测量与区域选择", stream_group)
        self.analysis_button.setToolTip(
            "打开离线测量界面，框选基准/障碍物区域并计算高度、长度"
        )
        stream_layout.addWidget(self.analysis_button)
        record_row = QHBoxLayout()
        self.record_count = _spin(stream_group, 1, 100000, 100)
        self.record_button = QPushButton("定长录制", stream_group)
        record_row.addWidget(self.record_count)
        record_row.addWidget(self.record_button)
        record_row.setSpacing(5)
        stream_layout.addLayout(record_row)
        layout.addWidget(stream_group)

        stats = QGroupBox("实时状态", panel)
        stats_layout = QFormLayout(stats)
        self._configure_result_form(stats_layout)
        self.state_label = QLabel("未连接", stats)
        self.capture_fps_label = QLabel("—", stats)
        self.process_fps_label = QLabel("—", stats)
        self.display_fps_label = QLabel("—", stats)
        self.processing_ms_label = QLabel("—", stats)
        self.drop_label = QLabel("—", stats)
        self.record_label = QLabel("未录制", stats)
        self.ground_source_label = QLabel("reference", stats)
        self.height_correction_mode_label = QLabel(
            self._height_correction_mode, stats
        )
        self.height_shadow_label = QLabel("未测量", stats)
        self.ground_reference_status_label = QLabel("未启用", stats)
        self.ground_reference_slope_label = QLabel("—", stats)
        self.ground_reference_intercept_label = QLabel("—", stats)
        self.ground_reference_rmse_label = QLabel("—", stats)
        self.ground_reference_range_label = QLabel("—", stats)
        self.ground_reference_points_label = QLabel("—", stats)
        self.ground_reference_coordinate_label = QLabel("—", stats)
        self.ground_reference_json_sha_label = QLabel("—", stats)
        self.ground_reference_runtime_label = QLabel("—", stats)
        self.session_ground_valid_label = QLabel("INVALID · 未标定", stats)
        self.session_ground_corner_label = QLabel("—", stats)
        self.session_ground_rmse_label = QLabel("—", stats)
        self.session_ground_delta_translation_label = QLabel("—", stats)
        self.session_ground_delta_rotation_label = QLabel("—", stats)
        self.session_ground_frames_label = QLabel("—", stats)
        self.session_ground_repeat_translation_label = QLabel("—", stats)
        self.session_ground_repeat_rotation_label = QLabel("—", stats)
        self.session_ground_qa_label = QLabel("Session PnP QA: —", stats)
        self.session_ground_quality_label = QLabel("棋盘质量：—", stats)
        self.session_ground_quality_label.setToolTip(
            "仅显示配置阈值产生的 warning；不会自动修改图像或标定参数"
        )
        self.session_sanity_status_label = QLabel(
            "SESSION_CALIBRATION = 未检查", stats
        )
        self.session_sanity_status_label.setStyleSheet("font-weight: 600;")
        self.session_sanity_bias_label = QLabel("—", stats)
        self.session_sanity_rmse_label = QLabel("—", stats)
        self.session_sanity_p95_label = QLabel("—", stats)
        self.session_sanity_max_label = QLabel("—", stats)
        self.session_sanity_slope_label = QLabel("—", stats)
        self.session_sanity_valid_count_label = QLabel("—", stats)
        self.session_sanity_mask_count_label = QLabel("—", stats)
        for title, label in (
            ("状态", self.state_label),
            ("采集 fps", self.capture_fps_label),
            ("处理 fps", self.process_fps_label),
            ("显示 fps", self.display_fps_label),
            ("单帧处理", self.processing_ms_label),
            ("丢帧/覆盖", self.drop_label),
            ("录制", self.record_label),
            ("ground 外参", self.ground_source_label),
            ("height correction", self.height_correction_mode_label),
            ("H1/H-B2 shadow", self.height_shadow_label),
            ("ground reference", self.ground_reference_status_label),
            ("reference slope", self.ground_reference_slope_label),
            ("reference intercept", self.ground_reference_intercept_label),
            ("reference RMSE", self.ground_reference_rmse_label),
            ("reference S range", self.ground_reference_range_label),
            ("reference points", self.ground_reference_points_label),
            ("reference coordinate", self.ground_reference_coordinate_label),
            ("Frozen JSON SHA256", self.ground_reference_json_sha_label),
            ("本帧 applied/out-of-range", self.ground_reference_runtime_label),
            ("Session 状态", self.session_ground_valid_label),
            ("角点数", self.session_ground_corner_label),
            ("重投影 RMSE", self.session_ground_rmse_label),
            ("Δtranslation", self.session_ground_delta_translation_label),
            ("Δrotation", self.session_ground_delta_rotation_label),
            ("PnP 帧数", self.session_ground_frames_label),
            ("帧间 translation", self.session_ground_repeat_translation_label),
            ("帧间 rotation", self.session_ground_repeat_rotation_label),
            ("Session PnP QA", self.session_ground_qa_label),
            ("Bias Zg", self.session_sanity_bias_label),
            ("Sanity RMSE", self.session_sanity_rmse_label),
            ("Sanity P95", self.session_sanity_p95_label),
            ("Sanity Max", self.session_sanity_max_label),
            ("Ground slope", self.session_sanity_slope_label),
            ("Board mask points", self.session_sanity_mask_count_label),
            ("Valid points", self.session_sanity_valid_count_label),
        ):
            _configure_wrapping_label(label)
            stats_layout.addRow(
                _configure_wrapping_label(
                    QLabel(title, stats),
                    selectable=False,
                    flexible=False,
                ),
                label,
            )
        _configure_wrapping_label(self.session_ground_quality_label)
        stats_layout.addRow(self.session_ground_quality_label)
        _configure_wrapping_label(self.session_sanity_status_label)
        stats_layout.addRow(self.session_sanity_status_label)
        layout.addWidget(stats)
        note = QLabel("历史点仅表示最近 1 秒时间轨迹，不是连续扫描表面。", panel)
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)

        # The panel is allowed to follow the scroll viewport.  Controls use
        # an ignored horizontal size hint so they shrink with it instead of
        # forcing a wide sidebar; labels and values handle the reflow.
        for widget in panel.findChildren(QPushButton):
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                widget.sizePolicy().verticalPolicy(),
            )
        for widget in panel.findChildren(QComboBox):
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                widget.sizePolicy().verticalPolicy(),
            )
        for widget in (
            *panel.findChildren(QDoubleSpinBox),
            *panel.findChildren(QSpinBox),
        ):
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                widget.sizePolicy().verticalPolicy(),
            )
        return panel

    @staticmethod
    def _configure_result_form(form: QFormLayout) -> None:
        """Keep compact title/value rows and wrap only content when needed."""
        form.setContentsMargins(6, 6, 6, 6)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(2)

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.connect_button.clicked.connect(self.connect_camera)
        self.disconnect_button.clicked.connect(self.disconnect_camera)
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.snapshot_button.clicked.connect(self.save_snapshot)
        self.session_ground_button.clicked.connect(
            self._prompt_session_ground_calibration
        )
        self.session_capture_button.clicked.connect(
            self._start_session_calibration_capture
        )
        self.ground_reference_button.clicked.connect(
            self.calibrate_session_ground_reference
        )
        self.frozen_session_ground_button.clicked.connect(
            lambda _checked=False: self.load_frozen_session_ground()
        )
        self.ground_sanity_button.clicked.connect(self.run_ground_sanity_check)
        self.export_button.clicked.connect(self._export_current_frame)
        self.analysis_button.clicked.connect(self.open_frame_analysis)
        self.height_correction_combo.currentIndexChanged.connect(
            lambda _index: self._set_height_correction_mode(
                self.height_correction_combo.currentData()
            )
        )
        self.record_button.clicked.connect(self.start_recording)
        self.exposure.editingFinished.connect(self._schedule_camera_control_update)
        self.gain.editingFinished.connect(self._schedule_camera_control_update)
        self.pixel_format.currentTextChanged.connect(
            self._schedule_camera_control_update
        )
        queued = Qt.ConnectionType.QueuedConnection
        self._controller.raw_frame_ready.connect(self._show_raw_frame, queued)
        self._controller.result_ready.connect(self._show_result, queued)
        self._controller.stats_updated.connect(self._show_stats, queued)
        self._controller.failed.connect(self._show_error, queued)
        self._controller.processing_failed.connect(
            self._show_processing_error, queued
        )
        self._controller.stopped.connect(self._on_stream_stopped, queued)
        self._session_calibration_frame_ready.connect(
            self._on_session_calibration_worker_result, queued
        )
        self._camera_config_ready.connect(
            self._on_camera_config_worker_result, queued
        )

    def _set_height_correction_mode(self, mode: object) -> None:
        """Set the scalar display mode; reconstruction remains unchanged."""
        try:
            normalized = normalize_correction_mode(str(mode))
        except ValueError as error:
            self.statusBar().showMessage(str(error))
            return
        self._height_correction_mode = normalized
        self._pipeline.set_height_correction_mode(normalized)
        self.height_correction_mode_label.setText(normalized)
        if normalized == HB2_CORRECTION_MODE:
            if self._config.correction.hb2_height_correction is None:
                message = "hb2 已选择但未配置 Frozen H-B2；输出将显式标记 not_configured"
            else:
                message = "高度修正模式 = hb2；H1 仅 shadow logging，不叠加"
        else:
            message = f"高度修正模式 = {normalized}；H1/H-B2 互斥"
        self.statusBar().showMessage(message)

    def _set_online_state(
        self, state: OnlineState, message: str | None = None
    ) -> None:
        self._online_state = state
        if message is None:
            message = {
                OnlineState.DISCONNECTED: "未连接",
                OnlineState.CONNECTING: "连接中",
                OnlineState.CONNECTED: "已连接",
                OnlineState.STARTING: "启动中",
                OnlineState.STREAMING: "取流中",
                OnlineState.STOPPING: "停止中",
                OnlineState.ERROR: "错误",
            }[state]
            if state is OnlineState.DISCONNECTED and self._device_count:
                message = f"未连接 · 发现 {self._device_count} 台设备"
            if self._session_calibration_active and state in {
                OnlineState.CONNECTED,
                OnlineState.STREAMING,
            }:
                message = f"{message} · Session 标定模式"
        self.state_label.setText(message)
        self._update_control_states()

    def _update_control_states(self) -> None:
        state = self._online_state
        has_session = self._session is not None
        disconnected = state is OnlineState.DISCONNECTED
        recoverable_disconnected = not has_session and state in {
            OnlineState.DISCONNECTED,
            OnlineState.ERROR,
        }
        idle = state is OnlineState.CONNECTED or (
            state is OnlineState.ERROR
            and has_session
            and not self._controller.running
        )
        streaming = state is OnlineState.STREAMING
        busy = state in {
            OnlineState.CONNECTING,
            OnlineState.STARTING,
            OnlineState.STOPPING,
        }
        camera_reconfigure_busy = (
            self._camera_reconfigure_in_progress
            or self._camera_config_worker_busy
        )
        self.device_combo.setEnabled(recoverable_disconnected)
        self.refresh_button.setEnabled(recoverable_disconnected)
        self.connect_button.setEnabled(
            recoverable_disconnected and self._device_count > 0
        )
        self.disconnect_button.setEnabled(has_session and not busy)
        required_ready = (
            self._session_ground_mode != "required"
            or self._active_session_ground_result is not None
        )
        self.start_button.setEnabled(idle and required_ready)
        self.stop_button.setEnabled(streaming)
        editable = disconnected or idle or (
            state is OnlineState.ERROR and not has_session
        )
        calibration_controls = (
            self._session_calibration_active
            and has_session
            and not busy
            and not camera_reconfigure_busy
            and not self._session_calibration_capturing
        )
        self.camera_settings_group.setEnabled(editable or calibration_controls)
        for control in (self.exposure, self.gain):
            control.setEnabled(editable or calibration_controls)
        for control in (
            self.pixel_format,
            self.offset_x,
            self.offset_y,
            self.roi_width,
            self.roi_height,
        ):
            control.setEnabled(editable and not self._session_calibration_active)
        self.processing_group.setEnabled(editable)
        self.snapshot_button.setEnabled(
            self._last_result is not None and not busy
        )
        session_available = (
            self._session_ground_mode != "disabled"
            and has_session
            and not busy
            and not camera_reconfigure_busy
            and not self._session_calibration_capturing
        )
        self.session_ground_button.setEnabled(session_available)
        ground_reference_available = (
            has_session
            and not busy
            and not camera_reconfigure_busy
            and not self._session_calibration_active
            and not self._session_calibration_capturing
        )
        self.ground_reference_button.setEnabled(ground_reference_available)
        self.ground_reference_source_combo.setEnabled(ground_reference_available)
        self.frozen_session_ground_button.setEnabled(
            ground_reference_available and self._has_valid_session_pnp()
        )
        capture_available = (
            self._session_ground_mode != "disabled"
            and self._session_calibration_active
            and has_session
            and streaming
            and not busy
            and not camera_reconfigure_busy
            and not self._session_calibration_capturing
        )
        self.session_capture_button.setEnabled(capture_available)
        if self._session_calibration_capturing:
            target = self._config.session_ground_calibration.quality.target_frames
            self.session_capture_button.setText(
                f"采集中… {len(self._session_calibration_frames)}/{target}"
            )
        else:
            self.session_capture_button.setText("采集 PnP 棋盘格（5 帧）")
        if self._session_ground_mode == "disabled":
            self.session_ground_button.setText("Session 基准标定（禁用）")
        elif self._session_calibration_active:
            self.session_ground_button.setText(
                "退出 Session 标定模式"
                if self._active_session_ground_result is not None
                else "取消 Session 标定模式"
            )
        elif self._active_session_ground_result is not None:
            self.session_ground_button.setText("重新 Session 基准标定")
        else:
            self.session_ground_button.setText("Session 基准标定")
        sanity_available = (
            self._session_ground_mode != "disabled"
            and self._active_session_ground_result is not None
            and has_session
            and not self._session_calibration_active
            and not busy
        )
        self.ground_sanity_button.setEnabled(sanity_available)
        self.export_button.setEnabled(
            self._last_result is not None and not busy
        )
        self.analysis_button.setEnabled(
            self._last_result is not None and not busy
        )
        self.record_count.setEnabled(streaming and not self._recorder.active)
        self.record_button.setEnabled(streaming and not self._recorder.active)

    def _has_valid_session_pnp(self) -> bool:
        """Return whether a usable Session PnP pose is currently active."""
        result = self._active_session_ground_result
        return bool(
            self._pipeline.ground_extrinsic_source == "session"
            and result is not None
            and result.status == "success"
            and result.R is not None
            and result.t is not None
            and self._pipeline.ground_extrinsic_generation > 0
        )

    def _on_stream_stopped(self) -> None:
        self.capture_fps_label.setText("0.0")
        self.process_fps_label.setText("0.0")
        self.display_fps_label.setText("0.0")
        if self._suppress_stream_stopped_once:
            self._suppress_stream_stopped_once = False
            if self._controller.running:
                self._set_online_state(OnlineState.STREAMING)
            else:
                self._set_online_state(
                    OnlineState.CONNECTED
                    if self._session is not None
                    else OnlineState.DISCONNECTED
                )
            return
        if self._pending_disconnect:
            self._pending_disconnect = False
            self._pending_camera_reconfigure = None
            self._camera_reconfigure_in_progress = False
            self._close_camera_session()
            return
        if self._pending_camera_reconfigure is not None:
            requested, was_streaming, callback = self._pending_camera_reconfigure
            self._pending_camera_reconfigure = None
            self._start_camera_config_worker(
                requested,
                was_streaming=was_streaming,
                callback=callback,
            )
            return
        if self._camera_reconfigure_in_progress:
            return
        if self._stop_due_to_error:
            self._stop_due_to_error = False
            self._set_online_state(
                OnlineState.ERROR,
                "错误，取流已停止；检查提示后可点击“开始”重试",
            )
            return
        self._set_online_state(
            OnlineState.CONNECTED
            if self._session is not None
            else OnlineState.DISCONNECTED
        )

    def _close_camera_session(self) -> bool:
        if self._recorder.active:
            self._cancel_recording()
        if self._recorder.cancelled:
            try:
                self._recorder.wait(2.0)
            except Exception:
                pass
        session = self._session
        if session is not None:
            try:
                session.close()
            except Exception as error:
                self._set_online_state(OnlineState.ERROR, f"关闭失败：{error}")
                return False
        self._session = None
        self._last_result = None
        self._clear_session_ground_runtime()
        self._set_online_state(OnlineState.DISCONNECTED)
        if self._closing:
            QTimer.singleShot(0, self.close)
        return True

    def refresh_devices(self) -> None:
        if self._session is not None:
            return
        self.device_combo.clear()
        try:
            devices = (
                [SyntheticCameraSession.device]
                if self._simulate
                else self._camera_backend.list_devices()
            )
        except Exception as error:
            self._device_count = 0
            self._set_online_state(
                OnlineState.ERROR,
                f"{self._camera_backend.display_name} SDK 不可用",
            )
            QMessageBox.warning(
                self,
                f"{self._camera_backend.display_name} SDK",
                str(error),
            )
            return
        for device in devices:
            self.device_combo.addItem(device.display_name, device)
        self._device_count = len(devices)
        self._set_online_state(OnlineState.DISCONNECTED)

    def _camera_config(self) -> CameraConfig:
        return CameraConfig(
            exposure_us=self.exposure.value(),
            gain_db=self.gain.value(),
            pixel_format=self.pixel_format.currentText(),
            offset_x=self.offset_x.value(),
            offset_y=self.offset_y.value(),
            width=self.roi_width.value(),
            height=self.roi_height.value(),
            timeout_ms=self._initial_camera_config.timeout_ms,
        )

    def _sync_camera_config(self, config: CameraConfig) -> None:
        self._camera_config_syncing = True
        try:
            self.pixel_format.setCurrentText(config.pixel_format)
            self.exposure.setValue(config.exposure_us)
            self.gain.setValue(config.gain_db)
            self.offset_x.setValue(config.offset_x)
            self.offset_y.setValue(config.offset_y)
            self.roi_width.setValue(config.width)
            self.roi_height.setValue(config.height)
        finally:
            self._camera_config_syncing = False

    def _schedule_camera_control_update(self, *_args) -> None:
        """Debounce live Daheng exposure/gain changes in calibration mode."""
        if self._camera_config_syncing or not self._session_calibration_active:
            return
        if self._session is None or self._online_state in {
            OnlineState.CONNECTING,
            OnlineState.STARTING,
            OnlineState.STOPPING,
        }:
            return
        self._session_calibration_debounce_timer.start()

    def _apply_session_calibration_camera_controls(self) -> None:
        if not self._session_calibration_active or self._session is None:
            return
        current = self._session.config
        requested = CameraConfig(
            exposure_us=self.exposure.value(),
            gain_db=self.gain.value(),
            pixel_format=self.pixel_format.currentText(),
            offset_x=0,
            offset_y=0,
            width=4096,
            height=3000,
            timeout_ms=current.timeout_ms,
        )
        if requested == current:
            self._on_session_camera_controls_applied(None)
            return
        try:
            if self._controller.running:
                self._request_camera_reconfigure(
                    requested,
                    callback=self._on_session_camera_controls_applied,
                )
            else:
                self._configure_session_preserving_runtime(requested)
                self._on_session_camera_controls_applied(None)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._on_session_camera_controls_applied(error)

    def _configure_session_preserving_runtime(self, requested: CameraConfig) -> None:
        """Pause/configure/resume without changing the Session PnP runtime."""
        if self._session is None:
            return
        was_streaming = self._controller.running
        if was_streaming:
            self._request_camera_reconfigure(requested)
            return
        self._camera_reconfigure_in_progress = True
        try:
            applied = self._session.configure(requested)
            self._sync_camera_config(applied)
            self._set_online_state(OnlineState.CONNECTED)
        finally:
            self._camera_reconfigure_in_progress = False
            self._update_control_states()

    def _request_camera_reconfigure(
        self,
        requested: CameraConfig,
        *,
        callback=None,
    ) -> None:
        """Queue pause/configure/resume without blocking the GUI thread."""
        if self._session is None:
            if callable(callback):
                callback(RuntimeError("相机尚未连接"))
            return
        if self._camera_reconfigure_in_progress:
            self._pending_camera_reconfigure = (
                requested,
                self._controller.running,
                callback,
            )
            return
        was_streaming = self._controller.running
        self._camera_reconfigure_in_progress = True
        if was_streaming:
            self._pending_camera_reconfigure = (requested, True, callback)
            self.statusBar().showMessage("正在停流应用相机参数，请稍候…")
            self.stop_stream()
            self._update_control_states()
            return
        self._start_camera_config_worker(
            requested,
            was_streaming=False,
            callback=callback,
        )

    def _start_camera_config_worker(
        self,
        requested: CameraConfig,
        *,
        was_streaming: bool,
        callback=None,
    ) -> None:
        if self._session is None:
            self._camera_reconfigure_in_progress = False
            if callable(callback):
                callback(RuntimeError("相机尚未连接"))
            return
        session = self._session
        self._camera_reconfigure_token += 1
        token = self._camera_reconfigure_token
        self._camera_config_worker_busy = True
        self._update_control_states()

        def configure() -> None:
            try:
                applied = session.configure(requested)
            except Exception as error:  # delivered to the GUI thread
                payload = (token, session, None, error, was_streaming, callback)
            else:
                payload = (token, session, applied, None, was_streaming, callback)
            self._camera_config_ready.emit(payload)

        threading.Thread(
            target=configure,
            name="camera-configure",
            daemon=True,
        ).start()

    def _on_camera_config_worker_result(self, payload) -> None:
        (
            token,
            session,
            applied,
            error,
            was_streaming,
            callback,
        ) = payload
        if token != self._camera_reconfigure_token or session is not self._session:
            return
        self._camera_config_worker_busy = False
        self._camera_reconfigure_in_progress = False
        if error is not None:
            self._set_online_state(
                OnlineState.CONNECTED
                if self._session is not None
                else OnlineState.DISCONNECTED
            )
            if callable(callback):
                callback(error)
            else:
                self.statusBar().showMessage(f"应用相机参数失败：{error}")
                QMessageBox.warning(self, "应用相机参数", str(error))
            self._update_control_states()
            return
        try:
            self._sync_camera_config(applied)
            if self._active_session_ground_result is not None:
                active = self._active_session_ground_result
                if active.R is not None and active.t is not None:
                    self._pipeline.apply_session_ground_extrinsic(
                        active.R,
                        active.t,
                        generation=self._session_ground_generation,
                    )
            if self._last_ground_reference is not None:
                self._pipeline.apply_session_ground_reference(
                    self._last_ground_reference
                )
            if (
                was_streaming
                and not self._pending_disconnect
                and self._session is session
                and not self._controller.running
            ):
                self._controller.start(self._session, self._pipeline, self._recorder)
                self._set_online_state(OnlineState.STREAMING)
            else:
                self._set_online_state(OnlineState.CONNECTED)
        except (OSError, RuntimeError, TypeError, ValueError) as apply_error:
            error = apply_error
        if callable(callback):
            callback(error)
        elif error is not None:
            self.statusBar().showMessage(f"应用相机参数失败：{error}")
            QMessageBox.warning(self, "应用相机参数", str(error))
        self._update_control_states()

    def _on_session_camera_controls_applied(self, error) -> None:
        if error is not None:
            self._session_calibration_capture_after_reconfigure = False
            self.statusBar().showMessage(f"标定预览参数应用失败：{error}")
            QMessageBox.warning(self, "标定预览参数", str(error))
            self._update_control_states()
            return
        self.statusBar().showMessage(
            "Session 标定预览参数已应用；标定模式和已完成的 PnP 保持有效"
        )
        if (
            self._session_calibration_capture_after_reconfigure
            and self._session_calibration_active
            and self._controller.running
        ):
            self._session_calibration_capture_after_reconfigure = False
            self._begin_session_calibration_capture()
        self._update_control_states()

    def _apply_camera_config(self) -> None:
        if self._session is None:
            return
        requested = self._camera_config()
        if requested == self._session.config:
            return
        try:
            applied = self._session.configure(requested)
        except RuntimeError as configure_error:
            device = self._session.device
            try:
                self._session.close()
            finally:
                self._session = None
            try:
                self._session = (
                    SyntheticCameraSession(requested)
                    if self._simulate
                    else self._camera_backend.open_session(device.serial_number, requested)
                )
            except Exception as reopen_error:
                raise RuntimeError(
                    "停流后应用采集参数失败，自动重连也未成功："
                    f"{reopen_error}"
                ) from configure_error
            applied = self._session.config
        self._sync_camera_config(applied)
        if self._active_session_ground_result is not None:
            active = self._active_session_ground_result
            if active.R is not None and active.t is not None:
                self._pipeline.apply_session_ground_extrinsic(
                    active.R,
                    active.t,
                    generation=self._session_ground_generation,
                )
        if self._last_ground_reference is not None:
            self._pipeline.apply_session_ground_reference(
                self._last_ground_reference
            )

    def connect_camera(self) -> None:
        if self._session is not None:
            return
        self._stop_due_to_error = False
        device: CameraDeviceInfo | None = self.device_combo.currentData()
        if device is None:
            QMessageBox.information(self, "未选择相机", "请先刷新并选择相机")
            return
        self._set_online_state(OnlineState.CONNECTING)
        QApplication.processEvents()
        try:
            config = self._camera_config()
            self._session = (
                SyntheticCameraSession(config)
                if self._simulate
                else self._camera_backend.open_session(device.serial_number, config)
            )
        except Exception as error:
            self._session = None
            self._show_error(str(error))
            return
        self._sync_camera_config(self._session.config)
        self._last_result = None
        self._clear_session_ground_runtime()
        self._set_online_state(OnlineState.CONNECTED)

    def disconnect_camera(self) -> None:
        self._cancel_recording()
        if self._online_state is OnlineState.STOPPING:
            self._pending_disconnect = True
            return
        if self._controller.running:
            self._pending_disconnect = True
            self.stop_stream()
            return
        self._close_camera_session()

    def start_stream(self) -> None:
        if self._session is None:
            self.connect_camera()
        if self._session is None or self._controller.running:
            return
        if (
            self._session_ground_mode == "required"
            and self._active_session_ground_result is None
        ):
            self._set_online_state(OnlineState.CONNECTED, "等待 Session 基准标定")
            self.statusBar().showMessage(
                "当前配置要求先完成 Session 基准标定；请连接相机后点击标定按钮"
            )
            self._update_control_states()
            return
        self._stop_due_to_error = False
        self._set_online_state(OnlineState.STARTING)
        QApplication.processEvents()
        try:
            self._apply_camera_config()
            self._replace_pipeline(
                FramePipeline(
                    self._config,
                    self.extraction_method_combo.currentText(),
                    system=self._camera_backend.name,
                )
            )
            if self._active_session_ground_result is not None:
                active = self._active_session_ground_result
                if active.R is not None and active.t is not None:
                    self._pipeline.apply_session_ground_extrinsic(
                        active.R,
                        active.t,
                        generation=self._session_ground_generation,
                    )
            if self._last_ground_reference is not None:
                self._pipeline.apply_session_ground_reference(
                    self._last_ground_reference
                )
            self._set_compensation_status()
            self._update_ground_source_label()
            self._trail.clear()
            self._reset_section_view()
            self._displayed_frames = 0
            self._display_started = time.monotonic()
            self._display_rate_history.clear()
            self._display_rate_history.append((self._display_started, 0))
            self._last_render_at = 0.0
            self._controller.start(self._session, self._pipeline, self._recorder)
            self._set_online_state(OnlineState.STREAMING)
        except Exception as error:
            self._show_error(str(error))

    def stop_stream(self) -> None:
        self._cancel_recording()
        if self._online_state is OnlineState.STOPPING:
            return
        if not self._controller.running:
            self._set_online_state(
                OnlineState.CONNECTED
                if self._session is not None
                else OnlineState.DISCONNECTED
            )
            return
        self._set_online_state(OnlineState.STOPPING)
        self._shutdown_thread = threading.Thread(
            target=self._controller.stop,
            name="online-controller-stop",
            daemon=True,
        )
        self._shutdown_thread.start()

    def _cancel_recording(self) -> None:
        if not self._recorder.active:
            return
        self._recorder.cancel()
        self.record_label.setText("取消中")
        self._update_control_states()

    def start_recording(self) -> None:
        if not self._controller.running or self._session is None:
            QMessageBox.information(self, "未取流", "请先连接相机并开始取流")
            return
        self._recording_error_reported = False
        root = (
            self._config.output.directory / "online_recordings"
            if self._config.output is not None
            else Path(__file__).resolve().parents[1] / "output" / "online_recordings"
        )
        try:
            target = self._recorder.start(
                root, self.record_count.value(), self._session.config
            )
            self.record_label.setText(f"录制中 → {target.name}")
            self._set_online_state(OnlineState.STREAMING, "录制中")
        except Exception as error:
            self.record_label.setText("失败")
            self._recording_error_reported = True
            self._show_error(str(error), stop_capture=False)

    def save_snapshot(self) -> None:
        if self._last_result is None:
            QMessageBox.information(self, "没有图像", "当前尚无可保存帧")
            return
        # TIFF is lossless for both Mono8 and Mono12/uint16 frames.  Keep PNG
        # in the save dialog as an explicit option, but make the default
        # snapshot format independent of the camera pixel depth.
        default_suffix = ".tif"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存当前帧",
            f"frame_{self._last_result.frame.camera_frame_number:06d}{default_suffix}",
            "图像 (*.png *.tif *.tiff)",
        )
        if not path:
            return
        if not cv2.imwrite(path, self._last_result.frame.image):
            self._show_error(f"无法保存图像: {path}")
            return

        frame = self._last_result.frame
        metadata = {
            "schema_version": 1,
            "source": "online_snapshot",
            "ground_extrinsic_source": self._pipeline.ground_extrinsic_source,
            "image": {
                "filename": Path(path).name,
                "width": int(frame.image.shape[1]),
                "height": int(frame.image.shape[0]),
                "dtype": str(frame.image.dtype),
            },
            "image_offset": {
                "u": int(frame.offset_x),
                "v": int(frame.offset_y),
            },
            "frame": {
                "camera_frame_number": int(frame.camera_frame_number),
                "camera_timestamp_ticks": (
                    None
                    if frame.camera_timestamp_ticks is None
                    else int(frame.camera_timestamp_ticks)
                ),
                "host_timestamp_ns": int(frame.host_timestamp_ns),
            },
        }
        try:
            Path(path).with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            QMessageBox.warning(
                self,
                "元数据保存失败",
                f"图像已保存，但无法保存硬件 ROI 偏移元数据：{error}",
            )

    def _prompt_session_ground_calibration(self) -> None:
        """Toggle the in-place five-frame Session calibration mode."""
        if self._session_ground_mode == "disabled":
            self.statusBar().showMessage("Session 基准标定已在配置中禁用")
            return
        if self._session_calibration_active:
            self._exit_session_calibration_mode()
            return
        self._enter_session_calibration_mode()

    def _enter_session_calibration_mode(self) -> None:
        if self._session is None:
            self.statusBar().showMessage("请先连接相机，再进入 Session 标定模式")
            return
        if self._camera_backend.name != "daheng":
            self.statusBar().showMessage("Session 全幅标定模式当前仅支持 Daheng 相机")
            return
        self._session_calibration_restore_config = self._session.config
        self._session_calibration_was_streaming = self._controller.running
        self._session_calibration_frames.clear()
        self._session_calibration_attempts = 0
        self._session_calibration_repeatability = None
        self._session_calibration_qa = None
        self._session_calibration_quality = None
        self._session_calibration_capturing = False
        self._session_calibration_capture_after_reconfigure = False
        self._session_calibration_capture_generation += 1
        self._session_calibration_capture_started_host_monotonic_ns = None
        self._session_calibration_worker_busy = False
        self._session_calibration_preview_image_initialized = False
        self._session_calibration_active = True
        self._extracted_view_shape = None
        self.extracted_image_view.setTitle("棋盘角点检测")
        self.extracted_corner_scatter.setData([], [])
        self.extracted_corner_boundary.setData([], [])
        self._update_session_calibration_quality_display()
        try:
            saved = self._session_calibration_restore_config
            full_frame = CameraConfig(
                exposure_us=saved.exposure_us,
                gain_db=saved.gain_db,
                pixel_format=saved.pixel_format,
                offset_x=0,
                offset_y=0,
                width=4096,
                height=3000,
                timeout_ms=saved.timeout_ms,
            )
            if self._controller.running:
                self._request_camera_reconfigure(
                    full_frame,
                    callback=self._on_session_calibration_prepare_complete,
                )
            else:
                self._configure_session_preserving_runtime(full_frame)
                self._on_session_calibration_prepare_complete(None)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._on_session_calibration_prepare_complete(error)
            return
        self._update_control_states()
        self.statusBar().showMessage("Session 标定模式：正在准备全幅预览，尚未开始采集")

    def _on_session_calibration_prepare_complete(self, error) -> None:
        if error is not None:
            self._session_calibration_active = False
            self._session_calibration_capturing = False
            self._session_calibration_capture_generation += 1
            self._session_calibration_restore_config = None
            self.extracted_image_view.setTitle("激光中心提取")
            self.statusBar().showMessage(f"无法进入 Session 标定模式：{error}")
            QMessageBox.warning(self, "Session 基准标定", str(error))
            self._update_control_states()
            return
        if not self._session_calibration_active or self._session is None:
            return
        try:
            if not self._controller.running:
                self._start_stream_for_session_calibration()
        except (OSError, RuntimeError, TypeError, ValueError) as start_error:
            self._on_session_calibration_prepare_complete(start_error)
            return
        self._update_control_states()
        self.statusBar().showMessage(
            "Session 标定预览已就绪；请调好曝光/增益后点击“采集 PnP 棋盘格”"
        )

    def _start_stream_for_session_calibration(self) -> None:
        if self._session is None or self._controller.running:
            return
        self._replace_pipeline(
            FramePipeline(
                self._config,
                self.extraction_method_combo.currentText(),
                system=self._camera_backend.name,
            )
        )
        if self._active_session_ground_result is not None:
            active = self._active_session_ground_result
            if active.R is not None and active.t is not None:
                self._pipeline.apply_session_ground_extrinsic(
                    active.R,
                    active.t,
                    generation=self._session_ground_generation,
                )
        if self._last_ground_reference is not None:
            self._pipeline.apply_session_ground_reference(
                self._last_ground_reference
            )
        self._controller.start(self._session, self._pipeline, self._recorder)
        self._set_online_state(OnlineState.STREAMING, "Session 标定预览中")

    def _exit_session_calibration_mode(self) -> None:
        saved = self._session_calibration_restore_config
        was_streaming = self._session_calibration_was_streaming
        self._session_calibration_active = False
        self._session_calibration_capturing = False
        self._session_calibration_capture_after_reconfigure = False
        self._session_calibration_capture_generation += 1
        self._session_calibration_capture_started_host_monotonic_ns = None
        self._session_calibration_debounce_timer.stop()
        if saved is None or self._session is None:
            self._session_calibration_restore_config = None
            self._restore_session_calibration_preview()
            self._update_control_states()
            return
        try:
            if self._controller.running:
                self._request_camera_reconfigure(
                    saved,
                    callback=lambda error: self._finish_session_calibration_exit(
                        error, was_streaming
                    ),
                )
            else:
                self._configure_session_preserving_runtime(saved)
                self._finish_session_calibration_exit(None, was_streaming)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._finish_session_calibration_exit(error, was_streaming)

    def _finish_session_calibration_exit(self, error, was_streaming: bool) -> None:
        self._session_calibration_restore_config = None
        if error is not None:
            self.statusBar().showMessage(f"退出标定模式并恢复采集参数失败：{error}")
            QMessageBox.warning(self, "恢复测量参数", str(error))
        else:
            self.statusBar().showMessage(
                "已退出 Session 标定模式；已恢复原 ROI、曝光、增益等测量参数"
            )
        self._restore_session_calibration_preview()
        self._update_control_states()

    def _restore_session_calibration_preview(self) -> None:
        self.extracted_image_view.setTitle("激光中心提取")
        self.extracted_corner_scatter.setData([], [])
        self.extracted_corner_boundary.setData([], [])

    def _start_session_calibration_capture(self) -> None:
        """Arm one continuous 5-frame PnP capture after preview adjustment."""
        if not self._session_calibration_active:
            return
        if self._session is None or not self._controller.running:
            self.statusBar().showMessage("请先等待 Session 全幅预览就绪")
            return
        if self._camera_reconfigure_in_progress or self._camera_config_worker_busy:
            self.statusBar().showMessage("正在应用曝光/增益，请稍候再采集")
            return
        if self._session_calibration_capturing:
            return
        # If the user clicks immediately after editing a spinbox, flush the
        # debounce timer first.  The capture starts only after the new camera
        # parameters have actually been applied.
        if self._session_calibration_debounce_timer.isActive():
            self._session_calibration_debounce_timer.stop()
            self._session_calibration_capture_after_reconfigure = True
            self._apply_session_calibration_camera_controls()
            if self._camera_reconfigure_in_progress or self._camera_config_worker_busy:
                self.statusBar().showMessage("正在应用最新曝光/增益，完成后自动开始采集")
                return
            if self._session_calibration_capturing:
                return
            self._session_calibration_capture_after_reconfigure = False
        self._begin_session_calibration_capture()

    def _begin_session_calibration_capture(self) -> None:
        if (
            not self._session_calibration_active
            or self._session is None
            or not self._controller.running
        ):
            return
        quality_config = self._config.session_ground_calibration.quality
        self._session_calibration_capture_generation += 1
        self._session_calibration_capture_started_host_monotonic_ns = (
            time.perf_counter_ns()
        )
        self._session_calibration_capturing = True
        self._session_calibration_worker_busy = False
        self._session_calibration_frames.clear()
        self._session_calibration_attempts = 0
        self._session_calibration_repeatability = None
        self._session_calibration_qa = None
        self._session_calibration_quality = None
        self.extracted_corner_scatter.setData([], [])
        self.extracted_corner_boundary.setData([], [])
        self._update_session_calibration_quality_display()
        self._update_control_states()
        self.statusBar().showMessage(
            "已开始采集 PnP 棋盘格；请保持棋盘静止，等待有效帧 "
            f"0/{quality_config.target_frames}"
        )

    def _handle_session_calibration_frame(self, frame: CapturedFrame) -> None:
        if not self._session_calibration_active or not self._session_calibration_capturing:
            return
        quality_config = self._config.session_ground_calibration.quality
        if len(self._session_calibration_frames) >= quality_config.target_frames:
            return
        if self._session_calibration_attempts >= quality_config.max_capture_attempts:
            return
        if self._session_calibration_worker_busy:
            return
        started_at = self._session_calibration_capture_started_host_monotonic_ns
        if started_at is not None and frame.host_monotonic_ns <= started_at:
            return
        self._session_calibration_attempts += 1
        calibration = self._pipeline.calibration_for_reconstruction()
        K = np.asarray(calibration["K"], dtype=np.float64).copy()
        K[0, 2] -= float(frame.offset_x)
        K[1, 2] -= float(frame.offset_y)
        captured = CapturedFrame(
            image=np.ascontiguousarray(frame.image.copy()),
            camera_frame_number=frame.camera_frame_number,
            camera_timestamp_ticks=frame.camera_timestamp_ticks,
            host_timestamp_ns=frame.host_timestamp_ns,
            host_monotonic_ns=frame.host_monotonic_ns,
            offset_x=frame.offset_x,
            offset_y=frame.offset_y,
        )
        token = self._session_calibration_capture_generation
        board = self._config.session_ground_calibration.board_config()
        self._session_calibration_worker_busy = True
        self._update_control_states()

        def detect() -> None:
            result = None
            quality = None
            error = None
            try:
                result = estimate_session_ground_extrinsic(
                    captured.image,
                    {"K": K, "D": np.asarray(calibration["D"]).copy()},
                    board,
                )
                expected = board.pattern_cols * board.pattern_rows
                if result.status != "success" or result.detected_corners is None:
                    raise ValueError(result.message)
                if len(result.detected_corners) != expected:
                    raise ValueError(
                        f"角点数 {len(result.detected_corners)}/{expected}，本帧拒绝"
                    )
                quality = assess_checkerboard_image_quality(
                    captured.image,
                    result.detected_corners,
                    pattern_cols=board.pattern_cols,
                    pattern_rows=board.pattern_rows,
                    saturation_ratio_warn=quality_config.saturation_ratio_warn,
                    dynamic_range_p95_p5_warn=quality_config.dynamic_range_p95_p5_warn,
                    edge_margin_warn_px=quality_config.edge_margin_warn_px,
                )
            except Exception as detect_error:  # delivered to the GUI thread
                error = detect_error
            self._session_calibration_frame_ready.emit(
                {
                    "token": token,
                    "frame": captured,
                    "result": result,
                    "quality": quality,
                    "error": error,
                }
            )

        threading.Thread(
            target=detect,
            name="session-checkerboard-detect",
            daemon=True,
        ).start()

    def _on_session_calibration_worker_result(self, payload) -> None:
        if payload.get("token") != self._session_calibration_capture_generation:
            return
        self._session_calibration_worker_busy = False
        frame = payload["frame"]
        result = payload["result"]
        quality = payload["quality"]
        error = payload["error"]
        if not self._session_calibration_active or not self._session_calibration_capturing:
            self._update_control_states()
            return
        # Update the lower preview once per submitted candidate, not once per
        # raw camera frame.  This keeps the two views responsive at 4096x3000.
        self._show_session_calibration_image(frame.image)
        quality_config = self._config.session_ground_calibration.quality
        expected = (
            self._config.session_ground_calibration.pattern_cols
            * self._config.session_ground_calibration.pattern_rows
        )
        if error is None and result is not None and quality is not None:
            self._session_calibration_frames.append((result, frame, quality))
            self._show_session_calibration_overlay(frame.image, result.detected_corners)
            self._update_session_calibration_quality_display()
            accepted = len(self._session_calibration_frames)
            self.statusBar().showMessage(
                f"Session 标定采集中：已接受 {accepted}/{quality_config.target_frames} 帧，"
                f"角点 {len(result.detected_corners)}/{expected}"
            )
            if accepted >= quality_config.target_frames:
                self._session_calibration_capturing = False
                self._finalize_session_calibration()
        else:
            message = str(error or "棋盘检测失败")
            remaining = quality_config.max_capture_attempts - self._session_calibration_attempts
            self.statusBar().showMessage(
                f"Session 标定本帧重试：{message}；剩余尝试 {max(remaining, 0)}"
            )
            if remaining <= 0:
                self._session_calibration_capturing = False
                self.statusBar().showMessage(
                    "Session 基准标定 INVALID：未能在配置的尝试次数内收集完整 5 帧"
                )
        self._update_control_states()

    def _show_session_calibration_overlay(
        self, image: np.ndarray, corners: np.ndarray
    ) -> None:
        self._show_session_calibration_image(image)
        corners_array = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        self.extracted_corner_scatter.setData(
            x=corners_array[:, 0], y=corners_array[:, 1]
        )
        rows = self._config.session_ground_calibration.pattern_rows
        cols = self._config.session_ground_calibration.pattern_cols
        grid = corners_array.reshape(rows, cols, 2)
        boundary = np.vstack((grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0], grid[0, 0]))
        self.extracted_corner_boundary.setData(boundary[:, 0], boundary[:, 1])

    def _show_session_calibration_image(self, image: np.ndarray) -> None:
        shape_changed = image.shape[:2] != self._extracted_view_shape
        levels = (0, 255) if image.dtype == np.uint8 else (0, 4095)
        self.extracted_image_item.setImage(image, autoLevels=False, levels=levels)
        _set_image_boundary(self.extracted_image_boundary, image)
        if shape_changed:
            self._extracted_view_shape = image.shape[:2]
            transform = _image_preview_transform(image, self._image_preview_rotated)
            self.extracted_image_item.setTransform(transform)
            self.extracted_image_boundary.setTransform(transform)
            self.extracted_corner_boundary.setTransform(transform)
            self.extracted_corner_scatter.setTransform(transform)
            _fit_image_view(
                self.extracted_image_view,
                image,
                self._image_view_mode,
                rotated=self._image_preview_rotated,
            )

    def _finalize_session_calibration(self) -> None:
        observations = self._session_calibration_frames
        quality_config = self._config.session_ground_calibration.quality
        try:
            calibration = self._pipeline.calibration_for_reconstruction()
            first_frame = observations[0][1]
            K = np.asarray(calibration["K"], dtype=np.float64).copy()
            K[0, 2] -= float(first_frame.offset_x)
            K[1, 2] -= float(first_frame.offset_y)
            frame_results = [item[0] for item in observations]
            final, repeatability = aggregate_session_ground_extrinsic(
                frame_results,
                {"K": K, "D": np.asarray(calibration["D"]).copy()},
                self._config.session_ground_calibration.board_config(),
                required_frames=quality_config.target_frames,
            )
            if (
                final.reprojection_rmse_px is None
                or final.reprojection_rmse_px > quality_config.max_reprojection_rmse_px
            ):
                raise ValueError(
                    "最终 PnP reprojection RMSE 超过配置阈值："
                    f"{final.reprojection_rmse_px} > {quality_config.max_reprojection_rmse_px} px"
                )
            try:
                if quality_config.target_frames != 5:
                    session_pnp_qa = SessionGroundPnPQA.failure(
                        "leave-one-frame-out QA requires target_frames=5; "
                        f"configured {quality_config.target_frames}",
                        fold_count=len(frame_results),
                    )
                else:
                    session_pnp_qa = assess_session_pnp_qa(
                        frame_results,
                        final,
                        {"K": K, "D": np.asarray(calibration["D"]).copy()},
                        self._config.session_ground_calibration.board_config(),
                        required_frames=5,
                        max_heldout_reprojection_rmse_px=quality_config.max_reprojection_rmse_px,
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as qa_error:
                # QA is diagnostic.  A QA implementation/data failure must be
                # recorded without replacing or invalidating the formal pose.
                session_pnp_qa = SessionGroundPnPQA.failure(
                    f"Session PnP QA failed: {qa_error}",
                    fold_count=len(frame_results),
                )
            reference_R, reference_t = self._pipeline.reference_ground_extrinsic
            delta = compare_ground_extrinsics(
                reference_R,
                reference_t,
                final.R,
                final.t,
            )
            self._session_ground_generation += 1
            self._invalidate_ground_reference_for_extrinsic_change(
                "Session PnP 已更新，激光地面基准需重新标定"
            )
            self._pipeline.apply_session_ground_extrinsic(
                final.R,
                final.t,
                generation=self._session_ground_generation,
            )
            self._invalidate_live_reconstruction_after_ground_update()
            self._active_session_ground_result = final
            self._last_session_ground_result = final
            self._session_calibration_repeatability = repeatability
            self._session_calibration_qa = session_pnp_qa
            self._session_calibration_quality = self._aggregate_session_quality()
            last_frame = observations[-1][1]
            self._session_ground_calibration_frame_number = int(
                last_frame.camera_frame_number
            )
            self._session_ground_calibration_host_monotonic_ns = int(
                last_frame.host_monotonic_ns
            )
            self._session_ground_calibration_offset = (
                int(last_frame.offset_x),
                int(last_frame.offset_y),
            )
            self._last_ground_sanity = None
            self._reset_ground_sanity_display()
            self._show_session_calibration_overlay(
                last_frame.image,
                final.detected_corners,
            )
            self._update_session_ground_display(final, delta)
            self._update_ground_source_label()
            reference_R, reference_t = self._pipeline.reference_ground_extrinsic
            payload = build_session_ground_payload(
                final,
                self._config.session_ground_calibration.board_config(),
                frame_number=int(last_frame.camera_frame_number),
                frame_offset=(int(last_frame.offset_x), int(last_frame.offset_y)),
                reference_R=reference_R,
                reference_t=reference_t,
                runtime_source=self._pipeline.ground_extrinsic_source,
                frame_host_monotonic_ns=int(last_frame.host_monotonic_ns),
                session_generation=self._session_ground_generation,
                repeatability=repeatability,
                quality=self._session_calibration_quality,
                session_pnp_qa=session_pnp_qa,
            )
            json_path = save_session_ground_payload(
                self._session_ground_json_path(), payload
            )
            self._update_session_calibration_quality_display()
            self.statusBar().showMessage(
                "正式 PnP VALID；"
                f"5/5 帧，RMSE {final.reprojection_rmse_px:.4f} px，"
                f"帧间 Δt max {repeatability.translation_max_mm:.4f} mm，"
                f"ΔR max {repeatability.rotation_max_deg:.4f}°；"
                f"Session PnP QA: reproj="
                f"{_format_qa_metric(session_pnp_qa.zg_propagation.get('heldout_reprojection_rmse_p95_px'), 'px')}，"
                f"ΔR max={_format_qa_metric(session_pnp_qa.sensitivity.get('rotation_max_deg'), '°')}，"
                f"Δt max={_format_qa_metric(session_pnp_qa.sensitivity.get('translation_max_mm'), 'mm')}，"
                f"predicted Zg P95={_format_qa_metric(session_pnp_qa.zg_propagation.get('p95_abs_mm'), 'mm')}，"
                f"{session_pnp_qa.status}；已保存 {json_path}"
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._last_session_ground_result = SessionGroundExtrinsic(
                status="invalid_quality",
                message=str(error),
                detected_corners=observations[-1][0].detected_corners
                if observations
                else None,
            )
            if observations:
                self._session_calibration_quality = self._aggregate_session_quality()
                try:
                    last_frame = observations[-1][1]
                    reference_R, reference_t = self._pipeline.reference_ground_extrinsic
                    payload = build_session_ground_payload(
                        self._last_session_ground_result,
                        self._config.session_ground_calibration.board_config(),
                        frame_number=int(last_frame.camera_frame_number),
                        frame_offset=(int(last_frame.offset_x), int(last_frame.offset_y)),
                        reference_R=reference_R,
                        reference_t=reference_t,
                        runtime_source=self._pipeline.ground_extrinsic_source,
                        frame_host_monotonic_ns=int(last_frame.host_monotonic_ns),
                        session_generation=self._session_ground_generation,
                        quality=self._session_calibration_quality,
                    )
                    save_session_ground_payload(self._session_ground_json_path(), payload)
                except (OSError, TypeError, ValueError):
                    pass
            self._update_session_ground_display(self._last_session_ground_result, None)
            self._update_session_calibration_quality_display()
            self.statusBar().showMessage(f"Session 基准标定 INVALID：{error}")

    def _aggregate_session_quality(self) -> dict[str, object]:
        frames = [item[2] for item in self._session_calibration_frames]
        warnings = sorted(
            {
                warning
                for item in frames
                for warning in item.get("warnings", [])
            }
        )
        return {
            "accepted_frames": len(frames),
            "saturation_ratio_max": max(
                float(item["saturation_ratio"]) for item in frames
            ),
            "dynamic_range_p95_p5_min": min(
                float(item["dynamic_range_p95_p5"]) for item in frames
            ),
            "edge_margin_min_px": min(float(item["edge_margin_px"]) for item in frames),
            "warnings": warnings,
            "frames": frames,
        }

    def _update_session_calibration_quality_display(self) -> None:
        if not hasattr(self, "session_ground_frames_label"):
            return
        target = self._config.session_ground_calibration.quality.target_frames
        self.session_ground_frames_label.setText(
            f"{len(self._session_calibration_frames)}/{target}"
        )
        repeatability = self._session_calibration_repeatability
        if repeatability is None:
            self.session_ground_repeat_translation_label.setText("—")
            self.session_ground_repeat_rotation_label.setText("—")
        else:
            self.session_ground_repeat_translation_label.setText(
                f"max {repeatability.translation_max_mm:.4f} mm"
            )
            self.session_ground_repeat_rotation_label.setText(
                f"max {repeatability.rotation_max_deg:.4f}°"
            )
        qa = self._session_calibration_qa
        if qa is None:
            self.session_ground_qa_label.setText("Session PnP QA: —")
        else:
            qa_p95 = qa.zg_propagation.get("heldout_reprojection_rmse_p95_px")
            qa_rotation = qa.sensitivity.get("rotation_max_deg")
            qa_translation = qa.sensitivity.get("translation_max_mm")
            qa_zg_p95 = qa.zg_propagation.get("p95_abs_mm")

            self.session_ground_qa_label.setText(
                "Session PnP QA: "
                f"{qa.status}; reproj={_format_qa_metric(qa_p95, 'px')}，"
                f"ΔR max={_format_qa_metric(qa_rotation, '°')}，"
                f"Δt max={_format_qa_metric(qa_translation, 'mm')}，"
                f"predicted Zg P95={_format_qa_metric(qa_zg_p95, 'mm')}"
            )
        quality = self._session_calibration_quality
        if quality is None:
            self.session_ground_quality_label.setText("棋盘质量：—")
        else:
            warnings = quality.get("warnings", [])
            self.session_ground_quality_label.setText(
                "棋盘质量：WARNING: " + ", ".join(str(item) for item in warnings)
                if warnings
                else "棋盘质量：OK"
            )

    def calibrate_session_ground(self) -> None:
        """Compatibility entry point for Session calibration.

        Daheng uses the live five-frame mode.  The legacy one-frame fallback is
        retained for non-Daheng/synthetic integrations that still call this
        method directly; the GUI button always enters the V2 mode above.
        """
        if self._session_ground_mode == "disabled":
            self.statusBar().showMessage("Session 基准标定已在配置中禁用")
            return
        if self._camera_backend.name == "daheng" and not self._session_calibration_active:
            self._enter_session_calibration_mode()
            return
        try:
            frame = self._session_calibration_frame()
            if frame is None:
                return
            calibration = self._pipeline.calibration_for_reconstruction()
            K = np.asarray(calibration["K"], dtype=np.float64).copy()
            # PnP 使用完整传感器内参；当前帧若是硬 ROI，需要将主点移到
            # ROI 局部坐标。D 不随纯平移 ROI 改变。
            K[0, 2] -= float(frame.offset_x)
            K[1, 2] -= float(frame.offset_y)
            result = estimate_session_ground_extrinsic(
                frame.image,
                {"K": K, "D": np.asarray(calibration["D"]).copy()},
                self._config.session_ground_calibration.board_config(),
            )
        except Exception as error:
            self._set_session_ground_attempt(
                None,
                f"Session 基准标定失败：{error}",
            )
            return

        self._last_session_ground_result = result
        delta: tuple[float, float] | None = None
        if result.status == "success" and result.R is not None and result.t is not None:
            try:
                reference_R, reference_t = self._pipeline.reference_ground_extrinsic
                delta = compare_ground_extrinsics(
                    reference_R,
                    reference_t,
                    result.R,
                    result.t,
                )
                self._session_ground_generation += 1
                self._invalidate_ground_reference_for_extrinsic_change(
                    "Session PnP 已更新，激光地面基准需重新标定"
                )
                self._pipeline.apply_session_ground_extrinsic(
                    result.R,
                    result.t,
                    generation=self._session_ground_generation,
                )
                self._invalidate_live_reconstruction_after_ground_update()
                self._active_session_ground_result = result
                self._session_ground_calibration_frame_number = int(
                    frame.camera_frame_number
                )
                self._session_ground_calibration_host_monotonic_ns = int(
                    frame.host_monotonic_ns
                )
                self._session_ground_calibration_offset = (
                    int(frame.offset_x),
                    int(frame.offset_y),
                )
                self._last_ground_sanity = None
                self._reset_ground_sanity_display()
            except (TypeError, ValueError) as error:
                self._set_session_ground_attempt(
                    result,
                    f"Session 外参未应用：{error}",
                )
                return

        self._update_session_ground_display(result, delta)
        self._update_ground_source_label()
        try:
            reference_R, reference_t = self._pipeline.reference_ground_extrinsic
            payload = build_session_ground_payload(
                result,
                self._config.session_ground_calibration.board_config(),
                frame_number=int(frame.camera_frame_number),
                frame_offset=(int(frame.offset_x), int(frame.offset_y)),
                reference_R=reference_R,
                reference_t=reference_t,
                runtime_source=self._pipeline.ground_extrinsic_source,
                frame_host_monotonic_ns=int(frame.host_monotonic_ns),
                session_generation=self._session_ground_generation,
            )
            json_path = save_session_ground_payload(
                self._session_ground_json_path(), payload
            )
        except (OSError, TypeError, ValueError) as error:
            self.statusBar().showMessage(f"Session 标定已处理，但 JSON 保存失败：{error}")
            self._update_control_states()
            return

        if result.status == "success" and self._active_session_ground_result is not None:
            assert delta is not None
            self.statusBar().showMessage(
                "Session 基准标定 VALID；"
                f"RMSE {result.reprojection_rmse_px:.4f} px，"
                f"Δt {delta[0]:.3f} mm，ΔR {delta[1]:.3f}°；"
                f"已保存 {json_path}"
            )
        else:
            self.statusBar().showMessage(
                f"Session 基准标定 INVALID：{result.message}；已保存 {json_path}"
            )
            QMessageBox.warning(
                self,
                "Session 基准标定失败",
                f"未检测到有效棋盘格外参：{result.message}\n"
                "请确认棋盘完整、无障碍且保持在当前相机 ROI 内。",
            )
        self._update_control_states()

    def run_ground_sanity_check(self) -> None:
        """用新的 laser-on 帧检查 Session 棋盘基准面的地面一致性。"""
        if self._session_ground_mode == "disabled":
            self.statusBar().showMessage("Session 基准标定已在配置中禁用")
            return
        if self._active_session_ground_result is None:
            QMessageBox.information(
                self,
                "尚未完成 Session 标定",
                "请先完成有效的 Session 基准标定，再执行激光地面一致性检查。",
            )
            return
        if self._pipeline.extraction_method.strip().lower() != "steger":
            QMessageBox.warning(
                self,
                "正式链路要求 Steger",
                "Laser Ground Sanity Check 必须使用 Steger。"
                "请停流后选择 Steger 并重新开始在线处理。",
            )
            return
        if not self._config.reconstruction.enable_laser_ray_correction:
            QMessageBox.warning(
                self,
                "正式链路缺少 Frozen C1",
                "Laser Ground Sanity Check 要求启用 Frozen C1。"
                "当前配置未启用 C1，检查已拒绝。",
            )
            return

        try:
            result = self._sanity_frame_result()
            if result is None:
                return
            active_generation = self._pipeline.ground_extrinsic_generation
            if (
                result.ground_extrinsic_source != "session"
                or result.ground_extrinsic_generation != active_generation
                or active_generation != self._session_ground_generation
            ):
                raise GroundPointAuditValidationError(
                    "当前 FrameResult 与 active Session ground generation 不一致"
                )
            calibration_host = self._session_ground_calibration_host_monotonic_ns
            if calibration_host is None or int(result.frame.host_monotonic_ns) <= int(
                calibration_host
            ):
                raise GroundPointAuditValidationError(
                    "当前激光帧不是 Session PnP 后的新帧"
                )
            (
                sanity_points,
                selected_indices,
                selected_mask,
                mask_metadata,
            ) = self._session_sanity_selection(result)
            sanity = evaluate_ground_sanity(
                sanity_points,
                ground_extrinsic_source=result.ground_extrinsic_source,
                frame_number=int(result.frame.camera_frame_number),
                session_calibration_frame_number=(
                    self._session_ground_calibration_frame_number
                ),
                frame_host_monotonic_ns=int(result.frame.host_monotonic_ns),
                session_calibration_host_monotonic_ns=(
                    self._session_ground_calibration_host_monotonic_ns
                ),
                session_generation=self._session_ground_generation,
                thresholds=self._config.session_ground_calibration.sanity,
                mask=mask_metadata,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._last_ground_sanity = None
            self._reset_ground_sanity_display()
            self.statusBar().showMessage(f"激光地面一致性检查失败：{error}")
            QMessageBox.warning(self, "激光地面一致性检查", str(error))
            return

        self._last_ground_sanity = sanity
        self._update_ground_sanity_display(sanity)
        sanity_payload = sanity.as_dict()
        sanity_payload["laser_state_assumption"] = "laser_on_assumed_from_user_action"
        sanity_payload["stage_a_height_scale_applied"] = False
        sanity_payload["frame"]["offset_x"] = int(result.frame.offset_x)
        sanity_payload["frame"]["offset_y"] = int(result.frame.offset_y)

        try:
            session_json_path = self._session_ground_json_path()
            session_plane = build_session_ground_plane_provenance(
                self._active_session_ground_result
            )
            frozen_provenance = build_frozen_chain_provenance(
                self._pipeline.package.manifest_path,
                calibration_package_id=result.calibration_package_id,
                calibration_manifest_sha256=result.calibration_manifest_sha256,
                algorithm_config_sha256=result.algorithm_config_sha256,
            )
            metric_points = None
            if (
                result.ground_reference_source != "none"
                or result.ground_reference_status != "inactive"
            ):
                metric_points = result.points_ground
            audit = export_ground_point_audit(
                session_json_path.parent / "ground_spatial_audit",
                session_id=session_json_path.parent.name,
                session_generation=self._session_ground_generation,
                ground_extrinsic_generation=result.ground_extrinsic_generation,
                frame_id=(
                    f"camera_{int(result.frame.camera_frame_number):06d}_"
                    f"host_{int(result.frame.host_monotonic_ns)}"
                ),
                camera_frame_number=int(result.frame.camera_frame_number),
                frame_host_monotonic_ns=int(result.frame.host_monotonic_ns),
                frame_offset=(int(result.frame.offset_x), int(result.frame.offset_y)),
                ground_extrinsic_source=result.ground_extrinsic_source,
                calibration_package_id=result.calibration_package_id,
                calibration_manifest_sha256=result.calibration_manifest_sha256,
                algorithm_config_sha256=result.algorithm_config_sha256,
                pixels_uv=result.pixels_uv,
                points_camera=result.points_camera,
                points_ground_raw=result.points_ground_raw,
                points_ground_metric=metric_points,
                selected_indices=selected_indices,
                selected_mask=selected_mask,
                sanity_points=sanity_points,
                sanity_result=sanity,
                mask_metadata=mask_metadata,
                ground_plane=session_plane["ground_plane"],
                session_pnp=session_plane["session_pnp"],
                frozen_provenance=frozen_provenance,
            )
        except (OSError, TypeError, ValueError, GroundPointAuditValidationError) as error:
            self._last_ground_sanity = None
            self._reset_ground_sanity_display()
            self.statusBar().showMessage(
                f"检查结果已计算，但 point-level audit 导出失败：{error}"
            )
            QMessageBox.warning(self, "Ground point audit 导出失败", str(error))
            self._update_control_states()
            return

        try:
            json_path = merge_session_ground_sanity(
                self._session_ground_json_path(), sanity_payload
            )
        except (OSError, TypeError, ValueError) as error:
            self.statusBar().showMessage(
                f"检查结果已计算，但 Session JSON 保存失败：{error}"
            )
            QMessageBox.warning(self, "Session JSON 保存失败", str(error))
            self._update_control_states()
            return

        if sanity.status == "VALID":
            self.statusBar().showMessage(
                "SESSION_CALIBRATION = VALID；"
                f"Bias {sanity.bias_zg_mm:.4f} mm，"
                f"RMSE {sanity.rmse_zg_mm:.4f} mm，"
                f"mask 点 {sanity.mask.get('selected_point_count', sanity.input_point_count)}，"
                f"有效点 {sanity.valid_point_count}；"
                f"audit 已保存 {audit.audit_dir}；Session JSON {json_path}"
            )
        else:
            warning_text = "; ".join(
                (*sanity.warnings, *sanity.threshold_violations)
            )
            self.statusBar().showMessage(
                f"SESSION_CALIBRATION = INVALID：{warning_text}；已保存 {json_path}"
            )
            QMessageBox.warning(
                self,
                "SESSION_CALIBRATION = INVALID",
                f"基准面激光检查异常：{sanity.message}\n"
                "未执行任何 Bias offset、a*S+b 或 Surface correction。",
            )
        self._update_control_states()

    def _session_sanity_points(
        self, result: FrameResult
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Apply the PnP-derived board interior mask before sanity metrics."""
        points, _, _, metadata = self._session_sanity_selection(result)
        return points, metadata

    def _session_sanity_selection(
        self, result: FrameResult
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        """Return sanity points plus their exact source-row mask linkage."""
        sanity_config = self._config.session_ground_calibration.sanity
        points = self._raw_ground_points(result)
        if not sanity_config.mask_enabled:
            selected_mask = np.ones(len(points), dtype=bool)
            return (
                points,
                np.arange(len(points), dtype=np.int64),
                selected_mask,
                {
                    "enabled": False,
                    "status": "disabled",
                    "source": "configuration",
                    "input_point_count": int(len(points)),
                    "selected_point_count": int(len(points)),
                },
            )

        return self._select_pnp_board_ground_points_with_mask(
            result,
            inset_mm=sanity_config.mask_inset_mm,
        )

    @staticmethod
    def _raw_ground_points(result: FrameResult) -> np.ndarray:
        """Return raw C0+C1+extrinsic points, never a corrected view."""
        return np.asarray(
            result.points_ground_raw
            if result.points_ground_raw is not None
            else result.points_ground,
            dtype=np.float64,
        )

    def _select_pnp_board_ground_points(
        self,
        result: FrameResult,
        *,
        inset_mm: float,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Shared PnP board-mask selection for sanity and ground reference."""
        points, _, _, metadata = self._select_pnp_board_ground_points_with_mask(
            result,
            inset_mm=inset_mm,
        )
        return points, metadata

    def _select_pnp_board_ground_points_with_mask(
        self,
        result: FrameResult,
        *,
        inset_mm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        """Select board points once while retaining exact source indices."""
        points = self._raw_ground_points(result)

        corners = (
            None
            if self._active_session_ground_result is None
            else self._active_session_ground_result.detected_corners
        )
        pixels = getattr(result, "pixels_uv", None)
        if corners is None or pixels is None:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.zeros(len(points), dtype=bool),
                {
                    "enabled": True,
                    "status": "unavailable",
                    "source": GROUND_SUPPORT_PNP_BOARD_MASK,
                    "reason": "reconstructed_source_pixels_missing",
                    "input_point_count": int(len(points)),
                    "selected_point_count": 0,
                },
            )

        mask_offset = self._session_ground_calibration_offset
        if mask_offset is None:
            mask_offset = (int(result.frame.offset_x), int(result.frame.offset_y))
        session_result = self._active_session_ground_result
        if session_result.rvec is None or session_result.tvec is None:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.zeros(len(points), dtype=bool),
                {
                    "enabled": True,
                    "status": "unavailable",
                    "source": GROUND_SUPPORT_PNP_BOARD_MASK,
                    "reason": "session_pnp_pose_missing",
                    "input_point_count": int(len(points)),
                    "selected_point_count": 0,
                },
            )
        try:
            calibration = self._pipeline.calibration_for_reconstruction()
            camera_matrix = np.asarray(calibration["K"], dtype=np.float64).copy()
            camera_matrix[0, 2] -= float(mask_offset[0])
            camera_matrix[1, 2] -= float(mask_offset[1])
            selection = select_board_ground_points_with_mask(
                pixels,
                points,
                rvec=session_result.rvec,
                tvec=session_result.tvec,
                camera_matrix=camera_matrix,
                dist_coeffs=np.asarray(calibration["D"], dtype=np.float64),
                pattern_cols=self._config.session_ground_calibration.pattern_cols,
                pattern_rows=self._config.session_ground_calibration.pattern_rows,
                square_size_mm=self._config.session_ground_calibration.square_size_mm,
                image_offset=mask_offset,
                inset_mm=inset_mm,
                detected_corners=corners,
            )
            return (
                selection.selected_points,
                selection.selected_indices,
                selection.selected_mask,
                selection.metadata,
            )
        except (TypeError, ValueError) as error:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.zeros(len(points), dtype=bool),
                {
                    "enabled": True,
                    "status": "unavailable",
                    "source": GROUND_SUPPORT_PNP_BOARD_MASK,
                    "reason": str(error),
                    "corner_count": int(len(np.asarray(corners).reshape(-1, 2))),
                    "input_point_count": int(len(points)),
                    "selected_point_count": 0,
                },
            )

    def _ground_reference_support_points(
        self,
        result: FrameResult,
    ) -> tuple[np.ndarray, dict[str, object], str, float | None]:
        """Resolve an explicit support source before fitting the reference."""
        support_config = self._config.session_ground_calibration.ground_reference
        source = str(
            self.ground_reference_source_combo.currentData()
            or support_config.support_source
        ).strip().lower()
        if source == GROUND_SUPPORT_PNP_BOARD_MASK:
            if self._active_session_ground_result is None:
                raise MeasurementError(
                    "pnp_board_mask 需要有效 Session PnP；"
                    "如不做 PnP，请切换为 manual_ground_roi 并先选择基准区域。"
                )
            points, metadata = self._select_pnp_board_ground_points(
                result,
                inset_mm=float(support_config.mask_inset_mm),
            )
            if metadata.get("status") != "applied":
                raise MeasurementError(
                    "pnp_board_mask 未能从当前帧选出棋盘物理 mask 内的激光点"
                )
            return points, metadata, source, float(support_config.mask_inset_mm)

        if source == GROUND_SUPPORT_MANUAL_ROI:
            analysis = self._analysis_window
            if analysis is None:
                self.open_frame_analysis()
                raise MeasurementError(
                    "请先在“单帧测量与区域选择”中添加基准区域，"
                    "再点击 Session 激光地面基准。"
                )
            roi_rects = getattr(analysis, "baseline_regions_full", ())
            if not roi_rects:
                raise MeasurementError(
                    "manual_ground_roi 尚未选择基准区域；请在单帧工具中添加基准区域。"
                )
            pixels = getattr(result, "pixels_uv", None)
            if pixels is None:
                raise MeasurementError("当前帧没有可用于 manual_ground_roi 的像素坐标")
            points, metadata = select_manual_ground_roi_points(
                pixels,
                self._raw_ground_points(result),
                roi_rects,
            )
            if metadata.get("status") != "applied":
                raise MeasurementError(
                    "manual_ground_roi 内没有当前帧的有效激光点"
                )
            return points, metadata, source, None

        raise MeasurementError(f"不支持的 ground support source: {source}")

    def _sanity_frame_result(self) -> FrameResult | None:
        """Return a post-PnP frame; idle mode captures and runs the formal pipeline."""
        if self._controller.running:
            result = self._last_result
            if result is None:
                QMessageBox.information(
                    self,
                    "等待激光帧",
                    "当前尚无实时帧，请等待新的激光-on 帧后再检查。",
                )
                return None
            frame_host = int(result.frame.host_monotonic_ns)
            calibration_host = self._session_ground_calibration_host_monotonic_ns
            if (
                result.ground_extrinsic_source != "session"
                or calibration_host is None
                or frame_host <= calibration_host
            ):
                QMessageBox.information(
                    self,
                    "等待 Session 后新帧",
                    "请保持棋盘不动并打开激光，等待 Session 外参生效后的新帧，"
                    "再点击检查。",
                )
                return None
            return result

        if self._session is None:
            QMessageBox.information(self, "未连接", "请先连接相机")
            return None
        try:
            self._session.start()
            frame = self._session.get_frame(self._session.config.timeout_ms)
            self._show_raw_frame(frame)
            # This is the same FramePipeline path used by online streaming:
            # Steger -> frozen C0/C1 -> current (Session) ground extrinsic.
            result = self._pipeline.run_frame(frame)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            QMessageBox.warning(
                self,
                "激光地面一致性检查",
                f"无法采集或重建新的 laser-on 帧：{error}",
            )
            return None
        finally:
            try:
                self._session.stop()
            except Exception:
                pass
        self._show_result(result)
        return result

    def _ground_reference_frame_result(self) -> FrameResult | None:
        """Return a current laser-on frame without requiring Session PnP."""
        if self._controller.running:
            result = self._last_result
            if result is None:
                QMessageBox.information(
                    self,
                    "等待激光帧",
                    "当前尚无实时帧，请等待新的无障碍激光-on 帧后再标定。",
                )
                return None
            return result

        if self._session is None:
            QMessageBox.information(self, "未连接", "请先连接相机")
            return None
        try:
            self._session.start()
            frame = self._session.get_frame(self._session.config.timeout_ms)
            self._show_raw_frame(frame)
            result = self._pipeline.run_frame(frame)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            QMessageBox.warning(
                self,
                "Session 激光地面基准",
                f"无法采集或重建当前基准面激光帧：{error}",
            )
            return None
        finally:
            try:
                self._session.stop()
            except Exception:
                pass
        self._show_result(result)
        return result

    def calibrate_session_ground_reference(self) -> None:
        """Fit and freeze an explicitly supported ground profile for this session.

        The fitter itself remains the existing ``a*S+b`` implementation.  This
        entry point is deliberately responsible for resolving provenance first,
        so an arbitrary full-frame point cloud can never become a runtime ground
        reference.
        """
        if self._pipeline.session_ground_reference is not None:
            QMessageBox.information(
                self,
                "Session 激光地面基准",
                "当前 Session 的线性地面基准已经冻结；如需重做，请断开并重新连接相机。",
            )
            return
        result = self._ground_reference_frame_result()
        if result is None:
            return
        try:
            support_points, support_metadata, support_source, mask_inset_mm = (
                self._ground_reference_support_points(result)
            )
            reference = fit_session_ground_reference_from_support(
                support_points,
                self._config.measurement,
                support_source=support_source,
                active_ground_extrinsic_source=self._pipeline.ground_extrinsic_source,
                ground_extrinsic_generation=(
                    self._pipeline.ground_extrinsic_generation
                ),
                frame_host_monotonic_ns=int(result.frame.host_monotonic_ns),
                mask_inset_mm=mask_inset_mm,
                support_metadata=support_metadata,
            )
            self._pipeline.apply_session_ground_reference(reference)
        except (MeasurementError, TypeError, ValueError) as error:
            QMessageBox.warning(
                self,
                "Session 激光地面基准失败",
                f"当前帧不能作为空基准面拟合：{error}",
            )
            return

        self._last_ground_reference = reference
        payload = reference.as_dict()
        payload["frame"] = {
            "camera_frame_number": int(result.frame.camera_frame_number),
            "host_monotonic_ns": int(result.frame.host_monotonic_ns),
            "offset_x": int(result.frame.offset_x),
            "offset_y": int(result.frame.offset_y),
            "ground_extrinsic_source": self._pipeline.ground_extrinsic_source,
            "ground_extrinsic_generation": (
                self._pipeline.ground_extrinsic_generation
            ),
        }
        payload["ground_extrinsic_source"] = self._pipeline.ground_extrinsic_source
        payload["ground_extrinsic_generation"] = (
            self._pipeline.ground_extrinsic_generation
        )
        payload["provenance"] = {
            "source": reference.provenance_source,
            "active_ground_extrinsic_source": (
                reference.active_ground_extrinsic_source
            ),
            "ground_extrinsic_generation": reference.ground_extrinsic_generation,
            "frame_host_monotonic_ns": reference.frame_host_monotonic_ns,
            "mask_inset_mm": reference.mask_inset_mm,
            "point_count": reference.point_count,
            "inlier_count": reference.inlier_count,
            "valid_s_range_mm": [
                float(reference.valid_s_range_mm[0]),
                float(reference.valid_s_range_mm[1]),
            ],
            "slope": reference.slope_z_per_mm,
            "intercept": reference.intercept_z_mm,
            "rmse_mm": reference.rmse_mm,
            "support": dict(reference.support_metadata),
        }
        try:
            merge_session_ground_reference(
                self._session_ground_json_path(),
                payload,
                ground_extrinsic_source=self._pipeline.ground_extrinsic_source,
            )
        except (OSError, TypeError, ValueError) as error:
            # Runtime activation is already complete; report persistence
            # separately so a writable-output problem cannot undo the session.
            self.statusBar().showMessage(f"地面基准已应用，但 Session JSON 保存失败：{error}")

        self._invalidate_live_reconstruction_after_ground_update()
        if not self._controller.running:
            try:
                refreshed = self._pipeline.run_frame(result.frame)
            except (RuntimeError, TypeError, ValueError) as error:
                self.statusBar().showMessage(
                    f"地面基准已应用；等待下一帧刷新点云：{error}"
                )
            else:
                self._show_result(refreshed)
        self._update_ground_reference_display()
        self._update_control_states()
        lower, upper = reference.valid_s_range_mm
        self.statusBar().showMessage(
            "SESSION_GROUND_REFERENCE = VALID | "
            f"source {reference.provenance_source} | "
            f"points {reference.point_count}/{reference.inlier_count} | "
            f"slope {reference.slope_z_per_mm:.7f} mm/mm | "
            f"RMSE {reference.rmse_mm:.4f} mm | "
            f"S {lower:.2f}~{upper:.2f} mm | "
            "已冻结到当前 Session（不修改 reference 标定）"
        )

    def load_frozen_session_ground(self, path: str | Path | None = None) -> bool:
        """Load the validated Ground-5C A-2 model for the active Session PnP.

        This is deliberately a load-only path.  It never calls a fitter and
        never writes the selected JSON.  If a frame is already displayed, its
        retained raw ground points are re-leveled in memory so the UI changes
        without re-running laser extraction.
        """
        if not self._has_valid_session_pnp():
            message = "请先完成有效的 Session PnP，再加载 Frozen Session Ground"
            self.statusBar().showMessage(message)
            QMessageBox.information(self, "Frozen Session Ground", message)
            return False

        if path is None:
            default_path = (
                Path(__file__).resolve().parents[2]
                / "outputs"
                / "ground5c_frozen_session_linear_0821"
                / "frozen_session_linear.json"
            )
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "加载 Frozen Session Ground",
                str(default_path if default_path.exists() else default_path.parent),
                "JSON 文件 (*.json)",
            )
            if not selected:
                return False
            path = selected

        previous_result = self._last_result
        try:
            reference = load_frozen_session_ground_reference(
                path,
                active_ground_extrinsic_source=self._pipeline.ground_extrinsic_source,
                ground_extrinsic_generation=(
                    self._pipeline.ground_extrinsic_generation
                ),
            )
            self._pipeline.apply_session_ground_reference(reference)
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.warning(
                self,
                "Frozen Session Ground 加载失败",
                str(error),
            )
            return False

        self._last_ground_reference = reference
        self._ground_reference_invalid_reason = None
        self._invalidate_live_reconstruction_after_ground_update()
        self._update_ground_reference_display()

        if previous_result is not None:
            raw_points = previous_result.points_ground_raw
            if raw_points is None:
                raw_points = previous_result.points_ground
            raw_points = np.ascontiguousarray(
                np.asarray(raw_points, dtype=np.float64).copy()
            )
            try:
                corrected, metadata = self._pipeline.apply_ground_reference_to_points(
                    raw_points
                )
                refreshed = replace(
                    previous_result,
                    points_ground=corrected,
                    points_ground_raw=raw_points,
                    section_xz=(
                        np.ascontiguousarray(corrected[:, (0, 2)])
                        if len(corrected)
                        else np.empty((0, 2), dtype=np.float64)
                    ),
                    ground_extrinsic_source=self._pipeline.ground_extrinsic_source,
                    ground_extrinsic_generation=(
                        self._pipeline.ground_extrinsic_generation
                    ),
                    **metadata,
                )
            except (TypeError, ValueError) as error:
                self.statusBar().showMessage(
                    f"Frozen Session Ground 已加载，等待下一帧刷新：{error}"
                )
            else:
                self._show_result(refreshed)

        self._update_control_states()
        lower, upper = reference.valid_s_range_mm
        self.statusBar().showMessage(
            "GROUND5C_FROZEN_SESSION = VALID | "
            f"physical_S a={reference.slope_z_per_mm:.9f} "
            f"b={reference.intercept_z_mm:.6f} mm | "
            f"S {lower:.2f}~{upper:.2f} mm | "
            f"JSON SHA256 {reference.frozen_json_sha256}"
        )
        return True

    def _session_calibration_frame(self) -> CapturedFrame | None:
        if self._last_result is not None:
            return self._last_result.frame
        if self._session is None:
            QMessageBox.information(self, "未连接", "请先连接相机")
            return None
        if self._controller.running:
            QMessageBox.information(
                self,
                "等待图像",
                "当前正在等待实时帧，请稍后再点击 Session 基准标定",
            )
            return None
        try:
            self._session.start()
            frame = self._session.get_frame(self._session.config.timeout_ms)
        except Exception as error:
            QMessageBox.warning(self, "Session 基准标定", f"无法采集棋盘格帧：{error}")
            return None
        finally:
            try:
                self._session.stop()
            except Exception:
                pass
        self._show_raw_frame(frame)
        return frame

    def _session_ground_json_path(self) -> Path:
        configured = self._config.session_ground_calibration.output
        if configured is not None:
            return configured
        if self._config.output is not None:
            return self._config.output.directory / "session_ground_calibration.json"
        return (
            Path(__file__).resolve().parents[1]
            / "output"
            / "session_ground_calibration.json"
        )

    def _set_session_ground_attempt(
        self,
        result: SessionGroundExtrinsic | None,
        message: str,
    ) -> None:
        if result is None:
            self._last_session_ground_result = SessionGroundExtrinsic(
                status="invalid_gui",
                message=message,
            )
        else:
            self._last_session_ground_result = result
        self._update_session_ground_display(self._last_session_ground_result, None)
        self._update_ground_source_label()
        self._update_control_states()
        self.statusBar().showMessage(message)
        QMessageBox.warning(self, "Session 基准标定", message)

    def _update_session_ground_display(
        self,
        result: SessionGroundExtrinsic | None,
        delta: tuple[float, float] | None,
    ) -> None:
        if result is None or result.status != "success":
            suffix = " · 保留 session" if self._pipeline.ground_extrinsic_source == "session" else ""
            self.session_ground_valid_label.setText(f"INVALID{suffix}")
            self.session_ground_corner_label.setText(
                str(len(result.detected_corners))
                if result is not None and result.detected_corners is not None
                else "—"
            )
            self.session_ground_rmse_label.setText("—")
            self.session_ground_delta_translation_label.setText("—")
            self.session_ground_delta_rotation_label.setText("—")
            self._update_session_calibration_quality_display()
            return
        self.session_ground_valid_label.setText("VALID")
        self.session_ground_corner_label.setText(
            str(len(result.detected_corners))
            if result.detected_corners is not None
            else "0"
        )
        self.session_ground_rmse_label.setText(
            "—"
            if result.reprojection_rmse_px is None
            else f"{result.reprojection_rmse_px:.4f} px"
        )
        self.session_ground_delta_translation_label.setText(
            "—" if delta is None else f"{delta[0]:.3f} mm"
        )
        self.session_ground_delta_rotation_label.setText(
            "—" if delta is None else f"{delta[1]:.3f}°"
        )
        self._update_session_calibration_quality_display()

    def _update_ground_sanity_display(
        self, result: GroundSanityResult | None
    ) -> None:
        if result is None:
            self._reset_ground_sanity_display()
            return
        self.session_sanity_status_label.setText(
            f"SESSION_CALIBRATION = {result.status}"
        )
        self.session_sanity_bias_label.setText(
            _format_mm(result.bias_zg_mm)
        )
        self.session_sanity_rmse_label.setText(
            _format_mm(result.rmse_zg_mm)
        )
        self.session_sanity_p95_label.setText(
            _format_mm(result.p95_abs_zg_mm)
        )
        self.session_sanity_max_label.setText(
            _format_mm(result.max_abs_zg_mm)
        )
        self.session_sanity_slope_label.setText(
            "—"
            if result.ground_slope_mm_per_mm is None
            else f"{result.ground_slope_mm_per_mm:.6f} mm/mm"
        )
        mask = result.mask
        if mask.get("enabled") is False:
            self.session_sanity_mask_count_label.setText("未启用")
        elif mask.get("status") == "applied":
            self.session_sanity_mask_count_label.setText(
                f"{mask.get('selected_point_count', 0)} / "
                f"{mask.get('input_point_count', result.input_point_count)}"
            )
        else:
            self.session_sanity_mask_count_label.setText("不可用")
        self.session_sanity_valid_count_label.setText(
            f"{result.valid_point_count} / {result.input_point_count}"
        )

    def _reset_ground_sanity_display(self) -> None:
        if not hasattr(self, "session_sanity_status_label"):
            return
        self.session_sanity_status_label.setText("SESSION_CALIBRATION = 未检查")
        self.session_sanity_bias_label.setText("—")
        self.session_sanity_rmse_label.setText("—")
        self.session_sanity_p95_label.setText("—")
        self.session_sanity_max_label.setText("—")
        self.session_sanity_slope_label.setText("—")
        self.session_sanity_mask_count_label.setText("—")
        self.session_sanity_valid_count_label.setText("—")

    def _update_ground_source_label(self) -> None:
        self.ground_source_label.setText(self._pipeline.ground_extrinsic_source)
        self._update_ground_reference_display()

    def _update_ground_reference_display(self) -> None:
        reference = self._pipeline.session_ground_reference
        if reference is None:
            self.ground_reference_status_label.setText(
                self._ground_reference_invalid_reason or "未启用"
            )
            self.ground_reference_slope_label.setText("—")
            self.ground_reference_intercept_label.setText("—")
            self.ground_reference_rmse_label.setText("—")
            self.ground_reference_range_label.setText("—")
            self.ground_reference_points_label.setText("—")
            self.ground_reference_coordinate_label.setText("—")
            self.ground_reference_json_sha_label.setText("—")
            self.ground_reference_runtime_label.setText("—")
            self.ground_reference_status_label.setToolTip("")
            self.ground_reference_json_sha_label.setToolTip("")
            self.ground_reference_runtime_label.setToolTip("")
            return
        self._ground_reference_invalid_reason = None
        coordinate = reference.coordinate or "legacy"
        frozen = bool(reference.frozen_json_sha256)
        self.ground_reference_status_label.setText(
            f"{reference.status} · {'Frozen' if frozen else reference.provenance_source}"
        )
        self.ground_reference_status_label.setToolTip(
            f"source={reference.provenance_source}; "
            f"extrinsic={reference.active_ground_extrinsic_source}; "
            f"generation={reference.ground_extrinsic_generation}; "
            f"coordinate={coordinate}; "
            f"a={reference.slope_z_per_mm:.9f}; "
            f"b={reference.intercept_z_mm:.6f}; "
            f"json_sha256={reference.frozen_json_sha256 or 'none'}"
        )
        self.ground_reference_slope_label.setText(
            f"{reference.slope_z_per_mm:.7f} mm/mm"
        )
        self.ground_reference_intercept_label.setText(
            f"{reference.intercept_z_mm:.4f} mm"
        )
        self.ground_reference_rmse_label.setText(
            f"{reference.rmse_mm:.4f} mm"
        )
        lower, upper = reference.valid_s_range_mm
        self.ground_reference_range_label.setText(
            f"{lower:.2f} ~ {upper:.2f} mm"
        )
        self.ground_reference_points_label.setText(
            (
                f"{reference.inlier_count} formal bins"
                if reference.frozen_json_sha256
                else f"{reference.inlier_count}/{reference.point_count}"
            )
        )
        self.ground_reference_coordinate_label.setText(
            reference.coordinate or "—"
        )
        self.ground_reference_json_sha_label.setText(
            _compact_sha256(reference.frozen_json_sha256) or "非 Frozen JSON"
        )
        self.ground_reference_json_sha_label.setToolTip(
            reference.frozen_json_sha256 or "非 Frozen JSON"
        )
        self.ground_reference_runtime_label.setText("等待新帧")
        self.ground_reference_runtime_label.setToolTip("等待新帧")

    def _update_ground_reference_runtime_display(
        self, result: FrameResult | None
    ) -> None:
        if result is None or result.ground_reference_source == "none":
            self.ground_reference_runtime_label.setText("未应用")
            self.ground_reference_runtime_label.setToolTip("未应用")
            return
        self.ground_reference_runtime_label.setText(
            f"{result.ground_reference_applied_count}/"
            f"{result.ground_reference_out_of_range_count} 点"
        )
        self.ground_reference_runtime_label.setToolTip(
            f"applied={result.ground_reference_applied_count}; "
            f"out_of_range={result.ground_reference_out_of_range_count}; "
            f"status={result.ground_reference_status}"
        )

    def _replace_pipeline(self, pipeline: FramePipeline) -> None:
        """Install a pipeline and drop session state when its package changes."""
        previous_identity = self._pipeline.calibration_package_identity
        pipeline.set_height_correction_mode(self._height_correction_mode)
        self._pipeline = pipeline
        if previous_identity == pipeline.calibration_package_identity:
            return

        # A new calibration package changes the camera/laser geometry.  Do not
        # replay either the old PnP pose or a ground model fitted from it into
        # the newly created pipeline.  Normal stop/start with the same package
        # takes the fast path above and preserves both runtime states.
        self._active_session_ground_result = None
        self._last_session_ground_result = SessionGroundExtrinsic(
            status="invalid_package",
            message="calibration package changed",
        )
        self._session_ground_generation = 0
        self._session_ground_calibration_frame_number = None
        self._session_ground_calibration_host_monotonic_ns = None
        self._session_ground_calibration_offset = None
        self._session_calibration_repeatability = None
        self._session_calibration_qa = None
        self._session_calibration_quality = None
        self._last_ground_sanity = None
        self._last_ground_reference = None
        self._ground_reference_invalid_reason = (
            "标定包已切换，激光地面基准需重新标定"
        )
        self._pipeline.reset_ground_extrinsic(generation=0)
        if hasattr(self, "session_ground_valid_label"):
            self._update_session_ground_display(self._last_session_ground_result, None)
            self._reset_ground_sanity_display()
            self._update_ground_source_label()

    def _invalidate_ground_reference_for_extrinsic_change(self, reason: str) -> None:
        """Invalidate the fitted ground model when active R/t changes."""
        self._last_ground_reference = None
        self._pipeline.reset_session_ground_reference()
        self._ground_reference_invalid_reason = reason
        self._update_ground_reference_display()
        self.statusBar().showMessage(reason)

    def _clear_session_ground_runtime(self) -> None:
        self._session_calibration_active = False
        self._session_calibration_capturing = False
        self._session_calibration_capture_after_reconfigure = False
        self._session_calibration_capture_generation += 1
        self._session_calibration_capture_started_host_monotonic_ns = None
        self._session_calibration_worker_busy = False
        self._session_calibration_restore_config = None
        self._session_calibration_debounce_timer.stop()
        self._pending_camera_reconfigure = None
        self._camera_reconfigure_token += 1
        self._camera_reconfigure_in_progress = False
        self._camera_config_worker_busy = False
        if hasattr(self, "extracted_image_view"):
            self._restore_session_calibration_preview()
        self._active_session_ground_result = None
        self._last_session_ground_result = None
        self._session_ground_calibration_frame_number = None
        self._session_ground_calibration_host_monotonic_ns = None
        self._session_ground_generation = 0
        self._session_ground_calibration_offset = None
        self._session_calibration_repeatability = None
        self._session_calibration_qa = None
        self._session_calibration_quality = None
        self._last_ground_sanity = None
        self._last_ground_reference = None
        self._ground_reference_invalid_reason = None
        self._pipeline.reset_ground_extrinsic(generation=0)
        self._pipeline.reset_session_ground_reference()
        if hasattr(self, "session_ground_valid_label"):
            self.session_ground_valid_label.setText("INVALID · 未标定")
            self.session_ground_corner_label.setText("—")
            self.session_ground_rmse_label.setText("—")
            self.session_ground_delta_translation_label.setText("—")
            self.session_ground_delta_rotation_label.setText("—")
            self.session_ground_frames_label.setText("—")
            self.session_ground_repeat_translation_label.setText("—")
            self.session_ground_repeat_rotation_label.setText("—")
            self.session_ground_qa_label.setText("Session PnP QA: —")
            self.session_ground_quality_label.setText("棋盘质量：—")
            self._reset_ground_sanity_display()
            self._update_ground_source_label()

    # Backward-compatible internal name for callers from older integrations;
    # only explicit disconnect/new-device paths call the clearing operation.
    def _reset_session_ground_runtime(self) -> None:
        self._clear_session_ground_runtime()

    def export_current_frame(self) -> Path:
        """导出当前实时帧的中心点、重建点云和地面系 PLY。

        导出目录采用与离线工具相同的 ``*_measure`` 命名规则，但放在
        ``output/online_measurements`` 下，避免覆盖已有离线结果。
        """
        result = self._last_result
        if result is None:
            raise RuntimeError("当前尚无可导出的实时帧")

        reconstruction = reconstruct_uv_to_ground(
            result.centers_uv_full,
            self._pipeline.calibration_for_reconstruction(),
            self._config.reconstruction,
        )
        points_ground, ground_reference_metadata = (
            self._pipeline.apply_ground_reference_to_points(
                reconstruction.points_ground
            )
        )
        root = (
            self._config.output.directory / "online_measurements"
            if self._config.output is not None
            else Path(__file__).resolve().parents[1]
            / "output"
            / "online_measurements"
        )
        image_name = f"frame_{result.frame.camera_frame_number:06d}.tiff"
        target_dir = next_measurement_dir(image_name, root)
        save_laser_centers_csv(
            target_dir / "laser_center.csv", result.centers_uv_full
        )
        save_reconstructed_points_csv(
            target_dir / "full_points.csv",
            reconstruction.pixels_uv,
            reconstruction.points_camera,
            points_ground,
        )
        save_ground_pointcloud_ply(
            target_dir / "full_laser_ground.ply",
            points_ground,
        )
        save_image_png(
            target_dir / "overlay.png",
            cv2.cvtColor(result.overlay_rgb, cv2.COLOR_RGB2BGR),
        )
        height_result = self._current_height_result()
        frame_shadow = {
            **height_result.as_dict(),
            "v_min": result.v_min,
            "v_median": result.v_median,
            "v_max": result.v_max,
            "point_count": int(reconstruction.point_count),
            "c1_clamp_status": result.c1_clamp_status,
            "ground_reference_status": ground_reference_metadata[
                "ground_reference_status"
            ],
        }
        payload = {
            "source": "online",
            "frame": {
                "camera_frame_number": int(result.frame.camera_frame_number),
                "camera_timestamp_ticks": (
                    None
                    if result.frame.camera_timestamp_ticks is None
                    else int(result.frame.camera_timestamp_ticks)
                ),
                "host_timestamp_ns": int(result.frame.host_timestamp_ns),
                "offset_x": int(result.frame.offset_x),
                "offset_y": int(result.frame.offset_y),
                "width": int(result.frame.image.shape[1]),
                "height": int(result.frame.image.shape[0]),
                "dtype": str(result.frame.image.dtype),
            },
            "extraction_method": self._pipeline.extraction_method,
            "calibration_package_id": result.calibration_package_id,
            "calibration_manifest_sha256": result.calibration_manifest_sha256,
            "algorithm_config_sha256": result.algorithm_config_sha256,
            "ground_extrinsic_source": self._pipeline.ground_extrinsic_source,
            "ground_extrinsic_generation": self._pipeline.ground_extrinsic_generation,
            "ground_reference": (
                None
                if self._pipeline.session_ground_reference is None
                else self._pipeline.session_ground_reference.as_dict()
            ),
            "ground_reference_runtime": {
                **ground_reference_metadata,
                "valid_s_range_mm": (
                    list(ground_reference_metadata["ground_reference_valid_s_range_mm"])
                    if ground_reference_metadata["ground_reference_valid_s_range_mm"]
                    is not None
                    else None
                ),
            },
            **height_result.as_dict(),
            "height_shadow": frame_shadow,
            "correction": {
                "mode": self._config.correction.mode,
                "active_mode": self._height_correction_mode,
                "stage_a_height_scale_enabled": (
                    self._config.correction.stage_a_height_scale_enabled
                ),
                "stage_a_height_scale_config": (
                    str(self._config.correction.stage_a_height_scale_config)
                    if self._config.correction.stage_a_height_scale_config
                    else None
                ),
                "hb2_height_correction_config": (
                    str(self._config.correction.hb2_height_correction_config)
                    if self._config.correction.hb2_height_correction_config
                    else None
                ),
                "hb2_q2_policy": self._config.correction.hb2_q2_policy,
            },
            "point_counts": {
                "laser_centers_2d": int(len(result.centers_uv_full)),
                "reconstructed": int(reconstruction.point_count),
            },
            "filtered": {key: int(value) for key, value in reconstruction.filtered.items()},
            "files": {
                "laser_center_csv": "laser_center.csv",
                "full_points_csv": "full_points.csv",
                "full_laser_ground_ply": "full_laser_ground.ply",
                "overlay_png": "overlay.png",
            },
        }
        save_measurement_json(target_dir / "result.json", payload)
        return target_dir

    def _current_height_result(self) -> HeightCorrectionResult:
        """Use an already completed ROI analysis when one exists.

        A live frame has no height ROI by itself, so its exported height fields
        remain explicitly unmeasured until the linked analysis window finishes
        a measurement.
        """
        measurements = getattr(self._analysis_window, "_last_measurements", ())
        height_raw = (
            measurements[0].height_mean_mm
            if measurements
            else None
        )
        analysis = self._analysis_window
        obstacle_recons = getattr(analysis, "_last_obstacle_reconstructions", ())
        if measurements and obstacle_recons and hasattr(
            analysis, "_height_correction_result"
        ):
            return analysis._height_correction_result(
                height_raw,
                obstacle_recons[0],
                mode_override=self._height_correction_mode,
            )
        result = self._last_result
        return resolve_height_correction(
            height_raw,
            q1=None if result is None else result.q1,
            q2=None if result is None else result.q2,
            q2_in_domain=None if result is None else result.q2_in_domain,
            system=self._pipeline.system,
            correction=self._config.correction,
            mode_override=self._height_correction_mode,
        )

    def _export_current_frame(self) -> None:
        try:
            target_dir = self.export_current_frame()
        except Exception as error:  # 导出失败应留在当前实时会话内
            QMessageBox.critical(self, "当前帧导出失败", str(error))
            return
        self.statusBar().showMessage(f"当前帧点云/CSV 已保存到 {target_dir}")

    def open_frame_analysis(self) -> None:
        """打开离线同款单帧分析窗口，支持 ROI 与高度/长度测量。"""
        result = self._last_result
        if result is None:
            QMessageBox.information(self, "没有图像", "当前尚无可分析帧")
            return
        if self._analysis_window is not None:
            self._analysis_window.show()
            self._analysis_window.raise_()
            self._analysis_window.activateWindow()
            return
        try:
            from gui.main_window import MainWindow

            analysis = MainWindow(
                self._config,
                system=self._camera_backend.name,
                runtime_calibration=self._pipeline.calibration_for_reconstruction(),
                ground_extrinsic_source=self._pipeline.ground_extrinsic_source,
                runtime_ground_reference=self._pipeline.session_ground_reference,
                height_correction_mode=self._height_correction_mode,
            )
            analysis.load_external_frame(
                result.frame.image,
                result.centers_uv_full,
                image_name=f"frame_{result.frame.camera_frame_number:06d}.tiff",
                image_offset=(result.frame.offset_x, result.frame.offset_y),
            )
        except Exception as error:  # 在线入口必须把配置/依赖错误转成界面提示
            QMessageBox.critical(self, "单帧分析启动失败", str(error))
            return
        self._analysis_window = analysis
        analysis.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        analysis.destroyed.connect(
            lambda _object=None: setattr(self, "_analysis_window", None)
        )
        analysis.show()
        analysis.raise_()
        analysis.activateWindow()

    def _set_image_view_mode(self, mode: str) -> None:
        if mode not in {"width", "fit"}:
            raise ValueError(f"未知图像视野模式: {mode}")
        self._image_view_mode = mode
        self._reset_image_views()

    def _set_image_preview_rotated(self, rotated: bool) -> None:
        """Rotate both previews without changing image or algorithm coordinates."""
        self._image_preview_rotated = bool(rotated)
        for item, boundary in (
            (self.raw_image_item, self.raw_image_boundary),
            (self.extracted_image_item, self.extracted_image_boundary),
        ):
            image = item.image
            transform = _image_preview_transform(image, self._image_preview_rotated)
            item.setTransform(transform)
            boundary.setTransform(transform)
        extracted_transform = _image_preview_transform(
            self.extracted_image_item.image, self._image_preview_rotated
        )
        self.extracted_corner_boundary.setTransform(extracted_transform)
        self.extracted_corner_scatter.setTransform(extracted_transform)
        self._reset_image_views()

    def _reset_image_views(self) -> None:
        for view, item in (
            (self.raw_image_view, self.raw_image_item),
            (self.extracted_image_view, self.extracted_image_item),
        ):
            image = item.image
            if isinstance(image, np.ndarray) and image.size:
                _fit_image_view(
                    view,
                    image,
                    self._image_view_mode,
                    rotated=self._image_preview_rotated,
                )

    def _show_raw_frame(self, frame: CapturedFrame) -> None:
        self._handle_session_calibration_frame(frame)
        if (
            self._session_calibration_active
            and not self._session_calibration_capturing
            and not self._session_calibration_preview_image_initialized
        ):
            self._show_session_calibration_image(frame.image)
            self._session_calibration_preview_image_initialized = True
        if self.tabs.currentIndex() != 0:
            return
        image_shape = frame.image.shape[:2]
        shape_changed = image_shape != self._raw_view_shape
        now = time.monotonic()
        if not shape_changed and now - self._last_raw_preview_at < 0.10:
            return
        self._last_raw_preview_at = now
        raw_levels = (0, 255) if frame.image.dtype == np.uint8 else (0, 4095)
        self.raw_image_item.setImage(
            frame.image,
            autoLevels=False,
            levels=raw_levels,
        )
        _set_image_boundary(self.raw_image_boundary, frame.image)
        if shape_changed:
            self._raw_view_shape = image_shape
            transform = _image_preview_transform(
                frame.image, self._image_preview_rotated
            )
            self.raw_image_item.setTransform(transform)
            self.raw_image_boundary.setTransform(transform)
            _fit_image_view(
                self.raw_image_view,
                frame.image,
                self._image_view_mode,
                rotated=self._image_preview_rotated,
            )

    def _show_result(self, result: FrameResult) -> None:
        # A result may already be queued when Session PnP switches the
        # pipeline from reference to session. Do not let that old result
        # repopulate the point-cloud view after it was invalidated.
        if (
            self._pipeline.ground_extrinsic_source == "session"
            and result.ground_extrinsic_source != "session"
        ):
            return
        active_generation = self._pipeline.ground_extrinsic_generation
        if (
            result.ground_extrinsic_generation is not None
            and result.ground_extrinsic_generation != active_generation
        ):
            return
        active_ground_reference = self._pipeline.session_ground_reference
        if active_ground_reference is not None:
            if (
                result.ground_reference_source
                != active_ground_reference.provenance_source
            ):
                return
            if (
                active_ground_reference.ground_extrinsic_generation is not None
                and result.ground_extrinsic_generation is not None
                and result.ground_extrinsic_generation
                != active_ground_reference.ground_extrinsic_generation
            ):
                return
        if (
            active_ground_reference is None
            and result.ground_reference_source != "none"
        ):
            return
        self._update_ground_reference_runtime_display(result)
        first_result = self._last_result is None
        self._last_result = result
        self._recorder.log_shadow(
            {
                "camera_frame_number": result.frame.camera_frame_number,
                "host_timestamp_ns": result.frame.host_timestamp_ns,
                "height_raw": result.height_raw,
                "height_h1": result.height_h1,
                "height_hb2": result.height_hb2,
                "active_height_correction": result.active_height_correction,
                "active_height": result.active_height,
                "active_height_valid": result.active_height_valid,
                "active_height_status": result.active_height_status,
                "q1": result.q1,
                "q2": result.q2,
                "q2_in_domain": result.q2_in_domain,
                "hb2_q2_status": result.hb2_q2_status,
                "v_min": result.v_min,
                "v_median": result.v_median,
                "v_max": result.v_max,
                "point_count": result.point_count,
                "c1_clamp_status": result.c1_clamp_status,
                "ground_reference_status": result.ground_reference_status,
            }
        )
        if first_result:
            self._update_control_states()
        if self._session_calibration_active:
            return
        now = time.monotonic()
        if now - self._last_render_at < 0.075:
            return
        self._last_render_at = now
        while self._trail and now - self._trail[0][0] > 1.0:
            self._trail.popleft()
        current = result.points_ground
        self._update_point_summary(current)
        self.height_correction_mode_label.setText(
            result.active_height_correction
        )
        stage_b_status = {
            "applied": "已应用",
            "HB2_Q2_OOD": "超出有效域",
            "HB2_Q2_MISSING": "q2 缺失",
            "HB2_Q2_INVALID": "q2 无效",
            "HB2_Q2_CLAMPED_DIAGNOSTIC": "诊断 clamp（已标记）",
            "not_measured": "未测量",
            "not_configured": "未配置",
            "unsupported_system": "非 Daheng，不适用",
            "invalid_height": "高度无效",
        }.get(result.hb2_q2_status, result.hb2_q2_status)
        self.height_shadow_label.setText(
            "raw={} | h1={} | hb2={} | q1={} | q2={} | q2_in_domain={} | "
            "v={:.1f}/{:.1f}/{:.1f} | points={} | C1={} | ground={} | "
            "Stage-B 状态: {}".format(
                _format_optional(result.height_raw),
                _format_optional(result.height_h1),
                _format_optional(result.height_hb2),
                _format_optional(result.q1),
                _format_optional(result.q2),
                result.q2_in_domain,
                result.v_min if result.v_min is not None else float("nan"),
                result.v_median if result.v_median is not None else float("nan"),
                result.v_max if result.v_max is not None else float("nan"),
                result.point_count,
                result.c1_clamp_status,
                result.ground_reference_status,
                stage_b_status,
            )
        )
        self._refresh_adaptive_control_layout()
        if len(current):
            self._trail.append((now, current[::4].copy()))
        current_tab = self.tabs.currentIndex()
        if current_tab == 0:
            image_shape = result.overlay_rgb.shape[:2]
            shape_changed = image_shape != self._extracted_view_shape
            self.extracted_image_item.setImage(
                result.overlay_rgb, autoLevels=False
            )
            _set_image_boundary(
                self.extracted_image_boundary, result.overlay_rgb
            )
            if shape_changed:
                self._extracted_view_shape = image_shape
                transform = _image_preview_transform(
                    result.overlay_rgb, self._image_preview_rotated
                )
                self.extracted_image_item.setTransform(transform)
                self.extracted_image_boundary.setTransform(transform)
                _fit_image_view(
                    self.extracted_image_view,
                    result.overlay_rgb,
                    self._image_view_mode,
                    rotated=self._image_preview_rotated,
                )
        elif current_tab == 1:
            trail_points: list[np.ndarray] = []
            trail_colors: list[np.ndarray] = []
            for timestamp, cloud in self._trail:
                if not len(cloud):
                    continue
                age = min(1.0, (now - timestamp) / 1.0)
                color = np.tile(
                    [0.04, 0.36, 0.72, max(0.14, 0.78 * (1.0 - age))],
                    (len(cloud), 1),
                )
                trail_points.append(cloud.astype(np.float32, copy=False))
                trail_colors.append(color.astype(np.float32))
            if trail_points:
                self.trail_point_item.setData(
                    pos=np.vstack(trail_points),
                    color=np.vstack(trail_colors),
                    size=2.0,
                )
            else:
                self.trail_point_item.setData(
                    pos=np.empty((0, 3), dtype=np.float32)
                )
            if len(current):
                self.current_point_item.setData(
                    pos=current.astype(np.float32, copy=False),
                    color=self._height_colors(current),
                    size=5.0,
                )
            else:
                self.current_point_item.setData(
                    pos=np.empty((0, 3), dtype=np.float32)
                )
        else:
            self._update_section_view(result.points_ground)
        self._displayed_frames += 1

    def _show_stats(self, stats: dict[str, float | int]) -> None:
        self.capture_fps_label.setText(f"{float(stats['capture_fps']):.1f}")
        self.process_fps_label.setText(f"{float(stats['process_fps']):.1f}")
        now = time.monotonic()
        self._display_rate_history.append((now, self._displayed_frames))
        cutoff = now - DISPLAY_FPS_WINDOW_S
        while (
            len(self._display_rate_history) > 1
            and self._display_rate_history[1][0] <= cutoff
        ):
            self._display_rate_history.popleft()
        base_time, base_count = self._display_rate_history[0]
        elapsed = max(now - base_time, 1e-9)
        display_fps = max(self._displayed_frames - base_count, 0) / elapsed
        self.display_fps_label.setText(f"{display_fps:.1f}")
        self.processing_ms_label.setText(f"{float(stats['processing_ms']):.1f} ms")
        self.drop_label.setText(
            f"{int(stats['camera_gaps'])} / {int(stats['queue_overwrites'])}"
        )
        self._refresh_adaptive_control_layout()

    def _poll_recording(self) -> None:
        if self._recorder.active:
            return
        self._update_control_states()
        recorder_error = self._recorder.error
        if recorder_error is not None:
            self.record_label.setText("失败")
            if not self._recording_error_reported:
                # Set the guard before opening the dialog.  The polling timer
                # may fire again while the first dialog is still visible.
                self._recording_error_reported = True
                # A failed final-directory commit belongs to the recording
                # task, not to the camera stream.  Keep acquisition alive so
                # the operator can still inspect the image, stop normally, or
                # retry recording without disconnecting the camera.
                self._show_error(str(recorder_error), stop_capture=False)
            return
        result = self._recorder.result
        if self._recorder.cancelled and result is None:
            if self.record_label.text() == "取消中":
                self.record_label.setText("已取消")
            return
        if result is not None and not self.record_label.text().startswith("完成"):
            self.record_label.setText(
                f"完成 {result.saved_frames} 帧，缺口 {result.detected_frame_gaps}"
            )
            if self._controller.running:
                self._set_online_state(OnlineState.STREAMING)

    def _clear_error_message_box(self, dialog: QMessageBox) -> None:
        if self._error_message_box is dialog:
            self._error_message_box = None
        dialog.deleteLater()

    def _show_processing_error(self, message: str) -> None:
        self._show_error(
            message,
            stop_capture=False,
            active_stream_message="处理异常，取流仍在运行",
        )

    def _show_error(
        self,
        message: str,
        *,
        stop_capture: bool = True,
        active_stream_message: str | None = None,
    ) -> None:
        self.statusBar().showMessage(message)
        if stop_capture:
            self._stop_due_to_error = True
            if self._controller.running:
                if self._online_state is not OnlineState.STOPPING:
                    self.stop_stream()
                self._set_online_state(OnlineState.STOPPING, "错误，正在停止")
            else:
                self._set_online_state(OnlineState.ERROR)
        else:
            # Recording finalization errors do not invalidate the camera
            # session.  Do not disturb an already-running explicit stop.
            if self._online_state is not OnlineState.STOPPING:
                self._stop_due_to_error = False
                if self._controller.running:
                    self._set_online_state(
                        OnlineState.STREAMING,
                        active_stream_message or "录制失败，取流仍在运行",
                    )
                elif self._session is not None:
                    self._set_online_state(
                        OnlineState.CONNECTED,
                        "录制失败；相机仍已连接，可重新开始取流",
                    )
                else:
                    self._set_online_state(OnlineState.ERROR)
        existing = self._error_message_box
        if existing is not None and existing.isVisible():
            return
        if existing is not None:
            existing.deleteLater()
            self._error_message_box = None

        # Keep the error notification non-modal.  A modal QMessageBox runs a
        # nested event loop; together with the recording poll timer that used
        # to create a new dialog every 250 ms and block disconnect/close.
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("在线相机错误")
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setText(message)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setModal(False)
        dialog.finished.connect(
            lambda _result, current=dialog: self._clear_error_message_box(current)
        )
        self._error_message_box = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._error_message_box is not None:
            self._error_message_box.close()
        if self._analysis_window is not None:
            self._analysis_window.close()
            self._analysis_window = None
        if (
            self._closing
            and self._session is None
            and not self._controller.running
        ):
            event.accept()
            return
        self._closing = True
        self._cancel_recording()
        if (
            self._controller.running
            or self._online_state is OnlineState.STOPPING
        ):
            self._pending_disconnect = True
            self.stop_stream()
            event.ignore()
            return
        if self._close_camera_session():
            event.accept()
        else:
            event.ignore()


def _spin(parent: QWidget, minimum: int, maximum: int, value: int) -> QSpinBox:
    widget = QSpinBox(parent)
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def _double_spin(
    parent: QWidget,
    minimum: float,
    maximum: float,
    value: float,
    suffix: str = "",
) -> QDoubleSpinBox:
    widget = QDoubleSpinBox(parent)
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    widget.setSuffix(suffix)
    return widget


def _format_mm(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f} mm"


def _format_qa_metric(value: object, suffix: str) -> str:
    return "—" if value is None else f"{float(value):.4f} {suffix}"


def _format_optional(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _compact_sha256(value: str | None) -> str:
    """Keep the sidebar readable while retaining the full SHA in the tooltip."""
    if not value:
        return ""
    if len(value) <= 24:
        return value
    return f"{value[:12]}…{value[-8:]}"


def _section_connection_mask(
    points_ground: np.ndarray,
    section_distance: np.ndarray,
    *,
    max_ds: float,
    max_dz: float,
    max_distance: float,
) -> np.ndarray:
    points = np.asarray(points_ground, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_ground 必须是形状为 (N, 3) 的数组")
    distance_along_line = np.asarray(section_distance, dtype=np.float64)
    if distance_along_line.shape != (len(points),):
        raise ValueError("section_distance 必须与 points_ground 等长")
    if min(max_ds, max_dz, max_distance) <= 0:
        raise ValueError("截面断线阈值必须大于 0")
    connections = np.zeros(len(points), dtype=np.int32)
    if len(points) < 2:
        return connections
    differences = np.diff(points, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    continuous = (
        (np.abs(np.diff(distance_along_line)) <= max_ds)
        & (np.abs(differences[:, 2]) <= max_dz)
        & (distances <= max_distance)
    )
    # PlotCurveItem uses item i to decide whether point i connects to i + 1.
    connections[:-1] = continuous.astype(np.int32)
    return connections


def _section_distance(points_xy: np.ndarray) -> np.ndarray:
    """Project ground XY points onto their dominant laser-line direction."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy 必须是形状为 (N, 2) 的数组")
    if not len(points):
        return np.empty(0, dtype=np.float64)
    centred = points - np.mean(points, axis=0)
    _, singular_values, right_vectors = np.linalg.svd(
        centred, full_matrices=False
    )
    if not len(singular_values) or singular_values[0] <= np.finfo(np.float64).eps:
        return np.zeros(len(points), dtype=np.float64)
    direction = right_vectors[0]
    if direction[0] < 0.0 or (direction[0] == 0.0 and direction[1] < 0.0):
        direction = -direction
    distance = centred @ direction
    return np.ascontiguousarray(distance - float(distance.min()))


def _section_height_range(
    heights: np.ndarray, *, robust: bool
) -> tuple[float, float]:
    """Return padded display limits; robust mode suppresses sparse tail points."""
    values = np.asarray(heights, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("heights 必须包含至少一个有限值")
    if robust and len(values) >= 20:
        minimum, maximum = np.quantile(
            values, (0.01, 0.99), method="nearest"
        )
        minimum = float(minimum)
        maximum = float(maximum)
    else:
        minimum = float(values.min())
        maximum = float(values.max())
    centre = (minimum + maximum) * 0.5
    span = max(maximum - minimum, 1.0)
    half_span = span * 0.58
    return centre - half_span, centre + half_span


def _set_image_boundary(boundary: pg.PlotDataItem, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    boundary.setData(
        [0, width, width, 0, 0],
        [0, 0, height, height, 0],
    )


def _image_preview_transform(
    image: np.ndarray | None, rotated: bool
) -> QTransform:
    """Return a display-only clockwise rotation with a positive scene extent."""
    transform = QTransform()
    if rotated and isinstance(image, np.ndarray) and image.size:
        height = image.shape[0]
        transform.translate(float(height), 0.0)
        transform.rotate(90.0)
    return transform


def _bounded_view_range(
    current_range: list[float] | tuple[float, float], extent: float
) -> tuple[float, float]:
    lower, upper = map(float, current_range)
    span = upper - lower
    if span >= extent:
        centre = extent * 0.5
        return centre - span * 0.5, centre + span * 0.5
    if lower < 0:
        return 0.0, span
    if upper > extent:
        return extent - span, extent
    return lower, upper


def _fit_image_view(
    view: pg.PlotWidget,
    image: np.ndarray,
    mode: str = "width",
    *,
    rotated: bool = False,
) -> None:
    height, width = image.shape[:2]
    if rotated:
        width, height = height, width
    view_box = view.getViewBox()
    if not isinstance(view_box, ConstrainedImageViewBox):
        raise TypeError("图像预览必须使用 ConstrainedImageViewBox")
    view_box.clear_image_constraints()
    view_box.setRange(
        xRange=(0, width),
        yRange=(0, height),
        padding=0,
    )
    if mode == "width":
        view_box.setXRange(0, width, padding=0)
    elif mode != "fit":
        raise ValueError(f"未知图像视野模式: {mode}")
    view_box.set_image_constraints(width, height)
