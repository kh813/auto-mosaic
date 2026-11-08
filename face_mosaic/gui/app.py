"""
face_mosaic GUI application using PySide6 (Qt for Python).
Features:
- Video Queue & Batch Processing with dynamic item addition/removal
- Granular Cancellation (skip/cancel current video or stop all)
- Real-time Video Preview (Pass 1 Face Detection & Pass 2 Mosaic Rendering)
- Real-time parameter tuning (Blur type, Strength, Confidence, Padding)
- Multi-threaded processing (UI remains responsive)
- Direct video playback & folder opening upon completion
- Automatic hardware spec diagnosis & model recommendation
"""

import sys
import os
import time
import traceback
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal, Slot, QRect, QSize, QPoint, QSettings, QMimeData, QItemSelectionModel, QUrl
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QAction, QDrag, QKeySequence, QShortcut, QIcon, QCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMenu, QSizePolicy, QCheckBox, QProgressDialog
)

from face_mosaic.config import AppConfig
from face_mosaic.pipeline import ProcessingPipeline
from face_mosaic.cli import collect_video_files
from face_mosaic.hardware import detect_hardware_profile
from face_mosaic import __version__
from face_mosaic.updater import check_for_updates, perform_self_update, is_git_repository


class UpdateCheckWorker(QThread):
    finished_signal = Signal(object)  # Optional[dict]

    def run(self):
        try:
            info = check_for_updates()
            self.finished_signal.emit(info)
        except Exception:
            self.finished_signal.emit(None)


class SelfUpdateWorker(QThread):
    progress_signal = Signal(str, int)     # (message, percent)
    finished_signal = Signal(bool, str)    # (success, message)

    def run(self):
        try:
            def on_progress(msg: str, cur: int, tot: int):
                pct = int((cur / tot) * 100) if tot > 0 else cur
                self.progress_signal.emit(msg, pct)

            success, msg = perform_self_update(on_progress)
            self.finished_signal.emit(success, msg)
        except Exception as e:
            self.finished_signal.emit(False, str(e))



@dataclass
class QueueItem:
    path: Path
    status: str = "Waiting"  # "Waiting", "Processing", "Completed", "Cancelled", "Error"
    progress: str = "0%"
    output_path: Optional[str] = None
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_removed: bool = False


def _write_to_log_file(text: str):
    try:
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "face_mosaic.log"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {text}\n")
    except Exception:
        pass


class VideoPreviewWidget(QWidget):
    """
    High-performance, flicker-free video preview widget.
    Renders live video frames during Pass 1 (Face Detection) and Pass 2 (Mosaic Rendering).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #121214; border: 1px solid #2c2c2e; border-radius: 6px;")
        self.pixmap: Optional[QPixmap] = None
        self.status_text: str = ""
        self.placeholder_text: str = "Live preview will appear here during processing\n(Pass 1: Face Detection / Pass 2: Mosaic Rendering)"

    def set_frame(self, frame: np.ndarray, status_text: str = ""):
        if frame is None or frame.size == 0:
            return
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.pixmap = QPixmap.fromImage(q_img)
        self.status_text = status_text
        self.update()

    def clear_preview(self, msg: Optional[str] = None):
        self.pixmap = None
        if msg:
            self.placeholder_text = msg
        self.status_text = ""
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor("#121214"))

        if self.pixmap and not self.pixmap.isNull():
            scaled = self.pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # Badge overlay
            if self.status_text:
                painter.setBrush(QColor(0, 0, 0, 190))
                painter.setPen(Qt.NoPen)
                badge_w = min(400, self.width() - 20)
                badge_rect = QRect(x + 10, y + 10, badge_w, 28)
                painter.drawRoundedRect(badge_rect, 4, 4)

                is_pass1 = "Pass 1" in self.status_text or "Detection" in self.status_text
                color = QColor("#00E676") if is_pass1 else QColor("#00B0FF")
                painter.setPen(color)
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.drawText(badge_rect.adjusted(10, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft, f"● {self.status_text}")
        else:
            painter.setPen(QColor("#8E8E93"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder_text)


class ProcessingWorker(QThread):
    item_updated_signal = Signal(str, str, str, str)   # (item_id, status_text, progress_text, output_path)
    preview_signal = Signal(object, str)               # (frame_ndarray, status_text)
    log_signal = Signal(str)
    overall_progress_signal = Signal(str, int, int)    # (phase_msg, current, total)
    finished_signal = Signal(bool, str)                # (success, summary_msg)

    def __init__(self, config: AppConfig, items: List[QueueItem]):
        super().__init__()
        self.config = config
        self.items = list(items)  # Working copy of queue items
        self._cancel_all = False
        self._cancel_current = False
        self.current_item_id: Optional[str] = None
        self._lock = threading.Lock()

    def add_items(self, new_items: List[QueueItem]):
        """Dynamically add new queue items while worker is running."""
        with self._lock:
            for item in new_items:
                if not any(it.item_id == item.item_id for it in self.items):
                    self.items.append(item)

    def cancel_all(self):
        self._cancel_all = True
        self._cancel_current = True

    def cancel_current_file(self):
        self._cancel_current = True

    def run(self):
        try:
            pipeline = ProcessingPipeline(self.config)
            provider_str = getattr(pipeline, "active_provider", None)
            if not provider_str or not isinstance(provider_str, str):
                provider_str = getattr(getattr(pipeline, "detector", None), "active_provider", "CPU")
            target_label = {
                "faces": f"Faces Only ({self.config.model.name})",
                "plates": f"License Plates Only ({self.config.plate_model.name})",
                "both": f"Faces + Plates ({self.config.model.name} & {self.config.plate_model.name})"
            }.get(self.config.target, self.config.model.name)
            self.log_signal.emit(f"Initialized pipeline [{target_label}] on {provider_str}")

            while not self._cancel_all:
                item = None
                total_items = 0
                item_display_idx = 0

                with self._lock:
                    valid_items = [it for it in self.items if not getattr(it, "is_removed", False)]
                    total_items = len(valid_items)

                    for pos, it in enumerate(valid_items):
                        if it.status in ("Waiting", "Queued"):
                            item = it
                            item_display_idx = pos + 1
                            break

                if item is None or self._cancel_all:
                    break

                self.current_item_id = item.item_id
                self._cancel_current = False
                video_path = item.path

                item.status = "Processing"
                item.progress = "0%"
                self.item_updated_signal.emit(item.item_id, "Processing", "0%", "")
                self.overall_progress_signal.emit(f"[{item_display_idx}/{total_items}] {video_path.name}", item_display_idx - 1, total_items)
                self.log_signal.emit(f"\n[{item_display_idx}/{total_items}] Starting: {video_path.name}")

                def cancel_check() -> bool:
                    return self._cancel_all or self._cancel_current or getattr(item, "is_removed", False)

                def on_progress(phase: str, current: int, total: int, elapsed: float):
                    pct = int((current / total) * 100) if total > 0 else 0
                    prog_str = f"{phase}: {pct}%"
                    item.progress = prog_str
                    self.item_updated_signal.emit(item.item_id, "Processing", prog_str, "")

                def on_preview(frame: np.ndarray, status_text: str = ""):
                    # Resize to 640w max for UI preview display efficiency
                    h, w = frame.shape[:2]
                    target_w = min(640, w)
                    target_h = max(1, int(h * (target_w / w)))
                    small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    self.preview_signal.emit(small, status_text)

                try:
                    res = pipeline.process_video(
                        str(video_path),
                        progress_callback=on_progress,
                        preview_callback=on_preview,
                        cancel_check=cancel_check
                    )
                    if res.get("cancelled", False) or self._cancel_current or self._cancel_all or getattr(item, "is_removed", False):
                        item.status = "Cancelled"
                        item.progress = "Cancelled"
                        self.item_updated_signal.emit(item.item_id, "Cancelled", "Cancelled", "")
                        self.log_signal.emit(f"⚠ Cancelled: {video_path.name}")
                    else:
                        out_p = res.get("output", "")
                        item.status = "Completed"
                        item.progress = "100%"
                        item.output_path = out_p
                        self.item_updated_signal.emit(item.item_id, "Completed", "100%", out_p)
                        self.log_signal.emit(
                            f"✓ Finished: {video_path.name} in {res['elapsed_seconds']:.1f}s "
                            f"({res['processing_fps']:.1f} fps, Color MAE: {res['color_mae']:.4f})"
                        )
                except Exception as e:
                    tb = traceback.format_exc()
                    item.status = "Error"
                    item.progress = "Error"
                    self.item_updated_signal.emit(item.item_id, "Error", "Failed", "")
                    self.log_signal.emit(f"✗ Failed {video_path.name}: {e}\n{tb}")
                    _write_to_log_file(f"Failed {video_path.name}: {e}\n{tb}")
                finally:
                    self._cancel_current = False
                    self.current_item_id = None

            with self._lock:
                if self._cancel_all:
                    self.log_signal.emit("All remaining queue processing cancelled by user.")
                    for it in self.items:
                        if it.status in ("Waiting", "Queued"):
                            it.status = "Cancelled"
                            it.progress = "-"
                            self.item_updated_signal.emit(it.item_id, "Cancelled", "-", "")

                final_total = len([it for it in self.items if not getattr(it, "is_removed", False)])
            self.overall_progress_signal.emit("Finished", final_total, final_total)
            self.finished_signal.emit(not self._cancel_all, "Batch processing finished." if not self._cancel_all else "Processing stopped by user.")
        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit(f"Fatal Error: {e}\n{tb}")
            _write_to_log_file(f"Fatal Error: {e}\n{tb}")
            self.finished_signal.emit(False, str(e))
        finally:
            self.current_item_id = None


class QueueTableWidget(QTableWidget):
    """
    Enhanced QTableWidget supporting:
    1. Direct mouse drag-and-drop row reordering
    2. External video file / folder drag-and-drop import from Explorer / Finder
    """
    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDropIndicatorShown(True)

    def startDrag(self, supportedActions):
        selected_rows = sorted({idx.row() for idx in self.selectedIndexes()})
        if not selected_rows:
            return
        # If any selected item is currently processing, do not allow dragging
        if any(self.main_window.queue_items[r].status == "Processing" for r in selected_rows if r < len(self.main_window.queue_items)):
            return

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-auto-mosaic-row", ",".join(map(str, selected_rows)).encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec(Qt.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasFormat("application/x-auto-mosaic-row"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasFormat("application/x-auto-mosaic-row"):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.main_window._handle_dropped_urls(event.mimeData().urls())
            event.acceptProposedAction()
            return

        if event.mimeData().hasFormat("application/x-auto-mosaic-row"):
            data_bytes = event.mimeData().data("application/x-auto-mosaic-row").data()
            try:
                selected_rows = [int(x) for x in data_bytes.decode("utf-8").split(",") if x]
            except Exception:
                selected_rows = []

            pos = event.position().toPoint()
            target_item = self.itemAt(pos)
            target_row = target_item.row() if target_item else (self.rowCount() - 1)

            if selected_rows and target_row >= 0:
                self.main_window._reorder_rows_to_target(selected_rows, target_row)
            event.acceptProposedAction()
            return

        super().dropEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("auto-mosaic: Automated Face & License Plate Blur Tool")
        self.resize(1100, 680)

        # Application & Window Icon
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if not icon_path.exists():
            icon_path = Path(__file__).resolve().parent.parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.hw_profile = detect_hardware_profile()
        self.config = AppConfig()
        self.queue_items: List[QueueItem] = []
        self.worker: Optional[ProcessingWorker] = None
        self.is_stopping: bool = False
        self.last_output_path: Optional[str] = None

        self.settings = QSettings("auto-mosaic", "auto-mosaic")
        saved_dir = self.settings.value("last_dir", "")
        if saved_dir and os.path.isdir(str(saved_dir)):
            self.last_open_dir = str(Path(saved_dir).resolve())
        else:
            videos_dir = Path.home() / "Videos"
            if videos_dir.is_dir():
                self.last_open_dir = str(videos_dir.resolve())
            else:
                self.last_open_dir = str(Path.home().resolve())

        self.update_checker_worker: Optional[UpdateCheckWorker] = None
        self.self_update_worker: Optional[SelfUpdateWorker] = None

        self.setAcceptDrops(True)
        self._init_menu_bar()
        self._init_ui()

        # Check updates quietly in background 2s after launch
        from PySide6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self._check_for_updates_ui(manual=False))

    def _init_menu_bar(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help (&H)")

        act_check_update = QAction("🔄 Check for Updates...", self)
        act_check_update.triggered.connect(lambda: self._check_for_updates_ui(manual=True))
        help_menu.addAction(act_check_update)

        help_menu.addSeparator()

        act_disclaimer = QAction("⚖ 免責事項・利用規約 (Disclaimer)...", self)
        act_disclaimer.triggered.connect(self._show_disclaimer)
        help_menu.addAction(act_disclaimer)

        act_about = QAction("ℹ About auto-mosaic", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(10)

        # =========================================================================
        # LEFT PANEL: 1. Video Processing Queue, Live Mosaic Preview, Execution Log
        # =========================================================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 1. Video Queue Group
        queue_group = QGroupBox("1. Video Processing Queue")
        queue_layout = QVBoxLayout(queue_group)
        queue_layout.setContentsMargins(8, 6, 8, 6)
        queue_layout.setSpacing(6)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        self.btn_select_files = QPushButton("+ Add File(s)...")
        self.btn_select_files.clicked.connect(self._select_files)
        self.btn_select_folder = QPushButton("+ Add Folder...")
        self.btn_select_folder.clicked.connect(self._select_folder)

        self.btn_move_up = QPushButton("▲")
        self.btn_move_up.setFixedWidth(32)
        self.btn_move_up.setToolTip("Move selected video(s) up in queue (Alt+Up)")
        self.btn_move_up.clicked.connect(self._move_selected_up)
        self.btn_move_up.setEnabled(False)

        self.btn_move_down = QPushButton("▼")
        self.btn_move_down.setFixedWidth(32)
        self.btn_move_down.setToolTip("Move selected video(s) down in queue (Alt+Down)")
        self.btn_move_down.clicked.connect(self._move_selected_down)
        self.btn_move_down.setEnabled(False)

        self.btn_remove_selected = QPushButton("✕ Remove")
        self.btn_remove_selected.setToolTip("Remove selected video(s) from queue")
        self.btn_remove_selected.clicked.connect(self._remove_selected_files)
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._clear_queue)

        btn_layout.addWidget(self.btn_select_files)
        btn_layout.addWidget(self.btn_select_folder)
        btn_layout.addWidget(self.btn_move_up)
        btn_layout.addWidget(self.btn_move_down)
        btn_layout.addWidget(self.btn_remove_selected)
        btn_layout.addWidget(self.btn_clear)
        queue_layout.addLayout(btn_layout)

        # Keyboard shortcuts for reordering
        QShortcut(QKeySequence("Alt+Up"), self, self._move_selected_up)
        QShortcut(QKeySequence("Alt+Down"), self, self._move_selected_down)

        # Queue Table
        self.table_queue = QueueTableWidget(self)
        self.table_queue.setColumnCount(4)
        self.table_queue.setHorizontalHeaderLabels(["#", "File Name", "Status", "Progress"])
        self.table_queue.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_queue.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_queue.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_queue.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_queue.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_queue.customContextMenuRequested.connect(self._show_table_context_menu)
        self.table_queue.itemDoubleClicked.connect(self._on_table_double_clicked)
        self.table_queue.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table_queue.setMinimumHeight(170)
        queue_layout.addWidget(self.table_queue)

        status_bar_layout = QHBoxLayout()
        self.lbl_file_count = QLabel("0 video(s) in queue.")
        status_bar_layout.addWidget(self.lbl_file_count)

        self.lbl_hw_badge = QLabel(f"⚡ {self.hw_profile.reason}")
        self.lbl_hw_badge.setStyleSheet("color: #007AFF; font-size: 11px;")
        self.lbl_hw_badge.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_hw_badge.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_hw_badge.setToolTip(self.hw_profile.reason)
        status_bar_layout.addWidget(self.lbl_hw_badge, stretch=1)
        queue_layout.addLayout(status_bar_layout)

        left_layout.addWidget(queue_group, stretch=4)

        # Live Preview Group (Bottom-Left)
        preview_group = QGroupBox("Live Mosaic Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(8, 6, 8, 6)
        preview_layout.setSpacing(6)

        self.preview_widget = VideoPreviewWidget()
        self.lbl_preview = self.preview_widget  # Alias for backward compatibility with tests
        preview_layout.addWidget(self.preview_widget, stretch=1)

        # Preview Action Buttons (Open Processed Video / Folder)
        action_layout = QHBoxLayout()
        action_layout.setSpacing(6)
        self.btn_open_video = QPushButton("▶ Open Processed Video")
        self.btn_open_video.setEnabled(False)
        self.btn_open_video.setToolTip("処理完了した動画を再生します（変換中は無効化されます）")
        self.btn_open_video.setStyleSheet("""
            QPushButton {
                height: 28px;
                font-weight: bold;
                background-color: #34C759;
                color: #FFFFFF;
                border: 1px solid #28A745;
                border-radius: 4px;
                padding: 0 10px;
            }
            QPushButton:hover {
                background-color: #2DB84D;
            }
            QPushButton:pressed {
                background-color: #24963E;
            }
            QPushButton:disabled {
                background-color: #E5E5EA;
                color: #8E8E93;
                border: 1px solid #D1D1D6;
            }
        """)
        self.btn_open_video.clicked.connect(self._open_processed_video)

        self.btn_open_folder = QPushButton("📁 Open Output Folder")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.setToolTip("処理完了した動画の保存先フォルダを開きます（変換中は無効化されます）")
        self.btn_open_folder.setStyleSheet("""
            QPushButton {
                height: 28px;
                font-weight: bold;
                background-color: #007AFF;
                color: #FFFFFF;
                border: 1px solid #0062CC;
                border-radius: 4px;
                padding: 0 10px;
            }
            QPushButton:hover {
                background-color: #0069D9;
            }
            QPushButton:pressed {
                background-color: #0051A8;
            }
            QPushButton:disabled {
                background-color: #E5E5EA;
                color: #8E8E93;
                border: 1px solid #D1D1D6;
            }
        """)
        self.btn_open_folder.clicked.connect(self._open_output_folder)

        action_layout.addWidget(self.btn_open_video)
        action_layout.addWidget(self.btn_open_folder)
        preview_layout.addLayout(action_layout)

        left_layout.addWidget(preview_group, stretch=5)

        main_layout.addWidget(left_panel, stretch=6)

        # =========================================================================
        # RIGHT PANEL: 2. Settings & Parameters, 3. Execution Controls, Execution Log
        # =========================================================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # 2. Parameters Group
        param_group = QGroupBox("2. Settings & Parameters")
        grid = QGridLayout(param_group)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        def _make_info(tooltip_text: str) -> QLabel:
            """Return a small ℹ label with a rich tooltip."""
            lbl = QLabel("ℹ")
            lbl.setStyleSheet(
                "color: #007AFF; font-size: 12px; font-weight: bold; padding: 0 2px;"
            )
            lbl.setToolTip(tooltip_text)
            lbl.setFixedWidth(18)
            lbl.setCursor(Qt.WhatsThisCursor)
            return lbl

        def _label_with_info(text: str, tooltip: str) -> QHBoxLayout:
            """Return an HBoxLayout with a text label + ℹ icon."""
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(2)
            lbl = QLabel(text)
            h.addWidget(lbl)
            h.addWidget(_make_info(tooltip))
            h.addStretch()
            return h

        def _make_compact_combo(combo: QComboBox, min_chars: int = 6):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(min_chars)

        # ---- Row 0: Target Mode (spans cols 1..3) ----
        grid.addLayout(_label_with_info(
            "Mosaic Target:",
            "<b>モザイク対象 (PC/Mac負荷最適化)</b><br>"
            "どの対象を検出してモザイク処理するかを選択します。<br>"
            "不要なAIモデルを動かさないことで、PC・MacのCPU/GPU/NPU負荷と処理時間を大幅に最適化できます。<br><br>"
            "<b>Faces Only (顔のみ):</b> 顔検出モデルのみを実行（遊歩道・屋内など車が少ない動画で高速・低負荷）。<br>"
            "<b>Faces + Plates (顔＋ナンバー両方):</b> 顔と車のナンバープレートの両方を検出・モザイク処理。<br>"
            "<b>Plates Only (ナンバーのみ):</b> ナンバープレート検出モデルのみを実行（車載動画・駐車場などで高速・低負荷）。"
        ), 0, 0)

        self.combo_target = QComboBox()
        self.combo_target.addItem("Faces Only (顔のみ - 低負荷)", "faces")
        self.combo_target.addItem("Faces + Plates (顔＋ナンバー両方)", "both")
        self.combo_target.addItem("Plates Only (ナンバーのみ - 低負荷)", "plates")
        self.combo_target.setCurrentIndex(0)
        self.combo_target.currentIndexChanged.connect(self._on_target_mode_changed)
        _make_compact_combo(self.combo_target)
        grid.addWidget(self.combo_target, 0, 1, 1, 3)

        # ---- Row 1: Face Detection Model (spans cols 1..3) ----
        grid.addLayout(_label_with_info(
            "Model:",
            "<b>Detection Model</b><br>"
            "顔検出AIモデルを選択します。<br><br>"
            "<b>scrfd_10g (Default):</b> 精度と速度のバランスが良い。ほとんどの用途に推奨。<br>"
            "<b>scrfd_2.5g (Fast):</b> 軽量・高速。PCスペックが低い場合やプレビュー確認向き。<br>"
            "<b>scrfd_34g (High Acc):</b> 最高精度。遠距離・小顔・斜め顔も検出しやすい。処理は遅い。"
        ), 1, 0)

        self.combo_model = QComboBox()
        model_defs = [
            ("scrfd_10g_bnkps.onnx", "scrfd_10g_bnkps.onnx (Default)"),
            ("scrfd_2.5g_bnkps.onnx", "scrfd_2.5g_bnkps.onnx (Fast)"),
            ("scrfd_34g_gnkps.onnx", "scrfd_34g_gnkps.onnx (High Acc)"),
        ]
        default_idx = 0
        for idx, (m_file, m_label) in enumerate(model_defs):
            if m_file == self.hw_profile.recommended_model:
                m_label += " ★ Auto-Selected"
                default_idx = idx
            self.combo_model.addItem(m_label)
        self.combo_model.setCurrentIndex(default_idx)
        _make_compact_combo(self.combo_model)
        grid.addWidget(self.combo_model, 1, 1, 1, 3)

        # ---- Row 2: Detection Confidence / Plate Confidence ----
        grid.addLayout(_label_with_info(
            "Face Confidence:",
            "<b>顔検出信頼度 (Confidence Threshold)</b><br>"
            "AIが「顔である」と判断するための信頼スコアの閾値（0〜1）。<br><br>"
            "<b>低い (0.3〜0.4):</b> 見逃しが減る。小さい・遠い顔も拾いやすいが誤検出が増える可能性。<br>"
            "<b>標準 (0.5):</b> バランス型。ほとんどの用途に適切。<br>"
            "<b>高い (0.7〜0.9):</b> 明確な顔のみ検出。誤検出を減らしたい場合に使用。"
        ), 2, 0)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.50)
        grid.addWidget(self.spin_conf, 2, 1)

        grid.addLayout(_label_with_info(
            "Plate Confidence:",
            "<b>ナンバープレート検出信頼度 (Plate Confidence)</b><br>"
            "AIが「車のナンバープレートである」と判断する閾値（0〜1）。<br><br>"
            "<b>低い (0.25〜0.35):</b> 遠方や斜めのナンバーも検出しやすい。<br>"
            "<b>標準 (0.40):</b> バランス型（推奨デフォルト）。<br>"
            "<b>高い (0.60〜0.80):</b> 誤検出を抑制し、明確なナンバーのみ検出。"
        ), 2, 2)

        self.spin_plate_conf = QDoubleSpinBox()
        self.spin_plate_conf.setRange(0.1, 0.99)
        self.spin_plate_conf.setSingleStep(0.05)
        self.spin_plate_conf.setValue(0.40)
        self.spin_plate_conf.setEnabled(False)  # Disabled when default is "Faces Only"
        grid.addWidget(self.spin_plate_conf, 2, 3)

        # ---- Row 3: Face Margin / Blur Strength ----
        grid.addLayout(_label_with_info(
            "Face Margin (%):",
            "<b>顔マージン (Face Margin)</b><br>"
            "顔の検出バウンディングボックスを周囲に何%拡大してモザイク処理するか。<br><br>"
            "値が大きいほど顔の周囲も広くぼかされ、輪郭や髪型からの特定を防ぎます。<br><br>"
            "<b>10〜20%:</b> 顔ギリギリ。<br>"
            "<b>30〜40%:</b> 標準。<br>"
            "<b>50%〜:</b> 顔全体・頭部・顎下・髪型まで完全カバー ★推奨<br><br>"
            "デフォルト: 50%"
        ), 3, 0)

        self.spin_margin = QSpinBox()
        self.spin_margin.setRange(0, 100)
        self.spin_margin.setSingleStep(5)
        self.spin_margin.setValue(50)
        grid.addWidget(self.spin_margin, 3, 1)

        self.lbl_strength = QLabel("Blur Strength (odd):")
        self._lbl_strength_info = _make_info(
            "<b>ぼかし強度</b><br>"
            "Gaussian モード時: カーネルサイズ（奇数）。大きいほど強くぼかします。<br>"
            "　推奨: 51〜99 / デフォルト: 51<br><br>"
            "Mosaic モード時: ブロックサイズ（ピクセル）。大きいほどモザイクが粗く強力になります。<br>"
            "　推奨: 24〜36（判別不能）/ デフォルト: 28"
        )
        _row_strength_right = QHBoxLayout()
        _row_strength_right.setContentsMargins(0, 0, 0, 0)
        _row_strength_right.setSpacing(2)
        _row_strength_right.addWidget(self.lbl_strength)
        _row_strength_right.addWidget(self._lbl_strength_info)
        _row_strength_right.addStretch()
        grid.addLayout(_row_strength_right, 3, 2)

        self.spin_strength = QSpinBox()
        self.spin_strength.setRange(3, 199)
        self.spin_strength.setSingleStep(2)
        self.spin_strength.setValue(51)
        grid.addWidget(self.spin_strength, 3, 3)

        # ---- Row 4: Blur Type / Blur Shape ----
        grid.addLayout(_label_with_info(
            "Blur Type:",
            "<b>ぼかし方式</b><br>"
            "顔・ナンバープレートへのモザイク処理方式を選択します。<br><br>"
            "<b>Gaussian (ガウスぼかし):</b> なめらかなぼかし。自然な仕上がりで目立ちにくい。<br>"
            "<b>Mosaic (ピクセルモザイク):</b> ブロック状のモザイク。SNS・報道向けの定番スタイル。"
        ), 4, 0)

        self.combo_blur_type = QComboBox()
        self.combo_blur_type.addItems(["Gaussian", "Mosaic"])
        self.combo_blur_type.currentTextChanged.connect(self._on_blur_type_changed)
        _make_compact_combo(self.combo_blur_type)
        grid.addWidget(self.combo_blur_type, 4, 1)

        grid.addLayout(_label_with_info(
            "Blur Shape:",
            "<b>モザイク・ぼかしの形状 (Blur Shape)</b><br>"
            "顔にかけるモザイクやぼかしの輪郭形状を選択します（ナンバープレートは常に四隅まで隠れる四角形で描画されます）。<br><br>"
            "<b>Round (Ellipse) ★推奨:</b><br>"
            "　顔の輪郭に沿った自然な楕円・丸型モザイク。背景への余計なかぶりを防ぎます。<br><br>"
            "<b>Rectangle (Square):</b><br>"
            "　従来の四角形モザイク。"
        ), 4, 2)

        self.combo_shape = QComboBox()
        self.combo_shape.addItems(["Round (Ellipse) ★推奨", "Rectangle (Square)"])
        _make_compact_combo(self.combo_shape)
        grid.addWidget(self.combo_shape, 4, 3)

        # ---- Row 5: Pad Backward / Pad Forward ----
        grid.addLayout(_label_with_info(
            "Pad Backward (f):",
            "<b>後方パディング (Pad Backward)</b><br>"
            "顔やナンバープレートが最初に検出されたフレームより何フレーム前からモザイクを適用するか。<br><br>"
            "正面から歩いてくる人物や接近する車両は検出される前から映り込みます。<br>"
            "この値を増やすことで、遠方から接近する被写体もしっかり先行モザイクできます。<br><br>"
            "<b>推奨: 8〜16フレーム</b> (30fps の場合 ≈ 0.3〜0.5秒分)<br>"
            "デフォルト: 12フレーム"
        ), 5, 0)

        self.spin_pad_back = QSpinBox()
        self.spin_pad_back.setRange(0, 120)
        self.spin_pad_back.setValue(12)
        grid.addWidget(self.spin_pad_back, 5, 1)

        grid.addLayout(_label_with_info(
            "Pad Forward (f):",
            "<b>前方パディング (Pad Forward)</b><br>"
            "顔やナンバープレートが最後に検出されたフレームより何フレーム後までモザイクを継続するか。<br><br>"
            "歩行中やすれ違い時に顔や車両が横を向いて検出が切れた場合も、<br>"
            "この値のフレーム数だけモザイクを延長して露出を防ぎます。<br><br>"
            "<b>推奨: 20〜36フレーム</b> (30fps基準で約0.7〜1.2秒)<br>"
            "デフォルト: 24フレーム (約0.8秒分)<br><br>"
            "<i>💡 60fps / 120fps / 240fps などの動画では、FPSに応じて自動的にフレーム数がスケールされるため、実時間でのモザイク継続時間は均一に維持されます。</i>"
        ), 5, 2)

        self.spin_pad_fwd = QSpinBox()
        self.spin_pad_fwd.setRange(0, 120)
        self.spin_pad_fwd.setValue(24)
        grid.addWidget(self.spin_pad_fwd, 5, 3)

        # ---- Row 6: Output Codec (spans cols 1..3) ----
        grid.addLayout(_label_with_info(
            "Output Codec:",
            "<b>出力コーデック</b><br>"
            "変換後の動画ファイルの圧縮形式を選択します。<br><br>"
            "<b>Auto (元動画と同じ) ★推奨:</b><br>"
            "　元の動画ファイルと同じコーデック（HEVC または H.264）・コンテナ・色空間で出力します。<br>"
            "　Windows / Mac / iPad / iPhone すべてで確実に再生できる標準設定です。<br><br>"
            "<b>HEVC (H.265):</b> iPhone 4K動画のネイティブ形式。<br>"
            "　H.264動画を約40〜50%小さいファイルサイズに高圧縮したい場合に選択。<br><br>"
            "<b>H.264:</b> 古い機器・レガシーソフトとの互換性を最優先する場合に選択。"
        ), 6, 0)

        self.combo_codec = QComboBox()
        self.combo_codec.addItem("Auto (元動画と同じ) ★推奨", "auto")
        self.combo_codec.addItem("HEVC (H.265)", "hevc")
        self.combo_codec.addItem("H.264", "h264")
        _make_compact_combo(self.combo_codec)
        grid.addWidget(self.combo_codec, 6, 1, 1, 3)

        # ---- Row 7: CRF Quality (spans cols 1..3) ----
        grid.addLayout(_label_with_info(
            "CRF Quality:",
            "<b>CRF (Constant Rate Factor) — 画質設定</b><br>"
            "数値が小さいほど高画質・大きいファイル。<br>"
            "数値が大きいほど低画質・小さいファイル。<br><br>"
            "<b>標準 (CRF 23) ★推奨:</b> 視覚的劣化なし・容量約50%削減<br>"
            "<b>最高画質 (CRF 18):</b> マスター保存向け（大容量）<br>"
            "<b>容量優先 (CRF 26):</b> さらに軽量化<br>"
            "<b>カスタム:</b> 0〜51まで自由指定"
        ), 7, 0)

        crf_h_layout = QHBoxLayout()
        crf_h_layout.setContentsMargins(0, 0, 0, 0)
        crf_h_layout.setSpacing(6)
        self.combo_crf_preset = QComboBox()
        self.combo_crf_preset.addItems([
            "標準 (23 - 推奨)",
            "最高画質 (18)",
            "容量優先 (26)",
            "カスタム"
        ])
        _make_compact_combo(self.combo_crf_preset)
        self.spin_crf = QSpinBox()
        self.spin_crf.setRange(0, 51)
        self.spin_crf.setValue(23)
        self.spin_crf.setFixedWidth(56)

        def _on_crf_preset_changed(idx: int):
            presets = [23, 18, 26]
            if idx < len(presets):
                self.spin_crf.blockSignals(True)
                self.spin_crf.setValue(presets[idx])
                self.spin_crf.blockSignals(False)

        def _on_spin_crf_changed(val: int):
            preset_map = {23: 0, 18: 1, 26: 2}
            self.combo_crf_preset.blockSignals(True)
            self.combo_crf_preset.setCurrentIndex(preset_map.get(val, 3))
            self.combo_crf_preset.blockSignals(False)

        self.combo_crf_preset.currentIndexChanged.connect(_on_crf_preset_changed)
        self.spin_crf.valueChanged.connect(_on_spin_crf_changed)

        crf_h_layout.addWidget(self.combo_crf_preset, stretch=1)
        crf_h_layout.addWidget(self.spin_crf)
        grid.addLayout(crf_h_layout, 7, 1, 1, 3)

        # ---- Row 8: Bit Depth / Filename Option ----
        grid.addLayout(_label_with_info(
            "Bit Depth:",
            "<b>ビット深度 (カラー深度)</b><br>"
            "<b>Auto (自動) ★推奨:</b> 元動画の深度を維持（HEVC時10-bit HDR維持）。<br>"
            "<b>8-bit:</b> iPhone 10-bit HDR動画でも8-bitに変換し、ファイルサイズ削減と再生互換性を最大化。<br>"
            "<b>10-bit:</b> 階調重視の10-bitエンコードを強制。"
        ), 8, 0)

        self.combo_bit_depth = QComboBox()
        self.combo_bit_depth.addItems(["Auto (自動)", "8-bit (互換性&軽量化)", "10-bit (高階調)"])
        _make_compact_combo(self.combo_bit_depth)
        grid.addWidget(self.combo_bit_depth, 8, 1)

        grid.addLayout(_label_with_info(
            "Filename Option:",
            "<b>ファイル名への設定情報付加</b><br>"
            "チェックを入れると、出力ファイル名に適用したぼかし強度・マージン・CRFを自動埋め込みします。<br>"
            "例: <code>_blurred_(G51-M50-CRF23).mp4</code>"
        ), 8, 2)

        self.chk_append_params = QCheckBox("パラメータを付与")
        self.chk_append_params.setChecked(False)
        grid.addWidget(self.chk_append_params, 8, 3)

        right_layout.addWidget(param_group)

        # 3. Execution Group (Right column below settings)
        exec_group = QGroupBox("3. Execution Controls")
        exec_layout = QVBoxLayout(exec_group)
        exec_layout.setContentsMargins(10, 8, 10, 8)
        exec_layout.setSpacing(6)

        exec_btn_layout = QHBoxLayout()
        exec_btn_layout.setSpacing(6)
        self.btn_start = QPushButton("▶ Start Processing")
        self.btn_start.setStyleSheet("font-size: 14px; font-weight: bold; height: 36px; background-color: #007AFF; color: white; border-radius: 6px;")
        self.btn_start.clicked.connect(lambda: self._toggle_processing())

        self.btn_cancel_current = QPushButton("⏭ Skip / Cancel Current File")
        self.btn_cancel_current.setEnabled(False)
        self.btn_cancel_current.setStyleSheet("font-size: 12px; font-weight: bold; height: 36px; background-color: #5856D6; color: white; border-radius: 6px;")
        self.btn_cancel_current.clicked.connect(self._cancel_current_file)

        exec_btn_layout.addWidget(self.btn_start, stretch=3)
        exec_btn_layout.addWidget(self.btn_cancel_current, stretch=2)
        exec_layout.addLayout(exec_btn_layout)

        self.lbl_progress_status = QLabel("Idle")
        exec_layout.addWidget(self.lbl_progress_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        exec_layout.addWidget(self.progress_bar)

        right_layout.addWidget(exec_group)

        # Execution Log Group (Right column below Execution Controls)
        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 6, 8, 6)
        log_layout.setSpacing(4)

        log_btn_layout = QHBoxLayout()
        log_btn_layout.setSpacing(4)
        self.btn_disclaimer = QPushButton("⚖ 免責事項")
        self.btn_disclaimer.setToolTip("免責事項・利用規約を表示します")
        self.btn_disclaimer.clicked.connect(self._show_disclaimer)
        self.btn_update = QPushButton("🔄 Check Update")
        self.btn_update.setToolTip("GitHubから最新バージョンを確認・更新します")
        self.btn_update.clicked.connect(lambda: self._check_for_updates_ui(manual=True))
        self.btn_copy_log = QPushButton("📋 Copy")
        self.btn_copy_log.setToolTip("実行ログをクリップボードにコピーします")
        self.btn_copy_log.clicked.connect(self._copy_log)
        self.btn_open_log = QPushButton("📁 Log File")
        self.btn_open_log.setToolTip("ログファイル (face_mosaic.log) を開きます")
        self.btn_open_log.clicked.connect(self._open_log_file)
        log_btn_layout.addWidget(self.btn_disclaimer)
        log_btn_layout.addWidget(self.btn_update)
        log_btn_layout.addStretch()
        log_btn_layout.addWidget(self.btn_copy_log)
        log_btn_layout.addWidget(self.btn_open_log)
        log_layout.addLayout(log_btn_layout)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(80)
        self.txt_log.append(f"System Check: {self.hw_profile.reason}\nReady for processing.")
        log_layout.addWidget(self.txt_log)

        right_layout.addWidget(log_group, stretch=1)

        # Subtle footer disclaimer banner
        lbl_footer_disclaimer = QLabel("⚠️ 本ツールは顔・ナンバープレート検出モザイクの自動化補助ツールです。公開前に必ず目視確認を行ってください。(Apache-2.0)")
        lbl_footer_disclaimer.setWordWrap(True)
        lbl_footer_disclaimer.setStyleSheet("color: #8E8E93; font-size: 10px; margin-top: 2px;")
        lbl_footer_disclaimer.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(lbl_footer_disclaimer)

        main_layout.addWidget(right_panel, stretch=5)

    def _on_target_mode_changed(self, idx: int = 0):
        target_val = self.combo_target.currentData() or "faces"
        use_faces = target_val in ("faces", "both")
        use_plates = target_val in ("plates", "both")
        if hasattr(self, "combo_model"):
            self.combo_model.setEnabled(use_faces)
        if hasattr(self, "spin_conf"):
            self.spin_conf.setEnabled(use_faces)
        if hasattr(self, "spin_plate_conf"):
            self.spin_plate_conf.setEnabled(use_plates)

        self._has_centered = False
        self._center_on_screen()

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_has_centered", False):
            self._has_centered = True
            self._center_on_screen()

    def _center_on_screen(self):
        """
        Center window on screen and ensure it fits comfortably within
        the available screen geometry (avoiding overflow on smaller/scaled displays).
        """
        screen = None
        try:
            cursor_pos = QCursor.pos()
            screen = QApplication.screenAt(cursor_pos)
        except Exception:
            pass
        if not screen:
            screen = QApplication.primaryScreen()
        if not screen:
            return

        avail = screen.availableGeometry()

        # Scale down if display resolution/scaling causes the window to overflow
        max_w = max(640, avail.width() - 40)
        max_h = max(480, avail.height() - 60)
        target_w = min(self.width(), max_w)
        target_h = min(self.height(), max_h)

        if target_w != self.width() or target_h != self.height():
            self.resize(target_w, target_h)

        geo = self.frameGeometry()
        geo.moveCenter(avail.center())
        self.move(geo.topLeft())

    def _on_blur_type_changed(self, text: str):
        if text.lower() == "mosaic":
            self.lbl_strength.setText("Mosaic Block Size:")
            self.spin_strength.setValue(28)
        else:
            self.lbl_strength.setText("Blur Strength (odd):")
            self.spin_strength.setValue(51)

    def _select_files(self):
        start_dir = self.last_open_dir
        if not (start_dir and os.path.isdir(start_dir)):
            videos_dir = Path.home() / "Videos"
            start_dir = str(videos_dir.resolve()) if videos_dir.is_dir() else str(Path.home().resolve())

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            start_dir,
            "Video Files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;All Files (*.*)"
        )
        if paths:
            new_dir = str(Path(paths[0]).parent.resolve())
            self.last_open_dir = new_dir
            self.settings.setValue("last_dir", new_dir)
            self.settings.sync()
            self._add_to_queue(collect_video_files(paths))

    def _select_folder(self):
        start_dir = self.last_open_dir
        if not (start_dir and os.path.isdir(start_dir)):
            videos_dir = Path.home() / "Videos"
            start_dir = str(videos_dir.resolve()) if videos_dir.is_dir() else str(Path.home().resolve())

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Videos",
            start_dir
        )
        if folder:
            new_dir = str(Path(folder).resolve())
            self.last_open_dir = new_dir
            self.settings.setValue("last_dir", new_dir)
            self.settings.sync()
            self._add_to_queue(collect_video_files([Path(new_dir)]))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self._handle_dropped_urls(event.mimeData().urls())
            event.acceptProposedAction()

    def _handle_dropped_urls(self, urls):
        raw_paths = [Path(url.toLocalFile()) for url in urls if url.toLocalFile()]
        if raw_paths:
            first = raw_paths[0]
            new_dir = str(first.resolve() if first.is_dir() else first.parent.resolve())
            if os.path.isdir(new_dir):
                self.last_open_dir = new_dir
                self.settings.setValue("last_dir", new_dir)
                self.settings.sync()
            self._add_to_queue(collect_video_files(raw_paths))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel_all()
            self.worker.wait(1000)
        if self.last_open_dir and os.path.isdir(self.last_open_dir):
            self.settings.setValue("last_dir", self.last_open_dir)
            self.settings.sync()
        event.accept()

    def _sync_worker_items_order(self):
        if self.worker and self.worker.isRunning():
            with self.worker._lock:
                curr_id = self.worker.current_item_id
                curr_idx = -1
                for idx, w_item in enumerate(self.worker.items):
                    if w_item.item_id == curr_id:
                        curr_idx = idx
                        break
                if curr_idx >= 0:
                    prefix = self.worker.items[:curr_idx + 1]
                    waiting_in_queue = [item for item in self.queue_items if item.status == "Waiting" and item not in prefix]
                    self.worker.items = prefix + waiting_in_queue
                else:
                    waiting_in_queue = [item for item in self.queue_items if item.status == "Waiting"]
                    self.worker.items = waiting_in_queue

    def _reselect_rows(self, rows: List[int]):
        self.table_queue.clearSelection()
        if not rows:
            self._on_table_selection_changed()
            return
        first_r = rows[0]
        if 0 <= first_r < self.table_queue.rowCount():
            self.table_queue.setCurrentCell(first_r, 0)
        sel_model = self.table_queue.selectionModel()
        for r in rows:
            if 0 <= r < self.table_queue.rowCount():
                idx = self.table_queue.model().index(r, 0)
                sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self._on_table_selection_changed()

    def _move_selected_up(self):
        selected_rows = sorted({idx.row() for idx in self.table_queue.selectedIndexes()})
        if not selected_rows or selected_rows[0] == 0:
            return

        if any(self.queue_items[r].status == "Processing" for r in selected_rows if r < len(self.queue_items)):
            self.txt_log.append("Cannot move a video that is currently being processed.")
            return

        if any(self.queue_items[r - 1].status == "Processing" for r in selected_rows if r > 0):
            self.txt_log.append("Cannot swap with a video that is currently being processed.")
            return

        for r in selected_rows:
            self.queue_items[r - 1], self.queue_items[r] = self.queue_items[r], self.queue_items[r - 1]

        new_selected_rows = [r - 1 for r in selected_rows]
        self._refresh_queue_table()
        self._reselect_rows(new_selected_rows)
        self._sync_worker_items_order()

    def _move_selected_down(self):
        selected_rows = sorted({idx.row() for idx in self.table_queue.selectedIndexes()})
        if not selected_rows or selected_rows[-1] >= len(self.queue_items) - 1:
            return

        if any(self.queue_items[r].status == "Processing" for r in selected_rows if r < len(self.queue_items)):
            self.txt_log.append("Cannot move a video that is currently being processed.")
            return

        if any(self.queue_items[r + 1].status == "Processing" for r in selected_rows if r < len(self.queue_items) - 1):
            self.txt_log.append("Cannot swap with a video that is currently being processed.")
            return

        for r in reversed(selected_rows):
            self.queue_items[r], self.queue_items[r + 1] = self.queue_items[r + 1], self.queue_items[r]

        new_selected_rows = [r + 1 for r in selected_rows]
        self._refresh_queue_table()
        self._reselect_rows(new_selected_rows)
        self._sync_worker_items_order()

    def _move_selected_to_top(self):
        selected_rows = sorted({idx.row() for idx in self.table_queue.selectedIndexes()})
        if not selected_rows or selected_rows[0] == 0:
            return

        if any(self.queue_items[r].status == "Processing" for r in selected_rows if r < len(self.queue_items)):
            self.txt_log.append("Cannot move a video that is currently being processed.")
            return

        top_idx = 1 if (self.queue_items and self.queue_items[0].status == "Processing") else 0
        if selected_rows[0] <= top_idx:
            return

        moving = [self.queue_items[r] for r in selected_rows]
        remaining = [item for r, item in enumerate(self.queue_items) if r not in selected_rows]

        prefix = remaining[:top_idx]
        suffix = remaining[top_idx:]
        self.queue_items = prefix + moving + suffix
        new_selected_rows = list(range(top_idx, top_idx + len(moving)))
        self._refresh_queue_table()
        self._reselect_rows(new_selected_rows)
        self._sync_worker_items_order()

    def _move_selected_to_bottom(self):
        selected_rows = sorted({idx.row() for idx in self.table_queue.selectedIndexes()})
        if not selected_rows or selected_rows[-1] >= len(self.queue_items) - 1:
            return

        if any(self.queue_items[r].status == "Processing" for r in selected_rows if r < len(self.queue_items)):
            self.txt_log.append("Cannot move a video that is currently being processed.")
            return

        moving = [self.queue_items[r] for r in selected_rows]
        remaining = [item for r, item in enumerate(self.queue_items) if r not in selected_rows]
        self.queue_items = remaining + moving
        new_selected_rows = list(range(len(remaining), len(self.queue_items)))
        self._refresh_queue_table()
        self._reselect_rows(new_selected_rows)
        self._sync_worker_items_order()

    def _reorder_rows_to_target(self, selected_rows: List[int], target_row: int):
        if not selected_rows or target_row < 0 or target_row >= len(self.queue_items):
            return
        if target_row in selected_rows:
            return

        if any(self.queue_items[r].status == "Processing" for r in selected_rows if r < len(self.queue_items)):
            self.txt_log.append("Cannot move a video that is currently being processed.")
            return

        if self.queue_items[target_row].status == "Processing":
            self.txt_log.append("Cannot move video before the currently processing video.")
            return

        moving = [self.queue_items[r] for r in selected_rows]
        target_item = self.queue_items[target_row]
        remaining = [item for r, item in enumerate(self.queue_items) if r not in selected_rows]

        idx = remaining.index(target_item)
        if target_row > selected_rows[-1]:
            insert_idx = idx + 1
        else:
            insert_idx = idx

        for i, item in enumerate(moving):
            remaining.insert(insert_idx + i, item)

        self.queue_items = remaining
        new_selected = [remaining.index(item) for item in moving]
        self._refresh_queue_table()
        self._reselect_rows(new_selected)
        self._sync_worker_items_order()

    def _add_to_queue(self, new_paths: List[Path]):
        existing = {str(item.path.resolve()) for item in self.queue_items}
        added_count = 0
        new_items = []
        for p in new_paths:
            rp = str(p.resolve())
            if rp not in existing:
                item = QueueItem(path=p)
                self.queue_items.append(item)
                new_items.append(item)
                existing.add(rp)
                added_count += 1
        if added_count > 0:
            self._refresh_queue_table()
            if self.worker and self.worker.isRunning():
                self.worker.add_items(new_items)
                self.txt_log.append(f"Added {added_count} video(s) to running processing queue.")

    def _remove_selected_files(self):
        selected_rows = sorted({idx.row() for idx in self.table_queue.selectedIndexes()}, reverse=True)
        if not selected_rows:
            return

        for row in selected_rows:
            if row < len(self.queue_items):
                item = self.queue_items[row]
                is_active = (
                    self.worker is not None and
                    self.worker.isRunning() and
                    self.worker.current_item_id == item.item_id
                )
                if is_active:
                    reply = QMessageBox.question(
                        self, "Cancel Processing?",
                        f"Video '{item.path.name}' is currently being processed.\nDo you want to cancel and remove it?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        continue
                    self.worker.cancel_current_file()

                item.is_removed = True
                self.queue_items.pop(row)

        self._refresh_queue_table()
        self._update_output_button_states()
        self._on_table_selection_changed()

    def _remove_single_row(self, row: int):
        if 0 <= row < len(self.queue_items):
            item = self.queue_items[row]
            is_active = (
                self.worker is not None and
                self.worker.isRunning() and
                self.worker.current_item_id == item.item_id
            )
            if is_active:
                reply = QMessageBox.question(
                    self, "Cancel Processing?",
                    f"Video '{item.path.name}' is currently being processed.\nDo you want to cancel and remove it?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
                self.worker.cancel_current_file()

            item.is_removed = True
            self.queue_items.pop(row)
            self._refresh_queue_table()
            self._update_output_button_states()
            self._on_table_selection_changed()

    def _clear_queue(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Processing Active", "Cannot clear queue while processing is running. Please stop processing first.")
            return
        for item in self.queue_items:
            item.is_removed = True
        self.queue_items.clear()
        self._refresh_queue_table()
        self._update_output_button_states()
        self._on_table_selection_changed()

    def _update_table_row(self, row: int, item: QueueItem):
        if row >= self.table_queue.rowCount():
            return
        # Column 2: Status
        item_status = self.table_queue.item(row, 2)
        if item_status:
            item_status.setText(item.status)
            if item.status == "Completed":
                item_status.setForeground(QColor("#34C759"))
            elif item.status == "Processing":
                item_status.setForeground(QColor("#007AFF"))
            elif item.status == "Cancelled":
                item_status.setForeground(QColor("#FF9500"))
            elif item.status == "Error":
                item_status.setForeground(QColor("#FF3B30"))
            else:
                item_status.setForeground(QColor("#8E8E93"))

        # Column 3: Progress
        item_prog = self.table_queue.item(row, 3)
        if item_prog:
            item_prog.setText(item.progress)

    def _refresh_queue_table(self):
        self.table_queue.setRowCount(len(self.queue_items))
        for idx, item in enumerate(self.queue_items):
            # Column 0: Index
            item_idx = QTableWidgetItem(str(idx + 1))
            item_idx.setTextAlignment(Qt.AlignCenter)
            self.table_queue.setItem(idx, 0, item_idx)

            # Column 1: File Name
            item_name = QTableWidgetItem(item.path.name)
            item_name.setToolTip(str(item.path))
            self.table_queue.setItem(idx, 1, item_name)

            # Column 2: Status
            item_status = QTableWidgetItem(item.status)
            item_status.setTextAlignment(Qt.AlignCenter)
            if item.status == "Completed":
                item_status.setForeground(QColor("#34C759"))
            elif item.status == "Processing":
                item_status.setForeground(QColor("#007AFF"))
            elif item.status == "Cancelled":
                item_status.setForeground(QColor("#FF9500"))
            elif item.status == "Error":
                item_status.setForeground(QColor("#FF3B30"))
            else:
                item_status.setForeground(QColor("#8E8E93"))
            self.table_queue.setItem(idx, 2, item_status)

            # Column 3: Progress
            item_prog = QTableWidgetItem(item.progress)
            item_prog.setTextAlignment(Qt.AlignCenter)
            self.table_queue.setItem(idx, 3, item_prog)

        self.lbl_file_count.setText(f"{len(self.queue_items)} video(s) in queue.")

    def _show_table_context_menu(self, pos: QPoint):
        item = self.table_queue.itemAt(pos)
        if not item:
            menu = QMenu(self)
            act_add_files = QAction("➕ Add Video File(s)...", self)
            act_add_files.triggered.connect(self._select_files)
            menu.addAction(act_add_files)

            act_add_folder = QAction("📁 Add Folder...", self)
            act_add_folder.triggered.connect(self._select_folder)
            menu.addAction(act_add_folder)

            menu.exec(self.table_queue.viewport().mapToGlobal(pos))
            return
        row = item.row()
        if row >= len(self.queue_items):
            return

        q_item = self.queue_items[row]
        menu = QMenu(self)

        is_running = self.worker is not None and self.worker.isRunning()
        is_active = is_running and self.worker.current_item_id == q_item.item_id

        if is_active:
            act_cancel = QAction("⏭ Cancel this video", self)
            act_cancel.triggered.connect(self._cancel_current_file)
            menu.addAction(act_cancel)
        else:
            if not is_running:
                selected_rows = sorted({idx.row() for idx in self.table_queue.selectedIndexes()})
                if len(selected_rows) > 1 and row in selected_rows:
                    selected_items = [self.queue_items[r] for r in selected_rows if 0 <= r < len(self.queue_items)]
                    act_process_sel = QAction(f"▶ Process Selected Videos ({len(selected_items)})", self)
                    act_process_sel.triggered.connect(lambda: self._toggle_processing(target_items_override=selected_items))
                    menu.addAction(act_process_sel)

                act_process_one = QAction("▶ Process This Video", self)
                act_process_one.triggered.connect(lambda: self._toggle_processing(target_items_override=[q_item]))
                menu.addAction(act_process_one)

                act_process_from_here = QAction("▶ Process From This Video to End", self)
                act_process_from_here.triggered.connect(lambda: self._toggle_processing(target_items_override=self.queue_items[row:]))
                menu.addAction(act_process_from_here)

            act_remove = QAction("✕ Remove from Queue", self)
            act_remove.triggered.connect(lambda: self._remove_single_row(row))
            menu.addAction(act_remove)

        menu.addSeparator()
        act_up = QAction("▲ Move Up", self)
        act_up.setEnabled(row > 0 and q_item.status != "Processing")
        act_up.triggered.connect(self._move_selected_up)
        menu.addAction(act_up)

        act_down = QAction("▼ Move Down", self)
        act_down.setEnabled(row < len(self.queue_items) - 1 and q_item.status != "Processing")
        act_down.triggered.connect(self._move_selected_down)
        menu.addAction(act_down)

        act_top = QAction("⤒ Move to Top", self)
        act_top.setEnabled(row > 0 and q_item.status != "Processing")
        act_top.triggered.connect(self._move_selected_to_top)
        menu.addAction(act_top)

        act_bottom = QAction("⤓ Move to Bottom", self)
        act_bottom.setEnabled(row < len(self.queue_items) - 1 and q_item.status != "Processing")
        act_bottom.triggered.connect(self._move_selected_to_bottom)
        menu.addAction(act_bottom)

        if q_item.output_path and os.path.exists(q_item.output_path):
            menu.addSeparator()
            act_play = QAction("▶ Open Processed Video", self)
            act_play.triggered.connect(lambda: self._open_path(q_item.output_path))
            menu.addAction(act_play)

            act_folder = QAction("📁 Open Containing Folder", self)
            act_folder.triggered.connect(lambda: self._open_path(str(Path(q_item.output_path).parent)))
            menu.addAction(act_folder)

        menu.exec(self.table_queue.viewport().mapToGlobal(pos))

    def _on_table_double_clicked(self, item: QTableWidgetItem):
        row = item.row()
        if 0 <= row < len(self.queue_items):
            q_item = self.queue_items[row]
            if q_item.output_path and os.path.exists(q_item.output_path):
                self._open_path(q_item.output_path)

    def _on_table_selection_changed(self):
        if self.is_stopping:
            return

        selected_rows = sorted({idx.row() for idx in self.table_queue.selectedIndexes()})
        num_selected = len(selected_rows)

        can_move_up = num_selected > 0 and selected_rows[0] > 0
        can_move_down = num_selected > 0 and selected_rows[-1] < len(self.queue_items) - 1
        if any(self.queue_items[r].status == "Processing" for r in selected_rows if r < len(self.queue_items)):
            can_move_up = False
            can_move_down = False
        if hasattr(self, "btn_move_up"):
            self.btn_move_up.setEnabled(can_move_up)
        if hasattr(self, "btn_move_down"):
            self.btn_move_down.setEnabled(can_move_down)

        if self.worker and self.worker.isRunning():
            self.btn_start.setText("⏹ Stop Processing")
            self.btn_start.setStyleSheet("font-size: 14px; font-weight: bold; height: 38px; background-color: #FF3B30; color: white; border-radius: 6px;")
            self.btn_start.setEnabled(True)
        else:
            if num_selected > 0:
                self.btn_start.setText(f"▶ Start Processing ({num_selected} selected)")
            else:
                self.btn_start.setText("▶ Start Processing")
            self.btn_start.setStyleSheet("font-size: 14px; font-weight: bold; height: 38px; background-color: #007AFF; color: white; border-radius: 6px;")
            self.btn_start.setEnabled(True)

        self._update_output_button_states()

    def _update_output_button_states(self):
        # If conversion is currently running, keep buttons disabled (greyed out)
        if self.worker is not None and self.worker.isRunning():
            self.btn_open_video.setEnabled(False)
            self.btn_open_folder.setEnabled(False)
            return

        selected_rows = {idx.row() for idx in self.table_queue.selectedIndexes()}
        for row in selected_rows:
            if row < len(self.queue_items):
                item = self.queue_items[row]
                if item.output_path and os.path.exists(item.output_path):
                    self.last_output_path = item.output_path
                    self.btn_open_video.setEnabled(True)
                    self.btn_open_folder.setEnabled(True)
                    return

        # Fall back to latest completed video in queue
        for item in reversed(self.queue_items):
            if item.output_path and os.path.exists(item.output_path):
                self.last_output_path = item.output_path
                self.btn_open_video.setEnabled(True)
                self.btn_open_folder.setEnabled(True)
                return

        self.last_output_path = None
        self.btn_open_video.setEnabled(False)
        self.btn_open_folder.setEnabled(False)

    def _toggle_processing(self, *args, target_items_override: Optional[List[QueueItem]] = None, **kwargs):
        try:
            # Discard boolean arguments sent by Qt signals like clicked(bool)
            if args and isinstance(args[0], (list, tuple)):
                target_items_override = args[0]
            elif target_items_override is not None and not isinstance(target_items_override, (list, tuple)):
                target_items_override = None

            if self.is_stopping:
                return

            if self.worker and self.worker.isRunning():
                # Stop requested
                self.is_stopping = True
                self.btn_start.setEnabled(False)
                self.btn_start.setText("⏳ Stopping...")
                self.btn_start.setStyleSheet(
                    "font-size: 14px; font-weight: bold; height: 38px; "
                    "background-color: #636366; color: #D1D1D6; border-radius: 6px;"
                )
                self.btn_cancel_current.setEnabled(False)
                self.lbl_progress_status.setText("Stopping processing, please wait...")
                self.txt_log.append("Stop requested. Waiting for active tasks to finalize...")
                self.worker.cancel_all()
                return

            if not self.queue_items:
                QMessageBox.warning(self, "No Videos", "Please add at least one video file or folder to the queue first.")
                return

            # Determine items to process:
            if target_items_override is not None:
                items_to_process = list(target_items_override)
            else:
                # Process all videos in the queue sequentially (non-completed items)
                items_to_process = [item for item in self.queue_items if item.status != "Completed"]
                if not items_to_process:
                    reply = QMessageBox.question(
                        self, "Re-process All?",
                        "All videos in the queue have already been completed.\nDo you want to re-process all videos?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        items_to_process = list(self.queue_items)
                    else:
                        return

            if not items_to_process:
                return

            # Reset status and progress for items to process
            for item in items_to_process:
                item.status = "Waiting"
                item.progress = "0%"

            self._refresh_queue_table()

            # Build config
            cfg = AppConfig()
            cfg.target = self.combo_target.currentData() or "faces"
            model_selection = self.combo_model.currentText().split()[0]
            cfg.model.name = model_selection
            cfg.model.conf_threshold = self.spin_conf.value()
            cfg.plate_model.conf_threshold = self.spin_plate_conf.value()

            cfg.blur.type = self.combo_blur_type.currentText().lower()
            if cfg.blur.type == "mosaic":
                cfg.blur.mosaic_block_size = self.spin_strength.value()
            else:
                val = self.spin_strength.value()
                cfg.blur.strength = val if val % 2 == 1 else val + 1

            cfg.tracking.padding_backward = self.spin_pad_back.value()
            cfg.tracking.padding_forward = self.spin_pad_fwd.value()
            margin_ratio = self.spin_margin.value() / 100.0
            cfg.blur.margin_x = margin_ratio
            cfg.blur.margin_y = margin_ratio

            shape_text = self.combo_shape.currentText().lower()
            cfg.blur.shape = "ellipse" if ("round" in shape_text or "ellipse" in shape_text) else "rect"
            cfg.filters.illustration_filter.enabled = True

            codec_data = self.combo_codec.currentData()
            if codec_data in ("auto", "hevc", "h264"):
                cfg.output.codec = codec_data
            else:
                ctext = self.combo_codec.currentText().upper()
                if "AUTO" in ctext:
                    cfg.output.codec = "auto"
                elif "HEVC" in ctext or "265" in ctext:
                    cfg.output.codec = "hevc"
                else:
                    cfg.output.codec = "h264"
            cfg.output.crf = self.spin_crf.value()

            bit_depth_text = self.combo_bit_depth.currentText().lower()
            if "8-bit" in bit_depth_text:
                cfg.output.bit_depth = "8bit"
            elif "10-bit" in bit_depth_text:
                cfg.output.bit_depth = "10bit"
            else:
                cfg.output.bit_depth = "auto"

            cfg.output.append_params_to_filename = self.chk_append_params.isChecked()

            self.btn_start.setText("⏹ Stop Processing")
            self.btn_start.setStyleSheet("font-size: 14px; font-weight: bold; height: 38px; background-color: #FF3B30; color: white; border-radius: 6px;")
            self.btn_start.setEnabled(True)
            self.btn_cancel_current.setEnabled(True)

            self.btn_open_video.setEnabled(False)
            self.btn_open_folder.setEnabled(False)

            self.worker = ProcessingWorker(cfg, items_to_process)
            self.worker.item_updated_signal.connect(self._on_item_updated)
            self.worker.preview_signal.connect(self._on_preview_frame)
            self.worker.log_signal.connect(self._on_log)
            self.worker.overall_progress_signal.connect(self._on_overall_progress)
            self.worker.finished_signal.connect(self._on_finished)
            self.worker.start()
        except Exception as e:
            tb = traceback.format_exc()
            _write_to_log_file(f"Error in _toggle_processing: {e}\n{tb}")
            self.txt_log.append(f"Error starting processing: {e}")
            QMessageBox.critical(self, "Processing Error", f"Failed to start processing:\n{e}")

    def _cancel_current_file(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel_current_file()
            self.txt_log.append("Cancelling current video...")

    @Slot(str, str, str, str)
    def _on_item_updated(self, item_id: str, status: str, progress: str, output_path: str = ""):
        for row, item in enumerate(self.queue_items):
            if item.item_id == item_id:
                item.status = status
                item.progress = progress
                if output_path:
                    item.output_path = output_path
                self._update_table_row(row, item)
                break

    @Slot(str, int, int)
    def _on_overall_progress(self, msg: str, current: int, total: int):
        self.lbl_progress_status.setText(f"{msg} ({current}/{total})")
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))

    @Slot(object, str)
    def _on_preview_frame(self, frame: np.ndarray, status_text: str):
        self.preview_widget.set_frame(frame, status_text)

    @Slot(str)
    def _on_log(self, text: str):
        self.txt_log.append(text)

    @Slot(bool, str)
    def _on_finished(self, success: bool, msg: str):
        self.is_stopping = False
        if self.worker:
            self.worker.wait(2000)
            self.worker = None

        self.btn_start.setEnabled(True)
        self.btn_cancel_current.setEnabled(False)
        self.lbl_progress_status.setText(msg)

        self.preview_widget.clear_preview("Live preview will appear here during processing\n(Pass 1: Face Detection / Pass 2: Mosaic Rendering)")

        self._update_output_button_states()
        self._on_table_selection_changed()

        if success:
            QMessageBox.information(self, "Completed", "Queue processing completed successfully!")

    def _open_processed_video(self):
        if self.last_output_path and os.path.exists(self.last_output_path):
            self._open_path(self.last_output_path)

    def _open_output_folder(self):
        if self.last_output_path:
            folder = str(Path(self.last_output_path).parent)
            self._open_path(folder)

    def _open_path(self, target_path: str):
        if not target_path or not os.path.exists(target_path):
            return
        if sys.platform == "win32":
            os.startfile(target_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", target_path])
        else:
            subprocess.run(["xdg-open", target_path])

    def _copy_log(self):
        text = self.txt_log.toPlainText()
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "Copied", "Log content copied to clipboard!")

    def _open_log_file(self):
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "face_mosaic.log"
        if not log_file.exists():
            log_file.touch()
        self._open_path(str(log_file))

    def _show_disclaimer(self):
        title = "利用規約・免責事項 (Disclaimer & License)"
        text = (
            "<h3>auto-mosaic 利用規約・免責事項</h3>"
            "<p><b>【ライセンス】</b><br>"
            "本ソフトウェアは <b>Apache License, Version 2.0</b> に基づき提供されます。</p>"
            "<p><b>【目視による最終確認の義務】</b><br>"
            "本ツールはAI（コンピュータビジョン）を用いて顔・ナンバープレート検出およびモザイク／ぼかし処理を補助するツールです。"
            "カメラアングル、被写体の動き、遮蔽物、極端な照明等の要因により、検出漏れが発生する可能性を技術的に100%排除することはできません。<br><br>"
            "処理後の動画をYouTubeやSNS等の公の場へ公開・共有する前に、<b>必ず利用者の自己責任において全編の目視確認を行い、必要に応じて手動での追加修正を行ってください。</b></p>"
            "<p><b>【保証の否認・責任の制限】</b><br>"
            "本ソフトウェアは「現状有姿 (AS IS)」で提供され、いかなる明示的・黙示的保証も行いません。"
            "本ツールの利用、誤動作、検出漏れ等に起因して生じたいかなる損害（肖像権侵害、プライバシー侵害、名誉毀損、第三者との紛争、営業上の損失等）"
            "についても、開発者およびコントリビューターは一切の責任を負いません。</p>"
        )
        QMessageBox.information(self, title, text)

    def _show_about(self):
        title = "About auto-mosaic"
        text = (
            f"<h3>auto-mosaic v{__version__}</h3>"
            f"<p>街中散歩・車載動画向け オフライン自動モザイクツール（顔・ナンバープレート対応）</p>"
            f"<p><b>リポジトリ:</b> <a href='https://github.com/kh813/auto-mosaic'>https://github.com/kh813/auto-mosaic</a></p>"
            f"<p><b>動作環境:</b> Python {sys.version.split()[0]} / {self.hw_profile.reason}</p>"
        )
        QMessageBox.about(self, title, text)

    def _check_for_updates_ui(self, manual: bool = True):
        if manual:
            self.txt_log.append("GitHubに接続して最新バージョンを確認中...")
            self.btn_update.setEnabled(False)

        worker = UpdateCheckWorker(self)

        def on_finished(info):
            self.update_checker_worker = None
            self.btn_update.setEnabled(True)
            if info and info.get("is_newer"):
                tag = info["tag_name"]
                self.txt_log.append(f"\n🎉 新しいバージョンが利用可能です: {tag} ({info['title']})")
                self.btn_update.setText(f"🎉 Update to {tag}")
                self.btn_update.setStyleSheet("""
                    QPushButton {
                        background-color: #34C759;
                        color: white;
                        font-weight: bold;
                        border-radius: 4px;
                        padding: 0 8px;
                    }
                    QPushButton:hover { background-color: #2DB84D; }
                """)

                reply = QMessageBox.question(
                    self,
                    "アップデートが見つかりました",
                    f"新しいバージョン <b>{tag}</b> が公開されています！\n\n"
                    f"【リリース内容】\n{info['title']}\n\n"
                    f"今すぐセルフアップデートを実行しますか？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self._start_self_update(info)
            else:
                if manual:
                    self.txt_log.append(f"✓ 現在最新バージョン (v{__version__}) を使用しています。")
                    QMessageBox.information(
                        self,
                        "最新バージョンです",
                        f"お使いの auto-mosaic は最新バージョン (v{__version__}) です。"
                    )

        worker.finished_signal.connect(on_finished)
        self.update_checker_worker = worker
        worker.start()

    def _start_self_update(self, info: dict):
        self.txt_log.append("\n=======================================================")
        self.txt_log.append("               SELF-UPDATE IN PROGRESS                 ")
        self.txt_log.append("=======================================================")
        self.btn_update.setEnabled(False)

        progress_dlg = QProgressDialog("アップデートを準備中...", None, 0, 100, self)
        progress_dlg.setWindowTitle("auto-mosaic 自動更新")
        progress_dlg.setWindowModality(Qt.WindowModal)
        progress_dlg.setAutoClose(False)
        progress_dlg.setValue(5)
        progress_dlg.show()

        worker = SelfUpdateWorker(self)

        def on_prog(msg: str, pct: int):
            progress_dlg.setLabelText(msg)
            progress_dlg.setValue(max(5, min(95, pct)))
            self.txt_log.append(f"  {msg}")

        def on_finished(success: bool, msg: str):
            self.self_update_worker = None
            self.btn_update.setEnabled(True)
            progress_dlg.close()
            if success:
                self.txt_log.append(f"✓ アップデート成功: {msg}")
                self.btn_update.setText("🔄 Check Update")
                self.btn_update.setStyleSheet("")
                QMessageBox.information(
                    self,
                    "アップデート完了",
                    f"最新バージョンへのアップデートが完了しました！\n\n"
                    f"{msg}\n\n"
                    f"変更を反映するため、アプリケーションを再起動してください。"
                )
            else:
                self.txt_log.append(f"✗ アップデート失敗: {msg}")
                reply = QMessageBox.warning(
                    self,
                    "アップデート失敗",
                    f"アップデートに失敗しました:\n{msg}\n\nブラウザでリリースページを開きますか？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    from PySide6.QtGui import QDesktopServices
                    from PySide6.QtCore import QUrl
                    QDesktopServices.openUrl(QUrl(info.get("html_url", "https://github.com/kh813/auto-mosaic/releases")))

        worker.progress_signal.connect(on_prog)
        worker.finished_signal.connect(on_finished)
        self.self_update_worker = worker
        worker.start()



def run_app():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("auto_mosaic.gui.app.1.0")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("auto-mosaic")

    icon_path = Path(__file__).parent / "assets" / "icon.png"
    if not icon_path.exists():
        icon_path = Path(__file__).resolve().parent.parent.parent / "assets" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception:
        import traceback
        err_msg = traceback.format_exc()
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "auto-mosaic エラー", f"エラーが発生しました:\n\n{err_msg}")
        except Exception:
            pass
        print(err_msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_app()
