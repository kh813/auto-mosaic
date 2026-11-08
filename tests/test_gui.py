import pytest
from unittest.mock import MagicMock
from PySide6.QtWidgets import QApplication
from face_mosaic.gui.app import MainWindow

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

@pytest.fixture(autouse=True)
def isolate_qsettings(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    test_ini = str(tmp_path / "test_settings.ini")
    monkeypatch.setattr(
        "face_mosaic.gui.app.QSettings",
        lambda *args, **kwargs: QSettings(test_ini, QSettings.IniFormat)
    )

def test_gui_window_initialization(qapp):
    window = MainWindow()
    assert window.windowTitle() == "auto-mosaic: Automated Face & License Plate Blur Tool"
    assert window.combo_model.count() >= 3
    assert window.spin_pad_back.value() == 12
    assert window.spin_pad_fwd.value() == 24
    assert window.combo_codec.currentData() == "auto"
    assert window.spin_crf.value() == 23
    assert window.combo_crf_preset.currentText() == "標準 (23 - 推奨)"
    assert window.combo_bit_depth.currentText() == "Auto (自動)"
    assert not window.chk_append_params.isChecked()
    assert not window.windowIcon().isNull()
    assert hasattr(window, "btn_update")
    assert "Check Update" in window.btn_update.text()

def test_gui_blur_type_switch(qapp):
    window = MainWindow()
    window.combo_blur_type.setCurrentText("Mosaic")
    assert window.lbl_strength.text() == "Mosaic Block Size:"
    window.combo_blur_type.setCurrentText("Gaussian")
    assert window.lbl_strength.text() == "Blur Strength (odd):"

def test_gui_preview_elements(qapp):
    window = MainWindow()
    assert hasattr(window, "lbl_preview")
    assert hasattr(window, "preview_widget")
    assert hasattr(window, "btn_open_video")
    assert hasattr(window, "btn_open_folder")
    assert hasattr(window, "table_queue")
    assert hasattr(window, "btn_cancel_current")
    assert not window.btn_open_video.isEnabled()
    assert not window.btn_open_folder.isEnabled()
    assert not window.btn_cancel_current.isEnabled()

def test_gui_queue_management(qapp, tmp_path):
    from pathlib import Path
    window = MainWindow()
    f1 = tmp_path / "video1.mp4"
    f2 = tmp_path / "video2.mp4"
    f1.touch()
    f2.touch()

    # Add to queue
    window._add_to_queue([f1, f2])
    assert len(window.queue_items) == 2
    assert window.table_queue.rowCount() == 2

    # Duplicate should not be added
    window._add_to_queue([f1])
    assert len(window.queue_items) == 2

    # Select row 0 and remove
    window.table_queue.selectRow(0)
    window._remove_selected_files()
    assert len(window.queue_items) == 1
    assert window.queue_items[0].path.name == "video2.mp4"

    # Clear queue
    window._clear_queue()
    assert len(window.queue_items) == 0
    assert window.table_queue.rowCount() == 0

def test_video_preview_widget_rendering(qapp):
    import numpy as np
    from face_mosaic.gui.app import VideoPreviewWidget

    widget = VideoPreviewWidget()
    assert widget.pixmap is None

    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    widget.set_frame(test_frame, "Pass 1/2: Detection (1/10)")
    assert widget.pixmap is not None
    assert not widget.pixmap.isNull()
    assert "Pass 1" in widget.status_text

    widget.clear_preview("Done")
    assert widget.pixmap is None
    assert widget.placeholder_text == "Done"

def test_gui_remember_last_directory(qapp, tmp_path):
    window = MainWindow()
    test_dir = str(tmp_path / "my_videos")
    import os
    os.makedirs(test_dir, exist_ok=True)

    window.last_open_dir = test_dir
    window.settings.setValue("last_dir", test_dir)

    # Re-initialize another window to verify persistence
    window2 = MainWindow()
    assert window2.last_open_dir == test_dir


def test_queue_item_id_and_granular_update(qapp, tmp_path):
    from face_mosaic.gui.app import QueueItem
    window = MainWindow()
    f1 = tmp_path / "v1.mp4"
    f2 = tmp_path / "v2.mp4"
    f1.touch()
    f2.touch()

    window._add_to_queue([f1, f2])
    item1 = window.queue_items[0]
    item2 = window.queue_items[1]

    # Verify unique IDs
    assert item1.item_id != item2.item_id

    # Emit update for item2 using item_id
    window._on_item_updated(item2.item_id, "Processing", "50%", "")
    assert item2.status == "Processing"
    assert item2.progress == "50%"
    assert window.table_queue.item(1, 2).text() == "Processing"
    assert window.table_queue.item(1, 3).text() == "50%"

    # Updating unknown item_id does nothing and does not crash
    window._on_item_updated("non-existent-uuid", "Completed", "100%", "")


def test_gui_start_stop_state_machine(qapp, tmp_path):
    window = MainWindow()
    f1 = tmp_path / "v1.mp4"
    f1.touch()
    window._add_to_queue([f1])

    # Initially idle
    assert window.btn_start.isEnabled()
    assert "Start Processing" in window.btn_start.text()
    assert not window.is_stopping

    # Mock running worker
    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = True
    window.worker = mock_worker

    # User clicks Stop Processing
    window._toggle_processing()
    assert window.is_stopping is True
    assert not window.btn_start.isEnabled()
    assert "Stopping" in window.btn_start.text()
    mock_worker.cancel_all.assert_called_once()

    # If clicked while stopping, nothing happens
    window._toggle_processing()
    assert window.is_stopping is True

    # Finished signal arrives
    window._on_finished(False, "Stopped by user.")
    assert window.is_stopping is False
    assert window.btn_start.isEnabled()
    assert "Start Processing" in window.btn_start.text()
    assert window.worker is None


def test_exact_user_workflow_delete_and_process_next(qapp, tmp_path, monkeypatch):
    from unittest.mock import patch
    window = MainWindow()
    f1 = tmp_path / "v1.mp4"
    f2 = tmp_path / "v2.mp4"
    f1.touch()
    f2.touch()

    window._add_to_queue([f1, f2])

    # Simulate earlier run was stopped/cancelled
    window.queue_items[0].status = "Cancelled"
    window.queue_items[0].progress = "Cancelled"
    window.queue_items[1].status = "Cancelled"
    window.queue_items[1].progress = "Cancelled"
    window._refresh_queue_table()

    # User selects row 0 (f1) and removes it
    window.table_queue.selectRow(0)
    window._remove_selected_files()

    # Queue now only has f2
    assert len(window.queue_items) == 1
    assert window.queue_items[0].path.name == "v2.mp4"
    assert window.table_queue.rowCount() == 1

    # User selects f2 (row 0 in current table)
    window.table_queue.selectRow(0)
    window._on_table_selection_changed()

    # Start button should indicate selected file
    assert "Start Processing (1 selected)" in window.btn_start.text()

    # Mock ProcessingWorker.start to prevent actual FFmpeg pipeline execution in unit test
    with patch("face_mosaic.gui.app.ProcessingWorker.start") as mock_start:
        window._toggle_processing()
        mock_start.assert_called_once()

    # Verify f2's status was reset to "Waiting" and progress to "0%" (not skipped!)
    assert window.queue_items[0].status == "Waiting"
    assert window.queue_items[0].progress == "0%"
    assert window.worker is not None
    assert len(window.worker.items) == 1
    assert window.worker.items[0].path.name == "v2.mp4"
    assert window.btn_start.text() == "⏹ Stop Processing"
    assert window.btn_cancel_current.isEnabled()


def test_gui_cancel_current_file(qapp):
    window = MainWindow()
    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = True
    window.worker = mock_worker

    window._cancel_current_file()
    mock_worker.cancel_current_file.assert_called_once()
    assert "Cancelling" in window.txt_log.toPlainText()


def test_gui_target_items_override(qapp, tmp_path):
    from unittest.mock import patch
    window = MainWindow()
    f1 = tmp_path / "v1.mp4"
    f2 = tmp_path / "v2.mp4"
    f3 = tmp_path / "v3.mp4"
    f1.touch()
    f2.touch()
    f3.touch()

    window._add_to_queue([f1, f2, f3])
    # Target only f2
    with patch("face_mosaic.gui.app.ProcessingWorker.start") as mock_start:
        window._toggle_processing(target_items_override=[window.queue_items[1]])
        mock_start.assert_called_once()

    assert len(window.worker.items) == 1
    assert window.worker.items[0].path.name == "v2.mp4"


def test_gui_btn_start_click_event(qapp, tmp_path):
    from unittest.mock import patch
    window = MainWindow()
    f1 = tmp_path / "v1.mp4"
    f1.touch()
    window._add_to_queue([f1])

    with patch("face_mosaic.gui.app.ProcessingWorker.start") as mock_start:
        # Simulate real QPushButton click event (which emits clicked(bool checked=False))
        window.btn_start.click()
        mock_start.assert_called_once()

    assert window.worker is not None
    assert len(window.worker.items) == 1
    assert window.worker.items[0].path.name == "v1.mp4"


def test_gui_btn_start_empty_queue_shows_warning(qapp):
    from unittest.mock import patch
    window = MainWindow()
    assert len(window.queue_items) == 0

    with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
        window.btn_start.click()
        mock_warn.assert_called_once()

    assert window.worker is None
    assert "Start Processing" in window.btn_start.text()


def test_gui_btn_clear_queue_idle_and_active(qapp, tmp_path):
    from unittest.mock import patch
    window = MainWindow()
    f1 = tmp_path / "v1.mp4"
    f1.touch()
    window._add_to_queue([f1])
    assert len(window.queue_items) == 1

    # Active processing: warning shown, queue NOT cleared
    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = True
    window.worker = mock_worker

    with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
        window.btn_clear.click()
        mock_warn.assert_called_once()
    assert len(window.queue_items) == 1

    # Idle: cleared cleanly
    window.worker = None
    window.btn_clear.click()
    assert len(window.queue_items) == 0
    assert window.table_queue.rowCount() == 0


def test_gui_btn_remove_active_file_prompt(qapp, tmp_path):
    from unittest.mock import patch
    from PySide6.QtWidgets import QMessageBox
    window = MainWindow()
    f1 = tmp_path / "v1.mp4"
    f2 = tmp_path / "v2.mp4"
    f1.touch()
    f2.touch()
    window._add_to_queue([f1, f2])

    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = True
    mock_worker.current_item_id = window.queue_items[0].item_id
    window.worker = mock_worker

    window.table_queue.selectRow(0)

    # If user answers No:
    with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.No):
        window.btn_remove_selected.click()
        mock_worker.cancel_current_file.assert_not_called()
        assert len(window.queue_items) == 2

    # If user answers Yes:
    with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.Yes):
        window.btn_remove_selected.click()
        mock_worker.cancel_current_file.assert_called_once()
        assert len(window.queue_items) == 1
        assert window.queue_items[0].path.name == "v2.mp4"


def test_gui_file_and_folder_dialog_actions(qapp, tmp_path):
    from unittest.mock import patch
    window = MainWindow()
    f1 = tmp_path / "v1.mp4"
    f2 = tmp_path / "v2.mov"
    f1.touch()
    f2.touch()

    # Test file dialog
    with patch("PySide6.QtWidgets.QFileDialog.getOpenFileNames", return_value=([str(f1), str(f2)], "filter")):
        window.btn_select_files.click()
        assert len(window.queue_items) == 2
        assert window.last_open_dir == str(tmp_path)

    # Test folder dialog
    sub_dir = tmp_path / "folder_videos"
    sub_dir.mkdir()
    f3 = sub_dir / "v3.mp4"
    f3.touch()
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(sub_dir)):
        window.btn_select_folder.click()
        assert len(window.queue_items) == 3
        assert window.last_open_dir == str(sub_dir)


def test_gui_multi_selection_and_deselection(qapp, tmp_path):
    from unittest.mock import patch
    from PySide6.QtCore import QItemSelectionModel
    window = MainWindow()
    files = [tmp_path / f"v{i}.mp4" for i in range(1, 5)]
    for f in files:
        f.touch()
    window._add_to_queue(files)

    # Multi-select rows 1 and 3 using QItemSelectionModel
    sel_model = window.table_queue.selectionModel()
    idx1 = window.table_queue.model().index(1, 0)
    idx3 = window.table_queue.model().index(3, 0)
    sel_model.select(idx1, QItemSelectionModel.Select | QItemSelectionModel.Rows)
    sel_model.select(idx3, QItemSelectionModel.Select | QItemSelectionModel.Rows)
    window._on_table_selection_changed()

    assert "Start Processing (2 selected)" in window.btn_start.text()

    # When Start is clicked, all non-completed items in the queue are processed
    # (selection is used only for display hint on button label, not actual queue filtering)
    with patch("face_mosaic.gui.app.ProcessingWorker.start") as mock_start:
        window.btn_start.click()
        mock_start.assert_called_once()

    assert window.worker is not None
    # All 4 non-completed items are processed (not just the 2 selected)
    assert len(window.worker.items) == 4

    # Deselect all
    window._on_finished(False, "Stopped")
    window.table_queue.clearSelection()
    window._on_table_selection_changed()
    assert window.btn_start.text() == "▶ Start Processing"


def test_gui_table_selection_while_running_preserves_stop_button(qapp, tmp_path):
    window = MainWindow()
    f1 = tmp_path / "v1.mp4"
    f1.touch()
    window._add_to_queue([f1])

    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = True
    window.worker = mock_worker

    # User clicks row while worker is running
    window.table_queue.selectRow(0)
    window._on_table_selection_changed()

    assert window.btn_start.text() == "⏹ Stop Processing"
    assert window.btn_start.isEnabled()


def test_gui_copy_log_button(qapp):
    from unittest.mock import patch, MagicMock
    window = MainWindow()
    window.txt_log.setPlainText("Test Diagnostic Log Output 12345")

    mock_clipboard = MagicMock()
    with patch("face_mosaic.gui.app.QApplication.clipboard", return_value=mock_clipboard):
        with patch("PySide6.QtWidgets.QMessageBox.information"):
            window.btn_copy_log.click()

    mock_clipboard.setText.assert_called_once_with("Test Diagnostic Log Output 12345")


def test_gui_disclaimer_button(qapp):
    from unittest.mock import patch
    window = MainWindow()
    with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
        window.btn_disclaimer.click()
        assert mock_info.called
        title, text = mock_info.call_args[0][1], mock_info.call_args[0][2]
        assert "免責事項" in title
        assert "Apache License, Version 2.0" in text
        assert "目視" in text


def test_gui_context_menu_actions(qapp, tmp_path):
    from unittest.mock import patch
    from PySide6.QtCore import QPoint
    window = MainWindow()
    files = [tmp_path / f"v{i}.mp4" for i in range(1, 5)]
    for f in files:
        f.touch()
    window._add_to_queue(files)

    # Test right-click context menu generates actions by mocking QMenu constructor in app.py
    with patch("face_mosaic.gui.app.QMenu") as mock_qmenu_cls:
        mock_menu = MagicMock()
        mock_qmenu_cls.return_value = mock_menu
        with patch.object(window.table_queue, "itemAt", return_value=window.table_queue.item(1, 1)):
            window._show_table_context_menu(QPoint(10, 10))
            mock_menu.exec.assert_called_once()
            assert mock_menu.addAction.call_count >= 2

    # Test single row removal from queue via context menu helper
    window._remove_single_row(0)
    assert len(window.queue_items) == 3
    assert window.queue_items[0].path.name == "v2.mp4"


def test_gui_os_open_path_dispatch_windows(qapp, tmp_path, monkeypatch):
    import sys
    window = MainWindow()
    test_file = tmp_path / "output.mp4"
    test_file.touch()

    # Windows dispatch test
    monkeypatch.setattr(sys, "platform", "win32")
    mock_startfile = MagicMock()
    monkeypatch.setattr("os.startfile", mock_startfile, raising=False)

    window._open_path(str(test_file))
    mock_startfile.assert_called_once_with(str(test_file))


def test_gui_os_open_path_dispatch_macos(qapp, tmp_path, monkeypatch):
    import sys
    window = MainWindow()
    test_file = tmp_path / "output.mp4"
    test_file.touch()

    # macOS dispatch test
    monkeypatch.setattr(sys, "platform", "darwin")
    mock_subrun = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_subrun)

    window._open_path(str(test_file))
    mock_subrun.assert_called_once_with(["open", str(test_file)])


def test_gui_os_open_path_dispatch_linux(qapp, tmp_path, monkeypatch):
    import sys
    window = MainWindow()
    test_file = tmp_path / "output.mp4"
    test_file.touch()

    # Linux dispatch test
    monkeypatch.setattr(sys, "platform", "linux")
    mock_subrun = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_subrun)

    window._open_path(str(test_file))
    mock_subrun.assert_called_once_with(["xdg-open", str(test_file)])


def test_subprocess_window_flags_windows_vs_unix(monkeypatch):
    import sys
    import subprocess
    from face_mosaic.io_utils import get_subprocess_kwargs

    # Windows check
    monkeypatch.setattr(sys, "platform", "win32")
    # Simulate Windows-specific attributes if running on Unix
    if not hasattr(subprocess, "CREATE_NO_WINDOW"):
        monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    if not hasattr(subprocess, "STARTUPINFO"):
        class DummyStartupInfo:
            dwFlags = 0
            wShowWindow = 0
        monkeypatch.setattr(subprocess, "STARTUPINFO", DummyStartupInfo, raising=False)
        monkeypatch.setattr(subprocess, "STARTF_USESHOWWINDOW", 1, raising=False)

    kwargs_win = get_subprocess_kwargs()
    assert kwargs_win.get("creationflags") in (getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000), 0x08000000)
    assert kwargs_win.get("startupinfo") is not None
    assert kwargs_win["startupinfo"].wShowWindow == 0

    # macOS / Linux check
    monkeypatch.setattr(sys, "platform", "darwin")
    kwargs_mac = get_subprocess_kwargs()
    assert kwargs_mac == {}

    monkeypatch.setattr(sys, "platform", "linux")
    kwargs_linux = get_subprocess_kwargs()
    assert kwargs_linux == {}


def test_worker_thread_resiliency_when_items_removed_externally(qapp, tmp_path):
    from unittest.mock import patch
    from face_mosaic.config import AppConfig
    from face_mosaic.gui.app import ProcessingWorker, QueueItem

    cfg = AppConfig()
    f1 = tmp_path / "v1.mp4"
    f2 = tmp_path / "v2.mp4"
    f3 = tmp_path / "v3.mp4"
    f1.touch()
    f2.touch()
    f3.touch()

    from pathlib import Path
    items = [QueueItem(path=f1), QueueItem(path=f2), QueueItem(path=f3)]
    # Mark item 1 as removed externally by GUI
    items[1].is_removed = True

    worker = ProcessingWorker(cfg, items)

    processed_names = []
    def mock_process_video(video_path, **kwargs):
        processed_names.append(Path(video_path).name)
        return {
            "cancelled": False,
            "output": str(video_path) + "_out.mp4",
            "elapsed_seconds": 0.1,
            "processing_fps": 30.0,
            "color_mae": 0.0
        }

    with patch("face_mosaic.gui.app.ProcessingPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.detector.active_provider = "CPU"
        mock_pipeline.process_video.side_effect = mock_process_video
        mock_pipeline_cls.return_value = mock_pipeline

        worker.run()

    # Verify v2 was skipped cleanly and v1 and v3 were processed!
    assert processed_names == ["v1.mp4", "v3.mp4"]
    assert items[0].status == "Completed"
    # items[1] is_removed=True so it's filtered from valid_items and keeps its original status
    assert items[1].is_removed is True
    assert items[2].status == "Completed"


def test_gui_last_directory_persistence_and_sync(qapp, tmp_path):
    from unittest.mock import patch
    import os
    window = MainWindow()

    folder_a = tmp_path / "folder_a"
    folder_a.mkdir()
    f1 = folder_a / "clip1.mp4"
    f1.touch()

    # When user selects files via dialog, last_open_dir and settings must be updated to folder_a
    with patch("PySide6.QtWidgets.QFileDialog.getOpenFileNames", return_value=([str(f1)], "filter")):
        window.btn_select_files.click()

    assert window.last_open_dir == str(folder_a.resolve())
    assert window.settings.value("last_dir") == str(folder_a.resolve())

    # Verify that launching another window retains folder_a
    window2 = MainWindow()
    assert window2.last_open_dir == str(folder_a.resolve())


def test_gui_drag_and_drop_imports(qapp, tmp_path):
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtGui import QDropEvent
    from PySide6.QtCore import QPointF, Qt

    window = MainWindow()
    drop_dir = tmp_path / "dropped_videos"
    drop_dir.mkdir()
    v1 = drop_dir / "vid1.mp4"
    v1.touch()

    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(v1))])

    event = QDropEvent(
        QPointF(10, 10),
        Qt.CopyAction,
        mime_data,
        Qt.LeftButton,
        Qt.NoModifier
    )
    window.dropEvent(event)

    # File must be added to queue and last_open_dir updated to drop_dir
    assert len(window.queue_items) == 1
    assert window.queue_items[0].path.name == "vid1.mp4"
    assert window.last_open_dir == str(drop_dir.resolve())
    assert window.settings.value("last_dir") == str(drop_dir.resolve())


def test_gui_close_event_persists_settings(qapp, tmp_path):
    from PySide6.QtGui import QCloseEvent
    window = MainWindow()
    custom_dir = tmp_path / "custom_dir"
    custom_dir.mkdir()
    window.last_open_dir = str(custom_dir.resolve())

    close_ev = QCloseEvent()
    window.closeEvent(close_ev)

    assert window.settings.value("last_dir") == str(custom_dir.resolve())


def test_gui_queue_reorder_up_down_buttons(qapp, tmp_path):
    window = MainWindow()
    v1 = tmp_path / "video1.mp4"
    v2 = tmp_path / "video2.mp4"
    v3 = tmp_path / "video3.mp4"
    v1.touch()
    v2.touch()
    v3.touch()
    window._add_to_queue([v1, v2, v3])

    # Select row 1 (video2)
    window.table_queue.selectRow(1)
    assert window.btn_move_up.isEnabled()
    assert window.btn_move_down.isEnabled()

    # Move video2 up
    window.btn_move_up.click()
    names = [item.path.name for item in window.queue_items]
    assert names == ["video2.mp4", "video1.mp4", "video3.mp4"]
    # Row 0 should now be selected and Move Up disabled
    assert window.table_queue.currentRow() == 0
    assert not window.btn_move_up.isEnabled()
    assert window.btn_move_down.isEnabled()

    # Move video2 down twice
    window.btn_move_down.click()
    assert [item.path.name for item in window.queue_items] == ["video1.mp4", "video2.mp4", "video3.mp4"]
    assert window.table_queue.currentRow() == 1

    window.btn_move_down.click()
    assert [item.path.name for item in window.queue_items] == ["video1.mp4", "video3.mp4", "video2.mp4"]
    assert window.table_queue.currentRow() == 2
    assert window.btn_move_up.isEnabled()
    assert not window.btn_move_down.isEnabled()


def test_gui_queue_reorder_top_bottom_context_menu(qapp, tmp_path):
    window = MainWindow()
    files = [tmp_path / f"video{i}.mp4" for i in range(1, 5)]
    for f in files:
        f.touch()
    window._add_to_queue(files)

    # Move row 3 (video4) to top
    window.table_queue.selectRow(3)
    window._move_selected_to_top()
    assert [item.path.name for item in window.queue_items] == ["video4.mp4", "video1.mp4", "video2.mp4", "video3.mp4"]
    assert window.table_queue.currentRow() == 0

    # Move row 0 (video4) to bottom
    window._move_selected_to_bottom()
    assert [item.path.name for item in window.queue_items] == ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
    assert window.table_queue.currentRow() == 3


def test_gui_queue_mouse_drag_and_drop_reorder(qapp, tmp_path):
    window = MainWindow()
    files = [tmp_path / f"clip{i}.mp4" for i in range(1, 4)]
    for f in files:
        f.touch()
    window._add_to_queue(files)

    # Simulate internal drop event from row 0 onto row 2 (clip1 -> bottom)
    window._reorder_rows_to_target([0], 2)
    assert [item.path.name for item in window.queue_items] == ["clip2.mp4", "clip3.mp4", "clip1.mp4"]
    assert window.table_queue.currentRow() == 2

    # Simulate internal drop event from row 2 onto row 0 (clip1 -> top)
    window._reorder_rows_to_target([2], 0)
    assert [item.path.name for item in window.queue_items] == ["clip1.mp4", "clip2.mp4", "clip3.mp4"]
    assert window.table_queue.currentRow() == 0


def test_gui_queue_reorder_with_active_processing_sync(qapp, tmp_path):
    from unittest.mock import MagicMock
    from face_mosaic.config import AppConfig

    window = MainWindow()
    files = [tmp_path / f"vid{i}.mp4" for i in range(1, 4)]
    for f in files:
        f.touch()
    window._add_to_queue(files)

    # Set item 0 as Processing
    window.queue_items[0].status = "Processing"
    window.queue_items[1].status = "Waiting"
    window.queue_items[2].status = "Waiting"

    # Mock worker
    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = True
    mock_worker.current_item_id = window.queue_items[0].item_id
    mock_worker.items = list(window.queue_items)
    window.worker = mock_worker

    # Attempting to move active item 0 should be rejected
    window.table_queue.selectRow(0)
    window._move_selected_down()
    assert [item.path.name for item in window.queue_items] == ["vid1.mp4", "vid2.mp4", "vid3.mp4"]

    # Moving waiting item 2 (vid3) up into position 1 should succeed
    window.table_queue.selectRow(2)
    window._move_selected_up()
    assert [item.path.name for item in window.queue_items] == ["vid1.mp4", "vid3.mp4", "vid2.mp4"]
    # And worker.items should be synchronized with the new waiting order
    assert [item.path.name for item in window.worker.items] == ["vid1.mp4", "vid3.mp4", "vid2.mp4"]


def test_gui_preview_action_buttons_styling_and_disabled_during_worker(qapp, tmp_path):
    from unittest.mock import MagicMock
    from face_mosaic.gui.app import QueueItem

    window = MainWindow()

    # Verify styling contains readable text color (#FFFFFF) and disabled state
    video_qss = window.btn_open_video.styleSheet()
    folder_qss = window.btn_open_folder.styleSheet()
    assert "#FFFFFF" in video_qss
    assert "#E5E5EA" in video_qss
    assert ":disabled" in video_qss
    assert "#FFFFFF" in folder_qss
    assert "#E5E5EA" in folder_qss
    assert ":disabled" in folder_qss

    # Initial state: no completed video, buttons disabled
    assert not window.btn_open_video.isEnabled()
    assert not window.btn_open_folder.isEnabled()

    # Add completed video item
    completed_video = tmp_path / "out.mp4"
    completed_video.touch()
    item = QueueItem(path=tmp_path / "in.mp4", status="Completed", output_path=str(completed_video))
    window.queue_items.append(item)

    # When idle, update enables buttons
    window._update_output_button_states()
    assert window.btn_open_video.isEnabled()
    assert window.btn_open_folder.isEnabled()

    # When worker is actively running, buttons MUST be disabled (greyed out)
    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = True
    window.worker = mock_worker

    window._update_output_button_states()
    assert not window.btn_open_video.isEnabled()
    assert not window.btn_open_folder.isEnabled()

    # When worker finishes, buttons become enabled again
    mock_worker.isRunning.return_value = False
    window._update_output_button_states()
    assert window.btn_open_video.isEnabled()
    assert window.btn_open_folder.isEnabled()

def test_gui_crf_presets_and_output_options(qapp, tmp_path, monkeypatch):
    window = MainWindow()

    # Test preset combo -> spinbox
    window.combo_crf_preset.setCurrentIndex(1)  # 最高画質 (18)
    assert window.spin_crf.value() == 18

    window.combo_crf_preset.setCurrentIndex(2)  # 容量優先 (26)
    assert window.spin_crf.value() == 26

    window.combo_crf_preset.setCurrentIndex(0)  # 標準 (23)
    assert window.spin_crf.value() == 23

    # Test spinbox -> preset combo
    window.spin_crf.setValue(18)
    assert "18" in window.combo_crf_preset.currentText()
    window.spin_crf.setValue(35)
    assert window.combo_crf_preset.currentText() == "カスタム"

    # Test Bit Depth & Param Suffix options
    window.combo_bit_depth.setCurrentIndex(1)  # 8-bit
    window.chk_append_params.setChecked(True)

    # Capture config launched into worker
    captured_cfg = []
    class MockWorker:
        def __init__(self, cfg, items):
            captured_cfg.append(cfg)
        def start(self): pass
        item_updated_signal = MagicMock()
        preview_signal = MagicMock()
        log_signal = MagicMock()
        overall_progress_signal = MagicMock()
        finished_signal = MagicMock()

    monkeypatch.setattr("face_mosaic.gui.app.ProcessingWorker", MockWorker)
    
    # Add dummy item via proper queue addition
    test_v = tmp_path / "dummy.mp4"
    test_v.touch()
    window._add_to_queue([test_v])

    window._toggle_processing()
    assert len(captured_cfg) == 1
    cfg = captured_cfg[0]
    assert cfg.output.crf == 35
    assert cfg.output.bit_depth == "8bit"
    assert cfg.output.append_params_to_filename is True


def test_gui_target_mode_selection_and_load_optimization_controls(qapp, tmp_path, monkeypatch):
    window = MainWindow()

    # Default is Faces Only ("faces")
    assert window.combo_target.currentData() == "faces"
    assert window.combo_model.isEnabled()
    assert window.spin_conf.isEnabled()
    assert not window.spin_plate_conf.isEnabled()

    # Switch to Faces + Plates ("both")
    window.combo_target.setCurrentIndex(1)
    assert window.combo_target.currentData() == "both"
    assert window.combo_model.isEnabled()
    assert window.spin_conf.isEnabled()
    assert window.spin_plate_conf.isEnabled()

    # Switch to Plates Only ("plates") -> face controls disabled, plate controls enabled
    window.combo_target.setCurrentIndex(2)
    assert window.combo_target.currentData() == "plates"
    assert not window.combo_model.isEnabled()
    assert not window.spin_conf.isEnabled()
    assert window.spin_plate_conf.isEnabled()
    window.spin_plate_conf.setValue(0.55)

    captured_cfg = []
    class MockWorker:
        def __init__(self, cfg, items):
            captured_cfg.append(cfg)
        def start(self): pass
        item_updated_signal = MagicMock()
        preview_signal = MagicMock()
        log_signal = MagicMock()
        overall_progress_signal = MagicMock()
        finished_signal = MagicMock()

    monkeypatch.setattr("face_mosaic.gui.app.ProcessingWorker", MockWorker)
    test_v = tmp_path / "plate_test.mp4"
    test_v.touch()
    window._add_to_queue([test_v])
    window._toggle_processing()

    assert len(captured_cfg) == 1
    assert captured_cfg[0].target == "plates"
    assert captured_cfg[0].plate_model.conf_threshold == pytest.approx(0.55)


def test_gui_layout_fits_full_hd(qapp):
    """Ensure MainWindow minimum and default dimensions fit comfortably within Full HD (even at 125-150% DPI scaling)."""
    window = MainWindow()
    min_hint = window.minimumSizeHint()
    assert min_hint.width() <= 1200, f"Window min width too wide for scaled Full HD: {min_hint.width()}"
    assert min_hint.height() <= 680, f"Window min height too tall for scaled Full HD: {min_hint.height()}"
    assert window.width() <= 1200
    assert window.height() <= 720


def test_gui_window_centering(qapp):
    window = MainWindow()
    assert hasattr(window, "_center_on_screen")
    window._center_on_screen()
    # Check that window fits within primary screen available geometry
    screen = QApplication.primaryScreen()
    if screen:
        avail = screen.availableGeometry()
        assert window.width() <= avail.width()
        assert window.height() <= avail.height()











