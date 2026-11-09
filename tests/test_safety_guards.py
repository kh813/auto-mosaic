import pytest
from unittest.mock import MagicMock
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QCloseEvent
from face_mosaic.gui.app import MainWindow

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_close_event_prompt_when_running(qapp, monkeypatch):
    window = MainWindow()

    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = True
    window.worker = mock_worker

    # User clicks "No" on confirmation
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)
    event = QCloseEvent()
    window.closeEvent(event)

    # Event should be rejected (ignored) and worker should NOT be cancelled
    assert not event.isAccepted()
    mock_worker.cancel_all.assert_not_called()

    # User clicks "Yes" on confirmation
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    event_yes = QCloseEvent()
    window.closeEvent(event_yes)

    # Event should be accepted and worker cancelled
    assert event_yes.isAccepted()
    mock_worker.cancel_all.assert_called_once()

    # Reset worker state before closing
    mock_worker.isRunning.return_value = False
    window.close()

def test_self_update_blocked_when_processing(qapp, monkeypatch):
    window = MainWindow()

    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = True
    window.worker = mock_worker

    warn_called = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warn_called.append(True))

    window._start_self_update({"tag_name": "v9.9.9", "title": "New Version"})

    assert len(warn_called) == 1
    assert window.self_update_worker is None

    # Reset worker state before closing
    mock_worker.isRunning.return_value = False
    window.close()
