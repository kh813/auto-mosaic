import pytest
from PySide6.QtWidgets import QApplication
from face_mosaic.gui.app import AutoMosaicSplashScreen, MainWindow

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_splash_screen_creation(qapp):
    splash = AutoMosaicSplashScreen()
    splash.show()
    assert splash.isVisible()

    splash.set_progress(50, "テスト中...")
    assert splash.progress_value == 50
    assert splash.status_message == "テスト中..."
    splash.close()

def test_main_window_with_splash(qapp):
    splash = AutoMosaicSplashScreen()
    splash.show()

    window = MainWindow(splash=splash)
    assert splash.progress_value >= 75
    splash.finish(window)
    assert window.windowTitle() != ""
    window.close()
