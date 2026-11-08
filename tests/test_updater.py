import io
import sys
import json
import zipfile
import tarfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from face_mosaic.updater import (
    parse_version,
    is_newer_version,
    is_git_repository,
    check_for_updates,
    update_via_git,
    update_via_release_asset,
    perform_self_update,
    get_project_root,
)


def test_parse_version():
    assert parse_version("0.2.0") == (0, 2, 0)
    assert parse_version("v0.2.0") == (0, 2, 0)
    assert parse_version("V1.4.10") == (1, 4, 10)
    assert parse_version("v2.1") == (2, 1, 0)
    assert parse_version("3") == (3, 0, 0)
    assert parse_version("invalid") == (0, 0, 0)
    assert parse_version("") == (0, 0, 0)


def test_is_newer_version():
    assert is_newer_version("v0.3.0", "0.2.0") is True
    assert is_newer_version("0.2.1", "0.2.0") is True
    assert is_newer_version("1.0.0", "0.9.9") is True
    assert is_newer_version("v0.2.0", "0.2.0") is False
    assert is_newer_version("0.1.9", "0.2.0") is False
    assert is_newer_version("0.0.1", "0.2.0") is False


def test_is_git_repository(tmp_path):
    assert not is_git_repository(tmp_path)
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    assert is_git_repository(tmp_path)


def test_check_for_updates_newer_release():
    mock_response_data = {
        "tag_name": "v0.9.0",
        "name": "Release 0.9.0",
        "body": "Exciting new features",
        "html_url": "https://github.com/kh813/auto-mosaic/releases/tag/v0.9.0",
        "published_at": "2026-10-11T20:00:00Z",
        "assets": [
            {
                "name": "auto-mosaic-windows.zip",
                "browser_download_url": "https://example.com/auto-mosaic-windows.zip",
            },
            {
                "name": "auto-mosaic-macos.zip",
                "browser_download_url": "https://example.com/auto-mosaic-macos.zip",
            },
        ],

    }
    raw_json = json.dumps(mock_response_data).encode("utf-8")

    cm = MagicMock()
    cm.__enter__.return_value.status = 200
    cm.__enter__.return_value.read.return_value = raw_json

    with patch("urllib.request.urlopen", return_value=cm):
        info = check_for_updates("0.2.0")
        assert info is not None
        assert info["version"] == "0.9.0"
        assert info["tag_name"] == "v0.9.0"
        assert info["is_newer"] is True
        expected_asset = (
            "https://example.com/auto-mosaic-windows.zip"
            if sys.platform == "win32"
            else "https://example.com/auto-mosaic-macos.zip"
        )
        assert info["asset_url"] == expected_asset



def test_check_for_updates_no_newer_release():
    mock_response_data = {
        "tag_name": "v0.2.0",
        "name": "Release 0.2.0",
        "assets": [],
    }
    raw_json = json.dumps(mock_response_data).encode("utf-8")

    cm = MagicMock()
    cm.__enter__.return_value.status = 200
    cm.__enter__.return_value.read.return_value = raw_json

    with patch("urllib.request.urlopen", return_value=cm):
        info = check_for_updates("0.2.0")
        assert info is None


def test_check_for_updates_network_error():
    with patch("urllib.request.urlopen", side_effect=Exception("Connection timed out")):
        info = check_for_updates("0.2.0")
        assert info is None


def test_update_via_git_success(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Already up to date.", stderr="")
        success, msg = update_via_git(tmp_path)
        assert success is True
        assert "Successfully updated" in msg
        mock_run.assert_called()


def test_update_via_git_failure(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Merge conflict")
        success, msg = update_via_git(tmp_path)
        assert success is False
        assert "Git pull failed" in msg


def test_update_via_release_asset_zip(tmp_path):
    target_dir = tmp_path / "app"
    target_dir.mkdir()
    (target_dir / "old_file.txt").write_text("old version")

    # Create an in-memory zip archive
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("new_file.txt", "new version contents")
        z.writestr("old_file.txt", "overwritten version")
    zip_bytes = buf.getvalue()

    cm = MagicMock()
    cm.__enter__.return_value.headers = {"content-length": str(len(zip_bytes))}
    cm.__enter__.return_value.read.side_effect = [zip_bytes, b""]

    progress_calls = []

    def progress(cur, tot):
        progress_calls.append((cur, tot))

    with patch("urllib.request.urlopen", return_value=cm):
        success, msg = update_via_release_asset(
            "https://example.com/release.zip",
            target_dir=target_dir,
            progress_callback=progress
        )
        assert success is True
        assert (target_dir / "new_file.txt").exists()
        assert (target_dir / "new_file.txt").read_text() == "new version contents"
        assert (target_dir / "old_file.txt").read_text() == "overwritten version"
        assert len(progress_calls) > 0


def test_perform_self_update_git_branch(tmp_path):
    with patch("face_mosaic.updater.get_project_root", return_value=tmp_path):
        with patch("face_mosaic.updater.is_git_repository", return_value=True):
            with patch("face_mosaic.updater.update_via_git", return_value=(True, "Git updated")) as mock_git:
                success, msg = perform_self_update()
                assert success is True
                assert msg == "Git updated"
                mock_git.assert_called_once_with(tmp_path)


def test_perform_self_update_standalone_no_update():
    with patch("face_mosaic.updater.is_git_repository", return_value=False):
        with patch("face_mosaic.updater.check_for_updates", return_value=None):
            success, msg = perform_self_update()
            assert success is True
            assert "Already running the latest version" in msg


def test_perform_self_update_standalone_with_update(tmp_path):
    update_info = {
        "version": "0.3.0",
        "tag_name": "v0.3.0",
        "asset_url": "https://example.com/asset.zip",
    }
    with patch("face_mosaic.updater.get_project_root", return_value=tmp_path):
        with patch("face_mosaic.updater.is_git_repository", return_value=False):
            with patch("face_mosaic.updater.check_for_updates", return_value=update_info):
                with patch("face_mosaic.updater.update_via_release_asset", return_value=(True, "Asset updated")) as mock_asset:
                    success, msg = perform_self_update()
                    assert success is True
                    assert msg == "Asset updated"
                    mock_asset.assert_called_once()


def test_cli_check_update_flag(capsys):
    from face_mosaic.cli import main
    with patch("sys.argv", ["face_mosaic", "--check-update"]):
        with patch("face_mosaic.updater.check_for_updates", return_value={"tag_name": "v0.9.0", "title": "New Version", "html_url": "http://example.com"}):
            main()
            captured = capsys.readouterr()
            assert "New version available: v0.9.0" in captured.out


def test_cli_self_update_flag(capsys):
    from face_mosaic.cli import main
    with patch("sys.argv", ["face_mosaic", "--update"]):
        with patch("face_mosaic.updater.perform_self_update", return_value=(True, "Update finished successfully")):
            main()
            captured = capsys.readouterr()
            assert "Update finished successfully" in captured.out

