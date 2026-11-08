"""
Self-update module for auto-mosaic.
Supports checking GitHub Releases, updating via Git or downloading release assets,
and platform-specific replacement.
"""

import sys
import os
import re
import json
import shutil
import tempfile
import tarfile
import zipfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable
import urllib.request
import urllib.error

from face_mosaic import __version__

GITHUB_REPO = "kh813/auto-mosaic"
GITHUB_API_LATEST_RELEASE = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

def parse_version(v_str: str) -> Tuple[int, ...]:
    """
    Parse a version string (e.g. 'v0.2.0', '0.1.2') into a tuple of ints.
    Returns (0, 0, 0) on parse failure.
    """
    cleaned = v_str.strip().lstrip("vV")
    # Extract only numbers from beginning, handling -rc1 or +meta
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", cleaned)
    if not match:
        return (0, 0, 0)
    groups = match.groups()
    return tuple(int(g) if g is not None else 0 for g in groups)

def is_newer_version(latest_tag: str, current_ver: str) -> bool:
    """Compare two version strings."""
    return parse_version(latest_tag) > parse_version(current_ver)

def get_project_root() -> Path:
    """Return the absolute path to the auto-mosaic project root."""
    return Path(__file__).resolve().parent.parent

def is_git_repository(project_root: Optional[Path] = None) -> bool:
    """Check if the project is run inside a git repository."""
    root = project_root or get_project_root()
    git_dir = root / ".git"
    return git_dir.exists()

def check_for_updates(current_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Query GitHub API for the latest release.
    Returns a dict with release details if a newer version is available, else None.
    """
    cur_ver = current_version or __version__
    req = urllib.request.Request(
        GITHUB_API_LATEST_RELEASE,
        headers={"User-Agent": f"auto-mosaic-updater/{cur_ver}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status != 200:
                return None
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        # Network unreachable, GitHub rate limit, or offline - fail quietly
        return None

    tag_name = data.get("tag_name", "")
    if not tag_name:
        return None

    if is_newer_version(tag_name, cur_ver):
        # Find matching asset for current platform
        if sys.platform == "win32":
            target_asset_names = ["auto-mosaic-windows.zip"]
        else:
            target_asset_names = ["auto-mosaic-macos.zip", "auto-mosaic-macos.tar.gz"]

        asset_url = None
        for asset in data.get("assets", []):
            if asset.get("name") in target_asset_names:
                asset_url = asset.get("browser_download_url")
                break

        return {
            "version": tag_name.lstrip("vV"),
            "tag_name": tag_name,
            "title": data.get("name") or tag_name,
            "body": data.get("body", ""),
            "html_url": data.get("html_url", ""),
            "asset_url": asset_url,
            "published_at": data.get("published_at", ""),
            "is_newer": True
        }

    return None

def update_via_git(project_root: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Perform update by running 'git pull' and updating pip requirements.
    """
    root = project_root or get_project_root()
    try:
        # 1. git pull
        pull_res = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False
        )
        if pull_res.returncode != 0:
            # Try plain git pull if ff-only fails
            pull_res = subprocess.run(
                ["git", "pull"],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False
            )
            if pull_res.returncode != 0:
                return False, f"Git pull failed: {pull_res.stderr.strip() or pull_res.stdout.strip()}"

        # 2. Update python requirements if venv exists
        venv_python = (
            root / "venv" / "Scripts" / "python.exe"
            if sys.platform == "win32"
            else root / "venv" / "bin" / "python"
        )
        if venv_python.exists():
            req_file = (
                root / "requirements" / "requirements-windows.txt"
                if sys.platform == "win32"
                else root / "requirements" / "requirements-mac.txt"
            )
            if req_file.exists():
                subprocess.run(
                    [str(venv_python), "-m", "pip", "install", "-r", str(req_file)],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    check=False
                )

        return True, "Successfully updated from Git repository."
    except Exception as e:
        return False, f"Update via Git encountered an error: {e}"

def update_via_release_asset(
    asset_url: str,
    target_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[bool, str]:
    """
    Download and extract release archive over target directory.
    """
    dest = target_dir or get_project_root()
    temp_dir = Path(tempfile.mkdtemp(prefix="face_mosaic_update_"))
    try:
        archive_name = "update.zip" if asset_url.endswith(".zip") else "update.tar.gz"
        download_path = temp_dir / archive_name

        # Download with chunked progress
        req = urllib.request.Request(asset_url, headers={"User-Agent": "auto-mosaic-updater"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(download_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size > 0:
                        progress_callback(downloaded, total_size)

        # Extract
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        if download_path.name.endswith(".zip"):
            with zipfile.ZipFile(download_path, "r") as z:
                z.extractall(extract_dir)
        else:
            with tarfile.open(download_path, "r:gz") as t:
                t.extractall(extract_dir)

        # Find the extracted root (sometimes archives wrap in a subfolder)
        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        source_content_dir = subdirs[0] if (len(subdirs) == 1 and len(list(extract_dir.iterdir())) == 1) else extract_dir

        # Copy extracted files over destination
        for item in source_content_dir.iterdir():
            target_item = dest / item.name
            if item.is_dir():
                if target_item.exists():
                    shutil.copytree(item, target_item, dirs_exist_ok=True)
                else:
                    shutil.copytree(item, target_item)
            else:
                shutil.copy2(item, target_item)

        # Ensure POSIX execute permissions on launch scripts on macOS/Linux
        if sys.platform != "win32":
            scripts_dir = dest / "scripts"
            executable_candidates = [
                dest / "start-app.command",
                *(list(scripts_dir.glob("*.sh")) if scripts_dir.exists() else [])
            ]
            for cand in executable_candidates:
                if cand.exists():
                    try:
                        cand.chmod(cand.stat().st_mode | 0o755)
                    except Exception:
                        pass

        return True, "Successfully installed latest release package."

    except Exception as e:
        return False, f"Release asset update failed: {e}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def perform_self_update(
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Tuple[bool, str]:
    """
    Smart self-update orchestrator:
    - If running in Git repo, updates via 'git pull'.
    - If running standalone, checks latest GitHub Release asset and updates files.
    """
    root = get_project_root()
    if is_git_repository(root):
        if progress_callback:
            progress_callback("Pulling latest code from Git...", 50, 100)
        return update_via_git(root)

    # Standalone mode: fetch latest GitHub Release
    if progress_callback:
        progress_callback("Checking GitHub for latest release...", 10, 100)
    info = check_for_updates()
    if not info:
        return True, "Already running the latest version."

    asset_url = info.get("asset_url")
    if not asset_url:
        return False, f"Release {info['tag_name']} found, but no matching binary asset available for this platform."

    def on_dl_progress(current: int, total: int):
        if progress_callback:
            pct = int((current / total) * 80) if total > 0 else 50
            progress_callback(f"Downloading update ({current // 1024} KB / {total // 1024} KB)...", 10 + pct, 100)

    return update_via_release_asset(asset_url, root, on_dl_progress)
