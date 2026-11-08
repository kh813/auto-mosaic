"""
Folder watcher service for auto-mosaic.
Monitors a directory for newly added video files and automatically executes blurring pipeline.
"""

import time
import os
import sys
from pathlib import Path
from typing import Set, Optional, Callable

from face_mosaic.config import AppConfig
from face_mosaic.pipeline import ProcessingPipeline
from face_mosaic.cli import VIDEO_EXTENSIONS

class FolderWatcher:
    def __init__(
        self,
        watch_dir: str,
        config: Optional[AppConfig] = None,
        poll_interval_seconds: float = 3.0,
        on_event_callback: Optional[Callable[[str], None]] = None
    ):
        self.watch_dir = Path(watch_dir).resolve()
        if not self.watch_dir.exists():
            raise FileNotFoundError(f"Watch directory does not exist: {self.watch_dir}")

        self.config = config or AppConfig()
        self.poll_interval = poll_interval_seconds
        self.on_event = on_event_callback or print
        
        self.processed_files: Set[str] = set()
        self._running = False
        
        # Populate initial existing files so we only process new arrivals (or specify in config)
        self._init_existing()

    def _init_existing(self):
        for f in self.watch_dir.glob("**/*"):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                self.processed_files.add(str(f.resolve()))

    def _is_file_ready(self, file_path: Path) -> bool:
        """Check if file has finished writing (size is constant over 1 sec)."""
        try:
            size1 = file_path.stat().st_size
            if size1 == 0:
                return False
            time.sleep(1.0)
            size2 = file_path.stat().st_size
            return size1 == size2
        except Exception:
            return False

    def scan_and_process(self, pipeline: ProcessingPipeline):
        for file_path in self.watch_dir.glob("**/*"):
            if not file_path.is_file() or file_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            stem = file_path.stem
            if "_blurred" in stem:
                continue
            if self.config.output.filename_suffix and self.config.output.filename_suffix in stem:
                continue

            abs_str = str(file_path.resolve())
            if abs_str in self.processed_files:
                continue

            if not self._is_file_ready(file_path):
                continue

            self.on_event(f"\n[Folder Watcher] New video detected: {file_path.name}")
            try:
                res = pipeline.process_video(abs_str)
                self.on_event(f"[Folder Watcher] Finished processing: {file_path.name} -> {res['output']}")
                self.processed_files.add(abs_str)
            except Exception as e:
                self.on_event(f"[Folder Watcher] Error processing {file_path.name}: {e}")
                self.processed_files.add(abs_str)

    def start(self):
        self._running = True
        self.on_event(f"Folder watcher started on: {self.watch_dir} (Polling every {self.poll_interval}s)")
        pipeline = ProcessingPipeline(self.config)
        
        try:
            while self._running:
                self.scan_and_process(pipeline)
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self._running = False
        self.on_event("Folder watcher stopped.")
