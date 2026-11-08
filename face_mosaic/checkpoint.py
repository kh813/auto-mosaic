"""
Checkpoint and resume manager for auto-mosaic pipeline.
Provides resilient processing: saves Pass 1 face detection progress incrementally,
caches completed tracklet/bounding box maps, and resumes interrupted tasks.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import shutil
import numpy as np

from .detector import FaceDetection

class VideoCheckpoint:
    def __init__(self, checkpoint_dir: Path):
        self.dir = Path(checkpoint_dir).resolve()
        self.state_file = self.dir / "state.json"
        self.detections_file = self.dir / "pass1_detections.jsonl"
        self.bboxes_file = self.dir / "pass1_frame_bboxes.json"
        self._det_buffer: List[str] = []

    @classmethod
    def for_video(cls, output_path: Path) -> "VideoCheckpoint":
        """
        Creates a checkpoint manager directory based on the output video path.
        Example: output.mp4 -> .output.mp4.checkpoint/
        """
        out_p = Path(output_path).resolve()
        chk_dir = out_p.parent / f".{out_p.name}.checkpoint"
        return cls(chk_dir)

    def exists(self) -> bool:
        return self.state_file.exists()

    def matches_input(
        self,
        input_path: Path,
        model_name: str = "",
        conf_threshold: float = 0.5,
        target_mode: Optional[str] = None
    ) -> bool:
        """Check if existing checkpoint matches the current input video and model settings."""
        if not self.exists():
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            stat = input_path.stat()
            target_ok = True
            if target_mode is not None:
                target_ok = state.get("target_mode", "faces") == target_mode
            return (
                state.get("input_path") == str(input_path.resolve())
                and state.get("input_size") == stat.st_size
                and abs(state.get("input_mtime", 0.0) - stat.st_mtime) < 1.0
                and state.get("model_name", "") == model_name
                and abs(state.get("conf_threshold", 0.5) - conf_threshold) < 0.01
                and target_ok
            )
        except Exception:
            return False

    def init_state(
        self,
        input_path: Path,
        total_frames: int,
        width: int,
        height: int,
        fps: float,
        model_name: str = "",
        conf_threshold: float = 0.5,
        target_mode: str = "faces"
    ):
        """Initialize or reset state file for a new processing run."""
        self.dir.mkdir(parents=True, exist_ok=True)
        stat = input_path.stat()
        state = {
            "input_path": str(input_path.resolve()),
            "input_size": stat.st_size,
            "input_mtime": stat.st_mtime,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "fps": fps,
            "model_name": model_name,
            "conf_threshold": conf_threshold,
            "target_mode": target_mode,
            "pass1_completed": False,
            "last_detected_frame": -1
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def is_pass1_completed(self) -> bool:
        """Check if Pass 1 detection has already completed and bounding boxes are saved."""
        if not self.exists() or not self.bboxes_file.exists():
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            return bool(state.get("pass1_completed", False))
        except Exception:
            return False

    def load_frame_bboxes_map(self) -> Optional[Dict[int, List[np.ndarray]]]:
        """Load final frame-to-bboxes map from JSON file."""
        if not self.bboxes_file.exists():
            return None
        try:
            with open(self.bboxes_file, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
            # Convert JSON data back to dict of int -> list of np.ndarray
            result: Dict[int, List[np.ndarray]] = {}
            for k, bboxes in raw_map.items():
                f_idx = int(k)
                result[f_idx] = [np.array(b, dtype=np.float32) for b in bboxes]
            return result
        except Exception:
            return None

    def save_pass1_completed(self, frame_bboxes_map: Dict[int, List[np.ndarray]], total_frames: int):
        """Save final Pass 1 frame bounding boxes map and mark Pass 1 as completed."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_detections()

        # Convert numpy arrays to float lists for json serialization
        serializable_map = {}
        for f_idx, bboxes in frame_bboxes_map.items():
            if bboxes:
                serializable_map[f_idx] = [
                    [round(float(c), 1) for c in (b.tolist() if hasattr(b, "tolist") else b)]
                    for b in bboxes
                ]
            else:
                serializable_map[f_idx] = []

        with open(self.bboxes_file, "w", encoding="utf-8") as f:
            json.dump(serializable_map, f)

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}
        state["pass1_completed"] = True
        state["last_detected_frame"] = total_frames - 1
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def append_detection(self, frame_idx: int, detections: List[FaceDetection]):
        """Buffers detected faces/plates for frame_idx and flushes periodically."""
        dets_data = []
        for d in detections:
            bbox_coords = [round(float(c), 1) for c in d.bbox]
            entry_d = {
                "bbox": bbox_coords,
                "score": round(float(d.score), 3)
            }
            lbl = getattr(d, "label", "face")
            if lbl != "face":
                entry_d["label"] = lbl
            dets_data.append(entry_d)
        entry = json.dumps({"f": frame_idx, "d": dets_data})
        self._det_buffer.append(entry)
        if len(self._det_buffer) >= 100:
            self.flush_detections(last_frame=frame_idx)

    def flush_detections(self, last_frame: Optional[int] = None):
        """Flush in-memory detection buffer to JSONL file."""
        if not self._det_buffer:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.detections_file, "a", encoding="utf-8") as f:
            for line in self._det_buffer:
                f.write(line + "\n")
        self._det_buffer.clear()

        if last_frame is not None:
            try:
                if self.state_file.exists():
                    with open(self.state_file, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    state["last_detected_frame"] = last_frame
                    with open(self.state_file, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2)
            except Exception:
                pass

    def load_cached_detections(self) -> Dict[int, List[FaceDetection]]:
        """Load already processed frame detections if resuming Pass 1."""
        if not self.detections_file.exists():
            return {}
        cached: Dict[int, List[FaceDetection]] = {}
        try:
            with open(self.detections_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    f_idx = int(obj["f"])
                    dets = [
                        FaceDetection(
                            bbox=np.array(d["bbox"], dtype=np.float32),
                            score=float(d["score"]),
                            label=d.get("label", "face")
                        )
                        for d in obj.get("d", [])
                    ]
                    cached[f_idx] = dets
        except Exception:
            pass
        return cached

    def cleanup(self):
        """Safely delete checkpoint directory when processing finishes successfully."""
        try:
            if self.dir.exists():
                shutil.rmtree(self.dir, ignore_errors=True)
        except Exception:
            pass
