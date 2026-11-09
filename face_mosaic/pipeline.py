"""
Processing pipeline orchestrator for auto-mosaic.
Performs 2-pass processing:
Pass 1: Detect and track faces across all frames to establish tracklets and temporal bounds.
Pass 2: Render blur/mosaic with forward/backward padding and pipe directly to ffmpeg encoder.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import time
import gc
import numpy as np
import cv2

from .config import AppConfig
from .io_utils import probe_video, VideoReader, VideoWriter, PrefetchedVideoReader, build_output_path
from .detector import SCRFDDetector
from .plate_detector import LicensePlateDetector
from .tracker import FaceTracker
from .blur import FaceBlurrer
from .color import compute_color_difference
from .checkpoint import VideoCheckpoint

class ProcessingPipeline:
    def __init__(self, config: AppConfig, model_dir: Optional[str] = None):
        self.config = config
        
        # Resolve model directory
        if model_dir:
            base_model_dir = Path(model_dir)
        else:
            base_model_dir = Path(__file__).resolve().parent.parent / "models"

        self.detector: Optional[SCRFDDetector] = None
        self.plate_detector: Optional[LicensePlateDetector] = None
        self.model_path = base_model_dir / config.model.name
        self.plate_model_path = base_model_dir / config.plate_model.name

        # Selective Model Loading: Only load face detector if target includes faces
        if self.config.detect_faces:
            if not self.model_path.exists():
                candidates = sorted(list(base_model_dir.glob("scrfd_*.onnx")))
                if candidates:
                    self.model_path = candidates[0]
                    print(f"Warning: {config.model.name} not found. Falling back to: {self.model_path.name}")
                else:
                    raise FileNotFoundError(
                        f"Face model file not found at {self.model_path}. "
                        f"Please run 'python scripts/download_models.py' first."
                    )

            print(f"Loading SCRFD Face Detector: {self.model_path.name}")
            self.detector = SCRFDDetector(
                model_path=str(self.model_path),
                conf_threshold=self.config.model.conf_threshold,
                nms_threshold=self.config.model.nms_threshold
            )

        # Selective Model Loading: Only load license plate detector if target includes plates
        if self.config.detect_plates:
            if not self.plate_model_path.exists():
                candidates = sorted(list(base_model_dir.glob("*plate*.onnx")))
                if candidates:
                    self.plate_model_path = candidates[0]
                    print(f"Warning: {config.plate_model.name} not found. Falling back to: {self.plate_model_path.name}")
                else:
                    # Attempt on-demand download of the license plate model
                    try:
                        from scripts.download_models import MODELS, download_file
                        p_name = config.plate_model.name
                        if p_name in MODELS:
                            info_m = MODELS[p_name]
                            print(f"Auto-downloading missing license plate model: {p_name}...")
                            download_file(
                                info_m["url"],
                                self.plate_model_path,
                                fallback_url=info_m.get("fallback_url"),
                                min_size=info_m.get("expected_min_size", 1_000_000)
                            )
                    except Exception as e:
                        print(f"Warning: Auto-download of license plate model failed: {e}")

            if not self.plate_model_path.exists():
                raise FileNotFoundError(
                    f"License plate model file not found at {self.plate_model_path}. "
                    f"Please run 'python scripts/download_models.py' first."
                )

            print(f"Loading License Plate Detector: {self.plate_model_path.name}")
            self.plate_detector = LicensePlateDetector(
                model_path=str(self.plate_model_path),
                conf_threshold=self.config.plate_model.conf_threshold,
                nms_threshold=self.config.plate_model.nms_threshold
            )

        self.blurrer = FaceBlurrer(
            blur_type=self.config.blur.type,
            strength=self.config.blur.strength,
            mosaic_block_size=self.config.blur.mosaic_block_size,
            margin_x=self.config.blur.margin_x,
            margin_y=self.config.blur.margin_y,
            shape=getattr(self.config.blur, "shape", "ellipse"),
            plate_margin_x=getattr(self.config.plate_model, "margin_x", 0.15),
            plate_margin_y=getattr(self.config.plate_model, "margin_y", 0.15),
            plate_shape=getattr(self.config.plate_model, "shape", "rect")
        )

    @property
    def active_provider(self) -> str:
        parts = []
        if self.config.detect_faces and self.detector is not None and hasattr(self.detector, "active_provider"):
            parts.append(f"Faces: {self.detector.active_provider}")
        if self.config.detect_plates and self.plate_detector is not None and hasattr(self.plate_detector, "active_provider"):
            parts.append(f"Plates: {self.plate_detector.active_provider}")
        return " | ".join(parts) if parts else "CPU"

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int, float], None]] = None,
        preview_callback: Optional[Callable[..., None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single video file.
        Supports resume via incremental checkpoints and caching.
        Returns a dict of metrics and results.
        """
        start_time = time.time()
        in_path = Path(input_path).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        # Determine output path
        if output_path is None:
            param_str = ""
            if getattr(self.config.output, "append_params_to_filename", False):
                t_abbr = "G" if self.config.blur.type == "gaussian" else "M"
                str_val = self.config.blur.strength if self.config.blur.type == "gaussian" else self.config.blur.mosaic_block_size
                m_val = int(self.config.blur.margin_x * 100)
                param_str = f"_({t_abbr}{str_val}-M{m_val}-CRF{self.config.output.crf})"
            suffix = f"{self.config.output.filename_suffix}{param_str}"
            out_path = build_output_path(
                input_path=in_path,
                output_dir=self.config.output.output_dir,
                filename_suffix=suffix,
            )
        else:
            out_path = Path(output_path).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)

        info = probe_video(str(in_path))
        needs_swap = abs(info.rotation) in (90, 270)
        out_width = info.height if needs_swap else info.width
        out_height = info.width if needs_swap else info.height

        tracker = FaceTracker(
            iou_threshold=self.config.tracking.iou_threshold,
            max_missing_frames=self.config.tracking.max_missing_frames,
            padding_backward=self.config.tracking.padding_backward,
            padding_forward=self.config.tracking.padding_forward,
            filters_config=self.config.filters,
            fps=info.fps,
            auto_scale_by_fps=getattr(self.config.tracking, "auto_scale_by_fps", True),
            base_fps=getattr(self.config.tracking, "base_fps", 30.0)
        )
        if tracker.auto_scale_by_fps and abs(info.fps - tracker.base_fps) > 2.0:
            print(
                f"[FPS-Adaptive Padding] {info.fps:.1f} fps detected: "
                f"Padding scaled to -{tracker.padding_backward}/+{tracker.padding_forward} frames "
                f"(approx {tracker.padding_forward / info.fps:.2f}s post-appearance retention)"
            )

        checkpoint = VideoCheckpoint.for_video(out_path)
        pass1_done = False
        cached_detections = {}

        # Check existing checkpoint
        if resume and checkpoint.matches_input(
            in_path,
            model_name=self.config.model.name,
            conf_threshold=self.config.model.conf_threshold,
            target_mode=self.config.target
        ):
            if checkpoint.is_pass1_completed():
                loaded_map = checkpoint.load_frame_bboxes_map()
                if loaded_map is not None:
                    print(f"[Resume] Found completed Pass 1 checkpoint for {in_path.name}. Skipping detection!")
                    frame_bboxes_map = loaded_map
                    total_frames = len(frame_bboxes_map)
                    pass1_done = True
                    if progress_callback:
                        progress_callback("Detection (Cached)", total_frames, total_frames, 0.0)

            if not pass1_done:
                cached_detections = checkpoint.load_cached_detections()
                if cached_detections:
                    last_c = max(cached_detections.keys())
                    print(f"[Resume] Resuming Pass 1 from frame {last_c + 1} (loaded {len(cached_detections)} cached frames).")
        else:
            if resume:
                checkpoint.init_state(
                    in_path,
                    info.total_frames,
                    info.width,
                    info.height,
                    info.fps,
                    model_name=self.config.model.name,
                    conf_threshold=self.config.model.conf_threshold,
                    target_mode=self.config.target
                )

        if not pass1_done:
            # ----------------- PASS 1: Detection & Tracking -----------------
            target_desc = {
                "faces": "faces",
                "plates": "license plates",
                "both": "faces & license plates"
            }.get(self.config.target, "targets")
            active_accel = self.active_provider
            print(f"\n[Pass 1/2] Detecting {target_desc} in {in_path.name} ({out_width}x{out_height}, {info.fps:.2f} fps) | Accelerator: {active_accel}...")
            
            # Optimal scale for Pass 1 detection (e.g. 640x360 for 4K 16:9) to maximize hardware throughput
            active_det = self.detector or self.plate_detector
            input_sz = getattr(active_det, "input_size", (640, 640))
            if isinstance(input_sz, (tuple, list)) and len(input_sz) == 2:
                det_w, det_h = int(input_sz[0]), int(input_sz[1])
            else:
                det_w, det_h = 640, 640

            im_ratio = float(out_height) / out_width
            model_ratio = float(det_h) / det_w
            if im_ratio > model_ratio:
                pass1_h = det_h
                pass1_w = int(pass1_h / im_ratio)
            else:
                pass1_w = det_w
                pass1_h = int(pass1_w * im_ratio)
            pass1_w = max(2, pass1_w - (pass1_w % 2))
            pass1_h = max(2, pass1_h - (pass1_h % 2))
            use_scale = (pass1_w, pass1_h) if (out_width > pass1_w or out_height > pass1_h) else None

            reader1 = PrefetchedVideoReader(str(in_path), scale=use_scale)
            frame_idx = 0
            pass1_start = time.time()
            last_preview_time = 0.0
            try:
                for frame in reader1:
                    if cancel_check and cancel_check():
                        if resume:
                            checkpoint.flush_detections(last_frame=frame_idx - 1)
                        return {
                            "input": str(in_path),
                            "status": "cancelled",
                            "cancelled": True,
                            "elapsed_seconds": time.time() - start_time
                        }

                    if frame_idx in cached_detections:
                        valid_dets = cached_detections[frame_idx]
                    else:
                        valid_dets = []
                        if self.config.detect_faces and self.detector is not None:
                            try:
                                face_dets = self.detector.detect(frame, orig_shape=(out_height, out_width))
                            except TypeError:
                                face_dets = self.detector.detect(frame)
                            valid_dets.extend([
                                d for d in face_dets
                                if min(d.width, d.height) >= self.config.model.min_face_size
                            ])

                        if self.config.detect_plates and self.plate_detector is not None:
                            try:
                                plate_dets = self.plate_detector.detect(frame, orig_shape=(out_height, out_width))
                            except TypeError:
                                plate_dets = self.plate_detector.detect(frame)
                            valid_dets.extend([
                                d for d in plate_dets
                                if min(d.width, d.height) >= self.config.plate_model.min_plate_size
                            ])

                        if resume:
                            checkpoint.append_detection(frame_idx, valid_dets)

                    tracker.update(frame_idx, frame, valid_dets, orig_shape=(out_height, out_width))

                    # Send preview frame in Pass 1 (throttled & memory-safe)
                    now = time.time()
                    if preview_callback and (frame_idx == 0 or now - last_preview_time >= 0.15):
                        last_preview_time = now
                        h, w = frame.shape[:2]
                        target_w = min(640, w)
                        target_h = max(1, int(h * (target_w / w)))
                        scale = target_w / out_width
                        if w == target_w and h == target_h:
                            disp_frame = frame.copy()
                        else:
                            disp_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        for d in valid_dets:
                            x1 = int(round(d.x1 * scale))
                            y1 = int(round(d.y1 * scale))
                            x2 = int(round(d.x2 * scale))
                            y2 = int(round(d.y2 * scale))
                            is_plate = getattr(d, "label", "face") == "plate"
                            box_color = (0, 165, 255) if is_plate else (0, 230, 118)
                            conf_text = f"LP {d.score:.2f}" if is_plate else f"{d.score:.2f}"
                            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(disp_frame, conf_text, (x1, max(15, y1 - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)
                        tot_str = str(info.total_frames) if info.total_frames else "?"
                        try:
                            preview_callback(disp_frame, f"Pass 1/2: Detection ({frame_idx + 1}/{tot_str})")
                        except TypeError:
                            preview_callback(disp_frame)

                    frame_idx += 1
                    if frame_idx % 1000 == 0:
                        gc.collect()

                    if progress_callback:
                        progress_callback("Detection", frame_idx, info.total_frames or frame_idx, time.time() - pass1_start)
            finally:
                reader1.close()

            total_frames = frame_idx
            print(f"Pass 1 complete. Total frames: {total_frames}. Establishing tracklets...")

            # Build blur bbox map for all frames (with padding and interpolation)
            frame_bboxes_map = tracker.get_frame_bboxes_map(total_frames, out_width, out_height)

            if resume:
                checkpoint.save_pass1_completed(frame_bboxes_map, total_frames)

        # ----------------- PASS 2: Blur Rendering & Encoding -----------------
        temp_out_path = out_path.with_name(f"{out_path.stem}.part{out_path.suffix}")
        if temp_out_path.exists():
            try:
                temp_out_path.unlink()
            except Exception:
                pass

        writer = VideoWriter(
            output_path=str(temp_out_path),
            input_info=info,
            width=out_width,
            height=out_height,
            fps=info.fps,
            codec=self.config.output.codec,
            crf=self.config.output.crf,
            preset=self.config.output.preset,
            preserve_color_tags=self.config.output.preserve_color_tags,
            bit_depth=self.config.output.bit_depth
        )

        enc_display = {
            "qsv": "Intel QSV Hardware Encoder (GPU)",
            "videotoolbox": "Apple VideoToolbox (Hardware)",
            "nvenc": "NVIDIA NVENC (GPU)",
            "amf": "AMD AMF (GPU)",
            "cpu": "CPU Software (Multi-threaded)"
        }.get(writer.active_encoder, writer.active_encoder.upper())
        print(f"[Pass 2/2] Rendering mosaic & encoding via {enc_display} to {out_path.name}...")

        reader2 = PrefetchedVideoReader(str(in_path))
        pass2_start = time.time()
        last_pass2_preview_time = 0.0

        mae_samples = []
        delta_e_samples = []

        try:
            for f_idx, frame in enumerate(reader2):
                if cancel_check and cancel_check():
                    if temp_out_path.exists():
                        try:
                            temp_out_path.unlink()
                        except Exception:
                            pass
                    return {
                        "input": str(in_path),
                        "status": "cancelled",
                        "cancelled": True,
                        "elapsed_seconds": time.time() - start_time
                    }

                bboxes = frame_bboxes_map.get(f_idx, [])
                blurred_frame, mask = self.blurrer.apply_blur(frame, bboxes)
                writer.write_frame(blurred_frame)

                # Send preview frame (throttled & downscaled)
                now = time.time()
                if preview_callback and (f_idx == 0 or now - last_pass2_preview_time >= 0.15):
                    last_pass2_preview_time = now
                    h, w = blurred_frame.shape[:2]
                    target_w = min(640, w)
                    target_h = max(1, int(h * (target_w / w)))
                    small_blurred = cv2.resize(blurred_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    try:
                        preview_callback(small_blurred, f"Pass 2/2: Mosaic ({f_idx + 1}/{total_frames})")
                    except TypeError:
                        preview_callback(small_blurred)

                # Sample color difference on 5% of frames
                if f_idx % max(1, total_frames // 20) == 0:
                    diff_metrics = compute_color_difference(frame, blurred_frame, mask)
                    mae_samples.append(diff_metrics["mae"])
                    delta_e_samples.append(diff_metrics["delta_e_approx"])

                if (f_idx + 1) % 1000 == 0:
                    gc.collect()

                if progress_callback:
                    progress_callback("Rendering", f_idx + 1, total_frames, time.time() - pass2_start)
        finally:
            reader2.close()
            writer.close()

        # Atomic move from temp_out_path to out_path
        if temp_out_path.exists():
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            temp_out_path.rename(out_path)

        # Processing succeeded completely - clean up checkpoint
        if resume:
            checkpoint.cleanup()

        total_elapsed = time.time() - start_time
        fps_proc = total_frames / total_elapsed if total_elapsed > 0 else 0.0

        avg_mae = float(np.mean(mae_samples)) if mae_samples else 0.0
        avg_delta_e = float(np.mean(delta_e_samples)) if delta_e_samples else 0.0

        result = {
            "input": str(in_path),
            "output": str(out_path),
            "total_frames": total_frames,
            "elapsed_seconds": total_elapsed,
            "processing_fps": fps_proc,
            "color_mae": avg_mae,
            "color_delta_e": avg_delta_e,
            "active_provider": self.active_provider,
            "active_encoder": writer.active_encoder,
            "status": "success"
        }

        print(f"Finished {in_path.name} -> {out_path.name} in {total_elapsed:.2f}s ({fps_proc:.2f} fps). Color MAE: {avg_mae:.4f}")
        return result
