"""
License Plate Detection Module using ONNX Runtime / OpenVINO.
Supports YOLOv8 / YOLOv11 license plate ONNX models (e.g., license-plate-finetune-v1n.onnx)
with multi-vendor hardware acceleration (Intel OpenVINO GPU/NPU, Apple Silicon CoreML,
Windows DirectML, NVIDIA CUDA, and multi-threaded CPU).
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import onnxruntime as ort

from .detector import FaceDetection, BaseFaceDetector


class LicensePlateDetector(BaseFaceDetector):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.40,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"License plate ONNX model file not found: {self.model_path}")

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # 1. Try Intel OpenVINO acceleration (Intel Arc GPU / Intel AI Boost NPU)
        self.ov_compiled = None
        self.session = None

        try:
            import openvino as ov
            core = ov.Core()
            devs = core.available_devices
            target_device = None
            # Direct GPU execution on Intel Arc / Iris Xe achieves ~4-6ms (160-250+ FPS).
            # Note: YOLO multi-scale architectures incur severe cross-device sync penalties on MULTI:GPU,NPU.
            # We prioritize GPU for peak throughput, then NPU, then CPU.
            if "GPU" in devs:
                target_device = "GPU"
            elif "NPU" in devs:
                target_device = "NPU"
            elif "CPU" in devs:
                target_device = "CPU"

            if target_device:
                ov_model = core.read_model(str(self.model_path))
                ov_model.reshape([1, 3, self.input_size[1], self.input_size[0]])
                self.ov_compiled = core.compile_model(ov_model, target_device)
                dev_label = "Intel Arc GPU" if target_device == "GPU" else ("Intel AI Boost NPU" if target_device == "NPU" else target_device)
                self.active_provider = f"OpenVINO ({dev_label})"
                self.output_names = [o.get_any_name() for o in ov_model.outputs]
                print(f"[Plate Detector Acceleration] Active: {self.active_provider}")
        except Exception:
            self.ov_compiled = None

        # 2. Fallback to ONNX Runtime (CoreML / DirectML / CUDA / OpenVINO / CPU)
        if self.ov_compiled is None:
            if providers is None:
                available = ort.get_available_providers()
                providers = []
                if "CoreMLExecutionProvider" in available:
                    providers.append("CoreMLExecutionProvider")
                if "DmlExecutionProvider" in available:
                    providers.append("DmlExecutionProvider")
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                if "OpenVINOExecutionProvider" in available:
                    providers.append("OpenVINOExecutionProvider")
                providers.append("CPUExecutionProvider")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            cpu_threads = min(os.cpu_count() or 4, 8)
            sess_options.intra_op_num_threads = cpu_threads

            self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
            active = self.session.get_providers()[0] if self.session.get_providers() else "Unknown"

            if "CoreML" in active:
                self.active_provider = "CoreML (Apple Silicon ANE & GPU)"
            elif "Dml" in active:
                self.active_provider = "DirectML (GPU Acceleration)"
            elif "CUDA" in active:
                self.active_provider = "CUDA (NVIDIA GPU)"
            elif "CPU" in active:
                self.active_provider = f"CPU Execution ({cpu_threads} threads, AVX/NEON)"
            else:
                self.active_provider = active

            print(f"[Plate Detector Acceleration] Active: {self.active_provider}")
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

    def _nms(self, bboxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        """Non-Maximum Suppression."""
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img: np.ndarray, orig_shape: Optional[Tuple[int, int]] = None) -> List[FaceDetection]:
        """
        Run license plate detection inference on image (BGR).
        Returns list of FaceDetection(label="plate") objects with coordinates scaled back to orig_shape.
        """
        if img is None or img.size == 0:
            return []

        h, w = img.shape[:2]
        orig_h, orig_w = orig_shape if orig_shape is not None else (h, w)
        input_w, input_h = self.input_size

        im_ratio = float(orig_h) / orig_w
        model_ratio = float(input_h) / input_w
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = max(1, int(new_h / im_ratio))
        else:
            new_w = input_w
            new_h = max(1, int(new_w * im_ratio))

        det_scale = float(new_h) / orig_h
        if h == new_h and w == new_w:
            det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
            det_img[:new_h, :new_w, :] = img
        else:
            resized_img = cv2.resize(img, (new_w, new_h))
            det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
            det_img[:new_h, :new_w, :] = resized_img

        # Normalize (BGR -> RGB, [0.0, 1.0])
        blob = (det_img[:, :, ::-1].astype(np.float32)) * (1.0 / 255.0)
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])

        if self.ov_compiled is not None:
            net_outs = list(self.ov_compiled([blob]).values())
        else:
            net_outs = self.session.run(self.output_names, {self.input_name: blob})

        raw = net_outs[0]
        if raw.ndim == 3:
            pred = raw[0]
        else:
            pred = raw

        # YOLOv8/v11 output format is (4 + num_classes, num_anchors), e.g., (5, 8400)
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 64:
            pred = pred.T  # -> (8400, 5)

        if pred.shape[1] < 5:
            return []

        cx = pred[:, 0]
        cy = pred[:, 1]
        bw = pred[:, 2]
        bh = pred[:, 3]
        if pred.shape[1] == 5:
            scores = pred[:, 4]
        else:
            scores = np.max(pred[:, 4:], axis=1)

        pos_inds = np.where(scores >= self.conf_threshold)[0]
        if len(pos_inds) == 0:
            return []

        cx = cx[pos_inds]
        cy = cy[pos_inds]
        bw = bw[pos_inds]
        bh = bh[pos_inds]
        scores = scores[pos_inds]

        x1 = (cx - 0.5 * bw) / det_scale
        y1 = (cy - 0.5 * bh) / det_scale
        x2 = (cx + 0.5 * bw) / det_scale
        y2 = (cy + 0.5 * bh) / det_scale

        x1 = np.clip(x1, 0.0, float(orig_w))
        y1 = np.clip(y1, 0.0, float(orig_h))
        x2 = np.clip(x2, 0.0, float(orig_w))
        y2 = np.clip(y2, 0.0, float(orig_h))

        bboxes = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)

        keep = self._nms(bboxes, scores, self.nms_threshold)
        results = []
        for i in keep:
            b = bboxes[i]
            bw_real = b[2] - b[0]
            bh_real = b[3] - b[1]
            if bw_real <= 0 or bh_real <= 0:
                continue

            ar = bw_real / bh_real
            # Japanese standard license plates have aspect ratio ~2.0 (330x165mm, 440x220mm).
            # With perspective tilt, valid range is [1.25, 2.70].
            # Reject station signboards/banners (ar >= 2.75 or ar < 1.20)
            if not (1.25 <= ar <= 2.70):
                continue

            # 1. Vertical position check: Vehicle license plates are strictly located in the lower half of the frame (road level).
            # Reject any candidate whose center is located in the upper 50% (station signs, platform boards, billboards, street banners).
            cy = (b[1] + b[3]) * 0.5
            if cy < (orig_h * 0.50):
                continue

            # 2. Maximum height check: license plates on vehicles never occupy > 20% of total video height.
            if bh_real > (orig_h * 0.20):
                continue

            results.append(FaceDetection(
                bbox=b,
                score=float(scores[i]),
                landmarks=None,
                label="plate"
            ))

        return results
