"""
SCRFD Face Detection Module using ONNX Runtime.
Supports arbitrary input sizes (or scaled 640x640), 2.5G/10G/34G models,
and returns bboxes, confidence scores, and 5-point facial landmarks.
"""

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import onnxruntime as ort

class FaceDetection:
    def __init__(
        self,
        bbox: np.ndarray,
        score: float,
        landmarks: Optional[np.ndarray] = None,
        label: str = "face"
    ):
        """
        bbox: [x1, y1, x2, y2]
        score: float
        landmarks: 5x2 array [[x, y], ...]
        label: "face" or "plate"
        """
        self.bbox = np.array(bbox, dtype=np.float32)
        self.score = float(score)
        self.landmarks = np.array(landmarks, dtype=np.float32) if landmarks is not None else None
        self.label = str(label)

    @property
    def x1(self) -> float:
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        return float(self.bbox[3])

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        pass

class SCRFDDetector(BaseFaceDetector):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"SCRFD model file not found: {self.model_path}")

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
            if "GPU" in devs and "NPU" in devs:
                target_device = "MULTI:GPU,NPU"
            elif "GPU" in devs:
                target_device = "GPU"
            elif "NPU" in devs:
                target_device = "NPU"

            if target_device:
                ov_model = core.read_model(str(self.model_path))
                ov_model.reshape([1, 3, self.input_size[1], self.input_size[0]])
                self.ov_compiled = core.compile_model(ov_model, target_device)
                self.active_provider = f"OpenVINO ({target_device})"
                self.output_names = [o.get_any_name() for o in ov_model.outputs]
                print(f"[Hardware Acceleration] Active: {self.active_provider}")
        except Exception:
            self.ov_compiled = None

        # 2. Fallback to ONNX Runtime (CoreML / DirectML / CUDA / OpenVINO / CPU)
        if self.ov_compiled is None:
            if providers is None:
                available = ort.get_available_providers()
                providers = []
                # Apple Silicon: CoreML is primary (Neural Engine & GPU)
                if "CoreMLExecutionProvider" in available:
                    providers.append("CoreMLExecutionProvider")
                # DirectML (AMD Radeon / Intel / NVIDIA on Windows)
                if "DmlExecutionProvider" in available:
                    providers.append("DmlExecutionProvider")
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                if "OpenVINOExecutionProvider" in available:
                    providers.append("OpenVINOExecutionProvider")
                providers.append("CPUExecutionProvider")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Multi-core CPU tuning (Ryzen AVX2/AVX-512, Intel, Apple NEON)
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

            print(f"[Hardware Acceleration] Active: {self.active_provider}")
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
        
        # SCRFD anchor strides
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = len(self.output_names) >= 9

    def _distance2bbox(self, points: np.ndarray, distance: np.ndarray, max_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Decode distance [l, t, r, b] to [x1, y1, x2, y2]."""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points: np.ndarray, distance: np.ndarray, max_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Decode distance [dx0, dy0, dx1, dy1, ...] to landmarks [[x0, y0], ...]."""
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = np.clip(px, 0, max_shape[1])
                py = np.clip(py, 0, max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def _nms(self, bboxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        """Non-Maximum Suppression."""
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
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
        Run inference on image (BGR).
        Returns list of FaceDetection objects with coordinates scaled back to original image size.
        """
        h, w = img.shape[:2]
        orig_h, orig_w = orig_shape if orig_shape is not None else (h, w)
        input_w, input_h = self.input_size

        # Preprocessing: resize preserving aspect ratio or direct resize
        im_ratio = float(orig_h) / orig_w
        model_ratio = float(input_h) / input_w
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_w
            new_h = int(new_w * im_ratio)

        det_scale = float(new_h) / orig_h
        if h == new_h and w == new_w:
            det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
            det_img[:new_h, :new_w, :] = img
        else:
            resized_img = cv2.resize(img, (new_w, new_h))
            det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
            det_img[:new_h, :new_w, :] = resized_img

        # Normalize (BGR -> RGB, mean 127.5, std 128.0)
        blob = (det_img[:, :, ::-1].astype(np.float32) - 127.5) * (1.0 / 128.0)
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])

        if self.ov_compiled is not None:
            net_outs = list(self.ov_compiled([blob]).values())
        else:
            net_outs = self.session.run(self.output_names, {self.input_name: blob})

        # Separate outputs
        scores_list = []
        bboxes_list = []
        kpss_list = []

        # Outputs are ordered by stride: 8, 16, 32
        # Format: [score8, score16, score32, bbox8, bbox16, bbox32, kps8, kps16, kps32]
        num_strides = len(self._feat_stride_fpn)
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + num_strides] * stride
            if self.use_kps:
                kps_preds = net_outs[idx + num_strides * 2] * stride

            height = input_h // stride
            width = input_w // stride
            
            # Generate anchor points
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            # Filter by confidence threshold
            pos_inds = np.where(scores >= self.conf_threshold)[0]
            if len(pos_inds) > 0:
                bboxes = self._distance2bbox(anchor_centers[pos_inds], bbox_preds[pos_inds])
                scores_list.append(scores[pos_inds])
                bboxes_list.append(bboxes)
                if self.use_kps:
                    kpss = self._distance2kps(anchor_centers[pos_inds], kps_preds[pos_inds])
                    kpss_list.append(kpss)

        if not scores_list:
            return []

        scores = np.vstack(scores_list).flatten()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list) if self.use_kps and kpss_list else None

        # Scale bboxes and kps back to original image
        bboxes[:, 0::2] = bboxes[:, 0::2] / det_scale
        bboxes[:, 1::2] = bboxes[:, 1::2] / det_scale
        if kpss is not None:
            kpss[:, 0::2] = kpss[:, 0::2] / det_scale
            kpss[:, 1::2] = kpss[:, 1::2] / det_scale

        # Apply NMS
        keep = self._nms(bboxes, scores, self.nms_threshold)

        results = []
        for i in keep:
            kps_i = kpss[i].reshape(5, 2) if kpss is not None else None
            results.append(FaceDetection(
                bbox=bboxes[i],
                score=scores[i],
                landmarks=kps_i
            ))

        return results
