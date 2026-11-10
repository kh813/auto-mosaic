"""
Vehicle Detection Module for auto-mosaic.
Provides vehicle validation (car, motorcycle, bus, truck) to strictly filter
out false-positive license plate detections (such as railway station signs,
platform boards, wall posters, and shop banners).
Supports hardware acceleration via OpenVINO (Intel Arc GPU/NPU) and ONNX Runtime.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import onnxruntime as ort

from .detector import FaceDetection

# COCO 80 classes relevant for license plates
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


@dataclass
class VehicleDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    cls_id: int
    label: str

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class VehicleDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Vehicle detection ONNX model file not found: {self.model_path}")

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        self.ov_compiled = None
        self.session = None

        # 1. Try Intel OpenVINO acceleration (Intel Arc GPU / Intel AI Boost NPU)
        try:
            import openvino as ov
            core = ov.Core()
            devs = core.available_devices
            target_device = None
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
                print(f"[Vehicle Detector Acceleration] Active: {self.active_provider}")
        except Exception:
            self.ov_compiled = None

        # 2. Fallback to ONNX Runtime
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
            else:
                self.active_provider = f"CPU Execution ({cpu_threads} threads)"

            print(f"[Vehicle Detector Acceleration] Active: {self.active_provider}")
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

    def _nms(self, bboxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
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

    def detect(self, img: np.ndarray, orig_shape: Optional[Tuple[int, int]] = None) -> List[VehicleDetection]:
        """
        Detect vehicles (cars, motorcycles, buses, trucks) in frame.
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

        blob = (det_img[:, :, ::-1].astype(np.float32)) * (1.0 / 255.0)
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])

        if self.ov_compiled is not None:
            net_outs = list(self.ov_compiled([blob]).values())
        else:
            net_outs = self.session.run(self.output_names, {self.input_name: blob})

        raw = net_outs[0]
        pred = raw[0] if raw.ndim == 3 else raw

        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 84:
            pred = pred.T  # e.g., (8400, 84)

        if pred.shape[1] < 84:
            return []

        cx = pred[:, 0]
        cy = pred[:, 1]
        bw = pred[:, 2]
        bh = pred[:, 3]

        # Vehicle classes submatrix (car: 2, motorcycle: 3, bus: 5, truck: 7)
        v_class_ids = np.array(list(VEHICLE_CLASSES.keys()), dtype=int)
        v_class_cols = v_class_ids + 4
        v_scores_matrix = pred[:, v_class_cols]

        best_v_idx = np.argmax(v_scores_matrix, axis=1)
        best_scores = np.max(v_scores_matrix, axis=1)
        assigned_cls_ids = v_class_ids[best_v_idx]

        pos_inds = np.where(best_scores >= self.conf_threshold)[0]
        if len(pos_inds) == 0:
            return []

        cx = cx[pos_inds]
        cy = cy[pos_inds]
        bw = bw[pos_inds]
        bh = bh[pos_inds]
        scores = best_scores[pos_inds]
        cls_ids = assigned_cls_ids[pos_inds]

        x1 = np.clip((cx - 0.5 * bw) / det_scale, 0.0, float(orig_w))
        y1 = np.clip((cy - 0.5 * bh) / det_scale, 0.0, float(orig_h))
        x2 = np.clip((cx + 0.5 * bw) / det_scale, 0.0, float(orig_w))
        y2 = np.clip((cy + 0.5 * bh) / det_scale, 0.0, float(orig_h))

        bboxes = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
        keep = self._nms(bboxes, scores, self.nms_threshold)

        results = []
        for i in keep:
            b = bboxes[i]
            # Vehicles must at least reach the lower half (y2 >= orig_h * 0.45)
            # to possibly contain a visible road license plate
            if b[3] < (orig_h * 0.45):
                continue
            cid = int(cls_ids[i])
            results.append(VehicleDetection(
                bbox=b,
                score=float(scores[i]),
                cls_id=cid,
                label=VEHICLE_CLASSES.get(cid, "vehicle")
            ))

        return results

    @staticmethod
    def is_plate_on_vehicle(
        plate_bbox: np.ndarray,
        vehicle_detections: List[VehicleDetection],
        orig_h: int
    ) -> bool:
        """
        Validate that the candidate license plate:
        1. Lies strictly in the lower half of the frame (pcy >= orig_h * 0.50).
        2. Is spatially located on/within a detected vehicle.
        """
        pcy = float((plate_bbox[1] + plate_bbox[3]) * 0.5)
        # 1. Strict vertical constraint: license plates strictly appear in lower half of screen
        if pcy < (orig_h * 0.50):
            return False

        if not vehicle_detections:
            return False

        pcx = float((plate_bbox[0] + plate_bbox[2]) * 0.5)
        px1, py1, px2, py2 = float(plate_bbox[0]), float(plate_bbox[1]), float(plate_bbox[2]), float(plate_bbox[3])
        p_area = max(1.0, (px2 - px1) * (py2 - py1))

        for v in vehicle_detections:
            vx1, vy1, vx2, vy2 = float(v.bbox[0]), float(v.bbox[1]), float(v.bbox[2]), float(v.bbox[3])
            vw = vx2 - vx1
            vh = vy2 - vy1

            # Expand vehicle box slightly to account for bumper overhang and plate mounts
            evx1 = vx1 - vw * 0.08
            evx2 = vx2 + vw * 0.08
            evy1 = vy1 - vh * 0.05
            evy2 = vy2 + vh * 0.15

            # Condition A: Center point inside expanded vehicle box
            if evx1 <= pcx <= evx2 and evy1 <= pcy <= evy2:
                return True

            # Condition B: High spatial overlap with vehicle box
            ix1 = max(px1, evx1)
            iy1 = max(py1, evy1)
            ix2 = min(px2, evx2)
            iy2 = min(py2, evy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if (inter / p_area) >= 0.35:
                return True

        return False
