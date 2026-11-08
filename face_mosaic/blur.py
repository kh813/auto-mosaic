"""
Face and license plate blur and mosaic rendering module for auto-mosaic.
Implements Gaussian blur and Mosaic pixelation with margin expansion
and round (elliptical/circular) or rectangular shapes.
"""

from typing import List, Tuple
import numpy as np
import cv2

class FaceBlurrer:
    def __init__(
        self,
        blur_type: str = "gaussian",
        strength: int = 51,
        mosaic_block_size: int = 28,
        margin_x: float = 0.50,
        margin_y: float = 0.50,
        shape: str = "ellipse",  # "ellipse" (round), "circle", or "rect"
        plate_margin_x: float = 0.15,
        plate_margin_y: float = 0.15,
        plate_shape: str = "rect"
    ):
        self.blur_type = blur_type.lower()
        self.strength = strength if strength % 2 == 1 else strength + 1  # ensure odd
        self.mosaic_block_size = max(2, mosaic_block_size)
        self.margin_x = max(0.0, margin_x)
        self.margin_y = max(0.0, margin_y)
        self.shape = shape.lower()
        self.plate_margin_x = max(0.0, plate_margin_x)
        self.plate_margin_y = max(0.0, plate_margin_y)
        self.plate_shape = plate_shape.lower()

    def expand_bbox(
        self,
        bbox: np.ndarray,
        img_w: int,
        img_h: int,
        margin_x: float = None,
        margin_y: float = None
    ) -> Tuple[int, int, int, int]:
        """
        Expand bounding box [x1, y1, x2, y2] by margin ratios and clip to image boundaries.
        """
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1

        mx = self.margin_x if margin_x is None else margin_x
        my = self.margin_y if margin_y is None else margin_y

        dx = w * mx
        dy = h * my

        ex1 = int(max(0, np.floor(x1 - dx)))
        ey1 = int(max(0, np.floor(y1 - dy)))
        ex2 = int(min(img_w, np.ceil(x2 + dx)))
        ey2 = int(min(img_h, np.ceil(y2 + dy)))

        return ex1, ey1, ex2, ey2

    def apply_mosaic_to_roi(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h <= 0 or w <= 0:
            return roi
        
        # Downscale then upscale with nearest neighbor
        small_w = max(1, w // self.mosaic_block_size)
        small_h = max(1, h // self.mosaic_block_size)
        
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return mosaic

    def apply_gaussian_to_roi(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h <= 0 or w <= 0:
            return roi
        
        ksize = min(self.strength, min(h, w))
        if ksize % 2 == 0:
            ksize -= 1
        if ksize < 3:
            ksize = 3
            
        return cv2.GaussianBlur(roi, (ksize, ksize), 0)

    def apply_blur(self, frame: np.ndarray, bboxes: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply blur/mosaic to frame at specified bboxes using round (ellipse) or rectangular shape.
        Returns:
            (processed_frame, mask)
            where mask is a uint8 binary image (255 where blurred, 0 elsewhere).
        """
        out_frame = frame.copy()
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for bbox in bboxes:
            is_plate = len(bbox) >= 5 and float(bbox[4]) == 1.0
            if is_plate:
                x1, y1, x2, y2 = self.expand_bbox(bbox, w, h, margin_x=self.plate_margin_x, margin_y=self.plate_margin_y)
                active_shape = self.plate_shape
            else:
                x1, y1, x2, y2 = self.expand_bbox(bbox, w, h)
                active_shape = self.shape

            if x2 <= x1 or y2 <= y1:
                continue

            roi = out_frame[y1:y2, x1:x2]
            rh, rw = roi.shape[:2]
            if rh <= 0 or rw <= 0:
                continue

            if self.blur_type == "mosaic":
                blurred_roi = self.apply_mosaic_to_roi(roi)
            else:
                blurred_roi = self.apply_gaussian_to_roi(roi)

            if active_shape in ("ellipse", "round", "circle"):
                # Generate smooth anti-aliased elliptical/circular mask
                ellipse_mask = np.zeros((rh, rw), dtype=np.uint8)
                center = (rw // 2, rh // 2)
                axes = (rw // 2, rh // 2)
                cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)

                # Smooth alpha blending along the boundary
                alpha = (ellipse_mask.astype(np.float32) / 255.0)[:, :, None]
                out_frame[y1:y2, x1:x2] = (blurred_roi.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], ellipse_mask)
            else:
                out_frame[y1:y2, x1:x2] = blurred_roi
                mask[y1:y2, x1:x2] = 255

        return out_frame, mask
