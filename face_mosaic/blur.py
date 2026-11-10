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
        plate_shape: str = "rounded_rect"  # "rounded_rect" (default: corner-rounded with soft edges), "ellipse", or "rect"
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
            
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)

        # Prevent darkening: preserve central face/plate natural luminance.
        # Strong blurs cause dark hair or dark surroundings to bleed into the face,
        # creating an unnatural blackish/muddy patch. We maintain central brightness.
        ch_y1, ch_y2 = int(h * 0.20), int(h * 0.80)
        ch_x1, ch_x2 = int(w * 0.20), int(w * 0.80)
        if ch_y2 > ch_y1 and ch_x2 > ch_x1:
            orig_center = roi[ch_y1:ch_y2, ch_x1:ch_x2].astype(np.float32)
            blur_center = blurred[ch_y1:ch_y2, ch_x1:ch_x2].astype(np.float32)
            orig_mean = np.mean(orig_center, axis=(0, 1))
            blur_mean = np.mean(blur_center, axis=(0, 1))

            # If blurred center is darker than original, lift brightness to preserve original color tone
            diff = orig_mean - blur_mean
            if np.any(diff > 3.0):
                lift = np.maximum(0.0, diff * 0.85)
                lifted = np.clip(blurred.astype(np.float32) + lift, 0.0, 255.0).astype(np.uint8)
                return lifted

        return blurred

    def generate_feathered_mask(self, rh: int, rw: int, shape: str) -> np.ndarray:
        """
        Generate smooth, soft-feathered gradient alpha mask (0.0 to 1.0).
        Blends gently into the background to eliminate harsh, conspicuous boundaries.
        Supports:
        - "ellipse" / "round" / "circle": soft-feathered ellipse
        - "rounded_rect" / "round_rect" / "rounded_rectangle": rectangle with rounded corners and softened edges
        - other shapes: soft feathered rectangle
        """
        if rh <= 0 or rw <= 0:
            return np.zeros((max(1, rh), max(1, rw)), dtype=np.float32)

        if shape in ("ellipse", "round", "circle"):
            # Solid core ellipse surrounded by gentle gradient feather
            ellipse_u8 = np.zeros((rh, rw), dtype=np.uint8)
            center = (rw // 2, rh // 2)
            axes = (max(1, int(rw * 0.44)), max(1, int(rh * 0.44)))
            cv2.ellipse(ellipse_u8, center, axes, 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)

            # Soft feathering via Gaussian blur
            feather_ksize = max(5, int(min(rh, rw) * 0.22))
            if feather_ksize % 2 == 0:
                feather_ksize += 1
            feathered = cv2.GaussianBlur(ellipse_u8.astype(np.float32) / 255.0, (feather_ksize, feather_ksize), 0)
            return np.clip(feathered * 1.20, 0.0, 1.0)

        elif shape in ("rounded_rect", "round_rect", "rounded_rectangle"):
            # Rounded rectangle (角を少し丸めた長方形) with feathered/softened edges
            mask_u8 = np.zeros((rh, rw), dtype=np.uint8)
            pad = max(1, int(min(rh, rw) * 0.04))
            radius = max(2, int(min(rh, rw) * 0.18))
            radius = min(radius, (rh - 2 * pad) // 2, (rw - 2 * pad) // 2)

            x1, y1 = pad, pad
            x2, y2 = rw - pad, rh - pad

            # Draw central cross rectangles
            cv2.rectangle(mask_u8, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
            cv2.rectangle(mask_u8, (x1, y1 + radius), (x2, y2 - radius), 255, -1)
            # Draw 4 corner circles
            cv2.circle(mask_u8, (x1 + radius, y1 + radius), radius, 255, -1)
            cv2.circle(mask_u8, (x2 - radius - 1, y1 + radius), radius, 255, -1)
            cv2.circle(mask_u8, (x1 + radius, y2 - radius - 1), radius, 255, -1)
            cv2.circle(mask_u8, (x2 - radius - 1, y2 - radius - 1), radius, 255, -1)

            # Soft edge feathering (縁のぼかしを少し緩くする)
            feather_ksize = max(3, int(min(rh, rw) * 0.12))
            if feather_ksize % 2 == 0:
                feather_ksize += 1
            feathered = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (feather_ksize, feather_ksize), 0)
            return np.clip(feathered * 1.15, 0.0, 1.0)

        else:
            # Soft-edged rectangle
            rect_u8 = np.zeros((rh, rw), dtype=np.uint8)
            pad_x = max(1, int(rw * 0.06))
            pad_y = max(1, int(rh * 0.06))
            rect_u8[pad_y:max(pad_y + 1, rh - pad_y), pad_x:max(pad_x + 1, rw - pad_x)] = 255

            feather_ksize = max(3, int(min(rh, rw) * 0.14))
            if feather_ksize % 2 == 0:
                feather_ksize += 1
            feathered = cv2.GaussianBlur(rect_u8.astype(np.float32) / 255.0, (feather_ksize, feather_ksize), 0)
            return np.clip(feathered * 1.15, 0.0, 1.0)

    def apply_blur(self, frame: np.ndarray, bboxes: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply blur/mosaic to frame at specified bboxes using feathered soft-edge blending.
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

            if active_shape in ("ellipse", "round", "circle", "rounded_rect", "round_rect", "rounded_rectangle"):
                # Soft feathered alpha mask to ensure edges blend seamlessly with no harsh boundary
                alpha_2d = self.generate_feathered_mask(rh, rw, active_shape)
                alpha = alpha_2d[:, :, None]

                # Blend softly with original video frame
                blended = (blurred_roi.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha))
                out_frame[y1:y2, x1:x2] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

                # Mask tracking (scale to uint8)
                mask_roi = (alpha_2d * 255.0).astype(np.uint8)
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], mask_roi)
            else:
                out_frame[y1:y2, x1:x2] = blurred_roi
                mask[y1:y2, x1:x2] = 255

        return out_frame, mask
