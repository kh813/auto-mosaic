"""
Tracklet association, temporal smoothing, track-level filtering,
and bidirectional frame padding (forward/backward blur extension).
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

from .detector import FaceDetection
from .filters.skin_color import SkinColorFilter
from .filters.animal_classifier import AnimalClassifier
from .filters.static_photo import StaticPhotoFilter
from .filters.illustration import IllustrationFilter
from .config import FiltersConfig

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

def compute_track_match_score(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute matching affinity between a tracklet box (predicted or latest) and a detection box.
    Combines:
    1. Standard Intersection over Union (IoU)
    2. Distance-IoU (DIoU) awareness for close objects with small/zero overlap
    3. Center-distance soft matching with adaptive search radius for fast-moving targets (bicycles, vehicles).
    """
    iou = compute_iou(box_a, box_b)
    if iou >= 0.20:
        return max(iou, 0.35 + 0.65 * iou)

    # Center distance & scale consistency check for fast moving faces / plates
    w1, h1 = box_a[2] - box_a[0], box_a[3] - box_a[1]
    w2, h2 = box_b[2] - box_b[0], box_b[3] - box_b[1]
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0

    c1_x, c1_y = (box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0
    c2_x, c2_y = (box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0
    dist = np.hypot(c1_x - c2_x, c1_y - c2_y)
    diag = np.hypot(max(w1, w2), max(h1, h2))

    # Scale consistency: ratio of bounding box areas
    scale_ratio = (w2 * h2) / (w1 * h1)
    if not (0.25 <= scale_ratio <= 4.0):
        return iou

    # Adaptive search radius: accommodates rapid motion of cyclists and vehicles
    # (allows tracking when object moves up to 2.5x diagonal or minimum 140px)
    max_search_dist = max(diag * 2.5, 140.0)

    if dist < max_search_dist:
        # Distance affinity mapped smoothly to [0.25, 0.55]
        norm_dist = dist / max_search_dist
        dist_affinity = 0.25 + 0.30 * (1.0 - norm_dist)
        
        # Penalize large scale difference slightly
        scale_penalty = min(abs(np.log(scale_ratio)), 1.0) * 0.08
        dist_affinity = max(0.0, dist_affinity - scale_penalty)

        # DIoU component: smallest enclosing box diagonal
        enc_x1 = min(box_a[0], box_b[0])
        enc_y1 = min(box_a[1], box_b[1])
        enc_x2 = max(box_a[2], box_b[2])
        enc_y2 = max(box_a[3], box_b[3])
        c_diag = np.hypot(enc_x2 - enc_x1, enc_y2 - enc_y1)
        if c_diag > 0:
            diou_penalty = (dist / c_diag) ** 2
            diou_score = max(0.0, 0.45 - 0.25 * diou_penalty)
            return max(iou, dist_affinity, diou_score)

        return max(iou, dist_affinity)

    return iou

class Tracklet:
    def __init__(self, track_id: int, start_frame: int, detection: FaceDetection, crop: Optional[np.ndarray]):
        self.track_id = track_id
        self.start_frame = start_frame
        self.last_frame = start_frame
        self.label: str = getattr(detection, "label", "face")
        
        # Frame index -> Bounding Box [x1, y1, x2, y2]
        self.frames: Dict[int, np.ndarray] = {start_frame: detection.bbox}
        # Frame index -> Score
        self.scores: Dict[int, float] = {start_frame: detection.score}
        # Frame index -> Landmarks
        self.landmarks: Dict[int, Optional[np.ndarray]] = {start_frame: detection.landmarks}
        # Sequence of face crops for track-level evaluation (capped to avoid memory exhaustion)
        self.crops: List[np.ndarray] = [crop] if crop is not None else []
        
        self.is_valid: Optional[bool] = None  # Judged at track level

    def add_detection(self, frame_idx: int, detection: FaceDetection, crop: np.ndarray):
        self.last_frame = frame_idx
        self.frames[frame_idx] = detection.bbox
        self.scores[frame_idx] = detection.score
        self.landmarks[frame_idx] = detection.landmarks
        if crop is not None and len(self.crops) < 35:
            self.crops.append(crop)

    @property
    def latest_bbox(self) -> np.ndarray:
        return self.frames[self.last_frame]

    def get_velocity(self) -> np.ndarray:
        """
        Estimate recent per-frame velocity [vx1, vy1, vx2, vy2].
        Uses recent detections with recency weighting to track acceleration/deceleration.
        """
        frame_keys = sorted(self.frames.keys())
        if len(frame_keys) < 2:
            return np.zeros(4, dtype=np.float32)

        recent_keys = frame_keys[-min(4, len(frame_keys)):]
        dt = recent_keys[-1] - recent_keys[0]
        if dt <= 0:
            return np.zeros(4, dtype=np.float32)

        # Weighted velocity giving higher weight to the most recent step
        if len(recent_keys) >= 3:
            dt_last = recent_keys[-1] - recent_keys[-2]
            dt_prev = recent_keys[-2] - recent_keys[0]
            v_last = (self.frames[recent_keys[-1]] - self.frames[recent_keys[-2]]) / float(dt_last) if dt_last > 0 else np.zeros(4, dtype=np.float32)
            v_prev = (self.frames[recent_keys[-2]] - self.frames[recent_keys[0]]) / float(dt_prev) if dt_prev > 0 else np.zeros(4, dtype=np.float32)
            delta = 0.65 * v_last + 0.35 * v_prev
        else:
            delta = (self.frames[recent_keys[-1]] - self.frames[recent_keys[0]]) / float(dt)

        # Realistic velocity cap for fast moving objects (bicycles, cars, runners):
        # Allow up to 2.0x object size or minimum 180px per frame (accommodates 4K/FHD motion)
        w = max(1.0, self.latest_bbox[2] - self.latest_bbox[0])
        h = max(1.0, self.latest_bbox[3] - self.latest_bbox[1])
        max_v = max(max(w, h) * 2.0, 180.0)
        return np.clip(delta, -max_v, max_v)

    def predict_bbox(self, target_frame: int) -> np.ndarray:
        """
        Predict bounding box at target_frame based on recent trajectory.
        """
        if target_frame == self.last_frame:
            return self.latest_bbox.copy()
        dt = target_frame - self.last_frame
        if dt <= 0 or len(self.frames) < 2:
            return self.latest_bbox.copy()
        vel = self.get_velocity()
        pred = self.latest_bbox.copy() + vel * dt
        # Ensure box remains valid and retains positive area
        w = max(5.0, self.latest_bbox[2] - self.latest_bbox[0])
        h = max(5.0, self.latest_bbox[3] - self.latest_bbox[1])
        pred[2] = max(pred[2], pred[0] + w * 0.5)
        pred[3] = max(pred[3], pred[1] + h * 0.5)
        return pred

class FaceTracker:
    """
    Online tracker with batch completion & track-level filtering.
    Associates frame detections across time, applies track-level filters,
    interpolates missing frames, and expands blur backward & forward.
    """
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missing_frames: int = 16,
        padding_backward: int = 12,
        padding_forward: int = 24,
        filters_config: Optional[FiltersConfig] = None,
        fps: Optional[float] = None,
        auto_scale_by_fps: bool = True,
        base_fps: float = 30.0
    ):
        self.iou_threshold = iou_threshold
        self.base_fps = base_fps
        self.video_fps = fps or base_fps
        self.auto_scale_by_fps = auto_scale_by_fps

        # Automatically adapt padding & missing frames according to video FPS
        # (e.g. 60fps = 2x frames, 120fps = 4x frames, 240fps = 8x frames)
        if auto_scale_by_fps and fps and fps > 0:
            scale = max(0.5, fps / base_fps)
            self.max_missing_frames = max(1, int(round(max_missing_frames * scale)))
            self.padding_backward = max(0, int(round(padding_backward * scale)))
            self.padding_forward = max(0, int(round(padding_forward * scale)))
        else:
            self.max_missing_frames = max_missing_frames
            self.padding_backward = padding_backward
            self.padding_forward = padding_forward
        
        self.next_track_id = 0
        self.active_tracks: List[Tracklet] = []
        self.finished_tracks: List[Tracklet] = []

        # Filters
        self.filters_config = filters_config or FiltersConfig()
        self.skin_filter = SkinColorFilter(min_skin_ratio=self.filters_config.skin_color_filter.min_skin_ratio)
        self.animal_filter = AnimalClassifier(human_prob_threshold=self.filters_config.animal_filter.human_prob_threshold)
        self.static_filter = StaticPhotoFilter(
            min_static_frames=self.filters_config.static_photo_filter.min_static_frames,
            motion_similarity_threshold=self.filters_config.static_photo_filter.motion_similarity_threshold,
            texture_diff_threshold=self.filters_config.static_photo_filter.texture_diff_threshold
        )
        self.illustration_filter = IllustrationFilter(
            flatness_threshold=self.filters_config.illustration_filter.flatness_threshold,
            min_crop_size=self.filters_config.illustration_filter.min_crop_size
        )

    def extract_crop(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        orig_shape: Optional[Tuple[int, int]] = None,
        max_dim: int = 128
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        if orig_shape is not None and (orig_shape[0] != h or orig_shape[1] != w):
            scale_x = w / orig_shape[1]
            scale_y = h / orig_shape[0]
            bx1, by1, bx2, by2 = bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y
        else:
            bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]

        x1 = max(0, int(np.floor(bx1)))
        y1 = max(0, int(np.floor(by1)))
        x2 = min(w, int(np.ceil(bx2)))
        y2 = min(h, int(np.ceil(by2)))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if max(ch, cw) > max_dim:
            scale = max_dim / max(ch, cw)
            nw = max(1, int(cw * scale))
            nh = max(1, int(ch * scale))
            return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA).copy()
        return crop.copy()

    def update(
        self,
        frame_idx: int,
        frame: np.ndarray,
        detections: List[FaceDetection],
        orig_shape: Optional[Tuple[int, int]] = None
    ):
        """
        Process detections for frame_idx.
        """
        matched_tracks = set()
        matched_dets = set()

        if self.active_tracks and detections:
            # Build match matrix using dual anchors:
            # Anchor 1: predicted bbox (with velocity extrapolation for moving subjects)
            # Anchor 2: latest observed bbox (classic legacy anchor, robust against turns, stops, and initial acceleration)
            ious = np.zeros((len(self.active_tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.active_tracks):
                pred_box = track.predict_bbox(frame_idx)
                latest_box = track.latest_bbox
                for d_idx, det in enumerate(detections):
                    det_label = getattr(det, "label", "face")
                    if det_label != track.label:
                        continue
                    score_pred = compute_track_match_score(pred_box, det.bbox)
                    score_latest = compute_track_match_score(latest_box, det.bbox)
                    ious[t_idx, d_idx] = max(score_pred, score_latest)

            # Greedy matching
            while True:
                max_iou = np.max(ious)
                if max_iou < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(ious), ious.shape)
                if t_idx in matched_tracks or d_idx in matched_dets:
                    ious[t_idx, d_idx] = 0.0
                    continue

                det = detections[d_idx]
                det_label = getattr(det, "label", "face")
                need_crop = (det_label != "plate") or self.filters_config.static_photo_filter.enabled
                crop = self.extract_crop(frame, det.bbox, orig_shape=orig_shape) if need_crop else None
                self.active_tracks[t_idx].add_detection(frame_idx, det, crop)
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                ious[t_idx, :] = 0.0
                ious[:, d_idx] = 0.0

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                det_label = getattr(det, "label", "face")
                need_crop = (det_label != "plate") or self.filters_config.static_photo_filter.enabled
                crop = self.extract_crop(frame, det.bbox, orig_shape=orig_shape) if need_crop else None
                track = Tracklet(self.next_track_id, frame_idx, det, crop)
                self.next_track_id += 1
                self.active_tracks.append(track)

        # Retire stale tracks
        still_active = []
        for t_idx, track in enumerate(self.active_tracks):
            if frame_idx - track.last_frame > self.max_missing_frames:
                self.finished_tracks.append(track)
            else:
                still_active.append(track)
        self.active_tracks = still_active

    def finish(self):
        """Called when video ends to finalize remaining active tracks."""
        self.finished_tracks.extend(self.active_tracks)
        self.active_tracks.clear()

    def evaluate_track_validity(self, track: Tracklet, img_w: int = 1920, img_h: int = 1080) -> bool:
        """
        Evaluate if track is a genuine human face to be blurred.
        Track-level judgment ensures no flickering.
        """
        # Exclusion masks check
        if self.filters_config.exclusion_masks:
            # Check center of track
            all_bboxes = list(track.frames.values())
            avg_center_x = np.mean([(b[0] + b[2]) / 2.0 for b in all_bboxes]) / img_w
            avg_center_y = np.mean([(b[1] + b[3]) / 2.0 for b in all_bboxes]) / img_h
            for mask in self.filters_config.exclusion_masks:
                if len(mask) == 4:
                    mx1, my1, mx2, my2 = mask
                    if mx1 <= avg_center_x <= mx2 and my1 <= avg_center_y <= my2:
                        return False

        # License plate tracks bypass human face-specific filters (skin color, animal, illustration)
        if getattr(track, "label", "face") == "plate":
            track.crops.clear()
            return True

        # Skin color check on sample crops
        if self.filters_config.skin_color_filter.enabled and track.crops:
            # Check if crops have non-zero content
            valid_crops = [c for c in track.crops if c is not None and np.any(c > 0)]
            if valid_crops:
                skin_valid_votes = []
                for crop in valid_crops[::max(1, len(valid_crops) // 5)]:
                    is_skin, _ = self.skin_filter.is_skin_tone(crop)
                    skin_valid_votes.append(is_skin)
                # Safe side: if at least some crops show human skin, consider valid
                if skin_valid_votes and not any(skin_valid_votes):
                    return False  # Pure statue/bronze without human skin tone

        # Animal filter
        if self.filters_config.animal_filter.enabled and track.crops:
            human_votes = []
            num_samples = min(len(track.crops), 5)
            sample_indices = np.linspace(0, len(track.crops) - 1, num_samples, dtype=int)
            sample_frames = list(track.frames.keys())
            for idx in sample_indices:
                f = sample_frames[min(idx, len(sample_frames) - 1)]
                lm = track.landmarks.get(f)
                crop = track.crops[idx]
                is_h, _ = self.animal_filter.is_human(crop, lm)
                human_votes.append(is_h)
            if human_votes and not any(human_votes):
                return False

        # Illustration / drawing filter (Anime, cartoon, drawing faces)
        if self.filters_config.illustration_filter.enabled and track.crops:
            valid_crops = [c for c in track.crops if c is not None and np.any(c > 0)]
            if valid_crops:
                human_votes = []
                for crop in valid_crops[::max(1, len(valid_crops) // 5)]:
                    is_h, _ = self.illustration_filter.is_human_face(crop)
                    human_votes.append(is_h)
                if human_votes and not any(human_votes):
                    track.crops.clear()
                    return False

        # Static photo filter
        if self.filters_config.static_photo_filter.enabled and track.crops:
            if self.static_filter.is_static_advertisement(track.crops):
                track.crops.clear()
                return False

        track.crops.clear()
        return True

    def get_frame_bboxes_map(self, total_frames: int, img_w: int, img_h: int) -> Dict[int, List[np.ndarray]]:
        """
        Build frame_idx -> list of [x1, y1, x2, y2] bounding boxes across all frames,
        applying:
        1. Track-level validity filtering
        2. Linear interpolation for missing frames inside track
        3. Backward (N) and Forward (M) padding extension with velocity extrapolation and safety margin
        """
        self.finish()
        
        frame_map: Dict[int, List[np.ndarray]] = {i: [] for i in range(total_frames)}

        for track in self.finished_tracks:
            # Check track-level validity
            if not self.evaluate_track_validity(track, img_w, img_h):
                continue

            detected_frames = sorted(track.frames.keys())
            if not detected_frames:
                continue

            min_f = detected_frames[0]
            max_f = detected_frames[-1]
            num_detections = len(detected_frames)
            is_plate = getattr(track, "label", "face") == "plate"

            def _tag_bbox(b: np.ndarray) -> np.ndarray:
                if is_plate:
                    return np.array([b[0], b[1], b[2], b[3], 1.0], dtype=np.float32)
                return b

            # Interpolate inside the track duration
            for f in range(min_f, max_f + 1):
                if 0 <= f < total_frames:
                    if f in track.frames:
                        bbox = track.frames[f]
                    else:
                        # Linear interpolation between closest prior and subsequent detections
                        prev_f = max(k for k in detected_frames if k < f)
                        next_f = min(k for k in detected_frames if k > f)
                        alpha = (f - prev_f) / (next_f - prev_f)
                        bbox = (1 - alpha) * track.frames[prev_f] + alpha * track.frames[next_f]
                    frame_map[f].append(_tag_bbox(bbox))

            # Adaptive padding based on track maturity:
            # Isolated single-frame detections (num_detections == 1) get minimal padding to prevent
            # stationary mosaic blocks lingering at old positions when subjects are moving.
            if num_detections == 1:
                active_pad_back = min(self.padding_backward, 3)
                active_pad_fwd = min(self.padding_forward, 4)
            elif num_detections == 2:
                active_pad_back = min(self.padding_backward, 6)
                active_pad_fwd = min(self.padding_forward, 8)
            else:
                active_pad_back = self.padding_backward
                active_pad_fwd = self.padding_forward

            vel = track.get_velocity()
            start_bbox = track.frames[min_f]
            has_velocity = np.any(np.abs(vel) > 1.0)

            # Apply backward padding (before start)
            # If moving, extrapolate in reverse direction with decay so mosaic leads the incoming subject
            for pad_i in range(1, active_pad_back + 1):
                target_f = min_f - pad_i
                if 0 <= target_f < total_frames:
                    if has_velocity and num_detections >= 3:
                        decay = 0.95 ** pad_i
                        extrapolated = start_bbox - vel * (pad_i * decay)
                        w = max(1.0, extrapolated[2] - extrapolated[0])
                        h = max(1.0, extrapolated[3] - extrapolated[1])
                        pad_box = np.array([
                            max(0.0, extrapolated[0] - w * 0.05),
                            max(0.0, extrapolated[1] - h * 0.05),
                            min(float(img_w), extrapolated[2] + w * 0.05),
                            min(float(img_h), extrapolated[3] + h * 0.05)
                        ], dtype=np.float32)
                        frame_map[target_f].append(_tag_bbox(pad_box))
                    else:
                        frame_map[target_f].append(_tag_bbox(start_bbox))

            # Apply forward padding (after end) with velocity extrapolation & safety margin
            end_bbox = track.frames[max_f]
            for pad_i in range(1, active_pad_fwd + 1):
                target_f = max_f + pad_i
                if 0 <= target_f < total_frames:
                    if has_velocity and num_detections >= 2:
                        decay = 0.96 ** pad_i
                        extrapolated = end_bbox + vel * (pad_i * decay)
                        # Expand by 5% safety margin to ensure no facial exposure
                        w = max(1.0, extrapolated[2] - extrapolated[0])
                        h = max(1.0, extrapolated[3] - extrapolated[1])
                        pad_box = np.array([
                            max(0.0, extrapolated[0] - w * 0.05),
                            max(0.0, extrapolated[1] - h * 0.05),
                            min(float(img_w), extrapolated[2] + w * 0.05),
                            min(float(img_h), extrapolated[3] + h * 0.05)
                        ], dtype=np.float32)
                        frame_map[target_f].append(_tag_bbox(pad_box))
                    else:
                        frame_map[target_f].append(_tag_bbox(end_bbox))

        return frame_map
