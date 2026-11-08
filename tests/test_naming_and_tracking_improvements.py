import pytest
from pathlib import Path
import numpy as np

from face_mosaic.detector import FaceDetection
from face_mosaic.tracker import FaceTracker, Tracklet, compute_track_match_score
from face_mosaic.io_utils import build_output_path
from face_mosaic.config import AppConfig
from face_mosaic.pipeline import ProcessingPipeline

# ==============================================================================
# (1) File naming tests: Verifying and debugging output file name algorithm
# ==============================================================================

def test_build_output_path_preserves_extensions(tmp_path):
    """
    Ensure original container extensions (.mov, .MOV, .mkv, .mp4, etc.)
    are properly preserved instead of being forced to .mp4.
    """
    f_mov = tmp_path / "walk_shibuya.mov"
    f_mov_upper = tmp_path / "IMG_1234.MOV"
    f_mkv = tmp_path / "night_walk.mkv"
    f_mp4 = tmp_path / "park.mp4"

    out_mov = build_output_path(f_mov)
    assert out_mov.name == "walk_shibuya_blurred.mov"
    assert out_mov.parent == tmp_path

    out_mov_upper = build_output_path(f_mov_upper)
    assert out_mov_upper.name == "IMG_1234_blurred.MOV"

    out_mkv = build_output_path(f_mkv)
    assert out_mkv.name == "night_walk_blurred.mkv"

    out_mp4 = build_output_path(f_mp4)
    assert out_mp4.name == "park_blurred.mp4"

def test_build_output_path_multiple_dots_in_filename(tmp_path):
    """
    Test files with multiple periods in filename like date/resolution stamps.
    """
    f_dots = tmp_path / "walk.tokyo.2026.10.11.4k.mov"
    out_dots = build_output_path(f_dots)
    assert out_dots.name == "walk.tokyo.2026.10.11.4k_blurred.mov"

def test_build_output_path_prevents_duplicate_suffix(tmp_path):
    """
    Ensure files that already end with _blurred do not get _blurred_blurred.
    """
    f_already = tmp_path / "clip_01_blurred.mp4"
    out = build_output_path(f_already, filename_suffix="_blurred")
    assert out.name == "clip_01_blurred.mp4"

def test_build_output_path_suffix_with_embedded_extension(tmp_path):
    """
    Handle user setting suffix with an extension like '_blurred.mp4'.
    Prevents double extensions such as 'video_blurred.mp4.mp4'.
    """
    f = tmp_path / "video.mov"
    out = build_output_path(f, filename_suffix="_blurred.mp4")
    assert out.name == "video_blurred.mp4"
    assert not out.name.endswith(".mp4.mp4")

def test_build_output_path_with_custom_output_dir(tmp_path):
    """
    Verify custom output directory placement.
    """
    in_file = tmp_path / "inputs" / "walk.mov"
    in_file.parent.mkdir(parents=True, exist_ok=True)
    in_file.touch()

    out_dir = tmp_path / "custom_outputs"
    out = build_output_path(in_file, output_dir=str(out_dir), filename_suffix="_mosaic")
    assert out == out_dir / "walk_mosaic.mov"
    assert out_dir.exists()

def test_build_output_path_collision_avoidance(tmp_path):
    """
    Verify collision avoidance numbering when avoid_collision=True.
    """
    f = tmp_path / "sample.mp4"
    f.touch()
    existing_out = tmp_path / "sample_blurred.mp4"
    existing_out.touch()

    out_coll = build_output_path(f, avoid_collision=True)
    assert out_coll.name == "sample_blurred_1.mp4"

# ==============================================================================
# (2) Tracking and Padding Improvement tests
# ==============================================================================

def test_tracklet_velocity_estimation_and_prediction():
    """
    Test Tracklet calculates velocity vector and extrapolates predicted bbox.
    """
    det1 = FaceDetection(bbox=np.array([100.0, 100.0, 150.0, 150.0]), score=0.9)
    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    tracklet = Tracklet(track_id=1, start_frame=0, detection=det1, crop=crop)

    # Move right by 10 pixels per frame
    det2 = FaceDetection(bbox=np.array([110.0, 100.0, 160.0, 150.0]), score=0.9)
    tracklet.add_detection(1, det2, crop)

    vel = tracklet.get_velocity()
    # Horizontal velocity vx1, vx2 should be approx 10.0
    assert abs(vel[0] - 10.0) < 1e-3
    assert abs(vel[2] - 10.0) < 1e-3

    # Predict at frame 3 (2 frames ahead of last_frame 1) -> +20px
    pred_box = tracklet.predict_bbox(3)
    assert abs(pred_box[0] - 130.0) < 1e-3
    assert abs(pred_box[2] - 180.0) < 1e-3

def test_fast_moving_face_tracking_soft_matching():
    """
    Verify distance-aware soft-matching allows tracking faces even with low IoU
    due to rapid motion.
    """
    # Box moving fast: box1: [0, 0, 100, 100], box2: [50, 0, 150, 100]
    b1 = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
    b2 = np.array([50.0, 0.0, 150.0, 100.0], dtype=np.float32)
    score = compute_track_match_score(b1, b2)
    # IoU is 50*100 / 150*100 = 0.333
    assert score >= 0.3

    # Faster box: [70, 0, 170, 100] -> IoU drops below 0.2
    b3 = np.array([70.0, 0.0, 170.0, 100.0], dtype=np.float32)
    score_fast = compute_track_match_score(b1, b3)
    # Distance aware soft score maintains continuity
    assert score_fast >= 0.25

def test_tracker_forward_padding_and_extrapolation():
    """
    Verify forward padding applies extended frames (default 24)
    with velocity extrapolation and safety expansion.
    """
    tracker = FaceTracker(
        iou_threshold=0.3,
        padding_backward=5,
        padding_forward=24  # Extended forward padding
    )
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # Face moving diagonally from frame 10 to 12
    det1 = FaceDetection(bbox=np.array([200.0, 200.0, 300.0, 300.0]), score=0.95)
    det2 = FaceDetection(bbox=np.array([210.0, 205.0, 310.0, 305.0]), score=0.95)
    det3 = FaceDetection(bbox=np.array([220.0, 210.0, 320.0, 310.0]), score=0.95)

    tracker.update(10, frame, [det1])
    tracker.update(11, frame, [det2])
    tracker.update(12, frame, [det3])

    frame_map = tracker.get_frame_bboxes_map(total_frames=50, img_w=1000, img_h=1000)

    # Forward padding covers frames 13 through 36 (12 + 24)
    for f in range(13, 37):
        assert len(frame_map[f]) == 1, f"Frame {f} should have a forward padding bbox"

    # Frame 37 should have no bbox (padding ended)
    assert len(frame_map[37]) == 0

    # Ensure extrapolated box moved in the direction of velocity
    pad_box_15 = frame_map[15][0]
    # Box should be shifted forward relative to end detection (220, 210)
    assert pad_box_15[0] > 210.0

def test_pipeline_output_extension_integration(tmp_path):
    """
    Test pipeline end-to-end output path generation inherits input extension.
    """
    config = AppConfig()
    pipeline = ProcessingPipeline(config)

    # Mock file input path with .mov
    fake_mov = tmp_path / "street_scene.mov"
    fake_mov.touch()

    out_p = build_output_path(fake_mov, filename_suffix=config.output.filename_suffix)
    assert out_p.suffix == ".mov"
    assert out_p.name == "street_scene_blurred.mov"

def test_fps_adaptive_padding_scaling():
    """
    Verify high-FPS video support (60fps, 120fps, 240fps):
    padding and max_missing_frames scale proportionally so real-time retention
    remains constant across different frame rates.
    """
    # Baseline 30fps: pad_fwd=24 (0.8s), pad_back=12 (0.4s), missing=16
    t_30 = FaceTracker(padding_backward=12, padding_forward=24, max_missing_frames=16, fps=30.0, base_fps=30.0)
    assert t_30.padding_forward == 24
    assert t_30.padding_backward == 12
    assert t_30.max_missing_frames == 16

    # 60fps: 2x frames
    t_60 = FaceTracker(padding_backward=12, padding_forward=24, max_missing_frames=16, fps=60.0, base_fps=30.0)
    assert t_60.padding_forward == 48
    assert t_60.padding_backward == 24
    assert t_60.max_missing_frames == 32

    # 120fps: 4x frames (e.g. iPhone 120fps slo-mo or 4K120)
    t_120 = FaceTracker(padding_backward=12, padding_forward=24, max_missing_frames=16, fps=120.0, base_fps=30.0)
    assert t_120.padding_forward == 96
    assert t_120.padding_backward == 48
    assert t_120.max_missing_frames == 64

    # 240fps: 8x frames (iPhone 240fps high-speed)
    t_240 = FaceTracker(padding_backward=12, padding_forward=24, max_missing_frames=16, fps=240.0, base_fps=30.0)
    assert t_240.padding_forward == 192
    assert t_240.padding_backward == 96
    assert t_240.max_missing_frames == 128

    # When auto_scale_by_fps is disabled, frames remain static
    t_no_scale = FaceTracker(
        padding_backward=12,
        padding_forward=24,
        max_missing_frames=16,
        fps=120.0,
        auto_scale_by_fps=False
    )
    assert t_no_scale.padding_forward == 24
    assert t_no_scale.padding_backward == 12
    assert t_no_scale.max_missing_frames == 16


def test_bicycle_rider_fast_motion_tracking():
    """
    Test rapid displacement of cyclist face (e.g. 50px/frame for 80x80 box).
    Verifies that dual-anchor matching and extended search radius maintain a SINGLE continuous track
    without splitting into isolated tracks that leave lingering mosaic boxes at the starting position.
    """
    tracker = FaceTracker(iou_threshold=0.3, padding_backward=12, padding_forward=24)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Cyclist face moves from x=200 to x=700 across 10 frames (50px per frame)
    w, h = 80.0, 80.0
    for f in range(10):
        cx = 200.0 + f * 50.0
        cy = 300.0 + (f % 2) * 5.0  # slight vertical bounce
        bbox = np.array([cx, cy, cx + w, cy + h], dtype=np.float32)
        det = FaceDetection(bbox=bbox, score=0.92, label="face")
        tracker.update(f, dummy_frame, [det])

    # Should establish exactly ONE tracklet across all 10 frames
    tracker.finish()
    valid_tracks = [t for t in tracker.finished_tracks if len(t.frames) > 1]
    assert len(valid_tracks) == 1, f"Expected 1 continuous track for cyclist, got {len(valid_tracks)}"
    track = valid_tracks[0]
    assert len(track.frames) == 10

    # Verify frame map accurately follows the cyclist's motion
    frame_map = tracker.get_frame_bboxes_map(total_frames=30, img_w=1920, img_h=1080)
    for f in range(10):
        assert len(frame_map[f]) >= 1
        box = frame_map[f][0]
        expected_x1 = 200.0 + f * 50.0
        # Mosaic bounding box must follow current position, NOT be stuck at initial x=200
        assert abs(box[0] - expected_x1) < 2.0, f"Frame {f}: bbox x1={box[0]} did not follow cyclist ({expected_x1})"


def test_moving_license_plate_tracking():
    """
    Test rapid displacement of moving vehicle license plate (aspect ratio ~ 2.5:1, 60px/frame).
    Verifies plates are continuously tracked and tagged with 5th dimension=1.0.
    """
    tracker = FaceTracker(iou_threshold=0.3, padding_backward=8, padding_forward=16)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Car license plate: width 120px, height 45px, moving right by 60px per frame
    pw, ph = 120.0, 45.0
    for f in range(8):
        px = 100.0 + f * 60.0
        py = 500.0
        bbox = np.array([px, py, px + pw, py + ph], dtype=np.float32)
        det = FaceDetection(bbox=bbox, score=0.95, label="plate")
        tracker.update(f, dummy_frame, [det])

    tracker.finish()
    valid_tracks = [t for t in tracker.finished_tracks if len(t.frames) > 1]
    assert len(valid_tracks) == 1, f"Expected 1 continuous plate track, got {len(valid_tracks)}"
    assert valid_tracks[0].label == "plate"

    frame_map = tracker.get_frame_bboxes_map(total_frames=25, img_w=1920, img_h=1080)
    for f in range(8):
        box = frame_map[f][0]
        assert len(box) == 5 and box[4] == 1.0, "License plate must be tagged with 5th element 1.0"
        expected_px = 100.0 + f * 60.0
        assert abs(box[0] - expected_px) < 2.0, f"Frame {f}: plate x1={box[0]} did not match vehicle position ({expected_px})"


def test_intermittent_detection_interpolation():
    """
    Test that when detection drops for 2 frames (motion blur / head turn),
    linear interpolation seamlessly bridges the gap and tracks the moving subject.
    """
    tracker = FaceTracker(iou_threshold=0.3, max_missing_frames=10)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Detection at frame 0, then missing frames 1 & 2, then detected at frame 3
    det0 = FaceDetection(bbox=np.array([100.0, 100.0, 180.0, 180.0]), score=0.9)
    det3 = FaceDetection(bbox=np.array([250.0, 100.0, 330.0, 180.0]), score=0.9)

    tracker.update(0, dummy_frame, [det0])
    tracker.update(1, dummy_frame, [])  # Drop
    tracker.update(2, dummy_frame, [])  # Drop
    tracker.update(3, dummy_frame, [det3])  # Reappear 150px away

    frame_map = tracker.get_frame_bboxes_map(total_frames=10, img_w=1920, img_h=1080)

    # Frame 1 and 2 must have interpolated bboxes advancing smoothly
    assert len(frame_map[1]) == 1
    assert len(frame_map[2]) == 1
    # Frame 1: 100 + (250-100)*(1/3) = 150
    assert abs(frame_map[1][0][0] - 150.0) < 1.0
    # Frame 2: 100 + (250-100)*(2/3) = 200
    assert abs(frame_map[2][0][0] - 200.0) < 1.0


def test_isolated_detection_adaptive_padding():
    """
    Verify that an isolated single-frame detection does not leave an excessive 24-frame
    stationary mosaic box at the old position.
    """
    tracker = FaceTracker(padding_backward=12, padding_forward=24)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    det = FaceDetection(bbox=np.array([400.0, 400.0, 480.0, 480.0]), score=0.88)
    tracker.update(10, dummy_frame, [det])

    frame_map = tracker.get_frame_bboxes_map(total_frames=50, img_w=1920, img_h=1080)

    # For single isolated detection, forward padding is capped to 4 frames (frames 11..14)
    assert len(frame_map[14]) == 1
    assert len(frame_map[15]) == 0, "Isolated detection should NOT persist beyond 4 forward frames"


