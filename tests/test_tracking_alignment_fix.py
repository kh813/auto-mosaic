import numpy as np
import pytest
from face_mosaic.tracker import FaceTracker
from face_mosaic.detector import FaceDetection

def test_two_adjacent_cars_license_plates_never_swap():
    """
    User Complaint Case 3:
    "2台の車のナンバープレートにかかるように左右に1つのぼかしが移動してかかり、その後2台にぼかしがかかる。"
    Verify that when 2 cars are adjacent, a tracklet never jumps between the two cars,
    and no single blur moves back and forth across both vehicles.
    """
    tracker = FaceTracker(iou_threshold=0.25, max_missing_frames=10)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Car 1 plate: x=[300, 420], y=[500, 560] (center 360, 530)
    # Car 2 plate: x=[480, 600], y=[500, 560] (center 540, 530)
    # Frame 0: Only Car 1 is detected
    d0 = [FaceDetection(bbox=np.array([300.0, 500.0, 420.0, 560.0]), score=0.92, label="plate")]
    tracker.update(0, dummy_frame, d0)

    # Frame 1: Car 1 momentarily missed, Car 2 is detected
    d1 = [FaceDetection(bbox=np.array([480.0, 500.0, 600.0, 560.0]), score=0.92, label="plate")]
    tracker.update(1, dummy_frame, d1)

    # Frame 2: Both Car 1 and Car 2 detected
    d2 = [
        FaceDetection(bbox=np.array([302.0, 500.0, 422.0, 560.0]), score=0.92, label="plate"),
        FaceDetection(bbox=np.array([482.0, 500.0, 602.0, 560.0]), score=0.92, label="plate")
    ]
    tracker.update(2, dummy_frame, d2)

    tracker.finish()
    # There must be 2 distinct tracks
    assert len(tracker.finished_tracks) == 2

    # Track 0 (Car 1) must NEVER contain Car 2's coordinates
    car1_track = next(t for t in tracker.finished_tracks if t.start_frame == 0)
    for f, box in car1_track.frames.items():
        cx = (box[0] + box[2]) / 2.0
        assert 300.0 <= cx <= 430.0, f"Car 1 track drifted to x={cx} at frame {f}!"

    # Track 1 (Car 2) must stay on Car 2
    car2_track = next(t for t in tracker.finished_tracks if t.start_frame == 1)
    for f, box in car2_track.frames.items():
        cx = (box[0] + box[2]) / 2.0
        assert 480.0 <= cx <= 610.0, f"Car 2 track drifted to x={cx} at frame {f}!"

def test_two_adjacent_faces_never_cross():
    """
    User Complaint Cases 1 & 4:
    "二名の顔とぼかしの場所がズレています。"
    "左の男性の顔とぼかしがズレている。"
    Verify that two people standing or walking next to each other maintain
    their own independent, accurate tracks without swapping or drifting.
    """
    tracker = FaceTracker(iou_threshold=0.25, max_missing_frames=10)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Person 1 (left): x=[100, 180]
    # Person 2 (right): x=[220, 300]
    for f in range(5):
        # Frame 2: Person 1 momentarily missed
        p2_det = FaceDetection(bbox=np.array([220.0 + f * 2, 200.0, 300.0 + f * 2, 280.0]), score=0.9, label="face")
        if f == 2:
            dets = [p2_det]
        else:
            p1_det = FaceDetection(bbox=np.array([100.0 + f * 2, 200.0, 180.0 + f * 2, 280.0]), score=0.9, label="face")
            dets = [p1_det, p2_det]
        tracker.update(f, dummy_frame, dets)

    tracker.finish()
    assert len(tracker.finished_tracks) == 2

    frame_map = tracker.get_frame_bboxes_map(total_frames=5, img_w=1920, img_h=1080)
    for f in range(5):
        assert len(frame_map[f]) == 2, f"Frame {f} should have exactly 2 boxes, got {len(frame_map[f])}"
        centers = sorted([(b[0] + b[2]) / 2.0 for b in frame_map[f]])
        # Left person center should be around 140..150
        assert 135.0 <= centers[0] <= 155.0
        # Right person center should be around 260..270
        assert 255.0 <= centers[1] <= 275.0

def test_padding_does_not_drift_away_with_extrapolated_velocity():
    """
    Verify backward and forward padding stay directly over the detection with safety margin,
    rather than extrapolating velocity vectors into empty space.
    """
    tracker = FaceTracker(padding_backward=4, padding_forward=6)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Face moving at 10px/frame across frames 10..15
    for f in range(10, 16):
        x = 500.0 + (f - 10) * 10.0
        det = FaceDetection(bbox=np.array([x, 300.0, x + 80.0, 380.0]), score=0.9, label="face")
        tracker.update(f, dummy_frame, [det])

    frame_map = tracker.get_frame_bboxes_map(total_frames=25, img_w=1920, img_h=1080)

    # Backward padding at frame 8 (2 frames before appearance at frame 10)
    assert len(frame_map[8]) == 1
    pad_b = frame_map[8][0]
    # Center should remain right at frame 10's starting position (540), NOT flying off to 460 or 420
    cx_back = (pad_b[0] + pad_b[2]) / 2.0
    assert abs(cx_back - 540.0) < 5.0, f"Backward padding shifted away to x={cx_back}"

    # Forward padding at frame 18 (3 frames after departure at frame 15, x=550..630, center=590)
    assert len(frame_map[18]) == 1
    pad_f = frame_map[18][0]
    cx_fwd = (pad_f[0] + pad_f[2]) / 2.0
    assert abs(cx_fwd - 590.0) < 5.0, f"Forward padding shifted away to x={cx_fwd}"

def test_moving_face_continuous_tracking_no_lagging_mosaic():
    """
    Verify that a moving face (e.g. running, walking, or camera pan where consecutive
    frame overlap is near zero IoU < 0.05) remains in a single continuous tracklet,
    and no orphaned/lagging stationary blur is left behind at the previous position.
    """
    tracker = FaceTracker(iou_threshold=0.25, max_missing_frames=16)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Face of size 60x60 moves 50px per frame (consecutive IoU < 0.05)
    # Frame 0: x=100..160, center=130
    # Frame 1: x=150..210, center=180
    # Frame 2: x=200..260, center=230
    detections_by_frame = {
        0: [FaceDetection(bbox=np.array([100.0, 300.0, 160.0, 360.0]), score=0.92, label="face")],
        1: [FaceDetection(bbox=np.array([150.0, 300.0, 210.0, 360.0]), score=0.92, label="face")],
        2: [FaceDetection(bbox=np.array([200.0, 300.0, 260.0, 360.0]), score=0.92, label="face")],
    }

    for f in range(3):
        tracker.update(f, dummy_frame, detections_by_frame[f])

    tracker.finish()
    # Must remain a single continuous track, not 3 fragmented tracks
    assert len(tracker.finished_tracks) == 1, (
        f"Expected 1 track for moving face, but got {len(tracker.finished_tracks)} tracks"
    )

    frame_map = tracker.get_frame_bboxes_map(total_frames=3, img_w=1920, img_h=1080)
    # In each frame, there should be exactly 1 mosaic box directly on the face
    for f in range(3):
        assert len(frame_map[f]) == 1, f"Frame {f} should have exactly 1 box, got {len(frame_map[f])}"
        b = frame_map[f][0]
        cx = (b[0] + b[2]) / 2.0
        expected_cx = 130.0 + f * 50.0
        assert abs(cx - expected_cx) < 2.0, (
            f"Frame {f} mosaic displaced! cx={cx}, expected={expected_cx}"
        )

def test_turning_head_missed_frames_interpolation():
    """
    Verify that when a person temporarily turns their head or looks away (missed for 3 frames),
    the tracker seamlessly re-associates the face and linearly interpolates the missing frames
    without losing the track or producing disconnected jumps.
    """
    tracker = FaceTracker(iou_threshold=0.25, max_missing_frames=16)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Frame 0: Detected at x=200..280 (center 240)
    tracker.update(0, dummy_frame, [FaceDetection(bbox=np.array([200.0, 200.0, 280.0, 280.0]), score=0.9, label="face")])
    # Frames 1-3: Head turned away, face missed
    for f in range(1, 4):
        tracker.update(f, dummy_frame, [])
    # Frame 4: Head turns back at x=280..360 (center 320, moved 80px over 4 frames)
    tracker.update(4, dummy_frame, [FaceDetection(bbox=np.array([280.0, 200.0, 360.0, 280.0]), score=0.9, label="face")])

    tracker.finish()
    # Must be recognized as the same person
    assert len(tracker.finished_tracks) == 1

    frame_map = tracker.get_frame_bboxes_map(total_frames=5, img_w=1920, img_h=1080)
    # Verify all frames (including missed 1, 2, 3) are populated via linear interpolation
    for f in range(5):
        assert len(frame_map[f]) == 1, f"Frame {f} missing interpolated mosaic!"
        b = frame_map[f][0]
        cx = (b[0] + b[2]) / 2.0
        expected_cx = 240.0 + f * 20.0  # (320 - 240) / 4 = 20px per frame
        assert abs(cx - expected_cx) < 2.0, f"Frame {f} interpolation off: cx={cx}, expected={expected_cx}"
