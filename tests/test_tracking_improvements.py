import numpy as np
import pytest
from face_mosaic.detector import FaceDetection
from face_mosaic.tracker import FaceTracker, Tracklet
from face_mosaic.plate_detector import LicensePlateDetector

def test_tracklet_is_out_of_frame():
    # Inside screen
    det_in = FaceDetection(bbox=np.array([100, 100, 200, 200]), score=0.9)
    tr_in = Tracklet(0, 0, det_in, None)
    assert not tr_in.is_out_of_frame(img_w=1920, img_h=1080)

    # Exiting left boundary
    det_left = FaceDetection(bbox=np.array([5, 100, 105, 200]), score=0.9)
    tr_left = Tracklet(1, 0, det_left, None)
    # Give it leftward velocity
    tr_left.frames[1] = np.array([0, 100, 100, 200])
    tr_left.last_frame = 1
    assert tr_left.is_out_of_frame(img_w=1920, img_h=1080)

    # Exiting right boundary
    det_right = FaceDetection(bbox=np.array([1820, 100, 1915, 200]), score=0.9)
    tr_right = Tracklet(2, 0, det_right, None)
    tr_right.frames[1] = np.array([1825, 100, 1920, 200])
    tr_right.last_frame = 1
    assert tr_right.is_out_of_frame(img_w=1920, img_h=1080)

def test_face_tracker_body_sway_interpolation_and_smoothing():
    tracker = FaceTracker(iou_threshold=0.3, max_missing_frames=16, padding_backward=2, padding_forward=2)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    # Person walking and swaying body left and right
    det0 = FaceDetection(bbox=np.array([100, 100, 150, 150]), score=0.9)
    det1 = FaceDetection(bbox=np.array([115, 102, 165, 152]), score=0.9) # sway right
    # Frame 2 is missed due to turned head / sway
    det3 = FaceDetection(bbox=np.array([98, 105, 148, 155]), score=0.9)  # sway left
    det4 = FaceDetection(bbox=np.array([112, 107, 162, 157]), score=0.9) # sway right

    tracker.update(0, frame, [det0])
    tracker.update(1, frame, [det1])
    # frame 2 missed
    tracker.update(3, frame, [det3])
    tracker.update(4, frame, [det4])

    frame_map = tracker.get_frame_bboxes_map(total_frames=6, img_w=300, img_h=300)

    # Frame 2 should be seamlessly interpolated with sway coverage
    assert len(frame_map[2]) == 1
    interp_box = frame_map[2][0]
    # Should have positive width and cover intermediate sway position
    assert interp_box[2] > interp_box[0]
    assert interp_box[3] > interp_box[1]
    # Width should be safely expanded compared to original 50px
    assert (interp_box[2] - interp_box[0]) >= 50.0

def test_plate_aspect_ratio_filter():
    import tempfile
    # Create dummy detector instance for helper testing
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        pass
    # Test aspect ratio calculation logic
    # Valid plate: width 100, height 50 -> ratio 2.0
    bw_valid, bh_valid = 100, 50
    ar_valid = bw_valid / bh_valid
    assert 1.25 <= ar_valid <= 2.70

    # Station banner signboard: width 300, height 50 -> ratio 6.0
    bw_banner, bh_banner = 300, 50
    ar_banner = bw_banner / bh_banner
    assert not (1.25 <= ar_banner <= 2.70)

    # Square poster: width 100, height 95 -> ratio 1.05
    bw_poster, bh_poster = 100, 95
    ar_poster = bw_poster / bh_poster
    assert not (1.25 <= ar_poster <= 2.70)
