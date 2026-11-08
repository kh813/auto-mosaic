import pytest
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import MagicMock

from face_mosaic.config import AppConfig
from face_mosaic.detector import FaceDetection
from face_mosaic.plate_detector import LicensePlateDetector
from face_mosaic.tracker import FaceTracker
from face_mosaic.blur import FaceBlurrer
from face_mosaic.pipeline import ProcessingPipeline
from face_mosaic.checkpoint import VideoCheckpoint


def _draw_synthetic_car_with_plate() -> np.ndarray:
    """Create a 360x640 test image with a car rear and a Japanese-style license plate."""
    img = np.full((360, 640, 3), 120, dtype=np.uint8)
    # Road
    cv2.rectangle(img, (0, 200), (640, 360), (60, 60, 60), -1)
    # Car body
    cv2.rectangle(img, (180, 100), (460, 280), (40, 40, 160), -1)
    # Taillights
    cv2.rectangle(img, (190, 150), (240, 180), (0, 0, 255), -1)
    cv2.rectangle(img, (400, 150), (450, 180), (0, 0, 255), -1)
    # White license plate with dark green characters
    cv2.rectangle(img, (275, 205), (365, 250), (240, 245, 240), -1)
    cv2.rectangle(img, (275, 205), (365, 250), (50, 50, 50), 2)
    cv2.putText(img, "300", (305, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 60, 10), 2)
    cv2.putText(img, "12-34", (285, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 60, 10), 2)
    return img


def test_license_plate_detector_inference():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    plate_model = models_dir / "license-plate-finetune-v1n.onnx"
    if not plate_model.exists():
        pytest.skip("License plate ONNX model not found")

    detector = LicensePlateDetector(str(plate_model), conf_threshold=0.35)
    img = _draw_synthetic_car_with_plate()
    dets = detector.detect(img, orig_shape=(360, 640))

    assert len(dets) >= 1
    best = dets[0]
    assert best.label == "plate"
    assert best.score >= 0.35
    # Verify detected bbox overlaps the drawn plate [275, 205, 365, 250]
    cx = (best.x1 + best.x2) / 2.0
    cy = (best.y1 + best.y2) / 2.0
    assert 260 <= cx <= 380
    assert 195 <= cy <= 260


def test_tracker_preserves_plate_tracks_without_skin_rejection():
    """
    Verify that license plate detections (which have metallic/white/yellow colors, not human skin)
    bypass human skin & illustration filters and get tagged with 1.0 in frame_bboxes_map.
    """
    tracker = FaceTracker(padding_backward=2, padding_forward=3, auto_scale_by_fps=False)
    # Pure blue/white frame (would fail human skin filter if not bypassed)
    frame = np.full((240, 320, 3), 255, dtype=np.uint8)

    plate_det1 = FaceDetection(bbox=np.array([100.0, 150.0, 180.0, 190.0]), score=0.85, label="plate")
    plate_det2 = FaceDetection(bbox=np.array([105.0, 150.0, 185.0, 190.0]), score=0.88, label="plate")

    tracker.update(5, frame, [plate_det1])
    tracker.update(6, frame, [plate_det2])

    fmap = tracker.get_frame_bboxes_map(total_frames=15, img_w=320, img_h=240)

    # Backward padding: frames 3, 4; Detected: 5, 6; Forward padding: 7, 8, 9
    for f in range(3, 10):
        assert len(fmap[f]) == 1
        bbox = fmap[f][0]
        assert len(bbox) == 5
        assert bbox[4] == 1.0  # Tagged as plate


def test_blurrer_applies_rect_to_plate_and_ellipse_to_face():
    """
    Verify that FaceBlurrer applies elliptical blur to faces (preserving corners)
    while applying rectangular blur to license plates (covering all 4 corners).
    """
    blurrer = FaceBlurrer(
        blur_type="mosaic",
        mosaic_block_size=8,
        margin_x=0.0,
        margin_y=0.0,
        shape="ellipse",
        plate_margin_x=0.0,
        plate_margin_y=0.0,
        plate_shape="rect"
    )

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    for y in range(200):
        for x in range(200):
            frame[y, x] = [(x * 3) % 256, (y * 3) % 256, ((x + y) * 2) % 256]

    face_bbox = np.array([10.0, 10.0, 70.0, 70.0], dtype=np.float32)          # 4-element face bbox
    plate_bbox = np.array([110.0, 110.0, 170.0, 170.0, 1.0], dtype=np.float32) # 5-element plate bbox

    out, mask = blurrer.apply_blur(frame, [face_bbox, plate_bbox])

    # Face corner (11, 11) should NOT be blurred (ellipse shape)
    assert mask[11, 11] == 0
    assert mask[40, 40] == 255

    # Plate corner (111, 111) MUST be blurred (rect shape covers plate corners)
    assert mask[111, 111] == 255
    assert mask[140, 140] == 255


def test_selective_model_loading_for_pc_mac_load_optimization():
    """
    Verify that ProcessingPipeline only loads the required detector(s) based on config.target
    to optimize PC/Mac CPU, GPU, NPU, and RAM load.
    """
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if not (models_dir / "scrfd_2.5g_bnkps.onnx").exists() or not (models_dir / "license-plate-finetune-v1n.onnx").exists():
        pytest.skip("Models not present for selective loading test")

    # 1. Faces Only -> detector loaded, plate_detector is None
    cfg_faces = AppConfig()
    cfg_faces.target = "faces"
    cfg_faces.model.name = "scrfd_2.5g_bnkps.onnx"
    pipe_faces = ProcessingPipeline(cfg_faces, model_dir=str(models_dir))
    assert pipe_faces.detector is not None
    assert pipe_faces.plate_detector is None

    # 2. Plates Only -> detector is None, plate_detector loaded
    cfg_plates = AppConfig()
    cfg_plates.target = "plates"
    pipe_plates = ProcessingPipeline(cfg_plates, model_dir=str(models_dir))
    assert pipe_plates.detector is None
    assert pipe_plates.plate_detector is not None

    # 3. Both -> both detector and plate_detector loaded
    cfg_both = AppConfig()
    cfg_both.target = "both"
    cfg_both.model.name = "scrfd_2.5g_bnkps.onnx"
    pipe_both = ProcessingPipeline(cfg_both, model_dir=str(models_dir))
    assert pipe_both.detector is not None
    assert pipe_both.plate_detector is not None


def test_pipeline_plates_only_and_both_e2e(tmp_path):
    video_path = tmp_path / "car_walk.mp4"
    fps = 30
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for _ in range(10):
        frame = np.full((height, width, 3), 100, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # Plates-only run with mocked plate_detector
    cfg = AppConfig()
    cfg.target = "plates"
    cfg.output.codec = "h264"
    cfg.output.preset = "ultrafast"
    pipeline = ProcessingPipeline(cfg)

    mock_plate_det = MagicMock()
    mock_plate_det.active_provider = "CPU"
    mock_plate_det.detect.return_value = [
        FaceDetection(bbox=np.array([80.0, 120.0, 160.0, 160.0]), score=0.91, label="plate")
    ]
    pipeline.plate_detector = mock_plate_det

    preview_statuses = []
    out_plates = tmp_path / "car_walk_plates.mp4"
    res = pipeline.process_video(
        str(video_path),
        output_path=str(out_plates),
        preview_callback=lambda f, s="": preview_statuses.append(s)
    )
    assert res["status"] == "success"
    assert out_plates.exists()
    assert mock_plate_det.detect.call_count == 10


def test_checkpoint_invalidated_when_target_mode_changes(tmp_path):
    in_video = tmp_path / "sample.mp4"
    in_video.write_bytes(b"video bytes content")
    out_video = tmp_path / "sample_blurred.mp4"

    chk = VideoCheckpoint.for_video(out_video)
    chk.init_state(
        in_video,
        total_frames=30,
        width=640,
        height=360,
        fps=30.0,
        model_name="scrfd_10g_bnkps.onnx",
        conf_threshold=0.5,
        target_mode="faces"
    )

    assert chk.matches_input(in_video, model_name="scrfd_10g_bnkps.onnx", conf_threshold=0.5, target_mode="faces")
    assert not chk.matches_input(in_video, model_name="scrfd_10g_bnkps.onnx", conf_threshold=0.5, target_mode="both")
    assert not chk.matches_input(in_video, model_name="scrfd_10g_bnkps.onnx", conf_threshold=0.5, target_mode="plates")
