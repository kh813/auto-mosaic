import numpy as np
import pytest
from pathlib import Path
from face_mosaic.blur import FaceBlurrer
from face_mosaic.vehicle_detector import VehicleDetector, VehicleDetection
from face_mosaic.config import AppConfig
from face_mosaic.pipeline import ProcessingPipeline

def test_rounded_rect_feathered_mask():
    """Verify that FaceBlurrer generates smooth feathered edges with rounded corners for license plates."""
    blurrer = FaceBlurrer(shape="ellipse", plate_shape="rounded_rect")
    
    # 50x100 plate
    mask = blurrer.generate_feathered_mask(50, 100, "rounded_rect")

    # Center is solid opaque
    assert mask[25, 50] == 1.0

    # The 4 sharp 90-deg outer corners must have 0 or near-0 opacity (rounded corner effect)
    assert mask[0, 0] < 0.05
    assert mask[0, 99] < 0.05
    assert mask[49, 0] < 0.05
    assert mask[49, 99] < 0.05

    # Interior edge feathering is soft (between 0.0 and 1.0)
    assert 0.0 < mask[25, 0] < 1.0

def test_blurrer_applies_rounded_rect_to_plates():
    """Verify FaceBlurrer applies rounded_rect to plate bboxes."""
    blurrer = FaceBlurrer(
        blur_type="gaussian",
        strength=25,
        margin_x=0.0,
        margin_y=0.0,
        plate_margin_x=0.0,
        plate_margin_y=0.0,
        plate_shape="rounded_rect"
    )

    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    plate_bbox = np.array([50.0, 50.0, 150.0, 110.0, 1.0], dtype=np.float32) # plate bbox (5 elements)

    out, mask = blurrer.apply_blur(frame, [plate_bbox])

    # Center of plate is blurred (mask > 200)
    assert mask[80, 100] > 200

    # Sharp outer corner (50, 50) is softly feathered / rounded
    assert mask[50, 50] < 100
    # Far outside is 0
    assert mask[10, 10] == 0

def test_vehicle_plate_spatial_validation():
    """Verify VehicleDetector.is_plate_on_vehicle logic."""
    orig_h = 1080
    orig_w = 1920

    # Car in lower half: x=[400, 1000], y=[500, 900]
    car = VehicleDetection(
        bbox=np.array([400.0, 500.0, 1000.0, 900.0], dtype=np.float32),
        score=0.92,
        cls_id=2,
        label="car"
    )

    # 1. Plate on car: center at (700, 850) -> Valid
    plate_on_car = np.array([650.0, 820.0, 750.0, 880.0], dtype=np.float32)
    assert VehicleDetector.is_plate_on_vehicle(plate_on_car, [car], orig_h) is True

    # 2. Plate in upper half (station board / overhead sign at y=250): Must be rejected
    overhead_plate = np.array([650.0, 200.0, 750.0, 300.0], dtype=np.float32)
    assert VehicleDetector.is_plate_on_vehicle(overhead_plate, [car], orig_h) is False

    # 3. Plate in lower half but away from vehicle (platform sign at x=1400..1500, y=700..800): Must be rejected
    sign_no_car = np.array([1400.0, 700.0, 1500.0, 800.0], dtype=np.float32)
    assert VehicleDetector.is_plate_on_vehicle(sign_no_car, [car], orig_h) is False

    # 4. No vehicles detected in frame at all: All plates rejected
    assert VehicleDetector.is_plate_on_vehicle(plate_on_car, [], orig_h) is False

def test_vehicle_detector_inference_if_model_present():
    """Verify VehicleDetector runs inference correctly if yolo11n.onnx is present."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "yolo11n.onnx"
    if not model_path.exists():
        pytest.skip("yolo11n.onnx not present")

    vd = VehicleDetector(str(model_path))
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = vd.detect(dummy_frame)
    assert isinstance(dets, list)
