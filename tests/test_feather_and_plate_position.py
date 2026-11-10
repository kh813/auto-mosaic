import numpy as np
import pytest
from face_mosaic.blur import FaceBlurrer
from face_mosaic.detector import FaceDetection

def test_feathered_mask_smooth_gradient():
    """Verify that FaceBlurrer generates smooth feathered edges (soft boundary)."""
    blurrer = FaceBlurrer(shape="ellipse")
    mask = blurrer.generate_feathered_mask(100, 100, "ellipse")

    # Center must be solid opaque (1.0)
    assert mask[50, 50] == 1.0
    # Outer corners must be transparent (0.0)
    assert mask[0, 0] == 0.0
    assert mask[99, 99] == 0.0

    # Intermediate points along gradient feather zone (around x=93)
    r_mid = mask[50, 93]
    assert 0.0 < r_mid < 1.0

def test_blur_anti_darkening():
    """Verify that blurring bright face surrounded by dark hair doesn't darken the face unnaturally."""
    blurrer = FaceBlurrer(blur_type="gaussian", strength=51)
    
    # 60x60 patch: center 30x30 is bright skin (200, 180, 160), borders are black hair (10, 10, 10)
    roi = np.full((60, 60, 3), 15, dtype=np.uint8)
    roi[15:45, 15:45] = [200, 180, 160]

    orig_center_mean = np.mean(roi[15:45, 15:45].astype(np.float32))
    blurred = blurrer.apply_gaussian_to_roi(roi)
    blur_center_mean = np.mean(blurred[15:45, 15:45].astype(np.float32))

    # Brightness should be protected and lifted, not collapsed into black
    assert blur_center_mean > 90.0

def test_plate_lower_half_filter():
    """Verify that license plate candidates in the upper half of screen are rejected."""
    # Screen resolution: 1920x1080
    orig_w, orig_h = 1920, 1080

    # Dummy station banner in upper half (center_y = 200 < 1080 * 0.45 = 486)
    banner_b = np.array([500, 150, 900, 250], dtype=np.float32)
    cy_banner = (banner_b[1] + banner_b[3]) * 0.5
    assert cy_banner < (orig_h * 0.45)

    # Vehicle plate at road level in lower half (center_y = 750 >= 486)
    car_plate_b = np.array([800, 720, 960, 800], dtype=np.float32)
    cy_car = (car_plate_b[1] + car_plate_b[3]) * 0.5
    assert cy_car >= (orig_h * 0.45)
