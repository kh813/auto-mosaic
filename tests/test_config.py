import pytest
from pathlib import Path
import tempfile
import yaml
from face_mosaic.config import AppConfig

def test_default_config():
    config = AppConfig()
    assert config.target == "faces"
    assert config.detect_faces is True
    assert config.detect_plates is False
    assert config.model.name == "scrfd_10g_bnkps.onnx"
    assert config.plate_model.name == "license-plate-finetune-v1n.onnx"
    assert config.plate_model.conf_threshold == pytest.approx(0.40)
    assert config.plate_model.shape == "rounded_rect"
    assert config.blur.type == "gaussian"
    assert config.tracking.padding_backward == 12
    assert config.tracking.padding_forward == 24
    assert config.tracking.max_missing_frames == 16
    assert config.output.codec == "auto"
    assert config.output.crf == 23
    assert config.output.bit_depth == "auto"
    assert config.output.append_params_to_filename is False

def test_load_yaml_config():
    yaml_data = {
        "model": {
            "name": "scrfd_2.5g_bnkps.onnx",
            "conf_threshold": 0.6
        },
        "blur": {
            "type": "mosaic",
            "mosaic_block_size": 20
        },
        "tracking": {
            "padding_backward": 5,
            "padding_forward": 5
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_data, f)
        temp_path = f.name

    try:
        loaded = AppConfig.load(temp_path)
        assert loaded.model.name == "scrfd_2.5g_bnkps.onnx"
        assert loaded.model.conf_threshold == 0.6
        assert loaded.blur.type == "mosaic"
        assert loaded.blur.mosaic_block_size == 20
        assert loaded.tracking.padding_backward == 5
        assert loaded.tracking.padding_forward == 5
    finally:
        Path(temp_path).unlink()
