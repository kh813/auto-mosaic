"""
Configuration loading and validation module for auto-mosaic.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

@dataclass
class ModelConfig:
    name: str = "scrfd_10g_bnkps.onnx"
    conf_threshold: float = 0.5
    nms_threshold: float = 0.4
    min_face_size: int = 15
    landmark_score_threshold: float = 0.3

@dataclass
class PlateModelConfig:
    name: str = "license-plate-finetune-v1n.onnx"
    vehicle_model: str = "yolo11n.onnx"
    require_vehicle: bool = True  # First detect vehicle, only blur plates on detected vehicles
    conf_threshold: float = 0.40
    nms_threshold: float = 0.45
    min_plate_size: int = 12
    margin_x: float = 0.15
    margin_y: float = 0.15
    shape: str = "rounded_rect"  # "rounded_rect" (recommended for plates), "ellipse", or "rect"

@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_missing_frames: int = 16
    padding_backward: int = 12
    padding_forward: int = 24
    auto_scale_by_fps: bool = True
    base_fps: float = 30.0

@dataclass
class BlurConfig:
    type: str = "gaussian"  # "gaussian" or "mosaic"
    strength: int = 71      # Gaussian blur kernel size (odd, enhanced default for stronger anonymity)
    mosaic_block_size: int = 36  # Mosaic block pixel size
    margin_x: float = 0.55  # Safe boundary margins to prevent exposure during body sway
    margin_y: float = 0.55
    shape: str = "ellipse"  # "ellipse" (round) or "rect"

@dataclass
class AnimalFilterConfig:
    enabled: bool = True
    human_prob_threshold: float = 0.5

@dataclass
class SkinColorFilterConfig:
    enabled: bool = True
    min_skin_ratio: float = 0.15

@dataclass
class IllustrationFilterConfig:
    enabled: bool = True
    flatness_threshold: float = 0.65
    min_crop_size: int = 16

@dataclass
class StaticPhotoFilterConfig:
    enabled: bool = False
    min_static_frames: int = 30
    motion_similarity_threshold: float = 0.95
    texture_diff_threshold: float = 0.05

@dataclass
class FiltersConfig:
    animal_filter: AnimalFilterConfig = field(default_factory=AnimalFilterConfig)
    skin_color_filter: SkinColorFilterConfig = field(default_factory=SkinColorFilterConfig)
    illustration_filter: IllustrationFilterConfig = field(default_factory=IllustrationFilterConfig)
    static_photo_filter: StaticPhotoFilterConfig = field(default_factory=StaticPhotoFilterConfig)
    exclusion_masks: List[List[float]] = field(default_factory=list)

@dataclass
class OutputConfig:
    codec: str = "auto"  # "auto" (match input video), "hevc", or "h264"
    crf: int = 23
    preset: str = "medium"
    bit_depth: str = "auto"  # "auto" (match input video), "8bit", "10bit"
    preserve_color_tags: bool = True
    output_dir: str = ""
    filename_suffix: str = "_blurred"
    append_params_to_filename: bool = False


@dataclass
class AppConfig:
    target: str = "faces"  # Options: "faces", "plates", "both"
    model: ModelConfig = field(default_factory=ModelConfig)
    plate_model: PlateModelConfig = field(default_factory=PlateModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    blur: BlurConfig = field(default_factory=BlurConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def detect_faces(self) -> bool:
        t = (self.target or "faces").strip().lower()
        return t in ("faces", "face", "both", "all")

    @property
    def detect_plates(self) -> bool:
        t = (self.target or "faces").strip().lower()
        return t in ("plates", "plate", "license_plate", "license_plates", "both", "all")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        cfg = cls()
        if not data:
            return cfg

        if "target" in data and isinstance(data["target"], str):
            t_val = data["target"].strip().lower()
            if t_val in ("faces", "face"):
                cfg.target = "faces"
            elif t_val in ("plates", "plate", "license_plate", "license_plates"):
                cfg.target = "plates"
            elif t_val in ("both", "all"):
                cfg.target = "both"

        if "model" in data and isinstance(data["model"], dict):
            cfg.model = ModelConfig(**{k: v for k, v in data["model"].items() if hasattr(ModelConfig, k)})

        if "plate_model" in data and isinstance(data["plate_model"], dict):
            cfg.plate_model = PlateModelConfig(**{k: v for k, v in data["plate_model"].items() if hasattr(PlateModelConfig, k)})

        if "tracking" in data and isinstance(data["tracking"], dict):
            cfg.tracking = TrackingConfig(**{k: v for k, v in data["tracking"].items() if hasattr(TrackingConfig, k)})

        if "blur" in data and isinstance(data["blur"], dict):
            cfg.blur = BlurConfig(**{k: v for k, v in data["blur"].items() if hasattr(BlurConfig, k)})

        if "filters" in data and isinstance(data["filters"], dict):
            f_data = data["filters"]
            animal_cfg = AnimalFilterConfig(**f_data.get("animal_filter", {})) if "animal_filter" in f_data else AnimalFilterConfig()
            skin_cfg = SkinColorFilterConfig(**f_data.get("skin_color_filter", {})) if "skin_color_filter" in f_data else SkinColorFilterConfig()
            illus_cfg = IllustrationFilterConfig(**f_data.get("illustration_filter", {})) if "illustration_filter" in f_data else IllustrationFilterConfig()
            static_cfg = StaticPhotoFilterConfig(**f_data.get("static_photo_filter", {})) if "static_photo_filter" in f_data else StaticPhotoFilterConfig()
            masks = f_data.get("exclusion_masks", [])
            cfg.filters = FiltersConfig(
                animal_filter=animal_cfg,
                skin_color_filter=skin_cfg,
                illustration_filter=illus_cfg,
                static_photo_filter=static_cfg,
                exclusion_masks=masks
            )

        if "output" in data and isinstance(data["output"], dict):
            cfg.output = OutputConfig(**{k: v for k, v in data["output"].items() if hasattr(OutputConfig, k)})

        return cfg

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "AppConfig":
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)

        # Look for default configs
        default_yaml = Path(__file__).resolve().parent.parent / "configs" / "config.default.yaml"
        if default_yaml.exists():
            with open(default_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)

        return cls()
