#!/usr/bin/env python3
"""
Model download script for auto-mosaic.
Downloads SCRFD ONNX face detection models and YOLOv11n ONNX license plate detection model.
Works identically on macOS and Windows.
"""

import os
import sys
from pathlib import Path
import urllib.request

MODELS = {
    "scrfd_2.5g_bnkps.onnx": {
        "url": "https://huggingface.co/hsuyabc/scrfd_2.5g_bnkps.onnx/resolve/main/scrfd_2.5g_bnkps.onnx",
        "fallback_url": "https://github.com/hpc203/scrfd-opencv/raw/main/weights/scrfd_2.5g_bnkps.onnx",
        "description": "SCRFD 2.5G lightweight model with keypoints (Fast)",
        "expected_min_size": 3_000_000,
    },
    "scrfd_10g_bnkps.onnx": {
        "url": "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
        "fallback_url": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx",
        "description": "SCRFD 10G standard model with keypoints (Default / Recommended)",
        "expected_min_size": 16_000_000,
    },
    "scrfd_34g_gnkps.onnx": {
        "url": "https://huggingface.co/immich-app/scrfd_34g_gnkps/resolve/main/detection/model.onnx",
        "fallback_url": None,
        "description": "SCRFD 34G high-accuracy model with keypoints",
        "expected_min_size": 25_000_000,
    },
    "license-plate-finetune-v1n.onnx": {
        "url": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.onnx",
        "fallback_url": "https://huggingface.co/wesleyerick/brazilian-license-plate-detector-yolov11-onnx/resolve/main/best.onnx",
        "description": "YOLOv11n License Plate Detection ONNX model (Fast & Accurate)",
        "expected_min_size": 8_000_000,
    },
}

def download_file(url: str, target_path: Path, fallback_url: str = None, min_size: int = 1_000_000) -> bool:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    
    urls_to_try = [url]
    if fallback_url:
        urls_to_try.append(fallback_url)
        
    for current_url in urls_to_try:
        try:
            print(f"Downloading {target_path.name} from {current_url}...")
            req = urllib.request.Request(
                current_url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            )
            with urllib.request.urlopen(req, timeout=60) as response, open(tmp_path, "wb") as out_file:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                block_size = 128 * 1024
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        print(f"\r  [{downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB] ({percent:.1f}%)", end="", flush=True)
                print()
            
            if tmp_path.stat().st_size < min_size:
                raise ValueError(f"Downloaded file too small: {tmp_path.stat().st_size} bytes")
                
            tmp_path.replace(target_path)
            print(f"Successfully saved to {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
            return True
        except Exception as e:
            print(f"\nFailed to download from {current_url}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
                
    return False

def main():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    success_count = 0
    
    for name in requested:
        if name not in MODELS:
            print(f"Warning: Unknown model '{name}'. Skipping.")
            continue
        
        info = MODELS[name]
        target = models_dir / name
        min_size = info.get("expected_min_size", 1_000_000)
        
        if target.exists() and target.stat().st_size >= min_size:
            print(f"Model already exists: {target.name} ({target.stat().st_size / (1024*1024):.2f} MB)")
            success_count += 1
            continue
            
        print(f"\nFetching {name}: {info['description']}")
        ok = download_file(info["url"], target, fallback_url=info.get("fallback_url"), min_size=min_size)
        if ok:
            success_count += 1
        else:
            print(f"Failed to fetch {name}.")
            
    print(f"\nModel setup complete: {success_count}/{len(requested)} models ready in {models_dir}")

if __name__ == "__main__":
    main()
