"""
Command Line Interface for auto-mosaic.
Handles single file, multiple files, and recursive folder batch processing.
Supports skip-on-error behavior and outputs summary reports.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List
from tqdm import tqdm

from .config import AppConfig
from .pipeline import ProcessingPipeline

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

def collect_video_files(
    inputs: List[str],
    recursive: bool = True,
    exclude_suffix: Optional[str] = None
) -> List[Path]:
    """
    Collect video files from single files, list of files, and directories.
    Skips previously processed output files matching exclude_suffix or '_blurred'.
    """
    collected = []
    for item in inputs:
        p = Path(item).resolve()
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                collected.append(p)
            else:
                print(f"[Warning] Skipping non-video file: {p.name}")
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for sub_p in p.glob(pattern):
                if sub_p.is_file() and sub_p.suffix.lower() in VIDEO_EXTENSIONS:
                    # Skip already blurred output files to avoid infinite loops
                    stem = sub_p.stem
                    if "_blurred" in stem:
                        continue
                    if exclude_suffix and exclude_suffix in stem:
                        continue
                    collected.append(sub_p.resolve())
        else:
            print(f"[Warning] Path does not exist: {item}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in collected:
        if str(f) not in seen:
            seen.add(str(f))
            unique_files.append(f)
            
    return unique_files

def main():
    parser = argparse.ArgumentParser(
        description="auto-mosaic: Automated offline face & license plate blurring/mosaic tool for videos"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="Input video file(s) or folder(s) to process"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical user interface (GUI)"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run in folder watch mode (continuously monitor input folder for new videos)"
    )


    parser.add_argument(
        "--target", "-t",
        choices=["faces", "plates", "both"],
        default=None,
        help="Detection target mode to optimize PC/Mac load: faces (faces only), plates (license plates only), or both"
    )
    parser.add_argument(
        "--faces-only",
        action="store_true",
        help="Shortcut for --target faces (blur human faces only)"
    )
    parser.add_argument(
        "--plates-only",
        action="store_true",
        help="Shortcut for --target plates (blur vehicle license plates only)"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Shortcut for --target both (blur both faces and license plates)"
    )
    parser.add_argument(
        "--plate-conf",
        type=float,
        default=None,
        help="License plate detection confidence threshold (0.0 - 1.0, default: 0.40)"
    )
    parser.add_argument(
        "--plate-model",
        type=str,
        default=None,
        help="License plate ONNX model name (default: license-plate-finetune-v1n.onnx)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="SCRFD model name (e.g. scrfd_2.5g_bnkps.onnx, scrfd_10g_bnkps.onnx, scrfd_34g_gnkps.onnx)"
    )
    parser.add_argument(
        "--blur-type", "-b",
        choices=["gaussian", "mosaic"],
        default=None,
        help="Blur algorithm (gaussian or mosaic)"
    )
    parser.add_argument(
        "--strength", "-s",
        type=int,
        default=None,
        help="Gaussian blur strength (odd int) or mosaic block size"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold (0.0 - 1.0)"
    )
    parser.add_argument(
        "--pad-back",
        type=int,
        default=None,
        help="Frames to pad backward before face appearance (default: 12)"
    )
    parser.add_argument(
        "--pad-fwd",
        type=int,
        default=None,
        help="Frames to pad forward after face disappearance (default: 24)"
    )
    parser.add_argument(
        "--codec",
        choices=["auto", "hevc", "h264"],
        default=None,
        help="Video encoder codec: auto (match input video format), hevc, or h264"
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="Constant Rate Factor (default: 23, lower means higher quality/larger size)"
    )
    parser.add_argument(
        "--bit-depth",
        choices=["auto", "8bit", "10bit"],
        default=None,
        help="Output bit depth: auto (match input), 8bit (reduce size & maximize compatibility), 10bit"
    )
    parser.add_argument(
        "--append-params",
        action="store_true",
        help="Append blur and quality parameters to output filename (e.g. _(G51-M50-CRF23))"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--shape",
        choices=["ellipse", "rect"],
        default=None,
        help="Blur/mosaic shape (ellipse for round, rect for rectangular)"
    )
    parser.add_argument(
        "--no-illustration-filter",
        action="store_true",
        help="Disable automatic illustration / anime face exclusion"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search in directories"
    )
    parser.add_argument(
        "--check-update",
        action="store_true",
        help="Check GitHub for newer versions of auto-mosaic"
    )
    parser.add_argument(
        "--update", "--self-update",
        action="store_true",
        dest="self_update",
        help="Perform self-update to the latest version"
    )

    args = parser.parse_args()

    # Self-update handling
    if args.check_update:
        from .updater import check_for_updates
        from face_mosaic import __version__
        print(f"Current version: v{__version__}")
        print("Checking GitHub for updates...")
        info = check_for_updates()
        if info:
            print(f"\n🎉 New version available: {info['tag_name']}")
            print(f"Release: {info['title']}")
            print(f"URL: {info['html_url']}")
            print("\nRun 'python -m face_mosaic.cli --update' to update.")
        else:
            print("✓ You are already using the latest version.")
        return

    if args.self_update:
        from .updater import perform_self_update
        from face_mosaic import __version__
        print(f"Current version: v{__version__}")
        print("Starting self-update...")
        def on_prog(msg: str, cur: int, tot: int):
            print(f"  [{cur}%] {msg}")
        success, message = perform_self_update(on_prog)
        if success:
            print(f"\n✓ {message}")
        else:
            print(f"\n✗ {message}")
            sys.exit(1)
        return

    if args.gui or len(args.inputs) == 0:
        from .gui import run_app
        run_app()
        return

    # Load configuration

    config = AppConfig.load(args.config)

    # CLI Overrides
    if args.target:
        config.target = args.target
    elif args.faces_only:
        config.target = "faces"
    elif args.plates_only:
        config.target = "plates"
    elif args.both:
        config.target = "both"

    if args.plate_conf is not None:
        config.plate_model.conf_threshold = args.plate_conf
    if args.plate_model:
        config.plate_model.name = args.plate_model

    if args.model:
        config.model.name = args.model
    elif args.config is None and config.detect_faces:
        from .hardware import detect_hardware_profile
        hw = detect_hardware_profile()
        config.model.name = hw.recommended_model
        print(f"[Hardware Auto-Select] {hw.reason}")
    if args.blur_type:
        config.blur.type = args.blur_type
    if args.strength:
        if config.blur.type == "mosaic":
            config.blur.mosaic_block_size = args.strength
        else:
            config.blur.strength = args.strength
    if args.conf is not None:
        config.model.conf_threshold = args.conf
    if args.pad_back is not None:
        config.tracking.padding_backward = args.pad_back
    if args.pad_fwd is not None:
        config.tracking.padding_forward = args.pad_fwd
    if args.codec:
        config.output.codec = args.codec
    if args.crf is not None:
        config.output.crf = args.crf
    if args.bit_depth:
        config.output.bit_depth = args.bit_depth
    if args.append_params:
        config.output.append_params_to_filename = True
    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.shape:
        config.blur.shape = args.shape
    if args.no_illustration_filter:
        config.filters.illustration_filter.enabled = False

    # Folder Watch Mode
    if args.watch:
        from .watcher import FolderWatcher
        target_dir = args.inputs[0] if args.inputs else "."
        print(f"Starting Folder Watcher on: {target_dir}")
        watcher = FolderWatcher(target_dir, config=config)
        watcher.start()
        return

    video_files = collect_video_files(
        args.inputs,
        recursive=not args.no_recursive,
        exclude_suffix=config.output.filename_suffix
    )

    if not video_files:
        print("Error: No video files found matching the specified inputs.")
        sys.exit(1)

    print(f"Found {len(video_files)} video file(s) for batch processing.")

    try:
        pipeline = ProcessingPipeline(config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)

    success_list = []
    failed_list = []

    for idx, video_path in enumerate(video_files, 1):
        print(f"\n=======================================================")
        print(f"[{idx}/{len(video_files)}] Processing: {video_path.name}")
        print(f"=======================================================")
        
        # Progress callback with tqdm
        pbar = None
        current_phase = [None]
        
        def on_progress(phase: str, current: int, total: int, elapsed: float):
            nonlocal pbar
            if current_phase[0] != phase:
                if pbar:
                    pbar.close()
                current_phase[0] = phase
                pbar = tqdm(total=total, desc=f"  {phase}", unit="frame")
            if pbar:
                pbar.n = current
                pbar.refresh()

        try:
            res = pipeline.process_video(str(video_path), progress_callback=on_progress)
            if pbar:
                pbar.close()
            success_list.append(res)
        except Exception as e:
            if pbar:
                pbar.close()
            print(f"\n[Error] Failed to process {video_path.name}: {e}")
            # Spec requirement: continue on error
            failed_list.append({
                "input": str(video_path),
                "error": str(e),
                "status": "failed"
            })

    # Summary Report
    print("\n=======================================================")
    print("                BATCH PROCESSING SUMMARY               ")
    print("=======================================================")
    print(f"Total videos queued: {len(video_files)}")
    print(f"Successfully processed: {len(success_list)}")
    print(f"Failed / Skipped: {len(failed_list)}")

    if success_list:
        print("\nSuccessful files:")
        for s in success_list:
            print(f"  ✓ {Path(s['input']).name} -> {Path(s['output']).name} ({s['elapsed_seconds']:.1f}s, {s['processing_fps']:.1f} fps)")

    if failed_list:
        print("\nFailed files:")
        for f in failed_list:
            print(f"  ✗ {Path(f['input']).name}: {f['error']}")

    print("=======================================================\n")
    if failed_list and not success_list:
        sys.exit(1)

if __name__ == "__main__":
    main()
