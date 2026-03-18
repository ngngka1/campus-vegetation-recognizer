#!/usr/bin/env python3
"""
Extract JPEG frames from .MOV files in raw_grouped into extracted_grouped.

The script mirrors the directory structure of raw_grouped and writes frames
every N milliseconds (configurable by CLI).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
import numpy as np
import cv2

def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent / "polyu_vegetation"

    parser = argparse.ArgumentParser(
        description=(
            "Mirror raw_grouped folder structure and extract frames from .MOV "
            "files into extracted_grouped every N milliseconds."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=(
            "Root path containing raw_grouped "
            "(default: <script_dir>/polyu_vegetation)."
        ),
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        required=True,
        help="Frame extraction interval in milliseconds (e.g., 200).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 0 to 100 (default: 95).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing .jpeg/.jpg files under extracted_grouped before extraction.",
    )
    return parser.parse_args()


def extract_frames_from_video(
    video_path: Path, out_dir: Path, interval_ms: int, jpeg_quality: int
) -> int:

    def open_capture(path: Path):
        capture = cv2.VideoCapture(str(path))
        temp_copy = None
        if capture.isOpened():
            return capture, temp_copy

        # Some OpenCV builds on Windows struggle with non-ASCII file paths.
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=path.suffix) as tf:
                temp_copy = Path(tf.name)
            shutil.copy2(path, temp_copy)
            capture = cv2.VideoCapture(str(temp_copy))
            if capture.isOpened():
                print(f"[INFO] Opened via temp copy for Unicode path: {path}")
                return capture, temp_copy
        except Exception:
            pass

        if temp_copy is not None and temp_copy.exists():
            temp_copy.unlink(missing_ok=True)
        return capture, None

    def write_jpeg(path: Path, image, quality: int) -> bool:
        # Use imencode + tofile for Unicode-safe writes on Windows.
        ok, encoded = cv2.imencode(
            ".jpeg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            return False
        try:
            encoded.astype(np.uint8).tofile(str(path))
            return True
        except Exception:
            return False

    cap, temp_video_copy = open_capture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0

    stem = video_path.stem
    saved = 0
    saved_idx = 0
    read_idx = 0
    next_capture_ms = 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_is_valid = fps and fps > 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read_idx += 1

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp_ms <= 0 and fps_is_valid:
            timestamp_ms = (read_idx * 1000.0) / fps

        if timestamp_ms + 1e-6 >= next_capture_ms:
            out_file = out_dir / f"{stem}_frame_{saved_idx:06d}.jpeg"
            if write_jpeg(out_file, frame, jpeg_quality):
                saved += 1
                saved_idx += 1
            else:
                print(f"[WARN] Failed to write JPEG: {out_file}")
            next_capture_ms += interval_ms

    cap.release()
    if temp_video_copy is not None and temp_video_copy.exists():
        temp_video_copy.unlink(missing_ok=True)
    return saved


def main() -> None:
    args = parse_args()

    if args.interval_ms <= 0:
        raise ValueError("--interval-ms must be > 0")
    if not (0 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be in [0, 100]")

    root = args.root.resolve()
    raw_root = root / "raw_grouped"
    out_root = root / "extracted_grouped"

    if not raw_root.exists() or not raw_root.is_dir():
        raise FileNotFoundError(f"raw_grouped not found: {raw_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    if args.clean:
        removed = 0
        for p in out_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg"}:
                p.unlink(missing_ok=True)
                removed += 1
        print(f"[INFO] --clean removed {removed} old extracted JPEG files.")

    total_videos = 0
    total_frames = 0
    total_copied_jpegs = 0

    # Mirror full directory structure under raw_grouped.
    for source_dir in [d for d in raw_root.rglob("*") if d.is_dir()]:
        rel_dir = source_dir.relative_to(raw_root)
        target_dir = out_root / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)

    # Include raw_root itself (for videos at top level).
    all_dirs = [raw_root] + [d for d in raw_root.rglob("*") if d.is_dir()]

    for source_dir in all_dirs:
        rel_dir = source_dir.relative_to(raw_root)
        target_dir = out_root / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        raw_jpegs = sorted(
            p
            for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg"}
        )
        for jpeg_path in raw_jpegs:
            target_path = target_dir / jpeg_path.name
            shutil.copy2(jpeg_path, target_path)
            total_copied_jpegs += 1

        mov_files = sorted(
            p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mov"
        )
        if not mov_files:
            continue

        for mov_path in mov_files:
            total_videos += 1
            print(f"[INFO] Processing: {mov_path}")
            saved = extract_frames_from_video(
                video_path=mov_path,
                out_dir=target_dir,
                interval_ms=args.interval_ms,
                jpeg_quality=args.jpeg_quality,
            )
            total_frames += saved
            print(f"[INFO] Saved {saved} frames -> {target_dir}")

    print(
        f"[DONE] Finished. Videos processed: {total_videos}, "
        f"frames extracted: {total_frames}, raw JPEGs copied: {total_copied_jpegs}, "
        f"output: {out_root}"
    )


if __name__ == "__main__":
    main()
