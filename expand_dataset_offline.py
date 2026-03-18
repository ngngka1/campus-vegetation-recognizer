#!/usr/bin/env python3
"""
Expand extracted vegetation dataset with offline augmentations.

Default behavior:
- Source: polyu_vegetation/extracted_grouped_1000ms (fallback: extracted_grouped)
- Target: polyu_vegetation/extracted_group_augmented
- Expansion fold: 5.0x (set 4.0 to get 4x, etc.)

This script copies original images and writes augmented images to disk.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from data_augmentation import IMAGE_SUFFIXES

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_data_root = project_root / "polyu_vegetation"
    parser = argparse.ArgumentParser(
        description=(
            "Create an offline-augmented dataset under extracted_group_augmented "
            "by expanding each class by a target fold."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="Root containing extracted_grouped/extracted_grouped_1000ms.",
    )
    parser.add_argument(
        "--source-subdir",
        type=str,
        default="auto",
        help=(
            "Source folder under data-root. Use 'auto' to prefer "
            "extracted_grouped_1000ms, then extracted_grouped."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="extracted_group_augmented",
        help="Output folder name under data-root.",
    )
    parser.add_argument(
        "--target-fold",
        type=float,
        default=5.0,
        help="Target total fold per class (e.g. 4.0 or 5.0).",
    )
    parser.add_argument(
        "--equalize-classes",
        action="store_true",
        help=(
            "Equalize classes to a shared target count computed as "
            "<base class count> * target-fold."
        ),
    )
    parser.add_argument(
        "--equalize-base",
        type=str,
        default="min",
        choices=["min", "median", "max"],
        help=(
            "Base used for equalization target when --equalize-classes is enabled. "
            "Example: base=min and fold=4 with class sizes [20,60] -> target 80 for all classes."
        ),
    )
    parser.add_argument(
        "--allow-downsample-for-equalize",
        action="store_true",
        help=(
            "Allow downsampling originals when a class has more than equalization target. "
            "Disabled by default (script raises an error instead)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for output images (0-100).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before generating new data.",
    )
    return parser.parse_args()


def resolve_source_dir(data_root: Path, source_subdir: str) -> Path:
    root = data_root.resolve()
    if source_subdir != "auto":
        source = root / source_subdir
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"Source dataset not found: {source}")
        return source

    candidates = [root / "extracted_grouped_1000ms", root / "extracted_grouped"]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(
        "No source dataset found. Expected extracted_grouped_1000ms or extracted_grouped."
    )


def imread_unicode(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 95) -> None:
    ok, encoded = cv2.imencode(
        ".jpeg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.astype(np.uint8).tofile(str(path))


def random_resized_crop(
    img: np.ndarray,
    rng: random.Random,
    *,
    min_side_fraction: float = 0.15,
    area_range: tuple[float, float] = (0.35, 1.00),
    ratio_range: tuple[float, float] = (0.65, 1.50),
) -> np.ndarray:
    h, w = img.shape[:2]
    area = h * w
    min_h = max(1, int(round(h * min_side_fraction)))
    min_w = max(1, int(round(w * min_side_fraction)))
    for _ in range(10):
        target = rng.uniform(area_range[0], area_range[1]) * area
        ratio = rng.uniform(ratio_range[0], ratio_range[1])
        crop_w = int(round(np.sqrt(target * ratio)))
        crop_h = int(round(np.sqrt(target / ratio)))
        if crop_w <= w and crop_h <= h and crop_h >= min_h and crop_w >= min_w:
            y = rng.randint(0, h - crop_h) if h > crop_h else 0
            x = rng.randint(0, w - crop_w) if w > crop_w else 0
            return img[y : y + crop_h, x : x + crop_w]

    crop_h = max(min_h, int(round(0.8 * h)))
    crop_w = max(min_w, int(round(0.8 * w)))
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)
    y = (h - crop_h) // 2
    x = (w - crop_w) // 2
    return img[y : y + crop_h, x : x + crop_w]


def augment_image(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    # Very strong geometric distortion/cropping.
    if rng.random() < 1.00:
        cropped = random_resized_crop(
            out,
            rng,
            min_side_fraction=0.10,
            area_range=(0.20, 1.00),
            ratio_range=(0.55, 1.80),
        )
        out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
    # A second crop pass occasionally stacks aggressive framing changes.
    if rng.random() < 0.35:
        cropped = random_resized_crop(
            out,
            rng,
            min_side_fraction=0.12,
            area_range=(0.30, 1.00),
            ratio_range=(0.65, 1.60),
        )
        out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.65:
        out = cv2.flip(out, 1)
    if rng.random() < 0.25:
        out = cv2.flip(out, 0)
    if rng.random() < 0.85:
        angle = rng.uniform(-60.0, 60.0)
        scale = rng.uniform(0.75, 1.25)
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        out = cv2.warpAffine(
            out,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    # Strong illumination changes without hue/saturation shift.
    if rng.random() < 0.85:
        alpha = rng.uniform(0.70, 1.40)  # contrast
        beta = rng.uniform(-45, 45)      # brightness
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.65:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Only adjust V channel so color stays consistent.
        hsv[..., 2] *= rng.uniform(0.55, 1.60)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Random gamma curve (bright/dark non-linear illumination).
    if rng.random() < 0.45:
        gamma = rng.uniform(0.55, 1.85)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lut)
    # Stronger blur/noise corruption.
    if rng.random() < 0.35:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=rng.uniform(0.4, 2.2))
    if rng.random() < 0.30:
        # Apply monochrome noise across channels to avoid color tint shifts.
        noise_map = np.random.normal(
            loc=0.0,
            scale=rng.uniform(6.0, 18.0),
            size=(out.shape[0], out.shape[1], 1),
        )
        out = np.clip(out.astype(np.float32) + noise_map, 0, 255).astype(np.uint8)
    return out


def distribute_aug_counts(n: int, target_fold: float) -> list[int]:
    if n <= 0:
        return []
    if target_fold < 1.0:
        raise ValueError("target_fold must be >= 1.0")
    extra_total = int(round(n * (target_fold - 1.0)))
    base = extra_total // n
    rem = extra_total % n
    counts = [base + (1 if i < rem else 0) for i in range(n)]
    return counts


def distribute_to_target(n: int, target_total: int) -> list[int]:
    """
    Return per-image augmentation counts to reach ``target_total`` samples for a class.

    Output count per image means number of augmented copies for that source image.
    """
    if n <= 0:
        return []
    if target_total < n:
        raise ValueError("target_total must be >= original class size when not downsampling.")
    extra_total = target_total - n
    base = extra_total // n
    rem = extra_total % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def collect_class_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not (0 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be in [0, 100]")

    source_root = resolve_source_dir(args.data_root, args.source_subdir)
    out_root = args.data_root.resolve() / args.output_subdir

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in source_root.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
    if not class_dirs:
        raise ValueError(f"No class folders under source dataset: {source_root}")

    class_to_images: dict[str, list[Path]] = {}
    for class_dir in class_dirs:
        imgs = collect_class_images(class_dir)
        if imgs:
            class_to_images[class_dir.name] = imgs
    if not class_to_images:
        raise ValueError(f"No images found under source dataset: {source_root}")

    equalize_target: int | None = None
    if args.equalize_classes:
        sizes = [len(v) for v in class_to_images.values()]
        if args.equalize_base == "min":
            base_n = min(sizes)
        elif args.equalize_base == "median":
            base_n = int(np.median(np.asarray(sizes)))
        else:
            base_n = max(sizes)
        equalize_target = int(round(base_n * args.target_fold))
        if equalize_target <= 0:
            raise ValueError("Equalize target computed <= 0. Check --target-fold.")
        max_n = max(sizes)
        if max_n > equalize_target and not args.allow_downsample_for_equalize:
            raise ValueError(
                "Equalization target is smaller than at least one class size. "
                "Increase --target-fold / change --equalize-base, or enable "
                "--allow-downsample-for-equalize."
            )

    total_original = 0
    total_augmented = 0
    total_dropped_for_downsample = 0

    for class_name in sorted(class_to_images.keys(), key=str.lower):
        src_images = class_to_images[class_name]

        dst_class = out_root / class_name
        dst_class.mkdir(parents=True, exist_ok=True)

        class_images_for_copy = src_images
        if args.equalize_classes and equalize_target is not None and len(src_images) > equalize_target:
            if not args.allow_downsample_for_equalize:
                raise RuntimeError("Internal guard failed: downsampling required but not allowed.")
            class_images_for_copy = sorted(rng.sample(src_images, equalize_target))
            total_dropped_for_downsample += len(src_images) - len(class_images_for_copy)

        # Copy originals.
        for src in class_images_for_copy:
            dst = dst_class / src.name
            shutil.copy2(src, dst)
        total_original += len(class_images_for_copy)

        # Generate augmented images to hit per-class target.
        if args.equalize_classes and equalize_target is not None:
            per_img_counts = distribute_to_target(len(class_images_for_copy), equalize_target)
        else:
            per_img_counts = distribute_aug_counts(len(class_images_for_copy), args.target_fold)

        iterable = zip(class_images_for_copy, per_img_counts)
        if tqdm is not None:
            iterable = tqdm(
                iterable,
                total=len(class_images_for_copy),
                desc=f"Augmenting [{class_name}]",
                unit="img",
            )
        for src, k in iterable:
            if k <= 0:
                continue
            base = src.stem
            img = imread_unicode(src)
            for idx in range(k):
                aug = augment_image(img, rng)
                out_name = f"{base}__aug_{idx:02d}.jpeg"
                out_path = dst_class / out_name
                # Avoid collisions when source has repeated stems.
                if out_path.exists():
                    salt = rng.randint(1000, 9999)
                    out_name = f"{base}__aug_{idx:02d}_{salt}.jpeg"
                    out_path = dst_class / out_name
                imwrite_unicode(out_path, aug, quality=args.jpeg_quality)
                total_augmented += 1

    total = total_original + total_augmented
    fold = (total / total_original) if total_original > 0 else 0.0
    print("[DONE] Offline augmentation complete.")
    print(f"[INFO] Source: {source_root}")
    print(f"[INFO] Output: {out_root}")
    print(f"[INFO] Original images copied: {total_original}")
    print(f"[INFO] Augmented images generated: {total_augmented}")
    if total_dropped_for_downsample > 0:
        print(f"[INFO] Original images dropped for equalization downsampling: {total_dropped_for_downsample}")
    if args.equalize_classes and equalize_target is not None:
        print(f"[INFO] Equalization target per class: {equalize_target}")
    print(f"[INFO] Total output images: {total} (~{fold:.2f}x)")


if __name__ == "__main__":
    main()

