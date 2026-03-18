#!/usr/bin/env python3
"""
Data augmentation and dataset balancing utilities.

This module provides:
1) Offline balancing by median-target downsampling (no file deletion/copying).
2) Online augmentation pipelines using torchvision transforms.
3) Helpers to reduce data leakage risk for video-frame-derived datasets.
4) Lightweight validation/sanity-check utilities.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def collect_class_samples(
    root_dir: Path | str,
    *,
    recursive: bool = True,
    class_sort: bool = True,
) -> dict[str, list[Path]]:
    """
    Collect image paths from class folders under ``root_dir``.

    Expected layout:
        root_dir/
          class_a/
            image_1.jpeg
            ...
          class_b/
            ...

    Returns:
        A mapping of class name -> sorted list of image paths.
    """
    root = Path(root_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")

    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    if class_sort:
        class_dirs = sorted(class_dirs, key=lambda p: p.name.lower())

    class_to_paths: dict[str, list[Path]] = {}
    for class_dir in class_dirs:
        if recursive:
            images = [
                p
                for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
        else:
            images = [
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
        class_to_paths[class_dir.name] = sorted(images)
    return class_to_paths


def class_counts(class_to_paths: dict[str, list[Path]]) -> dict[str, int]:
    """Return per-class sample counts."""
    return {class_name: len(paths) for class_name, paths in class_to_paths.items()}


def compute_median_target(class_to_paths: dict[str, list[Path]]) -> int:
    """
    Compute median class count target for downsampling.

    The result is floored to int for even-numbered class counts.
    """
    counts = [len(paths) for paths in class_to_paths.values()]
    if not counts:
        raise ValueError("No classes found when computing median target.")
    if any(count <= 0 for count in counts):
        # Keep behavior explicit: empty classes can be handled by caller.
        # Median still works, but this warning via exception avoids silent misconfiguration.
        raise ValueError("At least one class is empty. Remove empty classes before balancing.")
    return int(statistics.median(counts))


def _select_stride(paths: list[Path], target: int) -> list[Path]:
    """
    Select ``target`` paths by fixed-stride binning.

    This mirrors "skip every some samples" style selection and is deterministic.
    """
    n = len(paths)
    if target >= n:
        return list(paths)
    if target <= 0:
        return []

    # Pick the first element from each of target bins.
    selected = [paths[math.floor(i * n / target)] for i in range(target)]
    return selected


def _select_random(paths: list[Path], target: int, *, seed: int | None) -> list[Path]:
    """Select ``target`` paths randomly with a deterministic seed."""
    n = len(paths)
    if target >= n:
        return list(paths)
    if target <= 0:
        return []

    rng = random.Random(seed)
    picked_indices = sorted(rng.sample(range(n), target))
    return [paths[i] for i in picked_indices]


def build_balanced_index(
    class_to_paths: dict[str, list[Path]],
    *,
    target: int | None = None,
    method: str = "stride",
    seed: int | None = 42,
    shuffle: bool = True,
    shuffle_seed: int | None = 42,
) -> list[tuple[Path, str]]:
    """
    Build a balanced training index list: ``[(image_path, class_name), ...]``.

    Policy:
    - If class count <= target: keep all.
    - If class count > target: downsample to target.
    - No upsampling for low-frequency classes in this version.
    """
    if not class_to_paths:
        raise ValueError("class_to_paths is empty.")
    if target is None:
        target = compute_median_target(class_to_paths)
    if target <= 0:
        raise ValueError("target must be > 0.")

    method = method.lower().strip()
    if method not in {"stride", "random"}:
        raise ValueError("method must be one of: {'stride', 'random'}.")

    balanced_items: list[tuple[Path, str]] = []
    for class_name in sorted(class_to_paths.keys(), key=str.lower):
        paths = sorted(class_to_paths[class_name])
        if len(paths) <= target:
            selected = paths
        else:
            if method == "stride":
                selected = _select_stride(paths, target)
            else:
                # Slightly decorrelate classes while preserving deterministic runs.
                class_seed = None if seed is None else (seed + hash(class_name) % 10_000_019)
                selected = _select_random(paths, target, seed=class_seed)
        balanced_items.extend((p, class_name) for p in selected)

    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(balanced_items)
    return balanced_items


def summarize_balance(
    before_counts: dict[str, int],
    after_counts: dict[str, int],
) -> str:
    """Return a compact balance report table as a string."""
    classes = sorted(set(before_counts) | set(after_counts), key=str.lower)
    class_col_w = max(len("class"), *(len(c) for c in classes))
    before_col_w = max(len("before"), *(len(str(before_counts.get(c, 0))) for c in classes))
    after_col_w = max(len("after"), *(len(str(after_counts.get(c, 0))) for c in classes))

    header = f"{'class':<{class_col_w}}  {'before':>{before_col_w}}  {'after':>{after_col_w}}"
    divider = "-" * len(header)
    lines = [header, divider]
    for class_name in classes:
        lines.append(
            f"{class_name:<{class_col_w}}  "
            f"{before_counts.get(class_name, 0):>{before_col_w}}  "
            f"{after_counts.get(class_name, 0):>{after_col_w}}"
        )
    lines.append(divider)
    lines.append(
        f"{'TOTAL':<{class_col_w}}  "
        f"{sum(before_counts.values()):>{before_col_w}}  "
        f"{sum(after_counts.values()):>{after_col_w}}"
    )
    return "\n".join(lines)


def _to_hw_tuple(img_size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize img_size into (height, width)."""
    if isinstance(img_size, int):
        if img_size <= 0:
            raise ValueError("img_size must be > 0.")
        return (img_size, img_size)
    if len(img_size) != 2 or img_size[0] <= 0 or img_size[1] <= 0:
        raise ValueError("img_size tuple must be (height, width) with positive values.")
    return img_size


def _enforce_min_crop_scale(
    scale: tuple[float, float],
    ratio: tuple[float, float],
    *,
    min_side_fraction: float = 0.20,
) -> tuple[float, float]:
    """
    Raise RandomResizedCrop min area scale to guarantee crop sides are not too small.

    For RandomResizedCrop, with area fraction ``s`` and aspect ratio ``r = w / h``:
      w_frac = sqrt(s * r), h_frac = sqrt(s / r).
    To enforce both >= ``min_side_fraction``, we need:
      s >= f^2 / r_min and s >= f^2 * r_max.
    """
    if not (0.0 < min_side_fraction <= 1.0):
        raise ValueError("min_side_fraction must be in (0, 1].")
    r_min, r_max = ratio
    if r_min <= 0 or r_max <= 0 or r_min > r_max:
        raise ValueError("ratio must be positive with ratio[0] <= ratio[1].")

    required_min = max(
        (min_side_fraction * min_side_fraction) / r_min,
        (min_side_fraction * min_side_fraction) * r_max,
    )
    s_min, s_max = scale
    if s_min > s_max:
        raise ValueError("scale must satisfy scale[0] <= scale[1].")

    return (max(s_min, required_min), s_max)


def get_train_transforms(
    img_size: int | tuple[int, int],
    *,
    strength: str = "strong",
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
):
    """
    Build torchvision training transforms.

    strength:
    - 'light': conservative perturbations.
    - 'medium': balanced perturbations.
    - 'strong': stronger regularization.
    - 'very_strong': very aggressive crop/rotation-heavy regularization.
    """
    try:
        from torchvision import transforms as T
    except ImportError as exc:
        raise ImportError("torchvision is required for transform builders.") from exc

    h, w = _to_hw_tuple(img_size)
    strength = strength.lower().strip()
    min_side_fraction = 0.20

    if strength == "light":
        ratio = (0.9, 1.1)
        scale = _enforce_min_crop_scale((0.75, 1.0), ratio, min_side_fraction=min_side_fraction)
        aug = [
            T.RandomResizedCrop((h, w), scale=scale, ratio=ratio),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.5, 2.0), value="random"),
        ]
    elif strength == "medium":
        ratio = (0.8, 1.2)
        scale = _enforce_min_crop_scale((0.60, 1.0), ratio, min_side_fraction=min_side_fraction)
        aug = [
            T.RandomResizedCrop((h, w), scale=scale, ratio=ratio),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomApply(
                [T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)],
                p=0.8,
            ),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.4, 2.8), value="random"),
        ]
    elif strength == "strong":
        ratio = (0.75, 1.33)
        scale = _enforce_min_crop_scale((0.45, 1.0), ratio, min_side_fraction=min_side_fraction)
        aug = [
            T.RandomResizedCrop((h, w), scale=scale, ratio=ratio),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.15),
            T.RandomRotation(degrees=20),
            T.RandomApply(
                [T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08)],
                p=0.9,
            ),
            T.RandomGrayscale(p=0.10),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.30),
            T.RandomAutocontrast(p=0.20),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.35, scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"),
        ]
    elif strength == "very_strong":
        ratio = (0.55, 1.80)
        scale = _enforce_min_crop_scale((0.20, 1.0), ratio, min_side_fraction=min_side_fraction)
        aug = [
            T.RandomResizedCrop((h, w), scale=scale, ratio=ratio),
            T.RandomHorizontalFlip(p=0.6),
            T.RandomVerticalFlip(p=0.25),
            T.RandomRotation(degrees=55),
            T.RandomApply(
                [T.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.45, hue=0.10)],
                p=0.95,
            ),
            T.RandomGrayscale(p=0.15),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.5))], p=0.35),
            T.RandomAutocontrast(p=0.25),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.45, scale=(0.02, 0.25), ratio=(0.25, 3.8), value="random"),
        ]
    else:
        raise ValueError(
            "strength must be one of: {'light', 'medium', 'strong', 'very_strong'}."
        )

    return T.Compose(aug)


def get_eval_transforms(
    img_size: int | tuple[int, int],
    *,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
):
    """
    Build deterministic torchvision transforms for validation/test data.
    """
    try:
        from torchvision import transforms as T
    except ImportError as exc:
        raise ImportError("torchvision is required for transform builders.") from exc

    h, w = _to_hw_tuple(img_size)
    # Resize slightly larger before center crop to reduce border bias.
    resize_h = int(round(h / 0.875))
    resize_w = int(round(w / 0.875))
    return T.Compose(
        [
            T.Resize((resize_h, resize_w)),
            T.CenterCrop((h, w)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def infer_source_group(image_path: Path | str) -> str:
    """
    Infer source-video group id from frame filename.

    Expected extractor naming:
        <video_stem>_frame_000123.jpeg
    Falls back to image stem when pattern is unavailable.
    """
    p = Path(image_path)
    stem = p.stem
    if "_frame_" in stem:
        return stem.split("_frame_", 1)[0]
    return stem


def group_by_source(
    items: Iterable[tuple[Path, str]],
    *,
    group_fn: Callable[[Path], str] = infer_source_group,
) -> dict[str, list[tuple[Path, str]]]:
    """
    Group training index items by source identity (e.g., video stem).
    """
    grouped: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path, class_name in items:
        grouped[group_fn(path)].append((path, class_name))
    return dict(grouped)


def check_group_overlap(
    train_items: Iterable[tuple[Path, str]],
    val_items: Iterable[tuple[Path, str]],
    test_items: Iterable[tuple[Path, str]] | None = None,
    *,
    group_fn: Callable[[Path], str] = infer_source_group,
) -> dict[str, set[str]]:
    """
    Return overlapping source groups across splits.

    Non-empty overlap means potential leakage for video-frame-derived data.
    """
    train_groups = {group_fn(path) for path, _ in train_items}
    val_groups = {group_fn(path) for path, _ in val_items}
    test_groups = {group_fn(path) for path, _ in (test_items or [])}

    overlaps = {
        "train_val": train_groups & val_groups,
        "train_test": train_groups & test_groups,
        "val_test": val_groups & test_groups,
    }
    return overlaps


def assert_no_group_overlap(
    train_items: Iterable[tuple[Path, str]],
    val_items: Iterable[tuple[Path, str]],
    test_items: Iterable[tuple[Path, str]] | None = None,
    *,
    group_fn: Callable[[Path], str] = infer_source_group,
) -> None:
    """
    Raise ValueError if source groups overlap across data splits.
    """
    overlaps = check_group_overlap(
        train_items=train_items,
        val_items=val_items,
        test_items=test_items,
        group_fn=group_fn,
    )
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(
            "Data leakage risk: overlapping source groups across splits detected: "
            f"{bad}"
        )


def counts_from_index(items: Iterable[tuple[Path, str]]) -> dict[str, int]:
    """Count class frequencies from an index list."""
    counter: Counter[str] = Counter()
    for _, class_name in items:
        counter[class_name] += 1
    return dict(counter)


def quick_balance_sanity(
    class_to_paths: dict[str, list[Path]],
    *,
    target: int | None = None,
    method: str = "stride",
    seed: int | None = 42,
) -> dict[str, object]:
    """
    Run a concise sanity check for balancing behavior and reproducibility.

    Returns a dict with:
    - target
    - before_counts
    - after_counts
    - reproducible (bool)
    """
    if target is None:
        target = compute_median_target(class_to_paths)

    idx_1 = build_balanced_index(
        class_to_paths,
        target=target,
        method=method,
        seed=seed,
        shuffle=False,
    )
    idx_2 = build_balanced_index(
        class_to_paths,
        target=target,
        method=method,
        seed=seed,
        shuffle=False,
    )

    before = class_counts(class_to_paths)
    after = counts_from_index(idx_1)
    return {
        "target": target,
        "before_counts": before,
        "after_counts": after,
        "reproducible": idx_1 == idx_2,
    }
