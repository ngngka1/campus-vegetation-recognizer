#!/usr/bin/env python3
"""
Traditional ML vegetation classifier training/evaluation pipeline.

Task coverage:
  - Task 3: trains and compares multiple non-deep-learning classifiers.
  - Task 4: evaluates held-out test set with accuracy, confusion matrix,
            per-class metrics, and qualitative error analysis artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA

from data_augmentation import IMAGE_SUFFIXES, infer_source_group
from feature_extractor import (
    extract_hsv_hist_features,
    extract_color_moment_features,
    extract_hog_features,
    extract_glcm_features,
    extract_lbp_features,
    extract_sift_features,
    extract_surf_features,
)


@dataclass(frozen=True)
class DatasetSplit:
    train: np.ndarray
    val: np.ndarray


@dataclass(frozen=True)
class PipelineConfig:
    data_root: Path
    dataset_subdir: str = "auto"
    output_dir: Path | None = None
    seed: int = 42
    img_size: int = 128
    max_examples_per_group: int = 99999
    show_progress: bool = True
    # feature_combinations: tuple[str, ...] | None = None
    # include_surf_if_available: bool = False
    # If set, reduce each base feature block to this many components via PCA.
    # Set to None to disable. Helps balance dimensionality when concatenating.


@dataclass(frozen=True)
class PipelineDataset:
    images: np.ndarray
    labels: np.ndarray
    dataset_dir: Path | None = None
    paths: np.ndarray | None = None


def resolve_dataset_dir(data_root: Path, dataset_subdir: str) -> Path:
    data_root = data_root.resolve()
    if dataset_subdir != "auto":
        dataset_dir = data_root / dataset_subdir
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset subdirectory not found: {dataset_dir}")
        return dataset_dir

    candidates = [
        data_root / "extracted_group_augmented",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "No dataset folder found. Expected one of: " f"{candidates}"
    )


def collect_samples(
    dataset_dir: Path, max_examples_per_group: int
) -> tuple[np.ndarray, ...]:
    paths: list[Path] = []
    labels: list[str] = []

    class_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()
    )
    if not class_dirs:
        raise ValueError(f"No class folders found in dataset: {dataset_dir}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        all_imgs = sorted(
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not all_imgs:
            continue

        per_group_counter: dict[str, int] = {}
        for img_path in all_imgs:
            group_base = infer_source_group(str(img_path))
            group_id = f"{class_name}::{group_base}"
            seen = per_group_counter.get(group_id, 0)
            if seen >= max_examples_per_group:
                continue
            per_group_counter[group_id] = seen + 1
            paths.append(img_path)
            labels.append(class_name)

    if not paths:
        raise ValueError("No images collected after filtering.")
    return np.array(paths), np.array(labels)


def load_pipeline_dataset(config: PipelineConfig) -> PipelineDataset:
    dataset_dir = resolve_dataset_dir(config.data_root, config.dataset_subdir)
    paths, labels = collect_samples(
        dataset_dir=dataset_dir,
        max_examples_per_group=max(1, config.max_examples_per_group),
    )
    images = np.asarray(
        load_images(np.asarray(paths, dtype=object), size=config.img_size), dtype=np.uint8
    )
    return PipelineDataset(
        images=images,
        labels=np.asarray(labels, dtype=object),
        dataset_dir=dataset_dir,
        paths=np.asarray(paths, dtype=object),
    )


def load_bgr(path: Path, size: int = 128) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_images(paths: np.ndarray, size: int = 128) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for path in paths:
        images.append(load_bgr(Path(path), size=size))
    return images


def _parse_feature_combinations(
    feature_combinations: tuple[str, ...],
) -> list[str]:
    out = []
    for item in feature_combinations:
        key = item.strip().lower()
        if key:
            out.append(key)
    return out


def extract_base_feature_sets(
    images: list[np.ndarray],
    img_size: int,
    *,
    include_surf_if_available: bool = False,
) -> dict[str, np.ndarray]:
    """Extract reusable base features for each image."""
    base: dict[str, np.ndarray] = {
        "color_hist": extract_hsv_hist_features(images),
        "color_moment": extract_color_moment_features(images),
        "hog": extract_hog_features(images, img_size=img_size),
        "glcm": extract_glcm_features(images),
        "lbp": extract_lbp_features(images),
    }
    try:
        base["sift"] = extract_sift_features(images)
    except RuntimeError:
        pass

    # SURF is only available in opencv-contrib-python
    if include_surf_if_available:
        try:
            base["surf"] = extract_surf_features(images)
        except RuntimeError:
            pass
    return base


def reduce_base_features_with_pca(
    base_features: dict[str, np.ndarray],
    n_components: int = 50,
    *,
    features_to_reduce: list[str] | None = None,
    fitted_pcas: dict[str, PCA] | None = None,
    random_state: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, PCA]]:
    """
    Apply PCA to each base feature block to balance dimensionality.

    Use this before concatenation to prevent high-dimensional blocks (e.g. color_hist
    with 768 dims) from dominating distance-based models. Each block is reduced to
    n_components dimensions (or fewer if the block has fewer original dimensions).

    Args:
        base_features: Dict of feature_name -> (n_samples, n_dims).
        n_components: Target number of components per block. Blocks with fewer
            original dimensions are left unchanged.
        features_to_reduce: If provided, only apply PCA to these feature names.
            Features not in this list pass through unchanged. If None, all features
            are eligible for reduction.
        fitted_pcas: If provided, use these fitted PCA objects for transform only
            (no fit). Use when processing test/inference data.
        random_state: Random state for PCA reproducibility.

    Returns:
        (reduced_base_features, fitted_pcas). The fitted PCAs can be passed back
        as fitted_pcas when processing another dataset (e.g. test set).

    Example:
        # Training (reduce all features):
        train_base = extract_base_feature_sets(train_images, ...)
        train_reduced, pcas = reduce_base_features_with_pca(train_base, n_components=50)
        train_feature_sets = build_feature_sets(..., base_features=train_reduced)

        # Training (reduce only specific features):
        train_reduced, pcas = reduce_base_features_with_pca(
            train_base, n_components=50, features_to_reduce=["color_hist", "sift"]
        )

        # Test/inference:
        test_base = extract_base_feature_sets(test_images, ...)
        test_reduced, _ = reduce_base_features_with_pca(test_base, fitted_pcas=pcas)
        test_feature_sets = build_feature_sets(..., base_features=test_reduced)
    """
    reduced: dict[str, np.ndarray] = {}
    pcas_out: dict[str, PCA] = {}

    for name, arr in base_features.items():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        n_dims = arr.shape[1]
        effective_components = min(n_components, n_dims)

        if features_to_reduce is not None and name not in features_to_reduce:
            reduced[name] = arr
            continue

        if fitted_pcas is not None:
            if name in fitted_pcas:
                pca = fitted_pcas[name]
                reduced[name] = pca.transform(arr).astype(np.float32)
                pcas_out[name] = pca
            else:
                # Block was passed through during training (n_dims <= n_components)
                reduced[name] = arr
        elif effective_components < n_dims:
            pca = PCA(n_components=effective_components, random_state=random_state)
            reduced[name] = pca.fit_transform(arr).astype(np.float32)
            pcas_out[name] = pca
        else:
            # Block has fewer dims than n_components; pass through unchanged
            reduced[name] = arr
            # No PCA fitted; test data for this block will also pass through

    return reduced, pcas_out


def fit_pcas_on_indices(
    base_features: dict[str, np.ndarray],
    fit_indices: np.ndarray,
    n_components: int,
    *,
    features_to_reduce: list[str] | None = None,
    random_state: int | None = None,
) -> dict[str, PCA]:
    """
    Fit one PCA per eligible block using only rows ``fit_indices`` (e.g. CV fold train).

    Skips blocks not listed in ``features_to_reduce`` (when set) and blocks where
    dimension is already at or below the effective component count.
    """
    fitted: dict[str, PCA] = {}
    n_fit = int(fit_indices.size)
    for name, arr in base_features.items():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        n_dims = arr.shape[1]
        effective_components = min(n_components, n_dims, max(n_fit, 1))
        if features_to_reduce is not None and name not in features_to_reduce:
            continue
        if effective_components >= n_dims:
            continue
        subset = arr[fit_indices]
        pca = PCA(n_components=effective_components, random_state=random_state)
        pca.fit(subset)
        fitted[name] = pca
    return fitted


def concat_feature_combo(
    base_features: dict[str, np.ndarray],
    combo: str,
    row_indices: np.ndarray,
    fitted_pcas: dict[str, PCA],
    n_components: int,
    *,
    features_to_reduce: list[str] | None = None,
) -> np.ndarray:
    """
    Build one concatenated feature matrix for ``combo`` using pre-fitted PCAs per block.

    Must match the pass-through rules of :func:`reduce_base_features_with_pca`.
    """
    parts = [p.strip() for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty feature combo")
    chunks: list[np.ndarray] = []
    for p in parts:
        if p not in base_features:
            raise KeyError(f"Unknown base feature block: {p}")
        arr = np.nan_to_num(base_features[p], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        n_dims = arr.shape[1]
        effective_components = min(n_components, n_dims)
        sub = arr[row_indices]

        if features_to_reduce is not None and p not in features_to_reduce:
            chunks.append(sub)
            continue
        if p in fitted_pcas:
            chunks.append(fitted_pcas[p].transform(arr[row_indices]).astype(np.float32))
        elif effective_components >= n_dims:
            chunks.append(sub)
        else:
            raise ValueError(
                f"Missing fitted PCA for block '{p}' that requires reduction "
                f"(effective_components={effective_components}, n_dims={n_dims})"
            )
    return np.concatenate(chunks, axis=1)


def build_feature_sets(
    images: list[np.ndarray],
    img_size: int,
    *,
    feature_combinations: tuple[str, ...] | None = None,
    include_surf_if_available: bool = False,
    base_features: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate feature sets for each image.

    the output is a dictionary with:
    key: feature combination (e.g. hog+sift, sift, color_hist+hog+sift, etc.)
    value: numpy array containing the features for each image (note: features are concatenated for more than one feature)
    """
    base = (
        dict(base_features)
        if base_features is not None
        else extract_base_feature_sets(
            images,
            img_size=img_size,
            include_surf_if_available=include_surf_if_available,
        )
    )

    selected = _parse_feature_combinations(feature_combinations)
    # a feature set is a dictionary of feature names and their corresponding numpy arrays containing the features for each image
    feature_sets: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for combo in selected:
        # a combo is a string of feature names separated by "+"
        parts = [p.strip() for p in combo.split("+") if p.strip()]
        if not parts:
            continue
        # a part is a feature name
        # missing is a list of feature names that are not in the base dictionary (i.e. features that are not available for the implementation)
        missing = [p for p in parts if p not in base]
        if missing:
            skipped.append(f"{combo} (missing: {', '.join(missing)})")
            continue
        if len(parts) == 1:
            feature_sets[combo] = base[parts[0]]
        else:
            feature_sets[combo] = np.concatenate([base[p] for p in parts], axis=1)

    if not feature_sets:
        available = ", ".join(sorted(base.keys()))
        raise ValueError(
            f"No valid feature combinations available. Available base extractors: {available}"
        )
    if skipped:
        raise ValueError(
            "Some requested feature combinations are unavailable and were not trained: "
            + "; ".join(skipped)
        )
    # Make downstream models robust against occasional numeric instabilities.
    for k, v in feature_sets.items():
        feature_sets[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
    return feature_sets
