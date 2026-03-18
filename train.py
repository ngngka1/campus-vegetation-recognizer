from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score

import random
from pathlib import Path
from model import build_model_factories, save
from dataset_loader import (
    PipelineConfig,
    PipelineDataset,
    build_feature_sets,
    DatasetSplit,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class SplitLike(Protocol):
    train: np.ndarray
    val: np.ndarray


@dataclass
class MLPipelineOutput:
    train_results: list[TrainResult]
    train_results_by_model: dict[str, list[TrainResult]]
    labels: np.ndarray
    feature_sets: dict[str, np.ndarray]
    split: DatasetSplit
    train_idx_for_fit: np.ndarray
    val_idx_for_eval: np.ndarray
    dataset_dir: Path
    output_dir: Path
    seed: int
    split: SplitLike


@dataclass
class TrainResult:
    model_name: str
    feature_name: str
    test_accuracy: float
    val_accuracy: float
    train_accuracy: float
    train_seconds: float
    estimator: object


def _pair_progress_bar(total_pairs: int):
    if tqdm is None:
        return None
    return tqdm(
        total=total_pairs,
        desc="Training model-feature pairs",
        unit="pair",
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} pairs",
    )


def train_candidates(
    train_feature_sets: dict[str, np.ndarray],
    train_labels: np.ndarray,
    test_feature_sets: dict[str, np.ndarray],
    test_labels: np.ndarray,
    split: SplitLike,
    *,
    seed: int,
    show_progress: bool = True,
    save_model: bool = False,
    model_save_dir: Path | None = None,
) -> list[TrainResult]:
    model_factories = build_model_factories(seed=seed)
    candidates = [
        (model_name, feature_name)
        for feature_name in sorted(train_feature_sets.keys())
        for model_name in model_factories.keys()
    ]
    
    if save_model and model_save_dir is None:
        raise ValueError("model_save_dir must be provided when save_model is True.")

    results: list[TrainResult] = []
    best_scores_by_model: dict[str, tuple[float, float]] = {}
    bar = _pair_progress_bar(len(candidates)) if show_progress else None
    for model_name, feature_name in candidates:
        estimator = clone(model_factories[model_name])
        # get the feature for the images for index split.train
        x_train = train_feature_sets[feature_name][split.train]
        x_train = np.nan_to_num(
            x_train,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        # get the label for the images for index split.train
        y_train = train_labels[split.train]
        if np.unique(y_train).size < 2:
            raise ValueError(
                "Training split contains fewer than 2 classes after preprocessing/augmentation. "
                "Please adjust split settings or data sampling."
            )
        # get the feature for the images for index split.val
        x_val = train_feature_sets[feature_name][split.val]
        x_val = np.nan_to_num(
            x_val,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        # get the label for the images for index split.val
        y_val = train_labels[split.val]
        if bar is not None:
            bar.set_description(f"Training [{model_name} | {feature_name}]")
        t0 = time.perf_counter()

        estimator.fit(x_train, y_train)

        train_seconds = time.perf_counter() - t0

        # predict for training set
        y_train_pred = estimator.predict(x_train)
        # predict for validation set
        y_val_pred = estimator.predict(x_val)

        train_acc = float(accuracy_score(y_train, y_train_pred))
        val_acc = float(accuracy_score(y_val, y_val_pred))

        # finally, predict for test set and calculate accuracy
        y_test_pred = estimator.predict(test_feature_sets[feature_name])
        test_acc = float(accuracy_score(test_labels, y_test_pred))

        if show_progress:
            log_line = (
                f"[{model_name} | {feature_name}] \n"
                f"test_acc={test_acc:.4f}, val_acc={val_acc:.4f}, train_acc={train_acc:.4f}, time={train_seconds:.1f}s"
            )
            if tqdm is not None:
                tqdm.write(log_line)
            else:
                print(log_line)
        if bar is not None:
            bar.update(1)
        results.append(
            TrainResult(
                model_name=model_name,
                feature_name=feature_name,
                val_accuracy=val_acc,
                train_accuracy=train_acc,
                train_seconds=train_seconds,
                estimator=estimator,
                test_accuracy=test_acc,
            )
        )
        if save_model:
            if model_name not in best_scores_by_model or test_acc > best_scores_by_model[model_name][0]:
                best_scores_by_model[model_name] = (test_acc, val_acc, train_acc, train_seconds)
                save(estimator, model_save_dir / f"{model_name}_{feature_name}.pkl")
    if bar is not None:
        bar.close()
    # sort the results by test accuracy, or val accuracy if test accuracy is the same
    results.sort(key=lambda r: (r.test_accuracy, r.val_accuracy), reverse=True)
    return results


def _count_labels(y: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in y:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_traditional_ml_pipeline(
    config: PipelineConfig,
    train_dataset: PipelineDataset | None = None,
    train_base_features: dict[str, np.ndarray] | None = None,
    val_size: float = 0.2,
    test_dataset: PipelineDataset | None = None,
    test_base_features: dict[str, np.ndarray] | None = None,
    save_model: bool = False,
) -> MLPipelineOutput:
    set_seed(config.seed)

    output_dir = (
        config.output_dir.resolve()
        if config.output_dir is not None
        else (Path(__file__).resolve().parent / "ml_results")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # process the train dataset
    train_images = np.asarray(train_dataset.images, dtype=np.uint8)
    train_labels = np.asarray(train_dataset.labels, dtype=object)
    if len(train_images) != len(train_labels):
        raise ValueError(
            "Provided dataset must have equal lengths for paths, labels, and groups."
        )

    # process the test dataset
    test_images = np.asarray(test_dataset.images, dtype=np.uint8)
    test_labels = np.asarray(test_dataset.labels, dtype=object)
    if len(test_images) != len(test_labels):
        raise ValueError(
            "Provided test dataset must have equal lengths for paths, labels, and groups."
        )

    # split the train dataset into training and validation sets
    rng = np.random.default_rng(config.seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for cls in np.unique(train_labels):
        cls_idx = np.where(train_labels == cls)[0]
        cls_idx = np.asarray(cls_idx, dtype=np.int64)
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_size))
        n_val = min(max(n_val, 0), len(cls_idx))
        val_parts.append(cls_idx[:n_val])
        train_parts.append(cls_idx[n_val:])
    train_idx_for_fit = (
        np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    )
    val_idx_for_eval = (
        np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    )
    rng.shuffle(train_idx_for_fit)
    rng.shuffle(val_idx_for_eval)
    split = DatasetSplit(train=train_idx_for_fit, val=val_idx_for_eval)

    class_names = sorted(np.unique(train_labels).tolist())
    train_class_counts = _count_labels(train_labels)

    print("[INFO] Per-class sample size before training (full collected dataset):")
    for cls in sorted(train_class_counts.keys()):
        print(f"  - {cls}: {train_class_counts[cls]}")

    train_feature_sets = build_feature_sets(
        list(train_images),
        img_size=config.img_size,
        feature_combinations=config.feature_combinations,
        include_surf_if_available=config.include_surf_if_available,
        base_features=train_base_features,
    )
    test_feature_sets = build_feature_sets(
        list(test_images),
        img_size=config.img_size,
        feature_combinations=config.feature_combinations,
        include_surf_if_available=config.include_surf_if_available,
        base_features=test_base_features,
    )
    train_results = train_candidates(
        train_feature_sets=train_feature_sets,
        train_labels=train_labels,
        test_feature_sets=test_feature_sets,
        test_labels=test_labels,
        split=split,
        seed=config.seed,
        show_progress=config.show_progress,
        save_model=save_model,
        model_save_dir=output_dir / "models",
    )

    train_results_by_model: dict[str, list[TrainResult]] = {}
    for result in train_results:
        if result.model_name not in train_results_by_model:
            train_results_by_model[result.model_name] = []
        train_results_by_model[result.model_name].append(result)

    return MLPipelineOutput(
        dataset_dir=config.data_root / config.dataset_subdir,
        output_dir=output_dir,
        train_results=train_results,
        train_results_by_model=train_results_by_model,
        seed=config.seed,
        labels=train_labels,
        feature_sets=train_feature_sets,
        split=split,
        train_idx_for_fit=train_idx_for_fit,
        val_idx_for_eval=val_idx_for_eval,
    )
