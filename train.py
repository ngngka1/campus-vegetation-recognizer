from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import random
from pathlib import Path
from joblib import dump
from model import build_model_factories, save
from dataset_loader import (
    DatasetSplit,
    PipelineConfig,
    PipelineDataset,
    build_feature_sets,
    extract_base_feature_sets,
    reduce_base_features_with_pca,
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
    test_labels: np.ndarray
    test_feature_sets: dict[str, np.ndarray]
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
    val_folds: list[float]
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
    split: SplitLike,
    *,
    seed: int,
    show_progress: bool = True,
) -> list[TrainResult]:
    model_factories = build_model_factories(seed=seed)
    candidates = [
        (model_name, feature_name)
        for feature_name in sorted(train_feature_sets.keys())
        for model_name in model_factories.keys()
    ]

    # Use all available labeled training samples for CV, while keeping the same
    # function contract for callers.
    cv_pool_idx = np.concatenate([split.train, split.val]).astype(np.int64)
    if cv_pool_idx.size == 0:
        raise ValueError("No samples available for training/cross-validation.")

    cv_pool_labels = train_labels[cv_pool_idx]
    unique_labels, label_counts = np.unique(cv_pool_labels, return_counts=True)
    if unique_labels.size < 2:
        raise ValueError(
            "Training data contains fewer than 2 classes after preprocessing/augmentation. "
            "Please adjust split settings or data sampling."
        )

    # Prefer 5-fold CV, but adapt to small datasets safely.
    preferred_folds = 5
    max_possible_folds = int(label_counts.min())
    n_folds = min(preferred_folds, max_possible_folds)
    use_cv = n_folds >= 2

    results: list[TrainResult] = []
    bar = _pair_progress_bar(len(candidates)) if show_progress else None
    for model_name, feature_name in candidates:
        # Build the full train pool once and evaluate via CV folds.
        x_pool = train_feature_sets[feature_name][cv_pool_idx]
        x_pool = np.nan_to_num(
            x_pool,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        y_pool = cv_pool_labels
        if bar is not None:
            bar.set_description(f"Training [{model_name} | {feature_name}]")
        cv_train_scores: list[float] = []
        cv_val_scores: list[float] = []
        cv_fit_seconds = 0.0

        if use_cv:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold_train_idx, fold_val_idx in splitter.split(x_pool, y_pool):
                fold_estimator = clone(model_factories[model_name])
                x_fold_train = x_pool[fold_train_idx]
                y_fold_train = y_pool[fold_train_idx]
                x_fold_val = x_pool[fold_val_idx]
                y_fold_val = y_pool[fold_val_idx]

                t0 = time.perf_counter()
                fold_estimator.fit(x_fold_train, y_fold_train)
                cv_fit_seconds += time.perf_counter() - t0

                y_fold_train_pred = fold_estimator.predict(x_fold_train)
                y_fold_val_pred = fold_estimator.predict(x_fold_val)
                cv_train_scores.append(float(accuracy_score(y_fold_train, y_fold_train_pred)))
                cv_val_scores.append(float(accuracy_score(y_fold_val, y_fold_val_pred)))
        else:
            # Fallback for very small data where CV is not feasible.
            if split.train.size == 0:
                raise ValueError("Training split is empty and CV cannot be formed.")
            x_train = train_feature_sets[feature_name][split.train]
            x_train = np.nan_to_num(
                x_train,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
            y_train = train_labels[split.train]
            if np.unique(y_train).size < 2:
                raise ValueError(
                    "Training split contains fewer than 2 classes after preprocessing/augmentation. "
                    "Please adjust split settings or data sampling."
                )
            x_val = train_feature_sets[feature_name][split.val]
            x_val = np.nan_to_num(
                x_val,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
            y_val = train_labels[split.val]

            fold_estimator = clone(model_factories[model_name])
            t0 = time.perf_counter()
            fold_estimator.fit(x_train, y_train)
            cv_fit_seconds = time.perf_counter() - t0
            cv_train_scores.append(float(accuracy_score(y_train, fold_estimator.predict(x_train))))
            if y_val.size > 0:
                cv_val_scores.append(float(accuracy_score(y_val, fold_estimator.predict(x_val))))
            else:
                cv_val_scores.append(cv_train_scores[-1])

        train_acc = float(np.mean(cv_train_scores))
        val_acc = float(np.mean(cv_val_scores))
        train_seconds = cv_fit_seconds

        # Fit a final model on the full train pool used by CV for downstream use.
        estimator = clone(model_factories[model_name])
        estimator.fit(x_pool, y_pool)

        if show_progress:
            val_folds_str = ", ".join(f"{score:.4f}" for score in cv_val_scores)
            log_line = (
                f"[{model_name} | {feature_name}] \n"
                f"val_acc={val_acc:.4f}, "
                f"val_folds=[{val_folds_str}], train_acc={train_acc:.4f}, time={train_seconds:.1f}s"
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
                val_folds=cv_val_scores,
                train_accuracy=train_acc,
                train_seconds=train_seconds,
                estimator=estimator,
                test_accuracy=float("nan"),
            )
        )
    if bar is not None:
        bar.close()
    # Rank candidates by CV metrics only (primary: val, secondary: train).
    results.sort(key=lambda r: (r.val_accuracy, r.train_accuracy), reverse=True)
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
    clean_model_directory: bool = False,
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

    train_class_counts = _count_labels(train_labels)

    print("[INFO] Per-class sample size before training (full collected dataset):")
    for cls in sorted(train_class_counts.keys()):
        print(f"  - {cls}: {train_class_counts[cls]}")

    # Resolve base features (extract if not provided)
    if train_base_features is None:
        train_base_features = extract_base_feature_sets(
            list(train_images),
            img_size=config.img_size,
            include_surf_if_available=config.include_surf_if_available,
        )
    if test_base_features is None:
        test_base_features = extract_base_feature_sets(
            list(test_images),
            img_size=config.img_size,
            include_surf_if_available=config.include_surf_if_available,
        )

    # Apply PCA per block if enabled (balances dimensionality across feature types)
    fitted_pcas: dict | None = None
    if config.pca_per_block is not None:
        train_base_features, fitted_pcas = reduce_base_features_with_pca(
            train_base_features,
            n_components=config.pca_per_block,
            random_state=config.seed,
        )
        test_base_features, _ = reduce_base_features_with_pca(
            test_base_features,
            fitted_pcas=fitted_pcas,
        )
        if config.show_progress:
            print(f"[INFO] PCA per block enabled: n_components={config.pca_per_block}")

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
        split=split,
        seed=config.seed,
        show_progress=config.show_progress,
    )
    if not train_results:
        raise ValueError("No train results produced for model selection.")

    # Evaluate held-out test accuracy for all CV-ranked candidates, without
    # changing selection order (which remains CV-based). This is just
    # for reporting purposes, to show how robust the model is in
    # scenarios for recognizing vegetations outside the campus
    test_features_cache: dict[str, np.ndarray] = {}
    for result in train_results:
        if result.feature_name not in test_features_cache:
            test_features_cache[result.feature_name] = np.nan_to_num(
                test_feature_sets[result.feature_name],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
        y_test_pred = result.estimator.predict(test_features_cache[result.feature_name])
        result.test_accuracy = float(accuracy_score(test_labels, y_test_pred))

    if config.show_progress:
        best_result = train_results[0]
        print(
            f"[FINAL TEST] [{best_result.model_name} | {best_result.feature_name}] "
            f"test_acc={best_result.test_accuracy:.4f}"
        )

    train_results_by_model: dict[str, list[TrainResult]] = {}
    for result in train_results:
        if result.model_name not in train_results_by_model:
            train_results_by_model[result.model_name] = []
        train_results_by_model[result.model_name].append(result)

    model_save_dir = output_dir / "models"
    if save_model:
        if model_save_dir is None:
            raise ValueError("model_save_dir must be provided when save_model is True.")
        if model_save_dir.exists():
            if clean_model_directory:
                for file in model_save_dir.glob("*"):
                    file.unlink()
        else:
            model_save_dir.mkdir(parents=True, exist_ok=True)
        models_to_save = []
        for _, value in train_results_by_model.items():
            models_to_save.append(value[0])

        models_to_save.sort(key=lambda x: x.val_accuracy, reverse=True)

        for i, top_model in enumerate(models_to_save):
            save(
                top_model.estimator,
                model_save_dir / f"{1+i}_{top_model.model_name}_{top_model.feature_name}.pkl",
            )

        # Save fitted PCAs when PCA per block was used (needed for inference)
        if fitted_pcas is not None:
            dump(fitted_pcas, model_save_dir / "pcas.joblib")

    return MLPipelineOutput(
        dataset_dir=config.data_root / config.dataset_subdir,
        output_dir=output_dir,
        train_results=train_results,
        train_results_by_model=train_results_by_model,
        seed=config.seed,
        labels=train_labels,
        feature_sets=train_feature_sets,
        test_labels=test_labels,
        test_feature_sets=test_feature_sets,
        split=split,
        train_idx_for_fit=train_idx_for_fit,
        val_idx_for_eval=val_idx_for_eval,
    )
