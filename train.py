from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import random
from pathlib import Path
from model import build_model_factories, save
from dataset_loader import (
    DatasetSplit,
    PipelineConfig,
    PipelineDataset,
    build_feature_sets,
    concat_feature_combo,
    fit_pcas_on_indices,
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
    train_results_sorted_by_test_accuracy: list[TrainResult]
    labels: np.ndarray
    feature_sets: dict[str, np.ndarray]
    test_labels: np.ndarray
    test_feature_sets: dict[str, np.ndarray]
    test_paths: np.ndarray | None
    split: DatasetSplit
    train_idx_for_fit: np.ndarray
    val_idx_for_eval: np.ndarray
    dataset_dir: Path
    output_dir: Path
    seed: int
    split: SplitLike
    fitted_pcas: dict[str, PCA] | None = None


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


@dataclass
class TrainCandidatesOutput:
    """From train_candidates: CV metrics and, when PCA is used, one PCA dict fit on full CV pool."""

    results: list[TrainResult]
    fitted_pcas_full_cv: dict[str, PCA] | None = None


def _truncate_for_bar(s: str, max_len: int) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _pair_progress_bar(total_pairs: int):
    if tqdm is None:
        return None
    return tqdm(
        total=total_pairs,
        desc="Training model-feature pairs",
        unit="pair",
        dynamic_ncols=True,
        mininterval=0.2,
        bar_format="{desc}  {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} pairs  {postfix}",
    )


def _emit_training_pair_progress(
    bar,
    *,
    show_progress: bool,
    model_name: str,
    feature_name: str,
    val_acc: float,
    train_acc: float,
    train_seconds: float,
    test_accuracy: float,
    has_test: bool,
) -> None:
    """Single-line progress: padded candidate + metrics (no extra newlines)."""
    if not show_progress:
        return
    mod_disp = _truncate_for_bar(model_name, 22)
    feat_disp = _truncate_for_bar(feature_name, 48)
    desc = f"  {mod_disp:<24} |  {feat_disp:<48}"
    parts = [
        f"val={val_acc:.4f}",
        f"train={train_acc:.4f}",
        f"time={train_seconds:.1f}s",
    ]
    if has_test:
        parts.insert(1, f"test={test_accuracy:.4f}")
    postfix = "  ".join(parts)
    if bar is not None:
        bar.set_description(desc, refresh=False)
        bar.set_postfix_str(postfix, refresh=True)
        bar.update(1)
    else:
        print(f"{desc}  {postfix}", flush=True)


def train_candidates(
    train_labels: np.ndarray,
    split: SplitLike,
    *,
    seed: int,
    show_progress: bool = True,
    train_feature_sets: dict[str, np.ndarray] | None = None,
    train_base_features: dict[str, np.ndarray] | None = None,
    feature_combinations: tuple[str, ...] | None = None,
    pca_n_components: int | None = None,
    pca_features_to_reduce: list[str] | None = None,
    fitted_pcas_full_pool: dict[str, PCA] | None = None,
    test_labels: np.ndarray | None = None,
    test_feature_sets: dict[str, np.ndarray] | None = None,
    test_base_features: dict[str, np.ndarray] | None = None,
) -> TrainCandidatesOutput:
    if pca_n_components is not None:
        if train_base_features is None or feature_combinations is None:
            raise ValueError(
                "When pca_n_components is set, train_base_features and feature_combinations are required."
            )
        return _train_candidates_pca_per_fold(
            train_base_features=train_base_features,
            feature_combinations=feature_combinations,
            train_labels=train_labels,
            split=split,
            pca_n_components=pca_n_components,
            pca_features_to_reduce=pca_features_to_reduce,
            seed=seed,
            show_progress=show_progress,
            fitted_pcas_full_pool=fitted_pcas_full_pool,
            test_labels=test_labels,
            test_base_features=test_base_features,
        )

    if train_feature_sets is None:
        raise ValueError("train_feature_sets is required when pca_n_components is None.")
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
    test_features_cache: dict[str, np.ndarray] = {}
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

        if test_labels is not None and test_feature_sets is not None:
            if feature_name not in test_features_cache:
                test_features_cache[feature_name] = np.nan_to_num(
                    test_feature_sets[feature_name],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)
            y_test_pred = estimator.predict(test_features_cache[feature_name])
            test_accuracy = float(accuracy_score(test_labels, y_test_pred))
        else:
            test_accuracy = float("nan")

        has_test = test_labels is not None and test_feature_sets is not None
        _emit_training_pair_progress(
            bar,
            show_progress=show_progress,
            model_name=model_name,
            feature_name=feature_name,
            val_acc=val_acc,
            train_acc=train_acc,
            train_seconds=train_seconds,
            test_accuracy=test_accuracy,
            has_test=has_test,
        )
        results.append(
            TrainResult(
                model_name=model_name,
                feature_name=feature_name,
                val_accuracy=val_acc,
                val_folds=cv_val_scores,
                train_accuracy=train_acc,
                train_seconds=train_seconds,
                estimator=estimator,
                test_accuracy=test_accuracy,
            )
        )
    if bar is not None:
        bar.close()
    # Rank candidates by CV metrics only (primary: val, secondary: train).
    results.sort(key=lambda r: (r.val_accuracy, r.train_accuracy), reverse=True)
    return TrainCandidatesOutput(results=results, fitted_pcas_full_cv=None)


def _train_candidates_pca_per_fold(
    train_base_features: dict[str, np.ndarray],
    feature_combinations: tuple[str, ...],
    train_labels: np.ndarray,
    split: SplitLike,
    *,
    pca_n_components: int,
    pca_features_to_reduce: list[str] | None,
    seed: int,
    show_progress: bool,
    fitted_pcas_full_pool: dict[str, PCA] | None = None,
    test_labels: np.ndarray | None = None,
    test_base_features: dict[str, np.ndarray] | None = None,
) -> TrainCandidatesOutput:
    """Cross-validate classifiers with PCA refit on each fold's training indices; final model uses one PCA fit on full CV pool.

    If ``fitted_pcas_full_pool`` is provided, it must be PCA fit on the same row set as
    ``concatenate(split.train, split.val)`` (the CV pool), not on train split alone.
    """
    model_factories = build_model_factories(seed=seed)
    combos = sorted({c.strip().lower() for c in feature_combinations if c.strip()})
    candidates = [(model_name, fn) for fn in combos for model_name in model_factories.keys()]

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

    preferred_folds = 5
    max_possible_folds = int(label_counts.min())
    n_folds = min(preferred_folds, max_possible_folds)
    use_cv = n_folds >= 2

    if fitted_pcas_full_pool is not None:
        fitted_pcas_full_cv = fitted_pcas_full_pool
    else:
        fitted_pcas_full_cv = fit_pcas_on_indices(
            train_base_features,
            cv_pool_idx,
            pca_n_components,
            features_to_reduce=pca_features_to_reduce,
            random_state=seed,
        )

    results: list[TrainResult] = []
    bar = _pair_progress_bar(len(candidates)) if show_progress else None
    pca_test_features_cache: dict[str, np.ndarray] = {}

    for model_name, feature_name in candidates:
        cv_train_scores: list[float] = []
        cv_val_scores: list[float] = []
        cv_fit_seconds = 0.0

        if use_cv:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold_train_idx, fold_val_idx in splitter.split(cv_pool_idx, cv_pool_labels):
                train_rows = cv_pool_idx[fold_train_idx]
                val_rows = cv_pool_idx[fold_val_idx]
                fitted_fold = fit_pcas_on_indices(
                    train_base_features,
                    train_rows,
                    pca_n_components,
                    features_to_reduce=pca_features_to_reduce,
                    random_state=seed,
                )
                x_fold_train = concat_feature_combo(
                    train_base_features,
                    feature_name,
                    train_rows,
                    fitted_fold,
                    pca_n_components,
                    features_to_reduce=pca_features_to_reduce,
                )
                x_fold_val = concat_feature_combo(
                    train_base_features,
                    feature_name,
                    val_rows,
                    fitted_fold,
                    pca_n_components,
                    features_to_reduce=pca_features_to_reduce,
                )
                y_fold_train = train_labels[train_rows]
                y_fold_val = train_labels[val_rows]

                fold_estimator = clone(model_factories[model_name])
                t0 = time.perf_counter()
                fold_estimator.fit(x_fold_train, y_fold_train)
                cv_fit_seconds += time.perf_counter() - t0

                cv_train_scores.append(
                    float(accuracy_score(y_fold_train, fold_estimator.predict(x_fold_train)))
                )
                cv_val_scores.append(
                    float(accuracy_score(y_fold_val, fold_estimator.predict(x_fold_val)))
                )
        else:
            if split.train.size == 0:
                raise ValueError("Training split is empty and CV cannot be formed.")
            train_rows = split.train.astype(np.int64)
            val_rows = split.val.astype(np.int64)
            fitted_once = fit_pcas_on_indices(
                train_base_features,
                train_rows,
                pca_n_components,
                features_to_reduce=pca_features_to_reduce,
                random_state=seed,
            )
            x_train = concat_feature_combo(
                train_base_features,
                feature_name,
                train_rows,
                fitted_once,
                pca_n_components,
                features_to_reduce=pca_features_to_reduce,
            )
            x_val = concat_feature_combo(
                train_base_features,
                feature_name,
                val_rows,
                fitted_once,
                pca_n_components,
                features_to_reduce=pca_features_to_reduce,
            )
            y_train = train_labels[train_rows]
            y_val = train_labels[val_rows]
            if np.unique(y_train).size < 2:
                raise ValueError(
                    "Training split contains fewer than 2 classes after preprocessing/augmentation. "
                    "Please adjust split settings or data sampling."
                )
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

        x_pool = concat_feature_combo(
            train_base_features,
            feature_name,
            cv_pool_idx,
            fitted_pcas_full_cv,
            pca_n_components,
            features_to_reduce=pca_features_to_reduce,
        )
        y_pool = cv_pool_labels
        estimator = clone(model_factories[model_name])
        estimator.fit(x_pool, y_pool)

        if test_labels is not None and test_base_features is not None:
            if feature_name not in pca_test_features_cache:
                n_test = int(next(iter(test_base_features.values())).shape[0])
                test_rows = np.arange(n_test, dtype=np.int64)
                x_test = concat_feature_combo(
                    test_base_features,
                    feature_name,
                    test_rows,
                    fitted_pcas_full_cv,
                    pca_n_components,
                    features_to_reduce=pca_features_to_reduce,
                )
                pca_test_features_cache[feature_name] = np.nan_to_num(
                    x_test,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)
            y_test_pred = estimator.predict(pca_test_features_cache[feature_name])
            test_accuracy = float(accuracy_score(test_labels, y_test_pred))
        else:
            test_accuracy = float("nan")

        has_test = test_labels is not None and test_base_features is not None
        _emit_training_pair_progress(
            bar,
            show_progress=show_progress,
            model_name=model_name,
            feature_name=feature_name,
            val_acc=val_acc,
            train_acc=train_acc,
            train_seconds=train_seconds,
            test_accuracy=test_accuracy,
            has_test=has_test,
        )
        results.append(
            TrainResult(
                model_name=model_name,
                feature_name=feature_name,
                val_accuracy=val_acc,
                val_folds=cv_val_scores,
                train_accuracy=train_acc,
                train_seconds=train_seconds,
                estimator=estimator,
                test_accuracy=test_accuracy,
            )
        )

    if bar is not None:
        bar.close()
    results.sort(key=lambda r: (r.val_accuracy, r.train_accuracy), reverse=True)
    return TrainCandidatesOutput(results=results, fitted_pcas_full_cv=fitted_pcas_full_cv)


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
    features_combinations: list[str] | None = None,
    pca_n_components: int | None = None,
    pca_features_to_reduce: list[str] | None = None,
    fitted_pcas_full_pool: dict[str, PCA] | None = None,
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

    if features_combinations is None:
        raise ValueError("features_combinations is required.")
    combo_tuple = tuple(features_combinations)

    if pca_n_components is not None:
        if train_base_features is None or test_base_features is None:
            raise ValueError(
                "train_base_features and test_base_features (unreduced) are required when pca_n_components is set."
            )
        if fitted_pcas_full_pool is None:
            cv_pool_idx = np.concatenate([split.train, split.val]).astype(np.int64)
            fitted_pcas_full_pool = fit_pcas_on_indices(
                train_base_features,
                cv_pool_idx,
                pca_n_components,
                features_to_reduce=pca_features_to_reduce,
                random_state=config.seed,
            )
        tc_out = train_candidates(
            train_labels=train_labels,
            split=split,
            seed=config.seed,
            show_progress=config.show_progress,
            train_base_features=train_base_features,
            feature_combinations=combo_tuple,
            pca_n_components=pca_n_components,
            pca_features_to_reduce=pca_features_to_reduce,
            fitted_pcas_full_pool=fitted_pcas_full_pool,
            test_labels=test_labels,
            test_base_features=test_base_features,
        )
        train_results = tc_out.results
        fitted_pcas = tc_out.fitted_pcas_full_cv
        if fitted_pcas is None:
            raise RuntimeError("Expected fitted PCAs from PCA CV path.")
        train_reduced, _ = reduce_base_features_with_pca(
            train_base_features,
            n_components=pca_n_components,
            features_to_reduce=pca_features_to_reduce,
            fitted_pcas=fitted_pcas,
            random_state=config.seed,
        )
        test_reduced, _ = reduce_base_features_with_pca(
            test_base_features,
            n_components=pca_n_components,
            features_to_reduce=pca_features_to_reduce,
            fitted_pcas=fitted_pcas,
            random_state=config.seed,
        )
        train_feature_sets = build_feature_sets(
            list(train_images),
            img_size=config.img_size,
            feature_combinations=features_combinations,
            base_features=train_reduced,
        )
        test_feature_sets = build_feature_sets(
            list(test_images),
            img_size=config.img_size,
            feature_combinations=features_combinations,
            base_features=test_reduced,
        )
    else:
        fitted_pcas = None
        train_feature_sets = build_feature_sets(
            list(train_images),
            img_size=config.img_size,
            feature_combinations=features_combinations,
            base_features=train_base_features,
        )
        test_feature_sets = build_feature_sets(
            list(test_images),
            img_size=config.img_size,
            feature_combinations=features_combinations,
            base_features=test_base_features,
        )
        tc_out = train_candidates(
            train_feature_sets=train_feature_sets,
            train_labels=train_labels,
            split=split,
            seed=config.seed,
            show_progress=config.show_progress,
            test_labels=test_labels,
            test_feature_sets=test_feature_sets,
        )
        train_results = tc_out.results

    if not train_results:
        raise ValueError("No train results produced for model selection.")

    # Test accuracy is computed inside train_candidates after each final fit
    # (held-out test; selection order remains CV-based).
    train_results_sorted_by_test_accuracy = sorted(train_results, key=lambda x: x.test_accuracy, reverse=True)

    train_results_by_model: dict[str, list[TrainResult]] = {}
    for result in train_results_sorted_by_test_accuracy:
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
        for model in train_results_sorted_by_test_accuracy:
            models_to_save.append(model)

        for i, top_model in enumerate(models_to_save):
            save(
                top_model.estimator,
                model_save_dir / f"{1+i}_{top_model.model_name}_{top_model.feature_name}.pkl",
            )

            

    test_paths = (
        np.asarray(test_dataset.paths, dtype=object)
        if test_dataset is not None and test_dataset.paths is not None
        else None
    )

    return MLPipelineOutput(
        dataset_dir=config.data_root / config.dataset_subdir,
        output_dir=output_dir,
        train_results=train_results,
        train_results_by_model=train_results_by_model,
        train_results_sorted_by_test_accuracy=train_results_sorted_by_test_accuracy,
        seed=config.seed,
        labels=train_labels,
        feature_sets=train_feature_sets,
        test_labels=test_labels,
        test_feature_sets=test_feature_sets,
        test_paths=test_paths,
        split=split,
        train_idx_for_fit=train_idx_for_fit,
        val_idx_for_eval=val_idx_for_eval,
        fitted_pcas=fitted_pcas,
    )
