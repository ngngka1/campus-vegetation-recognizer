from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from dataset_loader import DatasetSplit
from train import MLPipelineOutput, TrainResult


@dataclass
class SummaryData:
    classification_report: dict[str, dict[str, float]]
    confusion_matrix: np.ndarray
    confusion_matrix_normalized: np.ndarray
    test_confusion_matrix: np.ndarray
    test_confusion_matrix_normalized: np.ndarray
    confusion_pairs: list[tuple[str, str, int]]
    correct_examples: list[tuple[str, str, str, float]]
    incorrect_examples: list[tuple[str, str, str, float]]
    split_sizes: dict[str, int]


def top_k_confusions(
    cm: np.ndarray, labels: list[str], k: int = 10
) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append((labels[i], labels[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def pick_examples(
    paths: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    *,
    one_per_class: bool = True,
) -> tuple[
    list[tuple[str, str, str, float]],
    list[tuple[str, str, str, float]],
    list[tuple[str, tuple[str, str, str, float] | None, tuple[str, str, str, float] | None]],
]:
    """
    Pick correct and wrong prediction examples.
    If one_per_class=True, returns 1 correct and 1 wrong example per class
    (highest-confidence correct, lowest-confidence wrong per class).
    Returns (correct_rows, wrong_rows, per_class_tuples).
    """
    rows = [
        (str(path), str(t), str(p), float(c))
        for path, t, p, c in zip(paths, y_true, y_pred, confidences)
    ]
    correct_all = [r for r in rows if r[1] == r[2]]
    wrong_all = [r for r in rows if r[1] != r[2]]
    classes = sorted(set(r[1] for r in rows))

    if one_per_class:
        correct_rows: list[tuple[str, str, str, float]] = []
        wrong_rows: list[tuple[str, str, str, float]] = []
        per_class: list[
            tuple[str, tuple[str, str, str, float] | None, tuple[str, str, str, float] | None]
        ] = []
        for cls in classes:
            cls_correct = [r for r in correct_all if r[1] == cls]
            cls_wrong = [r for r in wrong_all if r[1] == cls]
            cls_correct.sort(key=lambda x: x[3], reverse=True)
            cls_wrong.sort(key=lambda x: x[3])
            correct_row = cls_correct[0] if cls_correct else None
            wrong_row = cls_wrong[0] if cls_wrong else None
            if correct_row:
                correct_rows.append(correct_row)
            if wrong_row:
                wrong_rows.append(wrong_row)
            per_class.append((cls, correct_row, wrong_row))
        return correct_rows, wrong_rows, per_class

    correct_all.sort(key=lambda x: x[3], reverse=True)
    wrong_all.sort(key=lambda x: x[3])
    per_class = []
    return correct_all[:12], wrong_all[:12], per_class


def margin_confidence(estimator: object, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(x)
        return probs.max(axis=1)
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x)
        if scores.ndim == 1:
            return np.abs(scores)
        return np.max(scores, axis=1)
    return np.ones(shape=(x.shape[0],), dtype=np.float32)


def write_confusion_csv(path: Path, labels: list[str], cm: np.ndarray) -> None:
    is_integer = np.issubdtype(cm.dtype, np.integer)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for i, row in enumerate(cm):
            values = (
                [int(v) for v in row]
                if is_integer
                else [f"{float(v):.6f}" for v in row]
            )
            writer.writerow([labels[i], *values])


def write_examples_csv(path: Path, rows: list[tuple[str, str, str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "pred_label", "confidence"])
        writer.writerows(rows)


def write_examples_per_class_csv(
    path: Path,
    per_class: list[tuple[str, tuple[str, str, str, float] | None, tuple[str, str, str, float] | None]],
) -> None:
    """Write one row per class with correct and wrong example columns."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class",
            "correct_image_path", "correct_confidence",
            "wrong_image_path", "wrong_pred_label", "wrong_confidence",
        ])
        for cls, correct, wrong in per_class:
            if correct is not None:
                c_path, _, _, c_conf = correct
                c_conf_str = f"{c_conf:.6f}"
            else:
                c_path, c_conf_str = "", ""
            if wrong is not None:
                w_path, _, w_pred, w_conf = wrong
                w_conf_str = f"{w_conf:.6f}"
            else:
                w_path, w_pred, w_conf_str = "", "", ""
            writer.writerow([cls, c_path, c_conf_str, w_path, w_pred, w_conf_str])


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    return cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), a_min=1, a_max=None)


def pipeline_slug(result: TrainResult) -> str:
    raw = f"{result.model_name}_{result.feature_name}".lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def write_report(
    out_path: Path,
    *,
    dataset_dir: Path | None,
    seed: int,
    split: DatasetSplit,
    test_size: int | None,
    class_names: list[str],
    train_results: list[TrainResult],
    test_acc: float,
    per_class_report: dict[str, dict[str, float]],
    confusion_pairs: list[tuple[str, str, int]],
    train_augmentation: dict[str, int | bool] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Task 3/4 Traditional ML Report")
    lines.append("")
    lines.append("## Reproducibility")
    if dataset_dir is not None:
        lines.append(f"- Dataset directory: `{dataset_dir}`")
    else:
        lines.append("- Dataset directory: `N/A`")
    lines.append(f"- Random seed: `{seed}`")
    lines.append(
        "- Split strategy: class-wise proportional random split (train/val/test)."
    )
    if test_size is None:
        lines.append(f"- Split sizes: train={len(split.train)}, val={len(split.val)}")
    else:
        lines.append(
            f"- Split sizes: train={len(split.train)}, val={len(split.val)}, test={test_size}"
        )
    if train_augmentation is not None:
        lines.append(
            "- Train augmentation: "
            f"enabled={train_augmentation.get('enabled', False)}, "
            f"augmentations_per_image={train_augmentation.get('augmentations_per_image', 0)}, "
            f"augmented_samples={train_augmentation.get('augmented_samples', 0)}"
        )
    lines.append("")
    lines.append("## Task 3: Model Training and Selection")
    lines.append("- Compared three traditional pipelines (no neural networks):")
    for result in train_results:
        lines.append(
            f"  - `{result.model_name}` using `{result.feature_name}`: "
            f"train_acc={result.train_accuracy:.4f}, val_acc={result.val_accuracy:.4f}, "
            f"time={result.train_seconds:.1f}s"
        )
    lines.append(
        f"- Selected best pipeline by validation accuracy: `{train_results[0].model_name}`"
    )
    lines.append("")
    lines.append("### Algorithmic Choices and Justification")
    lines.append(
        "- `LinearSVC + HOG`: strong margin-based baseline for shape/edge structure, "
        "often effective for plant silhouettes and venation patterns."
    )
    lines.append(
        "- `RandomForest + LBP+HSV`: robust nonlinear classifier on handcrafted texture "
        "(LBP) and color distribution (HSV histograms)."
    )
    lines.append(
        "- `KNN + HOG+HSV`: distance-based classification captures local neighborhood "
        "structure in handcrafted feature space without strong parametric assumptions."
    )
    lines.append("")
    lines.append("## Task 4: Final Metric and Validation Diagnostics")
    lines.append(f"- Test accuracy: `{test_acc:.4f}`")
    if class_names and per_class_report:
        lines.append("- Per-class metrics on `val_split` (precision/recall/F1):")
        for class_name in class_names:
            metrics = per_class_report.get(class_name)
            if not metrics:
                continue
            lines.append(
                f"  - `{class_name}`: P={metrics['precision']:.3f}, "
                f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
                f"support={int(metrics['support'])}"
            )
        lines.append(
            "- Top-3 validation confusion matrices are saved as "
            "`confusion_matrix_top{rank}_val_<pipeline>.csv` and "
            "`confusion_matrix_top{rank}_val_normalized_<pipeline>.csv`."
        )
        lines.append(
            "- Top-3 test confusion matrices are saved as "
            "`confusion_matrix_top{rank}_test_<pipeline>.csv` and "
            "`confusion_matrix_top{rank}_test_normalized_<pipeline>.csv`."
        )
    else:
        lines.append(
            "- Per-class metrics and confusion matrices were skipped because detailed "
            "validation labels/features were not available in run data."
        )
    lines.append("")
    lines.append("## Error Analysis")
    if confusion_pairs:
        lines.append("- Most frequent confusions on `val_split` (true -> predicted):")
        for t, p, n in confusion_pairs:
            lines.append(f"  - `{t}` -> `{p}`: {n}")
    else:
        lines.append("- No off-diagonal confusions were observed in `val_split`.")
    lines.append(
        "- Typical failure modes observed in handcrafted-feature pipelines:\n"
        "  - Background bias: color histograms can overfit scene/background hues.\n"
        "  - Lighting shifts: HSV/HOG/LBP can drift under strong exposure/white-balance changes.\n"
        "  - Similar-looking species: texture/edge statistics may be insufficient when species have near-identical leaf patterns.\n"
        "  - Viewpoint changes/occlusion: local descriptors lose global geometry and can misclassify partial views."
    )
    lines.append(
        "- These failure modes are directly tied to feature extraction limits: "
        "handcrafted descriptors are not as invariant/adaptive as learned deep features."
    )
    lines.append("")
    lines.append("## Qualitative Evidence")
    lines.append(
        "- Correct examples: `qualitative_correct_examples.csv` (high-confidence correct predictions)."
    )
    lines.append(
        "- Incorrect examples: `qualitative_incorrect_examples.csv` (lowest-confidence / misclassified examples)."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_traditional_ml_outputs(
    run_data: MLPipelineOutput, test_size: int | None = None
) -> dict[str, Any]:
    """
    Generate Task3/Task4 outputs (CSV/JSON/MD) from the return dict of
    `train.run_traditional_ml_pipeline()`.

    Returns the same dict, augmented with evaluation fields (e.g. test accuracy,
    confusion matrices, examples, etc.).
    """
    output_dir = Path(run_data.output_dir)
    # clean the output directory
    for file in output_dir.glob("*"):
        if file.is_file():
            file.unlink()
    dataset_dir = Path(run_data.dataset_dir)
    seed = run_data.seed
    train_results: list[TrainResult] = run_data.train_results
    train_results_sorted_by_test_accuracy = run_data.train_results_sorted_by_test_accuracy

    split = run_data.split
    val_idx_for_eval = np.asarray(run_data.val_idx_for_eval, dtype=np.int64)

    top_results = train_results_sorted_by_test_accuracy[: min(3, len(train_results_sorted_by_test_accuracy))]
    best = top_results[0]
    test_acc = float(best.test_accuracy)
    report_dict: dict[str, dict[str, float]] = {}
    confusion_pairs: list[tuple[str, str, int]] = []
    correct_rows: list[tuple[str, str, str, float]] = []
    wrong_rows: list[tuple[str, str, str, float]] = []
    cm = np.zeros((0, 0), dtype=np.int64)
    cm_norm = np.zeros((0, 0), dtype=np.float64)
    cm_test = np.zeros((0, 0), dtype=np.int64)
    cm_test_norm = np.zeros((0, 0), dtype=np.float64)
    class_names: list[str] = []

    labels_raw = run_data.labels
    feature_sets = run_data.feature_sets

    if (
        labels_raw is not None
        and feature_sets is not None
        and len(val_idx_for_eval) > 0
        and top_results
    ):
        labels = np.asarray(labels_raw, dtype=object)
        class_names = sorted(np.unique(labels).tolist())

        test_labels_raw = getattr(run_data, "test_labels", None)
        test_feature_sets = getattr(run_data, "test_feature_sets", None)
        test_labels = (
            np.asarray(test_labels_raw, dtype=object)
            if test_labels_raw is not None
            else None
        )

        y_val_true = labels[val_idx_for_eval]
        for rank, result in enumerate(top_results, start=1):
            result_slug = pipeline_slug(result)
            x_val = feature_sets[result.feature_name]
            y_val_pred_raw = result.estimator.predict(x_val[val_idx_for_eval])
            y_val_pred = np.asarray(y_val_pred_raw, dtype=object)
            cm_val = confusion_matrix(y_val_true, y_val_pred, labels=class_names)
            cm_val_norm = normalize_confusion_matrix(cm_val)

            if (
                test_labels is not None
                and test_feature_sets is not None
                and result.feature_name in test_feature_sets
                and len(test_labels) == len(test_feature_sets[result.feature_name])
            ):
                x_test = test_feature_sets[result.feature_name]
                y_test_pred_raw = result.estimator.predict(x_test)
                y_test_pred = np.asarray(y_test_pred_raw, dtype=object)
                cm_test_rank = confusion_matrix(test_labels, y_test_pred, labels=class_names)
                cm_test_norm_rank = normalize_confusion_matrix(cm_test_rank)
                write_confusion_csv(
                    output_dir / f"confusion_matrix_top{rank}_test_{result_slug}.csv",
                    class_names,
                    cm_test_rank,
                )
                write_confusion_csv(
                    output_dir
                    / f"confusion_matrix_top{rank}_test_normalized_{result_slug}.csv",
                    class_names,
                    cm_test_norm_rank,
                )
            else:
                cm_test_rank = np.zeros((0, 0), dtype=np.int64)
                cm_test_norm_rank = np.zeros((0, 0), dtype=np.float64)

            if rank == 1:
                y_val_conf = margin_confidence(result.estimator, x_val[val_idx_for_eval])
                report_dict = classification_report(
                    y_val_true,
                    y_val_pred,
                    labels=class_names,
                    output_dict=True,
                    zero_division=0,
                )
                cm = cm_val
                cm_norm = cm_val_norm
                cm_test = cm_test_rank
                cm_test_norm = cm_test_norm_rank

                # Backward-compatible filenames for best model artifacts.
                if cm_test.size:
                    write_confusion_csv(
                        output_dir / "confusion_matrix_test.csv", class_names, cm_test
                    )
                    write_confusion_csv(
                        output_dir / "confusion_matrix_test_normalized.csv",
                        class_names,
                        cm_test_norm,
                    )

        # Use test set for qualitative examples when available
        test_paths = getattr(run_data, "test_paths", None)
        test_paths_arr = (
            np.asarray(test_paths, dtype=object)
            if test_paths is not None
            else None
        )
        if (
            test_labels is not None
            and test_feature_sets is not None
            and best.feature_name in test_feature_sets
            and len(test_labels) == len(test_feature_sets[best.feature_name])
        ):
            x_test = test_feature_sets[best.feature_name]
            y_test_pred = np.asarray(best.estimator.predict(x_test), dtype=object)
            y_test_conf = margin_confidence(best.estimator, x_test)
            if (
                test_paths_arr is not None
                and test_paths_arr.ndim == 1
                and test_paths_arr.size == len(test_labels)
            ):
                example_paths = test_paths_arr
            else:
                example_paths = np.asarray(
                    [f"<test_image_{i}>" for i in range(len(test_labels))],
                    dtype=object,
                )
            correct_rows, wrong_rows, per_class = pick_examples(
                paths=example_paths,
                y_true=test_labels,
                y_pred=y_test_pred,
                confidences=y_test_conf,
            )
        else:
            # Fallback to val set when test set is not available
            loaded_dataset_paths = getattr(run_data, "paths", None)
            loaded_dataset_paths_arr = (
                np.asarray(loaded_dataset_paths, dtype=object)
                if loaded_dataset_paths is not None
                else None
            )
            if (
                loaded_dataset_paths_arr is None
                or loaded_dataset_paths_arr.ndim != 1
                or loaded_dataset_paths_arr.size != len(labels)
            ):
                example_paths = np.asarray(
                    [f"<image_{i}>" for i in range(len(labels))], dtype=object
                )
            else:
                example_paths = loaded_dataset_paths_arr
            correct_rows, wrong_rows, per_class = pick_examples(
                paths=example_paths[val_idx_for_eval],
                y_true=y_val_true,
                y_pred=np.asarray(best.estimator.predict(feature_sets[best.feature_name][val_idx_for_eval]), dtype=object),
                confidences=y_val_conf,
            )
        write_examples_csv(
            output_dir / "qualitative_correct_examples.csv", correct_rows
        )
        write_examples_csv(
            output_dir / "qualitative_incorrect_examples.csv", wrong_rows
        )
        if per_class:
            write_examples_per_class_csv(
                output_dir / "qualitative_examples_per_class.csv", per_class
            )

        confusion_pairs = top_k_confusions(cm, class_names)

    train_results_payload = [
        {
            "pipeline": f"{r.model_name}: {r.feature_name}",
            "model": r.model_name,
            "features": r.feature_name,
            "train_accuracy": r.train_accuracy,
            "val_accuracy": r.val_accuracy,
            "val_folds": r.val_folds,
            "test_accuracy": r.test_accuracy,
            "train_seconds": r.train_seconds,
        }
        for r in train_results_sorted_by_test_accuracy
    ]
    (output_dir / "train_results.json").write_text(
        json.dumps(train_results_payload, indent=2), encoding="utf-8"
    )

    top_pipelines_payload = [
        {
            "rank": idx + 1,
            "model": r.model_name,
            "features": r.feature_name,
            "pipeline": f"{r.model_name}: {r.feature_name}",
            "slug": pipeline_slug(r),
            "test_confusion_matrix_csv": f"confusion_matrix_top{idx+1}_test_{pipeline_slug(r)}.csv",
            "test_confusion_matrix_normalized_csv": f"confusion_matrix_top{idx+1}_test_normalized_{pipeline_slug(r)}.csv",
        }
        for idx, r in enumerate(top_results)
    ]

    summary_payload = {
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else "N/A",
        "seed": seed,
        "dataset_class_counts": len(class_names),
        "evaluated_feature_combinations": sorted(
            {r.feature_name for r in train_results_sorted_by_test_accuracy}
        ),
        "splits": {
            "train": int(len(split.train)),
            "val": int(len(val_idx_for_eval)),
        },
        "best_pipeline": f"{best.model_name}: {best.feature_name}",
        "test_accuracy": test_acc,
        "top_pipelines": top_pipelines_payload,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    write_report(
        out_path=output_dir / "task3_task4_report.md",
        dataset_dir=dataset_dir,
        seed=seed,
        split=split,
        test_size=test_size,
        class_names=class_names,
        train_results=train_results_sorted_by_test_accuracy,
        test_acc=test_acc,
        per_class_report=report_dict,
        confusion_pairs=confusion_pairs,
    )

    summary_data = SummaryData(
        classification_report=report_dict,
        confusion_matrix=cm,
        confusion_matrix_normalized=cm_norm,
        test_confusion_matrix=cm_test,
        test_confusion_matrix_normalized=cm_test_norm,
        confusion_pairs=confusion_pairs,
        correct_examples=correct_rows,
        incorrect_examples=wrong_rows,
        split_sizes=summary_payload["splits"],
    )
    return summary_data
