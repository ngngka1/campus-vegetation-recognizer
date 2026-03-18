from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import cv2
import gradio as gr
import numpy as np
from joblib import load

# Ensure project-root imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import build_feature_sets  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
KNOWN_MODELS = ("random_forest", "svm", "knn")


def _iter_files(root: Path, suffixes: Iterable[str]) -> list[Path]:
    if not root.exists():
        return []
    allowed = {s.lower() for s in suffixes}
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed],
        key=lambda p: str(p).lower(),
    )


def _find_default_model_path() -> str:
    models_dir = PROJECT_ROOT / "ml_results" / "models"
    candidates = _iter_files(models_dir, {".pkl", ".joblib"})
    return str(candidates[0]) if candidates else ""


def _find_sample_images() -> list[str]:
    test_dir = PROJECT_ROOT / "polyu_vegetation" / "test"
    files = _iter_files(test_dir, IMAGE_SUFFIXES)
    return [str(p.relative_to(PROJECT_ROOT)) for p in files]


def _infer_feature_combo_from_model_name(model_path: str) -> str:
    stem = Path(model_path).stem
    # Remove ranking prefix from files like "1_svm_hog+sift".
    if "_" in stem and stem.split("_", 1)[0].isdigit():
        stem = stem.split("_", 1)[1]
    for model_name in sorted(KNOWN_MODELS, key=len, reverse=True):
        prefix = f"{model_name}_"
        if stem.startswith(prefix):
            return stem[len(prefix) :]
        if stem == model_name:
            return ""
    return ""


def _preprocess_to_bgr(image_rgb: np.ndarray, img_size: int) -> np.ndarray:
    if image_rgb is None:
        raise ValueError("No image was provided.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image.")
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


def _predict_from_bgr(
    image_bgr: np.ndarray,
    model_path: str,
    feature_combo: str,
    img_size: int,
    include_surf_if_available: bool,
) -> tuple[str, str]:
    model_file = Path(model_path).expanduser()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_file}")

    cleaned_feature_combo = feature_combo.strip()
    if not cleaned_feature_combo:
        inferred = _infer_feature_combo_from_model_name(model_path)
        if not inferred:
            raise ValueError(
                "Feature combination is empty and cannot be inferred from model filename."
            )
        cleaned_feature_combo = inferred

    model = load(model_file)
    feature_sets = build_feature_sets(
        [image_bgr],
        img_size=img_size,
        feature_combinations=(cleaned_feature_combo,),
        include_surf_if_available=include_surf_if_available,
    )
    x = feature_sets[cleaned_feature_combo]
    pred = model.predict(x)[0]

    details = [
        f"Predicted class: {pred}",
        f"Model: {model_file.name}",
        f"Feature combination: {cleaned_feature_combo}",
        f"Input size: {img_size}x{img_size}",
    ]

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None and len(classes) == len(probs):
                top_idx = np.argsort(probs)[::-1][:3]
                top_lines = [
                    f"- {classes[i]}: {float(probs[i]) * 100:.2f}%"
                    for i in top_idx
                ]
                details.append("Top confidence scores:")
                details.extend(top_lines)
        except Exception:
            # Some classifiers may expose predict_proba but fail in edge cases.
            pass

    return str(pred), "\n".join(details)


def predict_uploaded(
    image_rgb: np.ndarray,
    model_path: str,
    feature_combo: str,
    img_size: int,
    include_surf_if_available: bool,
) -> tuple[str, str]:
    bgr = _preprocess_to_bgr(image_rgb, img_size=img_size)
    return _predict_from_bgr(
        image_bgr=bgr,
        model_path=model_path,
        feature_combo=feature_combo,
        img_size=img_size,
        include_surf_if_available=include_surf_if_available,
    )


def load_sample_preview(sample_rel_path: str) -> np.ndarray | None:
    if not sample_rel_path:
        return None
    img_path = PROJECT_ROOT / sample_rel_path
    if not img_path.exists():
        return None
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def predict_sample(
    sample_rel_path: str,
    model_path: str,
    feature_combo: str,
    img_size: int,
    include_surf_if_available: bool,
) -> tuple[str, str]:
    if not sample_rel_path:
        raise ValueError("Please select a sample image.")
    img_path = PROJECT_ROOT / sample_rel_path
    if not img_path.exists():
        raise FileNotFoundError(f"Sample image not found: {img_path}")
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read selected sample image: {img_path}")
    resized = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return _predict_from_bgr(
        image_bgr=resized,
        model_path=model_path,
        feature_combo=feature_combo,
        img_size=img_size,
        include_surf_if_available=include_surf_if_available,
    )


def build_interface() -> gr.Blocks:
    default_model_path = _find_default_model_path()
    sample_images = _find_sample_images()
    inferred_feature = (
        _infer_feature_combo_from_model_name(default_model_path) if default_model_path else ""
    )

    with gr.Blocks(title="Vegetation Classifier") as demo:
        gr.Markdown(
            """
            # Vegetation Classifier
            Upload an image or select one from your local test set, then run inference.
            """
        )

        with gr.Row():
            model_path = gr.Textbox(
                label="Model path (.pkl or .joblib)",
                value=default_model_path,
                placeholder="e.g. ml_results/models/1_svm_hog+sift.pkl",
            )
            feature_combo = gr.Textbox(
                label="Feature combination",
                value=inferred_feature,
                placeholder="e.g. hog+sift",
            )
        with gr.Row():
            img_size = gr.Number(label="Image size", value=128, precision=0)
            include_surf = gr.Checkbox(
                label="Enable SURF feature support if available", value=False
            )

        pred_label = gr.Textbox(label="Predicted class")
        pred_details = gr.Textbox(label="Details", lines=8)

        with gr.Tab("Upload Image"):
            upload_image = gr.Image(label="Upload image", type="numpy")
            upload_btn = gr.Button("Predict from uploaded image", variant="primary")
            upload_btn.click(
                fn=predict_uploaded,
                inputs=[upload_image, model_path, feature_combo, img_size, include_surf],
                outputs=[pred_label, pred_details],
            )

        with gr.Tab("Select Existing Sample"):
            sample_dropdown = gr.Dropdown(
                label="Choose sample image from polyu_vegetation/test",
                choices=sample_images,
                value=sample_images[0] if sample_images else None,
            )
            sample_preview = gr.Image(label="Selected image preview", type="numpy")
            sample_dropdown.change(
                fn=load_sample_preview, inputs=sample_dropdown, outputs=sample_preview
            )
            sample_btn = gr.Button("Predict selected sample", variant="primary")
            sample_btn.click(
                fn=predict_sample,
                inputs=[sample_dropdown, model_path, feature_combo, img_size, include_surf],
                outputs=[pred_label, pred_details],
            )

        gr.Markdown(
            """
            Notes:
            - If *Feature combination* is left empty, the app tries to infer it from the model filename.
            - Make sure the model was trained with the same feature extraction settings.
            """
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="127.0.0.1", server_port=7860)
