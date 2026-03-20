from __future__ import annotations

print("Resolving imports...")

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

from dataset_loader import (  # noqa: E402
    build_feature_sets,
    extract_base_feature_sets,
    reduce_base_features_with_pca,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
KNOWN_MODELS = ("random_forest", "svm", "knn")
MODEL_SUFFIXES = {".pkl", ".joblib"}
_MODEL_CACHE: dict[str, object] = {}
_PCAS_CACHE: dict[str, object | None] = {}
DEFAULT_IMG_SIZE = 128
DEFAULT_INCLUDE_SURF = False


def _iter_files(root: Path, suffixes: Iterable[str]) -> list[Path]:
    if not root.exists():
        return []
    allowed = {s.lower() for s in suffixes}
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed],
        key=lambda p: str(p).lower(),
    )


def _find_default_model_path() -> str:
    candidates = _find_model_paths()
    return str(candidates[0]) if candidates else ""


def _find_model_paths() -> list[str]:
    models_dir = PROJECT_ROOT / "ml_results" / "models"
    candidates = _iter_files(models_dir, MODEL_SUFFIXES)
    return [str(p) for p in candidates]


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


def _get_cached_model(model_file: Path) -> object:
    key = str(model_file.resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load(model_file)
    return _MODEL_CACHE[key]


def _feature_combo_for_model(model_path: str) -> str:
    combo = _infer_feature_combo_from_model_name(model_path).strip()
    if not combo:
        raise ValueError(
            "Cannot infer feature combination from selected model filename."
        )
    return combo


def _get_pcas_for_model(model_path: str) -> object | None:
    """Load fitted PCAs if the model was trained with PCA per block."""
    models_dir = Path(model_path).expanduser().parent
    pcas_file = models_dir / "pcas.joblib"
    cache_key = str(models_dir.resolve())
    if cache_key not in _PCAS_CACHE:
        _PCAS_CACHE[cache_key] = load(pcas_file) if pcas_file.exists() else None
    return _PCAS_CACHE[cache_key]


def _build_features_for_inference(
    image_bgr: np.ndarray,
    feature_combo: str,
    img_size: int = DEFAULT_IMG_SIZE,
    fitted_pcas: object | None = None,
) -> np.ndarray:
    """Build features for inference, applying PCA if the model was trained with it."""
    base = extract_base_feature_sets(
        [image_bgr],
        img_size=img_size,
        include_surf_if_available=DEFAULT_INCLUDE_SURF,
    )
    if fitted_pcas is not None:
        base, _ = reduce_base_features_with_pca(base, fitted_pcas=fitted_pcas, n_components=400, features_to_reduce=["hog", "color_hist"])
    feature_sets = build_feature_sets(
        [image_bgr],
        img_size=img_size,
        feature_combinations=(feature_combo,),
        include_surf_if_available=DEFAULT_INCLUDE_SURF,
        base_features=base,
    )
    return feature_sets[feature_combo]


def _predict_from_bgr(
    image_bgr: np.ndarray,
    model_path: str,
) -> tuple[str, str]:
    model_file = Path(model_path).expanduser()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_file}")
    cleaned_feature_combo = _feature_combo_for_model(model_path)
    model = _get_cached_model(model_file)
    fitted_pcas = _get_pcas_for_model(model_path)
    x = _build_features_for_inference(
        image_bgr,
        cleaned_feature_combo,
        img_size=DEFAULT_IMG_SIZE,
        fitted_pcas=fitted_pcas,
    )
    pred = model.predict(x)[0]

    details = [
        f"Predicted class: {pred}",
        f"Model: {model_file.name}",
        f"Feature combination: {cleaned_feature_combo}",
        f"Input size: {DEFAULT_IMG_SIZE}x{DEFAULT_IMG_SIZE}",
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
) -> tuple[str, str]:
    bgr = _preprocess_to_bgr(image_rgb, img_size=DEFAULT_IMG_SIZE)
    return _predict_from_bgr(
        image_bgr=bgr,
        model_path=model_path,
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
) -> tuple[str, str]:
    if not sample_rel_path:
        raise ValueError("Please select a sample image.")
    img_path = PROJECT_ROOT / sample_rel_path
    if not img_path.exists():
        raise FileNotFoundError(f"Sample image not found: {img_path}")
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read selected sample image: {img_path}")
    resized = cv2.resize(
        bgr, (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), interpolation=cv2.INTER_AREA
    )
    return _predict_from_bgr(
        image_bgr=resized,
        model_path=model_path,
    )


def update_feature_combo(model_path: str) -> str:
    if not model_path:
        return ""
    try:
        return _feature_combo_for_model(model_path)
    except ValueError:
        return ""


def warm_up_default_model(default_model_path: str, img_size: int = 128) -> None:
    if not default_model_path:
        return
    try:
        model_file = Path(default_model_path).expanduser()
        _get_cached_model(model_file)
        combo = _feature_combo_for_model(default_model_path)
        fitted_pcas = _get_pcas_for_model(default_model_path)
        dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        features = _build_features_for_inference(
            dummy,
            combo,
            img_size=img_size,
            fitted_pcas=fitted_pcas,
        )
        _MODEL_CACHE[str(model_file.resolve())].predict(features)
    except Exception as exc:
        print(f"[WARN] Startup warm-up skipped: {exc}")


def build_interface() -> gr.Blocks:
    model_choices = _find_model_paths()
    default_model_path = model_choices[0] if model_choices else ""
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
            with gr.Column(scale=1):
                model_path = gr.Dropdown(
                    label="Select model (.pkl or .joblib)",
                    choices=model_choices,
                    value=default_model_path,
                    allow_custom_value=False,
                )
                feature_combo = gr.Textbox(
                    label="Feature combination",
                    value=inferred_feature,
                    interactive=False,
                )
                pred_label = gr.Textbox(label="Predicted class")
                pred_details = gr.Textbox(label="Details", lines=8)

            with gr.Column(scale=1):
                upload_image = gr.Image(label="Upload image", type="numpy")
                upload_btn = gr.Button("Predict from uploaded image", variant="primary")
                upload_btn.click(
                    fn=predict_uploaded,
                    inputs=[upload_image, model_path],
                    outputs=[pred_label, pred_details],
                )
                sample_preview = gr.Image(
                    label="Selected image preview (from polyu_vegetation/test)",
                    type="numpy",
                )
                sample_dropdown = gr.Dropdown(
                    label="Choose sample image",
                    choices=sample_images,
                    value=sample_images[0] if sample_images else None,
                )
                sample_dropdown.change(
                    fn=load_sample_preview, inputs=sample_dropdown, outputs=sample_preview
                )
                sample_btn = gr.Button("Predict selected sample", variant="secondary")
                sample_btn.click(
                    fn=predict_sample,
                    inputs=[sample_dropdown, model_path],
                    outputs=[pred_label, pred_details],
                )

        model_path.change(
            fn=update_feature_combo, inputs=model_path, outputs=feature_combo
        )

        gr.Markdown(
            """
            Notes:
            - Feature combination is inferred automatically from the selected model filename.
            - Make sure the model was trained with the same feature extraction settings.
            """
        )

    return demo

if __name__ == "__main__":
    print("Warming up default model...")
    warm_up_default_model(_find_default_model_path())
    print("Building interface...")
    app = build_interface()
    print("Launching app...")
    app.launch(server_name="127.0.0.1", server_port=7860)
