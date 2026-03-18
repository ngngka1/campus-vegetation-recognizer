# Vegetation Classification Assignment

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Preparation (existing pipeline)

```bash
python .\extract_grouped.py --interval-ms 1000 --clean
python .\expand_dataset_offline.py --source-subdir extracted_grouped --output-subdir extracted_group_augmented --target-fold 15.0 --clean --equalize-classes
```

## Train and Save Models

Run your training flow (e.g. notebook or script) and make sure models are saved under `ml_results/models/`.
The Gradio app can load `.pkl` / `.joblib` files from there.

## Task 5 Application (Local Gradio App)

```bash
python .\vegetation_classifier_app\app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860).

### How to use

1. Provide a model path (or keep the auto-detected one if available).
2. Provide a feature combination used when training (or leave blank to infer from model filename).
3. Either:
   - upload an image in **Upload Image**, or
   - pick an image from **Select Existing Sample**.
4. Click **Predict** to see the predicted plant class.
