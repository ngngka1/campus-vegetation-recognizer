from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import Any
from pathlib import Path
from joblib import dump

def save(model_pipeline: Pipeline, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model_pipeline, path)
    return path

def build_model_factories(seed: int) -> dict[str, Pipeline]:
    model_factories: dict[str, Any] = {
        "svm": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=10000,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    model_factories["knn"] = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",
                    metric="minkowski",
                    p=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    if not model_factories:
        raise ValueError(
            "No enabled models to train. Check model enable/disable flags."
        )
    return model_factories