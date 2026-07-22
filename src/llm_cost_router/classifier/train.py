import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from llm_cost_router.classifier.features import feature_vector


def load_labeled_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_feature_matrix(records: list[dict]) -> tuple[list[list[float]], list[int]]:
    X = [feature_vector(r["prompt"]) for r in records]
    y = [r["tier"] for r in records]
    return X, y


def fit_model(X: list[list[float]], y: list[int]) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def evaluate_model(model: LogisticRegression, X_test: list[list[float]], y_test: list[int]) -> dict:
    predictions = model.predict(X_test)
    labels = sorted(set(y_test))
    return {
        "accuracy": model.score(X_test, y_test),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "labels": labels,
    }


def train_and_evaluate(
    records: list[dict], test_size: float = 0.2, random_state: int = 42
) -> dict:
    X, y = build_feature_matrix(records)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = fit_model(X_train, y_train)
    eval_result = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "n_train": len(X_train),
        "n_test": len(X_test),
        **eval_result,
    }


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"No trained classifier model at {path}. Run "
            f"`python scripts/train_classifier.py` first."
        )
    return joblib.load(path)
