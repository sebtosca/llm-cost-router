"""Trains the sklearn tier classifier on data/labeled_prompts.json and saves
the model to data/classifier_model.joblib.

Usage: python scripts/train_classifier.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_cost_router import settings  # noqa: E402
from llm_cost_router.classifier.train import (  # noqa: E402
    load_labeled_dataset,
    save_model,
    train_and_evaluate,
)

DATASET_PATH = settings.PROJECT_ROOT / "data" / "labeled_prompts.json"


def main() -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"No labeled dataset at {DATASET_PATH}. Run "
            f"`python scripts/generate_labeled_dataset.py` first."
        )

    records = load_labeled_dataset(DATASET_PATH)
    result = train_and_evaluate(records)

    print(f"Trained on {result['n_train']} examples, evaluated on {result['n_test']} held-out.")
    print(f"Accuracy: {result['accuracy']:.1%}")
    print(f"Labels: {result['labels']}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    for label, row in zip(result["labels"], result["confusion_matrix"]):
        print(f"  Tier {label}: {row}")

    if result["accuracy"] < 0.8:
        print("\nWARNING: accuracy is below the 80% target for V1.")

    save_model(result["model"], settings.CLASSIFIER_MODEL_PATH)
    print(f"\nSaved model to {settings.CLASSIFIER_MODEL_PATH}")


if __name__ == "__main__":
    main()
