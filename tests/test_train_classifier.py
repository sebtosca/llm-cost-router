from llm_cost_router import settings
from llm_cost_router.classifier.train import (
    build_feature_matrix,
    load_labeled_dataset,
    train_and_evaluate,
)

DATASET_PATH = settings.PROJECT_ROOT / "data" / "labeled_prompts.json"


def test_load_labeled_dataset_has_200_plus_examples() -> None:
    records = load_labeled_dataset(DATASET_PATH)
    assert len(records) >= 200
    tiers = {r["tier"] for r in records}
    assert tiers == {1, 2, 3}


def test_build_feature_matrix_shapes() -> None:
    records = load_labeled_dataset(DATASET_PATH)
    X, y = build_feature_matrix(records)
    assert len(X) == len(y) == len(records)
    assert len(X[0]) == 6


def test_train_and_evaluate_meets_accuracy_target() -> None:
    records = load_labeled_dataset(DATASET_PATH)
    result = train_and_evaluate(records)

    assert result["accuracy"] > 0.8
    assert result["n_test"] > 0
    assert result["labels"] == [1, 2, 3]
