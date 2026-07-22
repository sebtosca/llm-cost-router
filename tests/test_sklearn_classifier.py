import pytest

from llm_cost_router import settings
from llm_cost_router.classifier.sklearn_classifier import SklearnClassifier
from llm_cost_router.classifier.train import load_labeled_dataset, save_model, train_and_evaluate
from llm_cost_router.models.types import Tier

DATASET_PATH = settings.PROJECT_ROOT / "data" / "labeled_prompts.json"


@pytest.fixture
def trained_model_path(tmp_path):
    records = load_labeled_dataset(DATASET_PATH)
    result = train_and_evaluate(records)
    model_path = tmp_path / "model.joblib"
    save_model(result["model"], model_path)
    return model_path


def test_sklearn_classifier_loads_and_classifies(trained_model_path) -> None:
    classifier = SklearnClassifier(trained_model_path)
    result = classifier.classify("What is the capital of France?")

    assert isinstance(result.tier, Tier)
    assert "features" in result.model_dump()
    assert result.reason


def test_sklearn_classifier_missing_model_raises_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="train_classifier.py"):
        SklearnClassifier(tmp_path / "does_not_exist.joblib")
