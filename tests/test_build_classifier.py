from llm_cost_router import settings
from llm_cost_router.classifier import build_classifier
from llm_cost_router.classifier.heuristic import HeuristicClassifier
from llm_cost_router.classifier.sklearn_classifier import SklearnClassifier
from llm_cost_router.classifier.train import load_labeled_dataset, save_model, train_and_evaluate

DATASET_PATH = settings.PROJECT_ROOT / "data" / "labeled_prompts.json"


def test_build_classifier_defaults_to_heuristic(monkeypatch) -> None:
    monkeypatch.setattr(settings, "CLASSIFIER", "heuristic")
    assert isinstance(build_classifier(), HeuristicClassifier)


def test_build_classifier_picks_sklearn_when_configured(monkeypatch, tmp_path) -> None:
    records = load_labeled_dataset(DATASET_PATH)
    result = train_and_evaluate(records)
    model_path = tmp_path / "model.joblib"
    save_model(result["model"], model_path)

    monkeypatch.setattr(settings, "CLASSIFIER", "sklearn")
    monkeypatch.setattr(settings, "CLASSIFIER_MODEL_PATH", model_path)

    assert isinstance(build_classifier(), SklearnClassifier)
