from llm_cost_router import settings
from llm_cost_router.classifier.base import Classifier
from llm_cost_router.classifier.heuristic import HeuristicClassifier


def build_classifier() -> Classifier:
    """Picks the Classifier implementation based on settings.CLASSIFIER
    ("heuristic" or "sklearn"). Kept in one place so the API startup and
    tests construct the classifier the same way."""
    if settings.CLASSIFIER == "sklearn":
        from llm_cost_router.classifier.sklearn_classifier import SklearnClassifier

        return SklearnClassifier(settings.CLASSIFIER_MODEL_PATH)
    return HeuristicClassifier()
