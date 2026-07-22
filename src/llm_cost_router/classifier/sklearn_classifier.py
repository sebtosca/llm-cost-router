from pathlib import Path

from llm_cost_router.classifier.base import ClassificationResult
from llm_cost_router.classifier.features import FEATURE_NAMES, feature_vector
from llm_cost_router.classifier.train import load_model
from llm_cost_router.models.types import Tier


class SklearnClassifier:
    def __init__(self, model_path: Path) -> None:
        self._model = load_model(model_path)

    def classify(self, prompt: str) -> ClassificationResult:
        vector = feature_vector(prompt)
        predicted_tier = int(self._model.predict([vector])[0])
        confidence = float(max(self._model.predict_proba([vector])[0]))
        features = dict(zip(FEATURE_NAMES, vector))

        reason = (
            f"sklearn LogisticRegression predicted Tier {predicted_tier} "
            f"(confidence={confidence:.2f}) from features {features}"
        )
        return ClassificationResult(tier=Tier(predicted_tier), reason=reason, features=features)
