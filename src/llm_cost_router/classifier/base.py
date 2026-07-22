from typing import Protocol

from pydantic import BaseModel

from llm_cost_router.models.types import Tier


class ClassificationResult(BaseModel):
    tier: Tier
    reason: str
    features: dict[str, float | int | bool | str]


class Classifier(Protocol):
    def classify(self, prompt: str) -> ClassificationResult: ...
