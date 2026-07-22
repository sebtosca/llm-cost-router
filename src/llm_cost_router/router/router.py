from llm_cost_router.classifier.base import ClassificationResult
from llm_cost_router.models import registry
from llm_cost_router.models.types import ModelConfig
from llm_cost_router.router.config import RoutingConfig


class Router:
    def __init__(self, routing_config: RoutingConfig) -> None:
        self._routing_config = routing_config

    def route(self, classification: ClassificationResult) -> tuple[ModelConfig, str]:
        model_id = self._routing_config.model_id_for(classification.tier)
        model_config = registry.get_model(model_id)
        reason = (
            f"{classification.reason}; routed to '{model_id}' "
            f"per routing.yaml tier_{classification.tier.value} mapping"
        )
        return model_config, reason
