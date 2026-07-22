from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from llm_cost_router.models import registry
from llm_cost_router.models.types import Tier

_TIER_KEYS = {"tier_1": Tier.TIER_1, "tier_2": Tier.TIER_2, "tier_3": Tier.TIER_3}


class RoutingConfig(BaseModel):
    routing: dict[str, str]

    @field_validator("routing")
    @classmethod
    def _check_tier_keys(cls, value: dict[str, str]) -> dict[str, str]:
        missing = _TIER_KEYS.keys() - value.keys()
        unknown = value.keys() - _TIER_KEYS.keys()
        if missing:
            raise ValueError(f"routing.yaml is missing tier keys: {sorted(missing)}")
        if unknown:
            raise ValueError(f"routing.yaml has unknown tier keys: {sorted(unknown)}")
        return value

    def model_id_for(self, tier: Tier) -> str:
        return self.routing[f"tier_{tier.value}"]


def validate_routing_config(config: RoutingConfig) -> None:
    """Cross-validates every model id referenced in the config against the
    model registry. Shared by the startup file loader and the runtime
    PUT /v1/routing-config endpoint so both fail with the same clear error."""
    for tier_key, model_id in config.routing.items():
        try:
            registry.get_model(model_id)
        except KeyError:
            raise ValueError(
                f"routing config '{tier_key}' references unknown model id "
                f"'{model_id}' (not in MODEL_REGISTRY)"
            ) from None


def load_routing_config(path: Path) -> RoutingConfig:
    raw = yaml.safe_load(path.read_text())
    config = RoutingConfig.model_validate(raw)
    validate_routing_config(config)
    return config
