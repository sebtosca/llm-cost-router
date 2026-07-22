import pytest

from llm_cost_router.classifier.base import ClassificationResult
from llm_cost_router.models.types import Tier
from llm_cost_router.router.config import RoutingConfig, load_routing_config
from llm_cost_router.router.router import Router
from llm_cost_router import settings


def test_load_routing_config_valid() -> None:
    config = load_routing_config(settings.ROUTING_CONFIG_PATH)
    assert config.model_id_for(Tier.TIER_1)
    assert config.model_id_for(Tier.TIER_2)
    assert config.model_id_for(Tier.TIER_3)


def test_load_routing_config_unknown_model_id_raises(tmp_path) -> None:
    bad_yaml = tmp_path / "routing.yaml"
    bad_yaml.write_text(
        "routing:\n  tier_1: not-a-real-model\n  tier_2: gpt-4o-mini\n  tier_3: gpt-4o\n"
    )
    with pytest.raises(ValueError, match="unknown model id"):
        load_routing_config(bad_yaml)


def test_routing_config_missing_tier_raises() -> None:
    with pytest.raises(Exception):
        RoutingConfig.model_validate({"routing": {"tier_1": "gpt-4o-mini"}})


def test_router_route_resolves_model_and_reason() -> None:
    config = RoutingConfig.model_validate(
        {
            "routing": {
                "tier_1": "gpt-4o-mini",
                "tier_2": "gemini-3-flash",
                "tier_3": "claude-sonnet-5",
            }
        }
    )
    router = Router(config)
    classification = ClassificationResult(tier=Tier.TIER_2, reason="test reason", features={})
    model_config, reason = router.route(classification)

    assert model_config.id == "gemini-3-flash"
    assert "test reason" in reason
    assert "tier_2" in reason
