from llm_cost_router.models.types import ModelConfig, Tier

# Pricing verified via web search 2026-07-22. OpenAI: openai.com/api/pricing.
# Anthropic: published per-1M-token rates for Claude Sonnet 5 / Haiku 4.5 (Anthropic
# is running introductory pricing of $2/$10 per 1M on Sonnet 5 through 2026-08-31;
# the standard $3/$15 rate below is used so costs don't silently drop after that date).
# Gemini: published per-1M-token rate for Gemini 3 Flash.
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o-mini": ModelConfig(
        id="gpt-4o-mini",
        provider="openai",
        provider_model_name="gpt-4o-mini",
        cost_per_input_token=0.15 / 1_000_000,
        cost_per_output_token=0.60 / 1_000_000,
        avg_latency_ms=800,
        quality_tier=Tier.TIER_1,
        max_output_tokens=1024,
    ),
    "gpt-4o": ModelConfig(
        id="gpt-4o",
        provider="openai",
        provider_model_name="gpt-4o",
        cost_per_input_token=2.50 / 1_000_000,
        cost_per_output_token=10.00 / 1_000_000,
        avg_latency_ms=1500,
        quality_tier=Tier.TIER_3,
        max_output_tokens=2048,
    ),
    "claude-haiku-4-5": ModelConfig(
        id="claude-haiku-4-5",
        provider="anthropic",
        provider_model_name="claude-haiku-4-5-20251001",
        cost_per_input_token=1.00 / 1_000_000,
        cost_per_output_token=5.00 / 1_000_000,
        avg_latency_ms=900,
        quality_tier=Tier.TIER_2,
        max_output_tokens=1024,
    ),
    "claude-sonnet-5": ModelConfig(
        id="claude-sonnet-5",
        provider="anthropic",
        provider_model_name="claude-sonnet-5",
        cost_per_input_token=3.00 / 1_000_000,
        cost_per_output_token=15.00 / 1_000_000,
        avg_latency_ms=1600,
        quality_tier=Tier.TIER_3,
        max_output_tokens=2048,
    ),
    "gemini-3-flash": ModelConfig(
        id="gemini-3-flash",
        provider="gemini",
        provider_model_name="gemini-3-flash",
        cost_per_input_token=0.50 / 1_000_000,
        cost_per_output_token=3.00 / 1_000_000,
        avg_latency_ms=1000,
        quality_tier=Tier.TIER_2,
        max_output_tokens=1024,
    ),
}


def get_model(model_id: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError:
        raise KeyError(f"Unknown model id '{model_id}'") from None


def list_models() -> list[ModelConfig]:
    return list(MODEL_REGISTRY.values())
