from enum import IntEnum

from pydantic import BaseModel


class Tier(IntEnum):
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3


class ModelConfig(BaseModel):
    model_config = {"frozen": True}

    id: str
    provider: str
    provider_model_name: str
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency_ms: float | None = None
    quality_tier: Tier
    max_output_tokens: int | None = None


class Response(BaseModel):
    output_text: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    raw_finish_reason: str | None = None


class ProviderRequestError(Exception):
    """Raised when a provider adapter fails to complete a request."""


class UnsupportedProviderError(Exception):
    """Raised when no adapter is registered for a ModelConfig's provider."""
