from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1)


class CompletionResponse(BaseModel):
    output: str
    model_used: str
    tier: int
    reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class ModelInfo(BaseModel):
    id: str
    provider: str
    cost_per_input_token: float
    cost_per_output_token: float
    quality_tier: int


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
