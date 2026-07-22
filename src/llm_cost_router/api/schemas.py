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
