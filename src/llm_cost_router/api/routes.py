from fastapi import APIRouter, HTTPException, Request

from llm_cost_router.api.schemas import CompletionRequest, CompletionResponse
from llm_cost_router.models.types import ProviderRequestError
from llm_cost_router.providers import send_request

router = APIRouter()


@router.post("/v1/completions", response_model=CompletionResponse)
def create_completion(body: CompletionRequest, request: Request) -> CompletionResponse:
    classifier = request.app.state.classifier
    app_router = request.app.state.router

    classification = classifier.classify(body.prompt)
    model_config, reason = app_router.route(classification)

    try:
        response = send_request(body.prompt, model_config)
    except ProviderRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return CompletionResponse(
        output=response.output_text,
        model_used=model_config.id,
        tier=classification.tier.value,
        reason=reason,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
    )
