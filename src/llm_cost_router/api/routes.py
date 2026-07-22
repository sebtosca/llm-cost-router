from fastapi import APIRouter, HTTPException, Request

from llm_cost_router.api.schemas import CompletionRequest, CompletionResponse
from llm_cost_router.models.types import ProviderRequestError
from llm_cost_router.providers import send_request
from llm_cost_router.storage.request_log import log_request

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
        log_request(
            prompt=body.prompt,
            tier=classification.tier.value,
            model_id=model_config.id,
            provider=model_config.provider,
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    log_request(
        prompt=body.prompt,
        tier=classification.tier.value,
        model_id=model_config.id,
        provider=model_config.provider,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
    )

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
