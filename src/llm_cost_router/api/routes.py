from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from llm_cost_router.api.schemas import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelsResponse,
)
from llm_cost_router.models.registry import list_models
from llm_cost_router.models.types import ProviderRequestError
from llm_cost_router.providers import send_request
from llm_cost_router.storage.request_log import log_request
from llm_cost_router.storage.stats import Stats, compute_stats
from llm_cost_router.verification.verifier import JUDGE_MODEL_ID, verify_response

router = APIRouter()


@router.post("/v1/completions", response_model=CompletionResponse)
def create_completion(
    body: CompletionRequest, request: Request, background_tasks: BackgroundTasks
) -> CompletionResponse:
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

    request_id = log_request(
        prompt=body.prompt,
        tier=classification.tier.value,
        model_id=model_config.id,
        provider=model_config.provider,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
    )

    # Verifying the judge model's own responses against itself would just
    # always score a perfect match while still costing two extra API calls.
    if model_config.id != JUDGE_MODEL_ID:
        background_tasks.add_task(
            verify_response,
            request_id,
            body.prompt,
            response.output_text,
            classification.tier.value,
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


@router.get("/v1/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    return ModelsResponse(
        models=[
            ModelInfo(
                id=m.id,
                provider=m.provider,
                cost_per_input_token=m.cost_per_input_token,
                cost_per_output_token=m.cost_per_output_token,
                quality_tier=m.quality_tier.value,
            )
            for m in list_models()
        ]
    )


@router.get("/v1/stats", response_model=Stats)
def get_stats() -> Stats:
    return compute_stats()
