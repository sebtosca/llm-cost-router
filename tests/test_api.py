from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_cost_router.api.app import create_app
from llm_cost_router.models.types import ProviderRequestError, Response


def test_completions_success() -> None:
    app = create_app()
    fake_response = Response(
        output_text="Paris",
        model_id="gpt-4o-mini",
        provider="openai",
        input_tokens=5,
        output_tokens=1,
        cost_usd=0.000002,
        latency_ms=42.0,
    )
    with patch("llm_cost_router.api.routes.send_request", return_value=fake_response):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/completions", json={"prompt": "What is the capital of France?"}
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["output"] == "Paris"
    assert body["model_used"] == "gpt-4o-mini"
    assert body["tier"] == 1


def test_completions_provider_error_returns_502() -> None:
    app = create_app()
    with patch(
        "llm_cost_router.api.routes.send_request",
        side_effect=ProviderRequestError("boom"),
    ):
        with TestClient(app) as client:
            resp = client.post("/v1/completions", json={"prompt": "hello"})

    assert resp.status_code == 502


def test_completions_empty_prompt_rejected() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/v1/completions", json={"prompt": ""})

    assert resp.status_code == 422
