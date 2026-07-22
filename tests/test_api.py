from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_cost_router.api.app import create_app
from llm_cost_router.models.types import ProviderRequestError, Response
from llm_cost_router.storage.db import get_connection


def _fake_response(model_id: str = "gpt-4o-mini", provider: str = "openai", text: str = "Paris") -> Response:
    return Response(
        output_text=text,
        model_id=model_id,
        provider=provider,
        input_tokens=5,
        output_tokens=1,
        cost_usd=0.000002,
        latency_ms=42.0,
    )


def test_completions_success() -> None:
    app = create_app()
    # The background verification job also calls send_request (via the
    # verifier module's own import) - mock it too so this test doesn't make
    # a real network call to the judge model.
    with patch("llm_cost_router.api.routes.send_request", return_value=_fake_response()):
        with patch(
            "llm_cost_router.verification.verifier.send_request",
            return_value=_fake_response(text="5"),
        ):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/completions", json={"prompt": "What is the capital of France?"}
                )

    assert resp.status_code == 200
    body = resp.json()
    assert body["output"] == "Paris"
    assert body["model_used"] == "gpt-4o-mini"
    assert body["tier"] == 1


def test_completions_triggers_background_verification() -> None:
    app = create_app()
    with patch(
        "llm_cost_router.api.routes.send_request",
        return_value=_fake_response(text="Paris"),
    ):
        with patch(
            "llm_cost_router.verification.verifier.send_request",
            side_effect=[_fake_response(text="Paris is the capital of France."), _fake_response(text="5")],
        ):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/completions", json={"prompt": "What is the capital of France?"}
                )

    assert resp.status_code == 200
    with get_connection() as conn:
        row = conn.execute(
            "SELECT quality_score FROM request_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row[0] == 5.0


def test_completions_adversarial_prompt_shows_up_escalated() -> None:
    app = create_app()
    reference_text = "The actual nuanced multi-step reasoning answer."
    with patch(
        "llm_cost_router.api.routes.send_request",
        return_value=_fake_response(text="a shallow, wrong answer"),
    ):
        with patch(
            "llm_cost_router.verification.verifier.send_request",
            side_effect=[_fake_response(text=reference_text), _fake_response(text="1")],
        ):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/completions", json={"prompt": "What is the capital of France?"}
                )

    assert resp.status_code == 200
    with get_connection() as conn:
        row = conn.execute(
            "SELECT quality_score, escalated, escalated_model_id FROM request_log "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row == (1.0, 1, "claude-sonnet-5")


def test_completions_skips_verification_when_judge_model_itself_is_routed() -> None:
    app = create_app()
    with patch(
        "llm_cost_router.api.routes.send_request",
        return_value=_fake_response(model_id="claude-sonnet-5", provider="anthropic"),
    ) as mock_send:
        with patch("llm_cost_router.verification.verifier.send_request") as mock_verify_send:
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/completions",
                    json={
                        "prompt": (
                            "Compare the trade-offs between a relational database and a "
                            "document store for a high-write telemetry system. Analyze "
                            "latency, consistency, and schema flexibility step by step, "
                            "then give a final recommendation. Output as JSON with fields: "
                            "options, tradeoffs, recommendation."
                        )
                    },
                )

    assert resp.status_code == 200
    assert mock_send.call_count == 1
    mock_verify_send.assert_not_called()


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


def test_get_models_lists_registry() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/v1/models")

    assert resp.status_code == 200
    models = resp.json()["models"]
    ids = {m["id"] for m in models}
    assert "gpt-4o-mini" in ids
    assert "claude-sonnet-5" in ids
    assert "gemini-3-flash" in ids


def test_get_stats_empty_db() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/v1/stats")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_requests"] == 0
    assert body["total_cost_usd"] == 0.0
