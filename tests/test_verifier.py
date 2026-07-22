from unittest.mock import patch

from llm_cost_router.models.types import ProviderRequestError, Response
from llm_cost_router.storage.db import get_connection, init_db
from llm_cost_router.storage.request_log import log_request
from llm_cost_router.verification.verifier import verify_response


def _fake_response(text: str) -> Response:
    return Response(
        output_text=text,
        model_id="claude-sonnet-5",
        provider="anthropic",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0001,
        latency_ms=100.0,
    )


def test_verify_response_writes_quality_score() -> None:
    init_db()
    request_id = log_request(
        prompt="What is the capital of France?",
        tier=1,
        model_id="gpt-4o-mini",
        provider="openai",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.000005,
        latency_ms=200.0,
    )

    with patch(
        "llm_cost_router.verification.verifier.send_request",
        side_effect=[_fake_response("Paris is the capital of France."), _fake_response("5")],
    ):
        verify_response(request_id, "What is the capital of France?", "Paris")

    with get_connection() as conn:
        row = conn.execute(
            "SELECT quality_score FROM request_log WHERE id = ?", (request_id,)
        ).fetchone()

    assert row[0] == 5.0


def test_verify_response_escalates_on_low_score() -> None:
    init_db()
    request_id = log_request(
        prompt="p", tier=1, model_id="gpt-4o-mini", provider="openai"
    )

    reference_response = _fake_response("a much better reference answer")
    with patch(
        "llm_cost_router.verification.verifier.send_request",
        side_effect=[reference_response, _fake_response("2")],
    ):
        verify_response(request_id, "p", "a weak candidate answer")

    with get_connection() as conn:
        row = conn.execute(
            "SELECT quality_score, escalated, escalated_model_id, cost_delta "
            "FROM request_log WHERE id = ?",
            (request_id,),
        ).fetchone()

    assert row == (2.0, 1, "claude-sonnet-5", reference_response.cost_usd)


def test_verify_response_no_escalation_on_high_score() -> None:
    init_db()
    request_id = log_request(
        prompt="p", tier=1, model_id="gpt-4o-mini", provider="openai"
    )

    with patch(
        "llm_cost_router.verification.verifier.send_request",
        side_effect=[_fake_response("a good reference answer"), _fake_response("4")],
    ):
        verify_response(request_id, "p", "a solid candidate answer")

    with get_connection() as conn:
        row = conn.execute(
            "SELECT quality_score, escalated FROM request_log WHERE id = ?",
            (request_id,),
        ).fetchone()

    assert row == (4.0, 0)


def test_verify_response_unparsable_judge_output_leaves_score_null() -> None:
    init_db()
    request_id = log_request(
        prompt="p", tier=1, model_id="gpt-4o-mini", provider="openai"
    )

    with patch(
        "llm_cost_router.verification.verifier.send_request",
        side_effect=[_fake_response("some reference"), _fake_response("not a score")],
    ):
        verify_response(request_id, "p", "candidate")

    with get_connection() as conn:
        row = conn.execute(
            "SELECT quality_score FROM request_log WHERE id = ?", (request_id,)
        ).fetchone()

    assert row[0] is None


def test_verify_response_provider_error_is_swallowed() -> None:
    init_db()
    request_id = log_request(
        prompt="p", tier=1, model_id="gpt-4o-mini", provider="openai"
    )

    with patch(
        "llm_cost_router.verification.verifier.send_request",
        side_effect=ProviderRequestError("boom"),
    ):
        verify_response(request_id, "p", "candidate")  # must not raise

    with get_connection() as conn:
        row = conn.execute(
            "SELECT quality_score FROM request_log WHERE id = ?", (request_id,)
        ).fetchone()

    assert row[0] is None
