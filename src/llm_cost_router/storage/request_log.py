import hashlib
from datetime import datetime, timezone

from llm_cost_router.storage.db import get_connection


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def log_request(
    *,
    prompt: str,
    tier: int | None,
    model_id: str | None,
    provider: str | None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    latency_ms: float | None = None,
    error: str | None = None,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO request_log
                (timestamp, prompt_hash, tier, model_id, provider,
                 input_tokens, output_tokens, cost_usd, latency_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                hash_prompt(prompt),
                tier,
                model_id,
                provider,
                input_tokens,
                output_tokens,
                cost_usd,
                latency_ms,
                error,
            ),
        )
        return cursor.lastrowid


def update_quality_score(request_id: int, quality_score: float) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE request_log SET quality_score = ? WHERE id = ?",
            (quality_score, request_id),
        )


def mark_escalated(request_id: int, escalated_model_id: str, cost_delta: float) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE request_log SET escalated = 1, escalated_model_id = ?, "
            "cost_delta = ? WHERE id = ?",
            (escalated_model_id, cost_delta, request_id),
        )
