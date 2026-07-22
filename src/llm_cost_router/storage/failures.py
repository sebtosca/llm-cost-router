from datetime import datetime, timezone

from llm_cost_router.storage.db import get_connection


def record_classifier_failure(
    prompt: str, original_tier: int, corrected_tier: int, quality_score: float
) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO classifier_failures "
            "(timestamp, prompt, original_tier, corrected_tier, quality_score) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                prompt,
                original_tier,
                corrected_tier,
                quality_score,
            ),
        )


def load_classifier_failures() -> list[dict]:
    """Returns failures in the same {"prompt", "tier"} shape as the base
    labeled dataset, so they can be merged directly into training data."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT prompt, corrected_tier FROM classifier_failures"
        ).fetchall()
    return [{"prompt": prompt, "tier": corrected_tier} for prompt, corrected_tier in rows]
