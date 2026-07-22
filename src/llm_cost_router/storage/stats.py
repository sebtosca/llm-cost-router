from pydantic import BaseModel

from llm_cost_router.models.registry import list_models
from llm_cost_router.storage.db import get_connection


class Stats(BaseModel):
    total_requests: int
    total_cost_usd: float
    cost_by_tier: dict[str, float]
    cost_by_model: dict[str, float]
    baseline_cost_usd: float
    savings_usd: float
    savings_pct: float


class DailyStat(BaseModel):
    date: str
    cost_usd: float
    baseline_usd: float
    requests: int
    escalations: int
    escalation_rate: float


def _priciest_model():
    # "What it would have cost if every request had gone to the priciest
    # registered model instead" - the headline cost-savings comparison.
    return max(list_models(), key=lambda m: m.cost_per_input_token + m.cost_per_output_token)


def compute_stats() -> Stats:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT tier, model_id, cost_usd, input_tokens, output_tokens "
            "FROM request_log WHERE error IS NULL"
        ).fetchall()

    total_requests = len(rows)
    total_cost_usd = 0.0
    cost_by_tier: dict[str, float] = {}
    cost_by_model: dict[str, float] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for tier, model_id, cost_usd, input_tokens, output_tokens in rows:
        cost_usd = cost_usd or 0.0
        total_cost_usd += cost_usd
        cost_by_tier[str(tier)] = cost_by_tier.get(str(tier), 0.0) + cost_usd
        cost_by_model[model_id] = cost_by_model.get(model_id, 0.0) + cost_usd
        total_input_tokens += input_tokens or 0
        total_output_tokens += output_tokens or 0

    priciest = _priciest_model()
    baseline_cost_usd = (
        total_input_tokens * priciest.cost_per_input_token
        + total_output_tokens * priciest.cost_per_output_token
    )
    savings_usd = baseline_cost_usd - total_cost_usd
    savings_pct = (savings_usd / baseline_cost_usd * 100) if baseline_cost_usd > 0 else 0.0

    return Stats(
        total_requests=total_requests,
        total_cost_usd=total_cost_usd,
        cost_by_tier=cost_by_tier,
        cost_by_model=cost_by_model,
        baseline_cost_usd=baseline_cost_usd,
        savings_usd=savings_usd,
        savings_pct=savings_pct,
    )


def compute_daily_series() -> list[DailyStat]:
    """One row per UTC date (from the ISO timestamp) for the dashboard's
    cost-over-time and escalation-rate-over-time charts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT timestamp, cost_usd, input_tokens, output_tokens, escalated "
            "FROM request_log WHERE error IS NULL"
        ).fetchall()

    priciest = _priciest_model()
    daily: dict[str, dict] = {}

    for timestamp, cost_usd, input_tokens, output_tokens, escalated in rows:
        date = timestamp[:10]
        bucket = daily.setdefault(
            date, {"cost_usd": 0.0, "baseline_usd": 0.0, "requests": 0, "escalations": 0}
        )
        bucket["cost_usd"] += cost_usd or 0.0
        bucket["baseline_usd"] += (input_tokens or 0) * priciest.cost_per_input_token + (
            output_tokens or 0
        ) * priciest.cost_per_output_token
        bucket["requests"] += 1
        bucket["escalations"] += escalated or 0

    return [
        DailyStat(
            date=date,
            cost_usd=bucket["cost_usd"],
            baseline_usd=bucket["baseline_usd"],
            requests=bucket["requests"],
            escalations=bucket["escalations"],
            escalation_rate=bucket["escalations"] / bucket["requests"] if bucket["requests"] else 0.0,
        )
        for date, bucket in sorted(daily.items())
    ]
