from llm_cost_router.models.registry import get_model
from llm_cost_router.storage.db import init_db
from llm_cost_router.storage.request_log import log_request
from llm_cost_router.storage.stats import compute_stats


def test_compute_stats_empty() -> None:
    init_db()
    stats = compute_stats()

    assert stats.total_requests == 0
    assert stats.total_cost_usd == 0.0
    assert stats.savings_pct == 0.0


def test_compute_stats_aggregates_and_savings() -> None:
    init_db()
    log_request(
        prompt="p1",
        tier=1,
        model_id="gpt-4o-mini",
        provider="openai",
        input_tokens=1000,
        output_tokens=500,
        cost_usd=1000 * get_model("gpt-4o-mini").cost_per_input_token
        + 500 * get_model("gpt-4o-mini").cost_per_output_token,
        latency_ms=100.0,
    )
    log_request(
        prompt="p2",
        tier=3,
        model_id="claude-sonnet-5",
        provider="anthropic",
        input_tokens=2000,
        output_tokens=1000,
        cost_usd=2000 * get_model("claude-sonnet-5").cost_per_input_token
        + 1000 * get_model("claude-sonnet-5").cost_per_output_token,
        latency_ms=200.0,
    )
    # A failed request (error set, no cost/tokens) must not be counted.
    log_request(
        prompt="p3",
        tier=2,
        model_id="gemini-3-flash",
        provider="gemini",
        error="boom",
    )

    stats = compute_stats()

    assert stats.total_requests == 2
    assert stats.total_cost_usd > 0
    assert set(stats.cost_by_tier.keys()) == {"1", "3"}
    assert set(stats.cost_by_model.keys()) == {"gpt-4o-mini", "claude-sonnet-5"}

    # baseline = cost if all 3000 input / 1500 output tokens had gone to the
    # priciest registered model (claude-sonnet-5, $3/$15 per 1M tokens).
    priciest = get_model("claude-sonnet-5")
    expected_baseline = 3000 * priciest.cost_per_input_token + 1500 * priciest.cost_per_output_token
    assert stats.baseline_cost_usd == expected_baseline
    assert stats.savings_usd == expected_baseline - stats.total_cost_usd
    assert stats.savings_pct > 0
