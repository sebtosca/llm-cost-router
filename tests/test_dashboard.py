from pathlib import Path

from streamlit.testing.v1 import AppTest

from llm_cost_router.models.registry import get_model
from llm_cost_router.storage.db import init_db
from llm_cost_router.storage.request_log import log_request

DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_dashboard_renders_without_exceptions_on_empty_db() -> None:
    init_db()
    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run()

    assert not at.exception


def test_dashboard_renders_metrics_with_seeded_data() -> None:
    init_db()
    m = get_model("gpt-4o-mini")
    log_request(
        prompt="p1",
        tier=1,
        model_id="gpt-4o-mini",
        provider="openai",
        input_tokens=1000,
        output_tokens=200,
        cost_usd=1000 * m.cost_per_input_token + 200 * m.cost_per_output_token,
        latency_ms=100.0,
    )

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run()

    assert not at.exception
    assert len(at.metric) == 3
    assert at.metric[0].value.startswith("$")
