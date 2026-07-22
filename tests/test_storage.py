from llm_cost_router.storage.db import get_connection, init_db
from llm_cost_router.storage.request_log import hash_prompt, log_request


def test_hash_prompt_deterministic() -> None:
    assert hash_prompt("hello") == hash_prompt("hello")
    assert hash_prompt("hello") != hash_prompt("world")


def test_init_db_creates_table() -> None:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='request_log'"
        ).fetchall()
    assert len(rows) == 1


def test_log_request_success_row() -> None:
    init_db()
    log_request(
        prompt="What is the capital of France?",
        tier=1,
        model_id="gpt-4o-mini",
        provider="openai",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.000005,
        latency_ms=250.0,
    )

    with get_connection() as conn:
        row = conn.execute("SELECT * FROM request_log").fetchone()

    assert row is not None
    assert row[2] == hash_prompt("What is the capital of France?")  # prompt_hash
    assert row[3] == 1  # tier
    assert row[4] == "gpt-4o-mini"  # model_id
    assert row[5] == "openai"  # provider
    assert row[6] == 10  # input_tokens
    assert row[7] == 5  # output_tokens
    assert row[11] == 0  # escalated defaults to false
    assert row[12] is None  # error


def test_log_request_error_row() -> None:
    init_db()
    log_request(
        prompt="hello",
        tier=2,
        model_id="gemini-3-flash",
        provider="gemini",
        error="boom",
    )

    with get_connection() as conn:
        row = conn.execute("SELECT * FROM request_log").fetchone()

    assert row[6] is None  # input_tokens
    assert row[12] == "boom"  # error
