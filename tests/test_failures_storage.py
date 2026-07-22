from llm_cost_router.storage.db import init_db
from llm_cost_router.storage.failures import load_classifier_failures, record_classifier_failure


def test_record_and_load_classifier_failures() -> None:
    init_db()
    record_classifier_failure(
        prompt="Should I take the job offer?",
        original_tier=1,
        corrected_tier=3,
        quality_score=1.0,
    )
    record_classifier_failure(
        prompt="Is this investment a good idea?",
        original_tier=2,
        corrected_tier=3,
        quality_score=2.0,
    )

    failures = load_classifier_failures()

    assert failures == [
        {"prompt": "Should I take the job offer?", "tier": 3},
        {"prompt": "Is this investment a good idea?", "tier": 3},
    ]


def test_load_classifier_failures_empty() -> None:
    init_db()
    assert load_classifier_failures() == []
