import pytest

from llm_cost_router.classifier.heuristic import HeuristicClassifier
from llm_cost_router.models.types import Tier

classifier = HeuristicClassifier()


@pytest.mark.parametrize(
    "prompt,expected_tier",
    [
        ("What is the capital of France?", Tier.TIER_1),
        ("Rewrite this sentence in passive voice: The dog chased the cat.", Tier.TIER_1),
        (
            "Summarize the following article in two sentences:\n\nContext: The city "
            "council voted 5-2 on Tuesday to approve a new zoning ordinance that will "
            "allow mixed-use development in the downtown corridor for the first time "
            "in twenty years. Supporters say the change will bring much-needed housing "
            "and foot traffic to a struggling business district, while critics worry "
            "about parking shortages and rising rents displacing long-time residents. "
            "The ordinance takes effect in 90 days, with a public review scheduled "
            "after the first year.",
            Tier.TIER_2,
        ),
        (
            "Compare the trade-offs between a relational database and a document "
            "store for a high-write telemetry system. Analyze latency, consistency, "
            "and schema flexibility step by step, then give a final recommendation. "
            "Output as JSON with fields: options, tradeoffs, recommendation.",
            Tier.TIER_3,
        ),
    ],
)
def test_classify_tiers(prompt: str, expected_tier: Tier) -> None:
    result = classifier.classify(prompt)
    assert result.tier == expected_tier, result.reason


def test_classification_result_has_features_and_reason() -> None:
    result = classifier.classify("Why does the sky appear blue during the day?")
    assert result.reason
    assert "word_count" in result.features
