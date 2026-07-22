from llm_cost_router.classifier.features import FEATURE_NAMES, feature_vector


def test_feature_vector_length_matches_names() -> None:
    vector = feature_vector("What is the capital of France?")
    assert len(vector) == len(FEATURE_NAMES) == 6


def test_feature_vector_simple_prompt() -> None:
    vector = feature_vector("What is the capital of France?")
    word_count, complex_kw, moderate_kw, instr_count, context, output_complexity = vector

    assert word_count == 6
    assert complex_kw == 0
    assert moderate_kw == 0
    assert context == 0.0
    assert output_complexity == 0


def test_feature_vector_complex_prompt_has_signal() -> None:
    vector = feature_vector(
        "Compare the trade-offs between a relational database and a document store for a "
        "high-write telemetry system. Analyze latency, consistency, and schema flexibility "
        "step by step, then give a final recommendation. Output as JSON with fields: "
        "options, tradeoffs, recommendation."
    )
    word_count, complex_kw, moderate_kw, instr_count, context, output_complexity = vector

    assert word_count > 20
    assert complex_kw > 0
    assert output_complexity == 2
