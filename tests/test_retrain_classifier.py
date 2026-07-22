from sklearn.model_selection import train_test_split

from llm_cost_router import settings
from llm_cost_router.classifier.features import feature_vector
from llm_cost_router.classifier.retrain import retrain_from_failures
from llm_cost_router.classifier.sklearn_classifier import SklearnClassifier
from llm_cost_router.classifier.train import build_feature_matrix, fit_model, load_labeled_dataset
from llm_cost_router.storage.db import init_db
from llm_cost_router.storage.failures import record_classifier_failure

DATASET_PATH = settings.PROJECT_ROOT / "data" / "labeled_prompts.json"

# Longer, no keyword/context/format signal - the base classifier calls this
# Tier 2, even though it's really a nuanced personal judgment call (Tier 3).
TRAP_PROMPT = (
    "My sister and I disagree about how to handle our aging parents care and "
    "I do not know what the right thing to do is"
)


def _baseline_model():
    records = load_labeled_dataset(DATASET_PATH)
    X, y = build_feature_matrix(records)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return fit_model(X_train, y_train)


def test_retrain_with_no_failures_reports_zero_and_does_not_swap(tmp_path) -> None:
    init_db()
    model_path = tmp_path / "model.joblib"

    result = retrain_from_failures(DATASET_PATH, model_path)

    assert result["n_failures"] == 0
    assert result["swapped"] is False
    assert not model_path.exists()


def test_retrain_from_failures_shifts_decision_boundary(tmp_path) -> None:
    init_db()

    baseline_tier = int(_baseline_model().predict([feature_vector(TRAP_PROMPT)])[0])
    assert baseline_tier != 3  # base model gets this wrong before retraining

    # Seed several repeated verification failures for the exact same trap
    # prompt, all corrected to Tier 3 - simulating the verifier repeatedly
    # catching this cheap-model routing mistake in production. 8 repeats
    # only nudges the decision to a 50/50 tie (verified empirically); 20
    # comfortably flips it - logistic regression on 6 shallow features needs
    # real weight-of-evidence to overcome the many genuinely-Tier-1/2
    # examples sharing a similar feature vector.
    for _ in range(20):
        record_classifier_failure(
            prompt=TRAP_PROMPT, original_tier=baseline_tier, corrected_tier=3, quality_score=1.0
        )

    model_path = tmp_path / "model.joblib"
    result = retrain_from_failures(DATASET_PATH, model_path)

    assert result["n_failures"] == 20
    assert result["swapped"] is True
    assert model_path.exists()

    retrained_classifier = SklearnClassifier(model_path)
    assert retrained_classifier.classify(TRAP_PROMPT).tier.value == 3
