from pathlib import Path

from sklearn.model_selection import train_test_split

from llm_cost_router.classifier.train import (
    build_feature_matrix,
    evaluate_model,
    fit_model,
    load_labeled_dataset,
    save_model,
)
from llm_cost_router.storage.failures import load_classifier_failures


def retrain_from_failures(
    dataset_path: Path, model_path: Path, test_size: float = 0.2, random_state: int = 42
) -> dict:
    """Retrains on the base labeled dataset plus accumulated verification
    failures, evaluates both the baseline (failures excluded) and merged
    model on the SAME held-out split of the base dataset for a fair
    comparison, and only swaps in the merged model file if it doesn't
    regress. This is meant to be run manually (see scripts/retrain_classifier.py)
    - an automated weekly schedule is a follow-up, not built here.
    """
    base_records = load_labeled_dataset(dataset_path)
    failures = load_classifier_failures()

    X_base, y_base = build_feature_matrix(base_records)
    X_train, X_test, y_train, y_test = train_test_split(
        X_base, y_base, test_size=test_size, random_state=random_state, stratify=y_base
    )

    baseline_model = fit_model(X_train, y_train)
    baseline_eval = evaluate_model(baseline_model, X_test, y_test)

    result = {
        "n_failures": len(failures),
        "baseline_accuracy": baseline_eval["accuracy"],
        "merged_accuracy": None,
        "swapped": False,
    }

    if not failures:
        return result

    X_failures, y_failures = build_feature_matrix(failures)
    merged_model = fit_model(X_train + X_failures, y_train + y_failures)
    merged_eval = evaluate_model(merged_model, X_test, y_test)
    result["merged_accuracy"] = merged_eval["accuracy"]

    if merged_eval["accuracy"] >= baseline_eval["accuracy"]:
        save_model(merged_model, model_path)
        result["swapped"] = True

    return result
