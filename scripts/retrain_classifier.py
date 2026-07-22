"""Retrains the sklearn classifier using accumulated verification failures
(the classifier_failures table, populated by the async verifier when a
cheap-tier response scores badly - see verification/verifier.py) merged with
the base labeled dataset. Swaps in the new model only if it doesn't regress
on a held-out split of the base dataset.

This is a manually-triggered script for this pass, not an automated weekly
cron - scheduling the retrain is a follow-up, not built here.

Usage: python scripts/retrain_classifier.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_cost_router import settings  # noqa: E402
from llm_cost_router.classifier.retrain import retrain_from_failures  # noqa: E402
from llm_cost_router.storage.db import init_db  # noqa: E402

DATASET_PATH = settings.PROJECT_ROOT / "data" / "labeled_prompts.json"


def main() -> None:
    init_db()
    result = retrain_from_failures(DATASET_PATH, settings.CLASSIFIER_MODEL_PATH)

    print(f"Accumulated verification failures: {result['n_failures']}")
    if result["n_failures"] == 0:
        print(
            "Nothing to retrain on yet. Run the API, let some cheap-tier "
            "responses get flagged by the verifier, then retry."
        )
        return

    print(f"Baseline accuracy (failures excluded): {result['baseline_accuracy']:.1%}")
    print(f"Merged accuracy (failures included):   {result['merged_accuracy']:.1%}")

    if result["swapped"]:
        print(f"\nMerged model did not regress - saved to {settings.CLASSIFIER_MODEL_PATH}")
    else:
        print("\nMerged model regressed on the held-out set - keeping the existing model file.")


if __name__ == "__main__":
    main()
