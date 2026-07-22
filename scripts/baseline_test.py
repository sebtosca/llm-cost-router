"""Phase-1 baseline: send every test prompt to every registered model.

Usage:
    python scripts/baseline_test.py [--yes]

Makes real, paid API calls. Requires OPENAI_API_KEY, ANTHROPIC_API_KEY, and
GEMINI_API_KEY in .env.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_cost_router.models import registry  # noqa: E402
from llm_cost_router.models.types import ProviderRequestError  # noqa: E402
from llm_cost_router.providers import init_providers, send_request  # noqa: E402
from llm_cost_router import settings  # noqa: E402

PROMPTS_PATH = settings.PROJECT_ROOT / "data" / "prompts" / "baseline_prompts.json"
RESULTS_DIR = settings.PROJECT_ROOT / "data" / "results"


def load_prompts() -> list[dict]:
    return json.loads(PROMPTS_PATH.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="skip cost confirmation prompt")
    args = parser.parse_args()

    prompts = load_prompts()
    models = registry.list_models()
    n_calls = len(prompts) * len(models)

    print(f"About to make {n_calls} calls ({len(prompts)} prompts x {len(models)} models).")
    print(f"Models: {', '.join(m.id for m in models)}")
    if not args.yes:
        confirm = input("This may incur real cost on OpenAI/Anthropic. Continue? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return

    init_providers()

    rows = []
    for model_config in models:
        for prompt_entry in prompts:
            prompt_text = prompt_entry["prompt"]
            try:
                response = send_request(prompt_text, model_config)
            except ProviderRequestError as exc:
                print(f"[WARN] {model_config.id} / {prompt_entry['id']}: {exc}")
                continue

            rows.append(
                {
                    "model_id": model_config.id,
                    "prompt_id": prompt_entry["id"],
                    "prompt_preview": prompt_text[:80],
                    "output_preview": response.output_text[:200],
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(
                f"{model_config.id:20s} {prompt_entry['id']:5s} "
                f"cost=${response.cost_usd:.6f} latency={response.latency_ms:.0f}ms"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows to {out_path}")

    print("\nPer-model summary:")
    for model_config in models:
        model_rows = [r for r in rows if r["model_id"] == model_config.id]
        if not model_rows:
            continue
        avg_cost = sum(r["cost_usd"] for r in model_rows) / len(model_rows)
        avg_latency = sum(r["latency_ms"] for r in model_rows) / len(model_rows)
        print(f"  {model_config.id:20s} avg_cost=${avg_cost:.6f} avg_latency={avg_latency:.0f}ms")


if __name__ == "__main__":
    main()
