"""Step 10 load test (docs/ROADMAP.md): sends the full diverse prompt set
through the live, running API (not direct provider calls) so classification,
routing, logging, and the async verifier all get exercised for real.

Requires the API to already be running (e.g. `docker-compose up` or
`uvicorn ...`) and reachable at --base-url.

Makes real, paid API calls. Every non-judge-tier request also triggers a
background verification call (Step 3) against the judge model, so the actual
live-call count is roughly 2x the prompt count.

Usage:
    python scripts/load_test.py [--base-url http://localhost:8000]
                                 [--concurrency 5] [--yes]
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_cost_router import settings  # noqa: E402

PROMPTS_PATH = settings.PROJECT_ROOT / "data" / "prompts" / "load_test_prompts.json"
RESULTS_DIR = settings.PROJECT_ROOT / "data" / "results"


def load_prompts() -> list[dict]:
    return json.loads(PROMPTS_PATH.read_text())


async def run_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, entry: dict) -> dict:
    async with sem:
        try:
            resp = await client.post("/v1/completions", json={"prompt": entry["prompt"]}, timeout=60.0)
            resp.raise_for_status()
            body = resp.json()
            return {
                "id": entry["id"],
                "prompt_preview": entry["prompt"][:80],
                "model_used": body["model_used"],
                "tier": body["tier"],
                "cost_usd": body["cost_usd"],
                "latency_ms": body["latency_ms"],
            }
        except httpx.HTTPError as exc:
            return {"id": entry["id"], "prompt_preview": entry["prompt"][:80], "error": str(exc)}


async def run_load_test(base_url: str, concurrency: int, prompts: list[dict]) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    results = []
    async with httpx.AsyncClient(base_url=base_url) as client:
        tasks = [run_one(client, sem, entry) for entry in prompts]
        done = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            done += 1
            status = result.get("error", f"tier={result.get('tier')} model={result.get('model_used')}")
            print(f"[{done}/{len(prompts)}] {result['id']:6s} {status}")
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--yes", action="store_true", help="skip cost confirmation prompt")
    args = parser.parse_args()

    prompts = load_prompts()

    print(f"About to send {len(prompts)} prompts to {args.base_url} (concurrency={args.concurrency}).")
    print("Each request also triggers an async verification call against the judge model.")
    if not args.yes:
        confirm = input("This will incur real cost on OpenAI/Anthropic/Gemini. Continue? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return

    results = asyncio.run(run_load_test(args.base_url, args.concurrency, prompts))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"load_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(results, indent=2))

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    print(f"\nWrote {len(results)} results to {out_path}")
    print(f"  Succeeded: {len(ok)}, Failed: {len(errors)}")
    if ok:
        total_cost = sum(r["cost_usd"] for r in ok)
        by_tier = {}
        for r in ok:
            by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
        print(f"  Total cost (primary completions only, excludes verifier): ${total_cost:.4f}")
        print(f"  Routed by tier: {by_tier}")

    print(
        "\nBackground verification jobs may still be finishing. Wait a bit, then "
        f"fetch the final report:\n  curl {args.base_url}/v1/stats"
    )


if __name__ == "__main__":
    main()
