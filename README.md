# llm-cost-router

Routes LLM completion requests to the cheapest model capable of handling them.
A heuristic classifier scores each prompt into a complexity tier (1=simple,
2=moderate, 3=complex), and a YAML-configured router maps each tier to a
model across OpenAI, Anthropic, and Gemini.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY
```

## Run the tests

```bash
pytest tests/
```

## Run the API

```bash
uvicorn llm_cost_router.api.app:app --reload --app-dir src --port 8000
```

```bash
curl -X POST localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is the capital of France?"}'
```

Routing tiers are configured in [config/routing.yaml](config/routing.yaml);
edit the tier -> model mapping there without touching code (validated against
the model registry at startup).

## Baseline test script

Sends a fixed prompt set to every registered model and reports cost/latency
per model. Makes real, paid API calls.

```bash
python scripts/baseline_test.py
```

## Not yet built

Scikit-learn classifier, async quality verification / auto-escalation /
feedback loop, SQLite request logging, cost dashboard, `GET /v1/models`,
`GET /v1/stats`, `PUT /v1/routing-config`, docker/docker-compose, load
testing.
