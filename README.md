# llm-cost-router

Routes LLM completion requests to the cheapest model capable of handling them.
A classifier scores each prompt into a complexity tier (1=simple, 2=moderate,
3=complex), and a YAML-configured router maps each tier to a model across
OpenAI, Anthropic, and Gemini.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dashboard,ml]"
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

Every request is logged to a local SQLite file (`data/router.db`, gitignored).
After each cheap-tier response returns, a background task re-answers the
prompt with the judge model (`claude-sonnet-5`), scores agreement via
LLM-as-judge, and writes `quality_score` back onto the log row - a score
below 3.0 marks the row `escalated=true` with the cost delta. See
[docs/ROADMAP.md](docs/ROADMAP.md) (local-only, gitignored) for the full
step-by-step design.

```bash
curl localhost:8000/v1/models
curl localhost:8000/v1/stats
```

Routing can also be hot-swapped at runtime without a restart:

```bash
curl -X PUT localhost:8000/v1/routing-config \
  -H 'Content-Type: application/json' \
  -d '{"routing": {"tier_1": "claude-haiku-4-5", "tier_2": "gemini-3-flash", "tier_3": "claude-sonnet-5"}}'
```

Validated the same way as `config/routing.yaml` at startup (unknown model ids
are rejected with `422` and the previous config stays in effect) - the change
only lives in memory for the running process, `config/routing.yaml` on disk
is untouched.

## Cost dashboard

```bash
streamlit run dashboard/app.py
```

Reads directly from `data/router.db`: headline cost-savings metrics, cost
per day vs. an all-priciest-model baseline, escalation rate over time,
routing distribution by model, and quality score distribution.

## Baseline test script

Sends a fixed prompt set to every registered model and reports cost/latency
per model. Makes real, paid API calls.

```bash
python scripts/baseline_test.py
```

## Classifier: heuristic vs. sklearn

The default classifier is a hand-coded heuristic (`classifier/heuristic.py`).
A scikit-learn LogisticRegression classifier is also available, trained on a
200+ example labeled dataset (`data/labeled_prompts.json`, tracked in git) built
from tier-targeted templates rather than manually typed one-by-one - see the
docstring in `scripts/generate_labeled_dataset.py` for why. To use it:

```bash
python scripts/generate_labeled_dataset.py   # regenerate the labeled dataset (optional, already tracked)
python scripts/train_classifier.py           # trains + prints accuracy/confusion matrix, saves data/classifier_model.joblib
CLASSIFIER=sklearn uvicorn llm_cost_router.api.app:app --app-dir src --port 8000
```

The trained model file is gitignored (a regenerable build artifact) - run
`train_classifier.py` once after cloning before setting `CLASSIFIER=sklearn`.

## Classifier feedback loop

When the async verifier (`verification/verifier.py`) escalates a cheap-tier
response (quality score below 3.0), it also records a `classifier_failures`
row: the actual prompt text, the tier the classifier picked, and the tier it
should have picked (the judge model's tier). This is a deliberate, narrow
exception to `request_log`'s hash-only privacy policy - only escalated
requests get their prompt text retained, and only for retraining.

```bash
python scripts/retrain_classifier.py
```

Merges accumulated failures into the base labeled dataset, evaluates both the
baseline and merged model on the same held-out split, and only swaps in the
new model file if it doesn't regress. This is manually triggered for now - an
automated weekly schedule is a follow-up, not built here.

## Docker

```bash
cp .env.example .env   # fill in real keys, or leave placeholders to smoke-test
docker compose up --build
```

```bash
curl localhost:8000/v1/models
```

`data/` and `config/` are bind-mounted, so `router.db` and the sklearn model
persist across container restarts, and `config/routing.yaml` can be edited on
the host without rebuilding. No separate worker container - verification runs
as an in-process FastAPI background task, not a real task queue. `.env` is
optional; the API starts fine without it and only fails at request time if a
key is missing.

## Not yet built

Load testing.
