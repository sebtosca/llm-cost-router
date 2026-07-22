FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[ml]"

COPY config/ ./config/
COPY scripts/ ./scripts/
COPY data/labeled_prompts.json ./data/labeled_prompts.json

EXPOSE 8000

CMD ["uvicorn", "llm_cost_router.api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
