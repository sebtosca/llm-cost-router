from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_cost_router import settings
from llm_cost_router.api.routes import router
from llm_cost_router.classifier.heuristic import HeuristicClassifier
from llm_cost_router.providers import init_providers
from llm_cost_router.router.config import load_routing_config
from llm_cost_router.router.router import Router
from llm_cost_router.storage.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    routing_config = load_routing_config(settings.ROUTING_CONFIG_PATH)
    init_providers()
    init_db()
    app.state.classifier = HeuristicClassifier()
    app.state.router = Router(routing_config)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Cost Router", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
