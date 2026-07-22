from typing import Protocol

from llm_cost_router.models.types import ModelConfig, Response, UnsupportedProviderError


class ProviderAdapter(Protocol):
    def send(self, prompt: str, model_config: ModelConfig) -> Response: ...


PROVIDER_ADAPTERS: dict[str, ProviderAdapter] = {}


def register_provider(name: str, adapter: ProviderAdapter) -> None:
    PROVIDER_ADAPTERS[name] = adapter


def send_request(prompt: str, model_config: ModelConfig) -> Response:
    adapter = PROVIDER_ADAPTERS.get(model_config.provider)
    if adapter is None:
        raise UnsupportedProviderError(
            f"No provider adapter registered for '{model_config.provider}' "
            f"(model '{model_config.id}')"
        )
    return adapter.send(prompt, model_config)
