from llm_cost_router import settings
from llm_cost_router.providers.anthropic_provider import AnthropicProvider
from llm_cost_router.providers.base import (
    PROVIDER_ADAPTERS,
    ProviderAdapter,
    register_provider,
    send_request,
)
from llm_cost_router.providers.gemini_provider import GeminiProvider
from llm_cost_router.providers.openai_provider import OpenAIProvider

__all__ = [
    "PROVIDER_ADAPTERS",
    "ProviderAdapter",
    "register_provider",
    "send_request",
    "init_providers",
]


def init_providers() -> None:
    """Construct and register the OpenAI, Anthropic, and Ollama adapters.

    Called once at process startup (API app factory, baseline script). Adapters
    are constructed even if their API key env var is unset - the underlying SDK
    only raises once a real call is made, which surfaces as a clean
    ProviderRequestError at request time rather than an import-time crash.
    """
    register_provider("openai", OpenAIProvider(settings.OPENAI_API_KEY))
    register_provider("anthropic", AnthropicProvider(settings.ANTHROPIC_API_KEY))
    register_provider("gemini", GeminiProvider(settings.GEMINI_API_KEY))
