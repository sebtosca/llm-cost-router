from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_cost_router.models.registry import get_model
from llm_cost_router.models.types import ProviderRequestError
from llm_cost_router.providers.anthropic_provider import AnthropicProvider
from llm_cost_router.providers.gemini_provider import GeminiProvider
from llm_cost_router.providers.openai_provider import OpenAIProvider


def test_openai_provider_send_success() -> None:
    provider = OpenAIProvider(api_key="fake-key")
    provider._client.chat.completions.create = MagicMock(
        return_value=SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello"), finish_reason="stop"
                )
            ],
        )
    )

    response = provider.send("hi", get_model("gpt-4o-mini"))

    assert response.output_text == "hello"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.cost_usd == pytest.approx(10 * 0.15e-6 + 5 * 0.60e-6)
    assert response.raw_finish_reason == "stop"


def test_openai_provider_send_error_raises_provider_request_error() -> None:
    import openai

    provider = OpenAIProvider(api_key="fake-key")
    provider._client.chat.completions.create = MagicMock(
        side_effect=openai.APIConnectionError(request=MagicMock())
    )

    with pytest.raises(ProviderRequestError):
        provider.send("hi", get_model("gpt-4o-mini"))


def test_anthropic_provider_send_success() -> None:
    provider = AnthropicProvider(api_key="fake-key")
    provider._client.messages.create = MagicMock(
        return_value=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=8, output_tokens=4),
            content=[SimpleNamespace(type="text", text="hi there")],
            stop_reason="end_turn",
        )
    )

    response = provider.send("hi", get_model("claude-haiku-4-5"))

    assert response.output_text == "hi there"
    assert response.input_tokens == 8
    assert response.output_tokens == 4
    assert response.raw_finish_reason == "end_turn"


def test_gemini_provider_send_success() -> None:
    provider = GeminiProvider(api_key="fake-key")
    provider._client.models.generate_content = MagicMock(
        return_value=SimpleNamespace(
            text="gemini output",
            usage_metadata=SimpleNamespace(prompt_token_count=6, candidates_token_count=3),
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        )
    )

    response = provider.send("hi", get_model("gemini-3-flash"))

    assert response.output_text == "gemini output"
    assert response.input_tokens == 6
    assert response.output_tokens == 3
    assert response.raw_finish_reason == "STOP"
