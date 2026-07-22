import time

import anthropic

from llm_cost_router.models.types import ModelConfig, ProviderRequestError, Response


class AnthropicProvider:
    def __init__(self, api_key: str | None) -> None:
        # A placeholder is used when unset so construction never fails at
        # import/startup time - a missing/invalid key surfaces as a clean
        # ProviderRequestError on the first real call instead.
        self._client = anthropic.Anthropic(api_key=api_key or "missing-anthropic-api-key")

    def send(self, prompt: str, model_config: ModelConfig) -> Response:
        start = time.monotonic()
        try:
            resp = self._client.messages.create(
                model=model_config.provider_model_name,
                max_tokens=model_config.max_output_tokens or 1024,
                messages=[{"role": "user", "content": prompt}],
                timeout=30,
            )
        except anthropic.AnthropicError as exc:
            raise ProviderRequestError(f"Anthropic request failed: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        input_tokens = resp.usage.input_tokens
        output_tokens = resp.usage.output_tokens
        output_text = "".join(
            block.text for block in resp.content if block.type == "text"
        )

        return Response(
            output_text=output_text,
            model_id=model_config.id,
            provider=model_config.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=input_tokens * model_config.cost_per_input_token
            + output_tokens * model_config.cost_per_output_token,
            latency_ms=latency_ms,
            raw_finish_reason=resp.stop_reason,
        )
