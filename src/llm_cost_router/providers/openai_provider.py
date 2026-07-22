import time

import openai

from llm_cost_router.models.types import ModelConfig, ProviderRequestError, Response


class OpenAIProvider:
    def __init__(self, api_key: str | None) -> None:
        # A placeholder is used when unset so construction never fails at
        # import/startup time - a missing/invalid key surfaces as a clean
        # ProviderRequestError on the first real call instead.
        self._client = openai.OpenAI(api_key=api_key or "missing-openai-api-key")

    def send(self, prompt: str, model_config: ModelConfig) -> Response:
        start = time.monotonic()
        try:
            resp = self._client.chat.completions.create(
                model=model_config.provider_model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=model_config.max_output_tokens,
                timeout=30,
            )
        except openai.OpenAIError as exc:
            raise ProviderRequestError(f"OpenAI request failed: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = resp.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        choice = resp.choices[0]

        return Response(
            output_text=choice.message.content or "",
            model_id=model_config.id,
            provider=model_config.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=input_tokens * model_config.cost_per_input_token
            + output_tokens * model_config.cost_per_output_token,
            latency_ms=latency_ms,
            raw_finish_reason=choice.finish_reason,
        )
