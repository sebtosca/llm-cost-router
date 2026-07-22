import time

from google import genai
from google.genai import errors as genai_errors

from llm_cost_router.models.types import ModelConfig, ProviderRequestError, Response


class GeminiProvider:
    def __init__(self, api_key: str | None) -> None:
        # A placeholder is used when unset so construction never fails at
        # import/startup time - a missing/invalid key surfaces as a clean
        # ProviderRequestError on the first real call instead.
        self._client = genai.Client(api_key=api_key or "missing-gemini-api-key")

    def send(self, prompt: str, model_config: ModelConfig) -> Response:
        start = time.monotonic()
        config = (
            {"max_output_tokens": model_config.max_output_tokens}
            if model_config.max_output_tokens
            else None
        )
        try:
            resp = self._client.models.generate_content(
                model=model_config.provider_model_name,
                contents=prompt,
                config=config,
            )
        except genai_errors.APIError as exc:
            raise ProviderRequestError(f"Gemini request failed: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = resp.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0
        finish_reason = None
        if resp.candidates:
            reason = resp.candidates[0].finish_reason
            finish_reason = reason.name if reason else None

        return Response(
            output_text=resp.text or "",
            model_id=model_config.id,
            provider=model_config.provider,
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            cost_usd=(input_tokens or 0) * model_config.cost_per_input_token
            + (output_tokens or 0) * model_config.cost_per_output_token,
            latency_ms=latency_ms,
            raw_finish_reason=finish_reason,
        )
