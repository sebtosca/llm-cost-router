import re

from llm_cost_router.models.registry import get_model
from llm_cost_router.models.types import ProviderRequestError
from llm_cost_router.providers import send_request
from llm_cost_router.storage.request_log import mark_escalated, update_quality_score

# The judge model doubles as the reference-answer generator: its own response
# to the prompt serves as the "what the best model would have said" baseline
# that the cheap model's candidate answer is graded against.
JUDGE_MODEL_ID = "claude-sonnet-5"

# Below this, the cheap model's answer is considered a routing failure and
# gets escalated.
ESCALATION_THRESHOLD = 3.0

_SCORE_RE = re.compile(r"[1-5]")


def _build_judge_prompt(prompt: str, candidate_answer: str, reference_answer: str) -> str:
    return (
        "You are grading whether a CANDIDATE answer adequately addresses a PROMPT, "
        "compared to a stronger REFERENCE answer from a top-tier model.\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"CANDIDATE:\n{candidate_answer}\n\n"
        f"REFERENCE:\n{reference_answer}\n\n"
        "Rate how well the CANDIDATE agrees with the REFERENCE on a scale of 1-5, "
        "where 5 means fully equivalent in correctness and completeness, and 1 "
        "means completely wrong or missing the point. Respond with ONLY the "
        "single digit, nothing else."
    )


def verify_response(request_id: int, prompt: str, candidate_output: str) -> None:
    """Runs as a FastAPI BackgroundTask after a cheap-tier response has already
    been returned to the caller. Best-effort: any provider failure or unparsable
    judge output is swallowed rather than raised, since nothing is listening for
    the result by the time this runs - the row simply keeps quality_score=NULL.

    Escalation here is necessarily after-the-fact: the original HTTP response
    already went out to the caller by the time this background job runs, so
    "escalating" can only mean logging that a stronger model's answer was
    available and how much extra it would have cost - not re-delivering a
    better response to a request that's already closed. This deviates from
    the original spec's "re-run and return the better result if latency
    permits," which assumed escalation happens inside the original request.
    """
    judge_model = get_model(JUDGE_MODEL_ID)
    try:
        reference = send_request(prompt, judge_model)
        judge_prompt = _build_judge_prompt(prompt, candidate_output, reference.output_text)
        judge_response = send_request(judge_prompt, judge_model)
    except ProviderRequestError:
        return

    match = _SCORE_RE.search(judge_response.output_text)
    if not match:
        return

    score = float(match.group())
    update_quality_score(request_id, score)

    if score < ESCALATION_THRESHOLD:
        mark_escalated(request_id, escalated_model_id=judge_model.id, cost_delta=reference.cost_usd)
