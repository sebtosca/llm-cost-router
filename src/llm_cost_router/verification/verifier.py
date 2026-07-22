import re

from llm_cost_router.models.registry import get_model
from llm_cost_router.models.types import ProviderRequestError
from llm_cost_router.providers import send_request
from llm_cost_router.storage.request_log import update_quality_score

# The judge model doubles as the reference-answer generator: its own response
# to the prompt serves as the "what the best model would have said" baseline
# that the cheap model's candidate answer is graded against.
JUDGE_MODEL_ID = "claude-sonnet-5"

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

    update_quality_score(request_id, float(match.group()))
