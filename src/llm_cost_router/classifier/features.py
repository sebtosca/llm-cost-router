import re

COMPLEX_KEYWORDS = {
    "analyze",
    "compare",
    "why",
    "step by step",
    "evaluate",
    "critique",
    "design",
    "recommend",
    "trade-off",
    "tradeoff",
    "reasoning",
    "implications",
}

MODERATE_KEYWORDS = {
    "summarize",
    "classify",
    "categorize",
    "extract",
    "list",
    "outline",
}

_IMPERATIVE_VERBS = (
    "explain",
    "write",
    "list",
    "compare",
    "summarize",
    "generate",
    "analyze",
    "translate",
    "convert",
    "create",
)

_LIST_LINE_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")
_CONTEXT_MARKERS = ("```", "context:", "document:", "article:", "text:")


def word_count(prompt: str) -> int:
    return len(prompt.split())


def token_estimate(prompt: str) -> int:
    """Rough token estimate (chars/4) — no tiktoken dependency for this pass."""
    return max(1, len(prompt) // 4)


def keyword_hits(prompt: str) -> tuple[list[str], list[str]]:
    lower = prompt.lower()
    complex_hits = [kw for kw in COMPLEX_KEYWORDS if kw in lower]
    moderate_hits = [kw for kw in MODERATE_KEYWORDS if kw in lower]
    return complex_hits, moderate_hits


def instruction_count(prompt: str) -> int:
    list_lines = len(_LIST_LINE_RE.findall(prompt))
    lower = prompt.lower()
    imperative_sentences = sum(
        1
        for sentence in re.split(r"[.?!\n]", lower)
        if any(sentence.strip().startswith(verb) for verb in _IMPERATIVE_VERBS)
    )
    return list_lines + imperative_sentences


def has_context_block(prompt: str) -> bool:
    lower = prompt.lower()
    if any(marker in lower for marker in _CONTEXT_MARKERS):
        return True
    return len(prompt) > 800


def requested_output_complexity(prompt: str) -> int:
    lower = prompt.lower()
    if any(kw in lower for kw in ("json", "table", "schema", "fields:", "format:")):
        return 2
    if any(kw in lower for kw in ("list", "bullet point", "bullet points")):
        return 1
    return 0


FEATURE_NAMES = (
    "word_count",
    "complex_keyword_count",
    "moderate_keyword_count",
    "instruction_count",
    "has_context_block",
    "output_complexity",
)


def feature_vector(prompt: str) -> list[float]:
    """Single source of truth for the numeric feature vector used both by the
    training script and the sklearn classifier at inference time - order must
    match FEATURE_NAMES."""
    complex_hits, moderate_hits = keyword_hits(prompt)
    return [
        float(word_count(prompt)),
        float(len(complex_hits)),
        float(len(moderate_hits)),
        float(instruction_count(prompt)),
        1.0 if has_context_block(prompt) else 0.0,
        float(requested_output_complexity(prompt)),
    ]
