from llm_cost_router.classifier import features as f
from llm_cost_router.classifier.base import ClassificationResult
from llm_cost_router.models.types import Tier

# Hand-tunable thresholds - adjust these based on observed misclassifications.
TIER_2_THRESHOLD = 2.0
TIER_3_THRESHOLD = 5.0


class HeuristicClassifier:
    def classify(self, prompt: str) -> ClassificationResult:
        wc = f.word_count(prompt)
        complex_hits, moderate_hits = f.keyword_hits(prompt)
        instr_count = f.instruction_count(prompt)
        context_block = f.has_context_block(prompt)
        output_complexity = f.requested_output_complexity(prompt)

        score = 0.0
        score += min(wc / 50, 3) * 1.0
        score += len(complex_hits) * 2.0
        score += len(moderate_hits) * 1.0
        score += instr_count * 1.0
        score += output_complexity * 1.5
        score += 1.0 if context_block else 0.0

        feature_dump = {
            "word_count": wc,
            "complex_keywords": ",".join(complex_hits),
            "moderate_keywords": ",".join(moderate_hits),
            "instruction_count": instr_count,
            "has_context_block": context_block,
            "output_complexity": output_complexity,
            "score": round(score, 2),
        }

        if complex_hits and wc > 30:
            tier = Tier.TIER_3
            reason = (
                f"score={score:.2f}, forced Tier 3 (complex keyword "
                f"{complex_hits} present with word_count={wc} > 30)"
            )
        elif wc < 15 and not complex_hits and not moderate_hits and not context_block:
            tier = Tier.TIER_1
            reason = (
                f"score={score:.2f}, forced Tier 1 (short prompt, word_count={wc} "
                f"< 15, no keyword/context signals)"
            )
        elif score >= TIER_3_THRESHOLD:
            tier = Tier.TIER_3
            reason = f"score={score:.2f} >= {TIER_3_THRESHOLD} -> Tier 3"
        elif score >= TIER_2_THRESHOLD:
            tier = Tier.TIER_2
            reason = f"score={score:.2f} >= {TIER_2_THRESHOLD} -> Tier 2"
        else:
            tier = Tier.TIER_1
            reason = f"score={score:.2f} < {TIER_2_THRESHOLD} -> Tier 1"

        reason += (
            f" (word_count={wc}, complex_keywords={complex_hits}, "
            f"moderate_keywords={moderate_hits}, instruction_count={instr_count}, "
            f"has_context_block={context_block}, output_complexity={output_complexity})"
        )

        return ClassificationResult(tier=tier, reason=reason, features=feature_dump)
