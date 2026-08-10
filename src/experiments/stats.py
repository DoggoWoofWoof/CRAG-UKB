"""
Shared significance testing for paired per-query comparisons.
==============================================================
Exact two-sided McNemar in log-space (overflow-safe at large n) + a helper that
takes two per-query binary outcome vectors (treatment vs baseline, e.g. per-query
FullCov@20) and returns the discordant counts + p-value. Used across the ablation
campaign so every "method A beats method B" claim carries a paired p-value.
"""
import math
from typing import List, Dict


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for discordant counts b, c (binomial(n,0.5))."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    log_half_n = -n * math.log(2.0)
    terms = [
        (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)) + log_half_n
        for i in range(k + 1)
    ]
    m = max(terms)
    log_tail = m + math.log(sum(math.exp(t - m) for t in terms))
    return float(min(1.0, 2.0 * math.exp(log_tail)))


def paired(treatment: List[int], baseline: List[int]) -> Dict:
    """Paired McNemar over per-query 0/1 outcomes (aligned same queries, same order).
    b = treatment wins where baseline fails; c = baseline wins where treatment fails."""
    if len(treatment) != len(baseline):
        return {"error": f"length mismatch {len(treatment)} vs {len(baseline)}"}
    b = sum(1 for t, base in zip(treatment, baseline) if t == 1 and base == 0)
    c = sum(1 for t, base in zip(treatment, baseline) if t == 0 and base == 1)
    p = mcnemar_exact(b, c)
    return {
        "n": len(treatment),
        "treatment_mean": round(sum(treatment) / max(len(treatment), 1) * 100, 2),
        "baseline_mean": round(sum(baseline) / max(len(baseline), 1) * 100, 2),
        "b_treatment_only": b, "c_baseline_only": c, "n_discordant": b + c,
        "p_value": round(p, 6), "significant_0.05": bool(p < 0.05),
    }
