"""Supersede PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json (v1, 17:50:31Z) after the first metaqa run died at a DIAGNOSTIC cross-check:
the section-17 population recomputation counted a missing gold node that is a frozen SEED as 'never scored', while section 17
(_l1c_h3patch_why.py) skips such a node; the assertion against h3patch_why_A_metaqa.json therefore failed (172 vs 173) after every
arm number of metaqa had already been computed and logged.  The fix touches only that diagnostic loop (the seed is now skipped exactly
as in section 17); the arms, the rule, the thresholds and the compression function are byte-identical.  The original record goes to
_history/ unchanged; this v2 carries the new module sha, the reason, and the metaqa numbers that had been seen when the fix was made.

    python -u scratchpad/_l1c_h2balanced_prereg_v2.py
"""
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402

X = S.X
OUT = os.path.join(X.REPO, "results", "L1_COVPART")
HIST = os.path.join(OUT, "_history")
FP = os.path.join(OUT, "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pin(p):
    return {"path": os.path.relpath(p, X.REPO).replace("\\", "/"), "sha256": sha_file(p), "bytes": os.path.getsize(p)}


v1 = json.load(open(FP, encoding="utf-8"))
assert "supersedes" not in v1 and v1["utc"] == "2026-09-15T17:50:31Z"
v1_sha = sha_file(FP)
os.makedirs(HIST, exist_ok=True)
hp = os.path.join(HIST, "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A__v1_2026-09-15T17-50-31Z.json")
assert not os.path.exists(hp)
shutil.move(FP, hp)
assert sha_file(hp) == v1_sha
assert not any(os.path.exists(os.path.join(OUT, "h2balanced_A_%s.json" % c)) for c in ("metaqa", "metaqa_phg", "squad", "squad_phg", "musique"))

mod = open(os.path.join(HERE, "_l1c_h2balanced.py"), encoding="utf-8").read()
rule_src = mod[mod.index("def rr_compress("):mod.index("def beam_balanced(")]
assert rule_src == v1["the_one_change_frontier_compression"]["code_verbatim"], "the compression function changed -- not allowed"

v2 = dict(v1)
v2["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
v2["code"] = dict(v1["code"])
v2["code"]["this_module"] = pin(os.path.join(HERE, "_l1c_h2balanced.py"))
v2["written_before_any_run"] = False
v2["supersedes"] = {
    "v1": {"path": os.path.relpath(hp, X.REPO).replace("\\", "/"), "sha256": v1_sha, "utc": v1["utc"], "module_sha256_v1": v1["code"]["this_module"]["sha256"]},
    "why": "the first metaqa run (log: results/L1_COVPART/h2balanced_A_metaqa__run1_failed_at_the_section17_crosscheck.log) computed every arm number and then died at the diagnostic cross-check of the section-17 population against h3patch_why_A_metaqa.json (172 vs 173 for hop2): the recomputation classified a missing gold node that is one of the frozen seeds as 'never scored', while _l1c_h3patch_why.py skips a seed gold node (neither scored nor visited by a hop). No result JSON was written.",
    "what_changed_in_the_module": "only the diagnostic loop (the section-17 population and node-level availability): a missing gold node that is a frozen seed is now counted in a 'missing_is_seed (skipped exactly as in section 17)' bucket and skipped, exactly as section 17 does. The compression function (asserted byte-identical above), the beam, the patch machinery, the arms, the metric, the decision rule, the thresholds and the summary are unchanged.",
    "numbers_seen_before_the_fix (metaqa DEV_A, from the failed run's log; stated so that nothing can be tuned to them)": {
        "L1": 0.6397, "SAFE": 0.6407, "H2_PATCH1": 0.7515, "BALANCED_H2_PATCH1": 0.7764,
        "BALANCED_vs_H2_PATCH1_hop2": {"n": 344, "H2_PATCH1": 0.7413, "BALANCED": 0.814, "gained": 30, "lost": 5, "p": 2.24e-05},
        "hop1_and_hop3_vs_H2_PATCH1": "0 / 0 on both",
        "beam_diversity": "hop-2 same kept set on 138 / 1002 queries, Jaccard 0.595, distinct parents 19.1 -> 24.0, largest parent share 0.234 -> 0.105, wall-clock 24.3 -> 20.0 ms",
        "flips": "30 newly covered (58 gold nodes newly visited by the balanced beam), 5 newly lost (8 gold nodes no longer visited)",
    },
    "consequence_for_the_status": "the metaqa (Mt-KaHyPar cache) numbers were seen before this record; the rule was fixed in v1 before any run and is unchanged here, so the decision is still the pre-registered one, but the record honestly states that the v2 re-run of metaqa is a reproduction of already-seen numbers; metaqa_phg, squad, squad_phg and musique had not been run.",
}
S.wj(FP, v2)
print("v1 ->", os.path.relpath(hp, X.REPO), v1_sha)
print("written", os.path.relpath(FP, X.REPO), sha_file(FP), os.path.getsize(FP), v2["utc"])
