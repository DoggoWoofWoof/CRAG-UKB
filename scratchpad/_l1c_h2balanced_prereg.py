"""Write-once pre-registration for BALANCED_H2_PATCH1 (twelfth ruling, 2026-09-15): the frozen H2_PATCH1 of section 16 with exactly one
change -- the beam's global top-100 frontier truncation replaced by a parameter-free per-parent rank round-robin.  Written before any
balanced beam has been run on any cache (the module has only been compiled and unit-tested on a synthetic 3-parent example).

    python -u scratchpad/_l1c_h2balanced_prereg.py   -> results/L1_COVPART/PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402
import _l1g_core  # noqa: E402,F401  (registers the squad_phg cache)

X = S.X
OUT = os.path.join(X.REPO, "results", "L1_COVPART")
FP = os.path.join(OUT, "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json")
assert not os.path.exists(FP), "write-once"
CACHES = ("metaqa", "metaqa_phg", "squad", "squad_phg", "musique")
for c in CACHES:
    assert not os.path.exists(os.path.join(OUT, "h2balanced_A_%s.json" % c)), "a result exists already"
assert not os.path.exists(os.path.join(OUT, "h2balanced_A_SUMMARY.json"))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pin(p):
    return {"path": os.path.relpath(p, X.REPO).replace("\\", "/"), "sha256": sha_file(p), "bytes": os.path.getsize(p)}


mod = open(os.path.join(HERE, "_l1c_h2balanced.py"), encoding="utf-8").read()
rule_src = mod[mod.index("def rr_compress("):mod.index("def beam_balanced(")]
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json", "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json", "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}

RULING = (
    "On the beam side ... the missing gold nodes are already in the candidate set before Top-100 pruning (173/174 MtK failures, 160/162 PHG failures) "
    "... we do not actually need to traverse more. We need a better vectorized compression of ~215 H2 candidates into the 100-node patch ... per-parent "
    "segmented Top-K before global packing ... However, do not choose k_local from MetaQA now. That creates another constant. There's an even cleaner "
    "parameter-free version: Round-robin by within-parent rank ... A1 B1 C1 ... A2 B2 C2 ... until the global budget is filled. No local K. No learned "
    "weights. No dataset parameter. The only existing constant remains the global beam/patch capacity ... B. BALANCED_H2_PATCH1: Keep the exact H2 graph "
    "expansion and 100-node patch, but replace global candidate truncation with a parameter-free per-parent rank interleave before patch packing. "
    "Question: Can we recover the already-present missing hop-2 candidates without increasing traversal depth, candidate generation, or patch exposure? "
    "This is the pseudo-L3/L1-booster workstream. And I would still leave frozen H2_PATCH1 untouched for WebQSP transfer. These are new exploratory "
    "arms, not modifications to the current candidate ... I would not execute either on MetaQA immediately without a clean preregistration.")

rec = {
    "RECORD": "PREREGISTRATION",
    "name": "BALANCED_H2_PATCH1_DEV_A",
    "STATUS": "POSTHOC_MECHANISM_TEST",
    "written_before_any_run": True,
    "written_before": ["any balanced beam on any cache", "any BALANCED_H2_PATCH1 number", "any h2balanced_A_*.json"],
    "what_was_run_before_this_record": "the module was compiled and its compression function unit-tested on a synthetic 3-parent example (no cache, no graph, no gold read)",
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ruling_2026-09-15_twelfth_verbatim_essentials_beam_side": RULING,
    "question": "Can the already-present missing hop-2 candidates (section 17: 630 / 624 missing gold nodes of the failing hop-2 queries are scored candidates cut by the global top-100) be recovered without increasing traversal depth, candidate generation or patch exposure, by replacing the global truncation with a parameter-free per-parent rank interleave?",
    "why_posthoc": "DEV_A has been read by sections 2-20 of this lane; this is a mechanism test on an already-read population. Promotion requires an untouched population (WebQSP or DEV_B by ruling). The frozen H2_PATCH1 is untouched and remains the WebQSP transfer candidate.",
    "frozen_and_reused_verbatim": {
        "canonical_L1": "served base_rank of each replay cache; unchanged",
        "canonical_SAFE": "the frozen B6_S4_F6_Ms64_Mr32 selector through its own code (_l1ps_router.build_cache + _l1kb_core.contexts / f6_select); its recomputed indicator must equal the frozen replay records' _ind_SAFE on every DEV_A row (asserted, else the run is invalid)",
        "H2_beam_candidate_generation": "the pinned MICRO_L3_H2 beam of scratchpad/_l1c_microl3.py (sha256 19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5, asserted): frozen 5 seeds, actual STRUCT adjacency under the frozen hub policy (out-edges always, in-edges only for non-hub sources), depth exactly 2, BEAM = 100 per hop, frozen transition geometry S_dst / S_dir / S_mid and frozen RRF (K0 = 60) fused transition order -- the module's own head is executed unchanged (import-by-exec, name substituted), so the pinned beams are produced by the pinned code and are asserted equal to a re-run that additionally records the candidates",
        "H2_PATCH1": "section 16 verbatim (scratchpad/_l1c_h2patch.py sha256 a1ed9caa906fc50a1cee573e346bec1fde70e1947badd68c9b86cd2ab455fa9d, asserted; copied patch machinery): SAFE minus its weakest admitted block (the sixth block of the F6 competition order) + one query-local virtual block of the same size = the beam's visited nodes not exposed by the 49 kept blocks in first-visit order (hop-1 beam then hop-2 beam, seeds excluded), topped up from the removed block (ret_rrf position ascending, then node id); |served| matched per query (asserted). The rebuilt H2_PATCH1 arm must reproduce h2patch_A_<cache>.json exactly (ALL, ANY, vs SAFE, per hop, novel-node counts, truncated queries) -- asserted, else the run is invalid",
        "partition": "the served hard partition of each cache (hard[v], part_sizes); blocks are disjoint and served whole",
    },
    "the_one_change_frontier_compression": {
        "pinned": "B(h) = the first BEAM = 100 distinct unvisited targets of the frozen fused transition order over the union of all parents' transitions (a global top-100)",
        "balanced": "(1) (parent, target) pairs in the frozen fused order, a pair reached twice keeping its FIRST occurrence; (2) within-parent rank r(m, u) = position of the pair among parent m's pairs in that order (the restriction of the frozen fused order to one parent; 0-based); (3) sequence = all pairs sorted by (r, position of the parent in the frontier order) = A1 B1 C1 ... A2 B2 C2 ...; (4) B(h) = the first BEAM distinct unvisited targets of that sequence, a target already taken by an earlier pair of the sequence being skipped; visited marks exactly as the pinned beam",
        "applied_at_every_hop": "hop 1: parents = the frozen seeds in their frozen order; hop 2: parents = the balanced hop-1 beam in its sequence order (a parent with no admissible transition contributes nothing)",
        "constants": "none new: BEAM = 100, DEPTH = 2, K0 = 60 and the patch budget (= the removed block's size) are the frozen constants of sections 14 / 16; no local K, no learned weight, no dataset parameter, no tie-break constant (ties in the fused order are resolved by the frozen fused order itself)",
        "unchanged": "admissibility, transition geometry, fused order, width, depth, seeds, patch rule, capacity, fill order, SAFE, L1, the metric",
        "code_verbatim": rule_src,
        "unit_test": "synthetic: frontier [A, B, C] with fused pairs (A,a1) (B,b1=a1) (A,a2) (C,c1) (B,b2) (A,a3) (A,a2 dup) -> budget 4 keeps [a1, c1, a2, b2] (B1 duplicate skipped, rounds 2); budget 100 keeps [a1, c1, a2, b2, a3] (rounds 3); frontier [B, A, C] with c1 visited and budget 3 keeps [a1, b2, a2]",
    },
    "pre_stated_risk_no_threshold": "a hop-2 question whose several answer nodes hang under ONE hop-1 parent (multi-answer branches) can lose slots to the round-robin that the global order would have given it; the balanced beam therefore trades within-branch depth for cross-branch coverage. This is reported (per-query flips vs H2_PATCH1 with their cause: gold cut by the balanced beam vs recovered by it), not thresholded; no hybrid, no k_local and no mixed order will be run after seeing it.",
    "arms": {"L1": "served P50 (reference only)", "SAFE": "canonical SAFE (reference)", "H2_PATCH1": "the frozen section-16 arm rebuilt in-process (must equal the frozen record)",
             "BALANCED_H2_PATCH1": "the new arm: the same patch rule over the balanced beam's visited nodes"},
    "population": "DEV_A of the five caches metaqa (Mt-KaHyPar), metaqa_phg, squad (Mt-KaHyPar), squad_phg, musique (PHG); DEV_B and TEST never read",
    "metric": "ALL-gold under matched unique-node exposure (node level; == block level for L1 / SAFE, asserted); ANY-gold reported; per hop on metaqa; McNemar exact two-sided on paired DEV_A 0/1 vectors (_l1x90_core.mcnemar)",
    "decision_rule": {
        "primary_comparison": "BALANCED_H2_PATCH1 vs H2_PATCH1 (the frozen arm), not vs SAFE: the question is whether the compression recovers what the global cut loses",
        "metaqa_required": "hop-2 ALL-gold BALANCED_H2_PATCH1 vs H2_PATCH1: gained > lost and p < 0.01 on BOTH metaqa caches",
        "no_regression_required": "hop-1 and hop-3 ALL-gold vs H2_PATCH1 on either metaqa cache: NOT (lost > gained and p < 0.05)",
        "text_safety_required": "on each of squad, squad_phg, musique: NOT (lost > gained and p < 0.05) for BALANCED_H2_PATCH1 vs H2_PATCH1 AND vs SAFE",
        "PASS": "MECHANISM_CONFIRMED_ON_DEV_A (POSTHOC; not promoted): the parameter-free compression recovers hop-2 candidates the global cut loses; the frozen H2_PATCH1 stays the WebQSP transfer candidate; no promotion, no composition",
        "FAIL": "NOT_CONFIRMED with the failing clause named (no hop-2 gain / text safety / hop-1-or-3 regression); reported with the flip attribution; no second compression rule is run",
        "magnitude_is_reported_not_thresholded": "hop-2 points vs H2_PATCH1, vs SAFE and vs L1; pooled hop 2+3; beam diversity (distinct parents represented, largest parent share, candidate-set overlap with the pinned beam); the section-17 within-depth-2 missing-gold population and how many of those nodes the balanced beam visits",
        "alpha": 0.01,
    },
    "also_recorded": ["per query: candidates unique / pairs / kept / truncated / rounds used per hop; same-set fraction and Jaccard between the pinned and balanced beams",
                      "flips vs H2_PATCH1 with causes (gold node newly visited by the balanced beam / gold node no longer visited / patch-capacity displacement)",
                      "the section-17 population reproduced against h3patch_why_A_<cache>.json (metaqa caches) and the coverage of its missing gold nodes by the balanced beam",
                      "wall time per query of the balanced beam (vectorised numpy; no Python per-candidate loop except the final sequential take)"],
    "forbidden": ["reading DEV_B or TEST", "any change to BEAM / DEPTH / K0 / seeds / admissibility / transition geometry / the fused order / the patch rule / the capacity / the fill order",
                  "a local K, a learned weight, a dataset parameter, a hybrid of the two orders, a second compression rule after seeing numbers",
                  "editing scratchpad/_l1c_microl3.py, scratchpad/_l1c_h2patch.py or any CONTRACT_FILE / pinned module; editing scratchpad/_l1c_h2balanced.py after this record (its sha is pinned below and asserted at run time)",
                  "composing with SAFE variants, H3, VCUT / R2 or any other arm", "tuning after seeing numbers; changing a label or threshold after a number is seen",
                  "writing under data/", "touching the WebQSP watcher or the foreign process"],
    "code": {
        "this_module": pin(os.path.join(HERE, "_l1c_h2balanced.py")),
        "usage": "python -u scratchpad/_l1c_h2balanced.py <cache> -> results/L1_COVPART/h2balanced_A_<cache>.json; --summary -> h2balanced_A_SUMMARY.json",
        "pinned_modules": {n: pin(os.path.join(HERE, n)) for n in ("_l1c_microl3.py", "_l1c_h2patch.py", "_l1kb_core.py", "_l1ps_router.py", "_l1g_core.py", "_l1s_core.py", "_l1x90_core.py", "_ta_prepartition.py")},
    },
    "records_read_only": {**{"h2patch_A_%s.json" % c: pin(os.path.join(OUT, "h2patch_A_%s.json" % c)) for c in CACHES},
                          **{"microl3_A_%s.json" % c: pin(os.path.join(OUT, "microl3_A_%s.json" % c)) for c in CACHES},
                          **{"h3patch_why_A_%s.json" % c: pin(os.path.join(OUT, "h3patch_why_A_%s.json" % c)) for c in ("metaqa", "metaqa_phg")},
                          "PREREGISTRATION_H2_PATCH1_DEV_A.json": pin(os.path.join(OUT, "PREREGISTRATION_H2_PATCH1_DEV_A.json")),
                          "PREREGISTRATION_MICROL3_DEV_A.json": pin(os.path.join(OUT, "PREREGISTRATION_MICROL3_DEV_A.json")),
                          "PREREGISTRATION_H3_PATCH1_DEV_A.json": pin(os.path.join(OUT, "PREREGISTRATION_H3_PATCH1_DEV_A.json"))},
    "SAFE_records_read_only": {c: pin(os.path.join(X.REPO, p)) for c, p in SAFE_RECORD.items()},
    "caches_read_only": {n: pin(X.CACHES[n]) for n in CACHES},
}
S.wj(FP, rec)
print("written", os.path.relpath(FP, X.REPO), sha_file(FP), os.path.getsize(FP), rec["utc"])
