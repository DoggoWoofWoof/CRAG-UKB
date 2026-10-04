"""Write-once pre-registration for COVSCORE_H2_PATCH1 (fourteenth ruling, item 2, 2026-09-16): the ONE coverage-then-score successor of the
section-21 frontier compression, on the five DEV_A caches.  Written after the unit test and the identity-only dry run and before any coverage
number of the new arm exists.

    python -u scratchpad/_l1c_covscore_prereg.py <dry_run_log>   -> results/L1_COVPART/PREREGISTRATION_COVSCORE_H2_PATCH1.json
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402
import _l1g_core  # noqa: E402,F401  (registers the squad_phg cache)
import _l1c_br2_lib as L  # noqa: E402  (sha_file / pin helpers only)

X = S.X
OUT = os.path.join(X.REPO, "results", "L1_COVPART")
FP = os.path.join(OUT, "PREREGISTRATION_COVSCORE_H2_PATCH1.json")
assert not os.path.exists(FP), "write-once"
assert not any(f.startswith("covscore") for f in os.listdir(OUT)), "a COVSCORE_H2_PATCH1 result exists already"
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
sha_file, pin = L.sha_file, L.pin
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json", "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json", "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}

# ---- what the dry run printed (gold-free), taken from its log verbatim
dry_log = open(sys.argv[1], encoding="utf-8").read()
assert "DRY RUN:" in dry_log and "rc=0" in dry_log, "the dry run did not complete"
dry_lines = [ln for ln in dry_log.splitlines() if any(k in ln for k in ("unit test OK", "beam done:", "beams done", "balanced beam reproduced", "covscore beam (gold-free)",
                                                                          "canonical SAFE reproduced", "DRY RUN:"))]
cov_beam = json.loads(re.search(r"covscore beam \(gold-free\) (\{.*\})", dry_log).group(1))

pre21 = json.load(open(os.path.join(OUT, "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json"), encoding="utf-8"))
sum21 = json.load(open(os.path.join(OUT, "h2balanced_A_SUMMARY.json"), encoding="utf-8"))
sum16 = json.load(open(os.path.join(OUT, "h2patch_A_SUMMARY.json"), encoding="utf-8"))
r21 = {c: json.load(open(os.path.join(OUT, "h2balanced_A_%s.json" % c), encoding="utf-8")) for c in CACHES}
for c in CACHES:
    assert r21[c]["code"]["sha256"] == pre21["code"]["this_module"]["sha256"] == sha_file(os.path.join(HERE, "_l1c_h2balanced.py"))
seen = {}
for c in CACHES:
    a = r21[c]["arms"]
    seen[c] = {"n_DEV_A": r21[c]["n_DEV_A"], "L1": a["L1"]["ALL"], "SAFE": a["SAFE"]["ALL"], "H2_PATCH1": a["H2_PATCH1"]["ALL"], "BALANCED_H2_PATCH1": a["BALANCED_H2_PATCH1"]["ALL"],
               "BALANCED_vs_H2_PATCH1": a["BALANCED_H2_PATCH1"]["ALL_vs_H2_PATCH1"], "BALANCED_vs_SAFE": a["BALANCED_H2_PATCH1"]["ALL_vs_SAFE"],
               "per_hop_BALANCED": {h: e["ALL"] for h, e in a["BALANCED_H2_PATCH1"]["per_hop"].items()},
               "per_hop_BALANCED_vs_H2_PATCH1": {h: [e["vs_H2_PATCH1"]["gained"], e["vs_H2_PATCH1"]["lost"]] for h, e in a["BALANCED_H2_PATCH1"]["per_hop"].items()},
               "beam_size_mean_balanced (hop1 / hop2)": r21[c]["beam_diversity_DEV_A"]["beam_size_mean (hop1 / hop2)"]["balanced"],
               "queries_truncated_at_hop_balanced": r21[c]["beam_diversity_DEV_A"]["queries_truncated_at_hop (hop1 / hop2) balanced"]}
# item 1 (QMAX_BALANCED_H2_PATCH1) records that exist at the time of writing: seen, but they concern the composed chain; the reference arms in them equal section 21
item1 = {}
for c in CACHES:
    p = os.path.join(OUT, "qmaxbal_A_%s.json" % c)
    if os.path.exists(p):
        r = json.load(open(p, encoding="utf-8"))
        item1[c] = {"QMAX_BALANCED_H2_PATCH1": r["arms"]["QMAX_BALANCED_H2_PATCH1"]["ALL"], "cell": r["per_cache_outcome"]["cell"]}
    else:
        item1[c] = "not yet run at the time of this record (its chain is running; its numbers do not enter this test)"

RULING = (
    "2. BALANCED is good, but its sparse/dense tradeoff can probably be improved once ... I would try exactly one parameter-free rule: Coverage-then-score frontier: First give "
    "every active parent one child (A1 B1 C1 D1 ...), then fill the remaining global budget using the original frozen global candidate ranking (A2 A3 B2 A4 ...). No k_local. No "
    "lambda. No learned weighting. No dataset branch. B = {best child of each parent} u TopK(remaining candidates) until capacity is reached. ... I would not run five diversity "
    "methods. One coverage-then-score successor is enough. ... 3. Only after fixing H2 compression should you reopen H3 ... If coverage-then-score or BALANCED keeps more correct "
    "H2 parents, then run one H3_PATCH1 successor using that frontier. No wider patch initially. No H3-specific scoring. Same real-edge maths. ... We have over-read DEV_A. So "
    "don't use MetaQA/MuSiQue indefinitely to select every next rule.")

src = open(os.path.join(HERE, "_l1c_covscore.py"), encoding="utf-8").read()
code_verbatim = src[src.index("def cs_compress_core"):src.index("def unit_test")].rstrip().splitlines()

rec = {
    "RECORD": "PREREGISTRATION",
    "name": "COVSCORE_H2_PATCH1",
    "STATUS": "PREREGISTERED_SUCCESSOR_TEST_DEV_A",
    "written_before_any_run": True,
    "written_before": ["any coverage (ALL / ANY) number of COVSCORE_H2_PATCH1 on any cache", "any paired comparison of the new arm", "any per-hop number of the new arm",
                       "any gold-bearing diagnostic of the covscore beam", "any number of any kind for the new arm on metaqa_phg, squad, squad_phg or musique"],
    "what_was_run_before_this_record": {
        "unit_test": "python -u scratchpad/_l1c_covscore.py --unit: five synthetic examples (no cache, no graph, no gold) assert the hand-derived sequences, including the example "
                     "that separates the rule from the section-21 round-robin ([a1 b1 c1 a2 a3] vs [a1 b1 c1 a2 b2]) and a permuted fused order",
        "dry_run": "python -u scratchpad/_l1c_covscore.py metaqa --dry: the pinned beam with candidate recording == the exec'd pinned beam; the balanced beam reproduces every "
                   "deterministic figure of h2balanced_A_metaqa.json beam_diversity_DEV_A; canonical SAFE / BASE == the L1_REPLAY record on DEV_A rows; the covscore beam built "
                   "and its gold-free statistics printed; no coverage number of the new arm; no record",
        "dry_run_log_lines_verbatim": dry_lines,
        "covscore_beam_gold_free_statistics_metaqa_DEV_A": cov_beam,
        "coverage_of_the_new_arm": "NEVER computed anywhere before this record",
    },
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ruling_2026-09-16_fourteenth_item_2_verbatim_essentials": RULING,
    "question": "Does the ruling's coverage-then-score frontier (one child per active parent, then the frozen global ranking) keep more correct hop-2 parents / gold children than "
                "the section-21 round-robin (BALANCED_H2_PATCH1) -- i.e. is it a better sparse/dense compromise -- with the frozen H2_PATCH1 patch, budget, beam and SAFE unchanged?",
    "frozen_and_reused_verbatim": {
        "beam": "the pinned uL3 beam of _l1c_microl3.py (sha 19774617...): frozen admissibility (out always, in for non-hubs; hub = deg_u + 1 > N / k), frozen transition geometry "
                "S_dst / S_dir / S_mid, frozen RRF fused order (K0 = 60), BEAM = 100, DEPTH = 2, seeds = the frozen top-5 RRF nodes; exec'd verbatim; the section-16 pinned beam and "
                "the section-21 balanced beam are recomputed and asserted against their records",
        "patch": "section 16 verbatim (_l1c_h2patch.py sha a1ed9caa...; build_patches copied from _l1c_h2balanced.py sha 7aed6b1d...): SAFE minus its weakest admitted block + one "
                 "query-local block of the same size = the beam's visited nodes not exposed by the 49 kept blocks in first-visit order (hop 1 then hop 2), topped up from the removed "
                 "block (ret_rrf position, then node id); matched unique-node exposure (asserted)",
        "SAFE": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select, BASE_CFG unchanged (B = 6, M_struct = 64, M_ret = 32, S4, F6); served BASE ranking",
        "constants": "K0 = 60, K_LOCK = 100, P_MAIN = 50, B = 6, BEAM = 100, DEPTH = 2 -- all frozen before; nothing new, nothing fitted, no dataset branch, no weight",
    },
    "the_one_change_frontier_compression": {
        "rule": "(1) (parent, target) pairs in the frozen fused transition order, a pair reached twice keeping its first occurrence; (2) within-parent rank r = position of the pair "
                "among its parent's pairs in that order (section 21's definition); (3) phase 1 = the pairs with r = 0 in frontier order (A1 B1 C1 D1 ...: the best child of every active "
                "parent); phase 2 = ALL remaining pairs in the frozen global fused order (A2 A3 B2 A4 ...); (4) B(h) = the first BEAM distinct unvisited targets of phase 1 then phase 2, "
                "a target already taken being skipped, visited marked exactly as the pinned beam. A parent whose best child is already taken contributes nothing in phase 1. "
                "B = {best child of each parent} u TopK(remaining candidates) until capacity.",
        "what_differs_from_section_21": "section 21 continues round-robin after round 1 (A2 B2 C2 ... A3 B3 ...); this rule fills the remainder in the frozen global order. Both rules "
                                        "share round 1 exactly. Where a hop has at most BEAM distinct unvisited candidates, all three beams (pinned, balanced, covscore) are identical.",
        "no_constant": "no k_local, no lambda, no weight, no dataset branch; the only constants are the frozen BEAM / DEPTH / K0 and the frozen patch budget",
        "code_verbatim (scratchpad/_l1c_covscore.py, cs_compress_core)": code_verbatim,
        "unit_test": "five synthetic examples asserted at every start of the runner (see what_was_run_before_this_record)",
    },
    "arms": {
        "reference (recomputed; asserted equal to the frozen section-16 / 21 records)": ["L1", "SAFE", "H2_PATCH1", "BALANCED_H2_PATCH1"],
        "new": {"COVSCORE_H2_PATCH1": "the ONLY decided-on arm: the section-16 patch over the covscore beam"},
        "not_run": ["any k_local / lambda / weighted variant", "any other diversity method", "COVSCORE over the QMAX_F0 ranking (item 1 is a separate pre-registration)",
                    "depth 3 (item 3 is conditional on this outcome and separately pre-registered)", "any change of the patch, budget, beam width, SAFE or fusion"],
    },
    "seen_numbers_DEV_A (section 21 records; the reference arms of this test)": seen,
    "item1_records_existing_at_write_time (QMAX_BALANCED_H2_PATCH1; not used by this test)": item1,
    "section21_and_16_outcomes_read": {"BALANCED_H2_PATCH1": {"verdict": sum21["verdict"], "label": sum21["label"]}, "H2_PATCH1": {"verdict": sum16.get("verdict"), "label": sum16.get("label")}},
    "predictions_stated_now": {
        "mechanism": "the rule differs from BALANCED only where a hop has more than 100 distinct candidates: at hop 1 (5 seeds) it keeps one child per seed and then the global top of the "
                     "rest (BALANCED keeps ~20 per seed); at hop 2 (up to 100 parents) phase 1 alone can fill the budget, in which case it equals BALANCED's round 1. The metaqa dry run "
                     "shows the covscore beam closer to the pinned beam than to the balanced beam at hop 2 (Jaccard %s vs %s), identical to both for most queries (%s / %s of 1,002 same "
                     "set as pinned / balanced at hop 2)." % (cov_beam["jaccard_kept_vs_pinned_mean (hop1 / hop2)"][1], cov_beam["jaccard_kept_vs_balanced_mean (hop1 / hop2)"][1],
                                                                cov_beam["same_kept_set_as_pinned (queries; hop1 / hop2)"][1], cov_beam["same_kept_set_as_balanced (queries; hop1 / hop2)"][1]),
        "direction": "uncertain, stated honestly: section 21's hop-2 gain (+30 / -5 on metaqa) came from representing more hop-1 parents at hop 2 and from balancing the 5 seeds at hop 1. "
                     "If the gain came from round 1 (coverage), COVSCORE keeps it and may add the global fill's precision (PASS_SUCCESSOR); if it came from the later rounds (the second "
                     "and third children of the right parents), COVSCORE lands between H2_PATCH1 and BALANCED (FAIL_NOT_BETTER or FAIL_WORSE). The dry run's identity (covscore closer to "
                     "the pinned beam) makes the second reading more likely on metaqa; text caches: within noise of BALANCED (squad at the ceiling; musique differences of a few queries).",
    },
    "population": {c: "served %s cache rows; DEV_A = sha1(query_id)[:8] & 1 == 0 exactly as _l1x90_core; hop labels as present in the cache" % c for c in CACHES}
                  | {"DEV_B": "never read", "TEST": "never read", "WebQSP / 2Wiki / HotpotQA": "not used"},
    "statistics": {"test": "exact two-sided McNemar on paired DEV_A 0/1 ALL vectors (_l1x90_core.mcnemar): gained / lost / p",
                   "alpha_gain": 0.01, "alpha_regression_and_safety": 0.05,
                   "labels": {"GAIN": "gained > lost and p < 0.01", "LOSS": "lost > gained and p < 0.01", "NEUTRAL": "otherwise"},
                   "decision_comparisons": "COVSCORE_H2_PATCH1 vs BALANCED_H2_PATCH1 (hop 2 and hop 1 / hop 3 on the two metaqa caches; ALL on the three text caches) and, for text safety only, "
                                           "vs H2_PATCH1 and vs SAFE; everything else (vs L1, per hop on text, flips, beam statistics, the gold-bearing-parent diagnostic) is descriptive"},
    "decision_rule_pre_registered": {
        "metaqa_required": "hop-2 ALL of COVSCORE_H2_PATCH1 vs BALANCED_H2_PATCH1 with gained > lost and p < 0.01 on BOTH metaqa and metaqa_phg",
        "no_regression_required": "hop-1 and hop-3 vs BALANCED_H2_PATCH1 on both metaqa caches: not (lost > gained and p < 0.05)",
        "text_safety_required": "squad, squad_phg, musique: ALL vs BALANCED_H2_PATCH1, vs H2_PATCH1 and vs SAFE: not (lost > gained and p < 0.05)",
        "OUTCOME": {"PASS_SUCCESSOR": "metaqa_required AND no_regression_required AND text_safety_required",
                    "FAIL_WORSE": "not PASS and (a LOSS vs BALANCED_H2_PATCH1 on a metaqa cache at hop 2 or on ALL (p < 0.01) OR a hop-1 / hop-3 regression OR a text-safety violation)",
                    "FAIL_NOT_BETTER": "otherwise (neither the required gain nor a loss)"},
        "frontier_for_item_3": "COVSCORE_H2_PATCH1 if PASS_SUCCESSOR, else BALANCED_H2_PATCH1 (section 21 already showed BALANCED keeps more correct hop-2 parents than the pinned beam: "
                               "hop 2 +30 / -5 and +31 / -6 vs H2_PATCH1); the H3 successor of item 3 is a separate pre-registration on that frontier",
        "what_follows": {
            "PASS_SUCCESSOR": "COVSCORE_H2_PATCH1 = the current best experimental successor (MECHANISM_CONFIRMED_ON_DEV_A, pre-registered; not promoted; NOT transfer-confirmed); "
                              "BALANCED_H2_PATCH1 keeps its section-21 record; H2_PATCH1 stays the frozen WebQSP transfer reference exactly as section 16; item 3 runs on the covscore frontier",
            "FAIL_WORSE": "the coverage-then-score rule is recorded as worse than the round-robin on DEV_A and CLOSED (no k_local, no lambda, no other diversity rule -- 'one coverage-then-score "
                          "successor is enough'); BALANCED_H2_PATCH1 keeps the frontier role; item 3 runs on the balanced frontier",
            "FAIL_NOT_BETTER": "the rule is recorded as not better than the round-robin on DEV_A and CLOSED; BALANCED_H2_PATCH1 keeps the frontier role; item 3 runs on the balanced frontier",
        },
        "no_promotion": "nothing here changes the served L1 or any frozen record; sections 16-24 are unchanged; the WebQSP primary transfer remains frozen H2_PATCH1 exactly as section 16",
    },
    "also_recorded (descriptive, never decided on)": [
        "COVSCORE vs H2_PATCH1, vs SAFE, vs L1 (ALL, ANY, per hop, hop-2 + hop-3 pooled)",
        "flips vs BALANCED and vs H2_PATCH1 with their cause (newly visited / cut by the reference patch; no longer visited / beyond the patch capacity)",
        "the ruling's diagnostic: on SAFE failures, missing gold nodes visited / scored-but-not-kept / never scored per beam, gold-bearing hop-1 parents and how many keep a gold child at hop 2, "
        "failures with every missing node visited, failures covered by each arm (per hop on metaqa)",
        "gold-free beam statistics: beam sizes, truncation, Jaccard vs pinned / balanced, distinct parents, largest-parent share, phase-1 / phase-2 composition, wall clock",
    ],
    "forbidden": [
        "reading DEV_B or TEST; using WebQSP, 2Wiki or HotpotQA",
        "any second compression rule, any k_local / lambda / weight, any dataset branch, any change of the patch, budget, beam width, depth, SAFE, F6, RRF or hub definition",
        "editing any CONTRACT_FILE or pinned module; editing the two new modules after this record (their shas are pinned below and asserted at run time)",
        "signalling the foreign process; touching the WebQSP watcher; writing under data/",
        "tuning after seeing numbers; changing a label, clause or threshold after a number is seen; re-running a cache after seeing its number",
    ],
    "code": {
        "new_modules": {n: pin(os.path.join(HERE, n)) for n in ("_l1c_covscore.py", "_l1c_covscore_summary.py")},
        "usage": ["python -u scratchpad/_l1c_covscore.py <cache> for metaqa, metaqa_phg, squad, squad_phg, musique (chain: scratchpad/_l1c_covscore_chain.sh)", "python -u scratchpad/_l1c_covscore_summary.py"],
        "imports_unchanged": {**{n: pin(os.path.join(HERE, n)) for n in ("_l1g_core.py", "_l1s_core.py", "_l1x90_core.py", "_ta_prepartition.py", "_l1ps_router.py", "_l1kb_core.py",
                                                                        "_l1c_microl3.py", "_l1c_h2patch.py", "_l1c_h2balanced.py")},
                              **{"src/l1_canonical/%s.py" % n: pin(os.path.join(X.REPO, "src", "l1_canonical", n + ".py")) for n in ("adapter", "hypergraph")}},
        "pins_asserted_at_run_time": "the runner asserts the pinned module shas, this record's new-module / import / cache / record shas and write-once outputs; the new arm is computed only "
                                     "after the reference arms are reproduced against the section-16 / 21 records",
    },
    "records_read_only": {**{"h2patch_A_%s.json" % c: pin(os.path.join(OUT, "h2patch_A_%s.json" % c)) for c in CACHES},
                          **{"h2balanced_A_%s.json" % c: pin(os.path.join(OUT, "h2balanced_A_%s.json" % c)) for c in CACHES},
                          **{os.path.basename(SAFE_RECORD[c]): pin(os.path.join(X.REPO, SAFE_RECORD[c])) for c in CACHES},
                          "h2balanced_A_SUMMARY.json": pin(os.path.join(OUT, "h2balanced_A_SUMMARY.json")), "h2patch_A_SUMMARY.json": pin(os.path.join(OUT, "h2patch_A_SUMMARY.json")),
                          "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json": pin(os.path.join(OUT, "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json")),
                          "PREREGISTRATION_H2_PATCH1_DEV_A.json": pin(os.path.join(OUT, "PREREGISTRATION_H2_PATCH1_DEV_A.json"))},
    "caches_read_only": {n: pin(X.CACHES[n]) for n in CACHES},
    "outputs": {"per_cache": "results/L1_COVPART/covscore_A_<cache>.{json,log} (write-once)", "summary": "results/L1_COVPART/covscore_SUMMARY.{json,log} (write-once)"},
}
S.wj(FP, rec)
print("written", os.path.relpath(FP, X.REPO), sha_file(FP), os.path.getsize(FP), rec["utc"])
