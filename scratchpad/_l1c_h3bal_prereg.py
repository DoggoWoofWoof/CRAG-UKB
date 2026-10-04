"""Write-once pre-registration for BALANCED_H3_PATCH1 (fourteenth ruling, item 3, 2026-09-16): the ONE H3_PATCH1 successor on the balanced
frontier, on the five DEV_A caches.  Written after the identity-only dry run (metaqa; gold-free) and before any coverage number of the new
arm exists.

    python -u scratchpad/_l1c_h3bal_prereg.py <dry_run_log>   -> results/L1_COVPART/PREREGISTRATION_BALANCED_H3_PATCH1.json
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
FP = os.path.join(OUT, "PREREGISTRATION_BALANCED_H3_PATCH1.json")
assert not os.path.exists(FP), "write-once"
assert not any(f.startswith("h3bal") for f in os.listdir(OUT)), "a BALANCED_H3_PATCH1 result exists already"
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
sha_file, pin = L.sha_file, L.pin
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json", "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json", "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}

# ---- what the dry run printed (gold-free), taken from its log verbatim
dry_log = open(sys.argv[1], encoding="utf-8").read()
assert "DRY RUN:" in dry_log and "rc=0" in dry_log, "the dry run did not complete"
dry_lines = [ln for ln in dry_log.splitlines() if any(k in ln for k in ("beam done:", "beams done", "balanced beam hops 1-2 reproduced", "depth-3 beams (gold-free)",
                                                                          "balanced depth-3 patch capacity (gold-free)", "canonical SAFE reproduced", "DRY RUN:"))]
d3 = json.loads(re.search(r"depth-3 beams \(gold-free\) (\{.*\})", dry_log).group(1))
cap = json.loads(re.search(r"balanced depth-3 patch capacity \(gold-free\) (\{.*\})", dry_log).group(1))

pre17 = json.load(open(os.path.join(OUT, "PREREGISTRATION_H3_PATCH1_DEV_A.json"), encoding="utf-8"))
pre21 = json.load(open(os.path.join(OUT, "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json"), encoding="utf-8"))
sum17 = json.load(open(os.path.join(OUT, "h3patch_A_SUMMARY.json"), encoding="utf-8"))
sum21 = json.load(open(os.path.join(OUT, "h2balanced_A_SUMMARY.json"), encoding="utf-8"))
sum25 = json.load(open(os.path.join(OUT, "covscore_SUMMARY.json"), encoding="utf-8"))
r17 = {c: json.load(open(os.path.join(OUT, "h3patch_A_%s.json" % c), encoding="utf-8")) for c in CACHES}
r21 = {c: json.load(open(os.path.join(OUT, "h2balanced_A_%s.json" % c), encoding="utf-8")) for c in CACHES}
why17 = {c: json.load(open(os.path.join(OUT, "h3patch_why_A_%s.json" % c), encoding="utf-8")) for c in ("metaqa", "metaqa_phg")}
for c in CACHES:
    assert r21[c]["code"]["sha256"] == pre21["code"]["this_module"]["sha256"] == sha_file(os.path.join(HERE, "_l1c_h2balanced.py"))
    assert r17[c]["beam_module_sha256"] == sha_file(os.path.join(HERE, "_l1c_microl3.py"))
seen = {}
for c in CACHES:
    a17, a21 = r17[c]["arms"], r21[c]["arms"]
    h3, b2 = a17["H3_PATCH1"], a21["BALANCED_H2_PATCH1"]
    seen[c] = {"n_DEV_A": r17[c]["n_DEV_A"], "L1": a17["L1"]["ALL"], "SAFE": a17["SAFE"]["ALL"], "H2_PATCH1": a17["H2_PATCH1"]["ALL"], "H3_PATCH1": h3["ALL"], "BALANCED_H2_PATCH1": b2["ALL"],
               "H3_PATCH1_vs_H2_PATCH1": h3["ALL_vs_H2_PATCH1"], "BALANCED_H2_vs_H2_PATCH1": b2["ALL_vs_H2_PATCH1"],
               "per_hop": {h: {"H2_PATCH1": a17["H2_PATCH1"]["per_hop"][h]["ALL"], "H3_PATCH1": h3["per_hop"][h]["ALL"], "BALANCED_H2_PATCH1": b2["per_hop"][h]["ALL"],
                               "H3_vs_H2 (gained / lost)": [h3["per_hop"][h]["vs_H2_PATCH1"]["gained"], h3["per_hop"][h]["vs_H2_PATCH1"]["lost"]],
                               "BALANCED_vs_H2 (gained / lost)": [b2["per_hop"][h]["vs_H2_PATCH1"]["gained"], b2["per_hop"][h]["vs_H2_PATCH1"]["lost"]]} for h in h3["per_hop"]},
               "H3_PATCH1_patch": {k: h3[k] for k in ("novel_per_hop_mean", "patch_composition_mean (hop-1 / hop-2 / hop-3 novel / fill from removed block)", "queries_truncated",
                                                      "gold_nodes_gained_vs_SAFE_by_hop (1 / 2 / 3)", "gold_nodes_lost_with_removed_block")},
               "cost_depth3_pinned": r17[c]["cost_depth3"]}
why = {c: {"hop3": {k: why17[c]["groups"]["hop3"][k] for k in ("SAFE_failures", "missing_gold_nodes", "visited_at_hop (1 / 2 / 3)", "scored_but_not_selected_first_at_hop (1 / 2 / 3)",
                                                               "never_scored (not adjacent to any beam node under the frozen admissibility)",
                                                               "failures_where_all_missing_nodes_are_scored_within_depth3", "missing_nodes_per_failure_mean")}} for c in why17}

RULING = (
    "3. Only after fixing H2 compression should you reopen H3 ... If coverage-then-score or BALANCED keeps more correct H2 parents, then run one H3_PATCH1 successor using that "
    "frontier. No wider patch initially. No H3-specific scoring. Same real-edge maths. ... We have over-read DEV_A. So don't use MetaQA/MuSiQue indefinitely to select every next rule.")
CONDITION = ("met by BALANCED_H2_PATCH1 (section 25's diagnostic on SAFE's metaqa hop-2 failures: gold-bearing hop-1 parents with a gold child kept 410 vs 357 pinned of 450; PHG 384 vs "
             "330 of 420; section 21 hop 2 +30 / -5 and +31 / -6 vs H2_PATCH1); the coverage-then-score frontier FAILED (section 25, FAIL_WORSE) so the frontier is the round-robin")

src = open(os.path.join(HERE, "_l1c_h3bal.py"), encoding="utf-8").read()
code_verbatim = src[src.index("def rr_compress"):src.index("def diversity")].rstrip().splitlines()
patch_verbatim = src[src.index("def build_patches"):src.index("patches = {ARM_H2")].rstrip().splitlines()

rec = {
    "RECORD": "PREREGISTRATION",
    "name": "BALANCED_H3_PATCH1",
    "STATUS": "PREREGISTERED_SUCCESSOR_TEST_DEV_A",
    "written_before_any_run": True,
    "written_before": ["any coverage (ALL / ANY) number of BALANCED_H3_PATCH1 on any cache", "any paired comparison of the new arm", "any per-hop number of the new arm",
                       "any gold-bearing diagnostic of the balanced depth-3 beam (hop 3)", "any number of any kind for the new arm on metaqa_phg, squad, squad_phg or musique"],
    "what_was_run_before_this_record": {
        "dry_run": "python -u scratchpad/_l1c_h3bal.py metaqa --dry: the pinned uL3 head exec'd at DEPTH = 3 with section 17's substitutions (asserted equal to the section-17 record); "
                   "the pinned beam with candidate recording == the exec'd pinned depth-3 beam; the balanced depth-3 beam built with the section-21 compression at every hop; its hops 1-2 "
                   "reproduce every deterministic figure of h2balanced_A_metaqa.json beam_diversity_DEV_A; the pinned depth-3 transition counts == h3patch_A_metaqa.json cost_depth3; "
                   "canonical SAFE / BASE == the L1_REPLAY record on DEV_A rows; the gold-free depth-3 beam statistics and the gold-free patch-capacity arithmetic printed; "
                   "no coverage number of the new arm; no record",
        "dry_run_log_lines_verbatim": dry_lines,
        "depth3_beams_gold_free_statistics_metaqa_DEV_A": d3,
        "balanced_depth3_patch_capacity_gold_free_metaqa_DEV_A": cap,
        "coverage_of_the_new_arm": "NEVER computed anywhere before this record",
    },
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ruling_2026-09-16_fourteenth_item_3_verbatim_essentials": RULING,
    "condition_of_the_ruling": CONDITION,
    "question": "Does the section-17 depth-3 arm (the frozen real-edge beam one hop deeper under the identical one-block patch), run on the balanced frontier instead of the pinned "
                "global-order frontier, reach the hop-3 gold that H3_PATCH1 could not -- i.e. is the hop-3 residual of section 17 a frontier effect (the wrong hop-2 parents kept) "
                "rather than a width / capacity effect -- with the patch, budget, beam width, SAFE and fusion unchanged?",
    "frozen_and_reused_verbatim": {
        "beam": "the pinned uL3 beam of _l1c_microl3.py (sha 19774617...): frozen admissibility (out always, in for non-hubs; hub = deg_u + 1 > N / k), frozen transition geometry "
                "S_dst / S_dir / S_mid, frozen RRF fused order (K0 = 60), BEAM = 100, seeds = the frozen top-5 RRF nodes; exec'd with DEPTH = 3 exactly as section 17 (the same five "
                "head substitutions, asserted equal to PREREGISTRATION_H3_PATCH1_DEV_A.json and to h3patch_A_<cache>.json); the pinned depth-3 beam and the section-21 balanced beam "
                "are recomputed and H2_PATCH1 / H3_PATCH1 / BALANCED_H2_PATCH1 asserted against their records before the new arm is evaluated",
        "head_substitutions": pre17["frozen_and_reused_verbatim"]["head_substitutions"],
        "compression": "section 21 verbatim (rr_compress; _l1c_h2balanced.py sha 7aed6b1d...): (parent, target) pairs in the frozen fused order (first occurrence), within-parent rank r, "
                       "sequence sorted by (r, frontier position) = A1 B1 C1 ... A2 B2 C2 ..., the first BEAM distinct unvisited targets; applied at hop 1, hop 2 AND hop 3 -- one rule, "
                       "no hop branch (hops 1-2 of the balanced depth-3 beam ARE the section-21 balanced beam; the depth changes nothing before hop 3)",
        "patch": "section 16 / 17 verbatim (_l1c_h2patch.py sha a1ed9caa...; the depth-3 build_patches of _l1c_h3patch.py): SAFE minus its weakest admitted block + one query-local block "
                 "of the same size = the beam's visited nodes not exposed by the 49 kept blocks in first-visit order (hop 1, hop 2, hop 3 -- the order section 17 pre-registered, which "
                 "preserves the depth-2 patch and admits hop-3 nodes only into the capacity the depth-2 novel nodes leave), topped up from the removed block (ret_rrf position, then node id); "
                 "matched unique-node exposure (asserted)",
        "SAFE": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select, BASE_CFG unchanged (B = 6, M_struct = 64, M_ret = 32, S4, F6); served BASE ranking",
        "constants": "K0 = 60, K_LOCK = 100, P_MAIN = 50, B = 6, BEAM = 100, DEPTH = 3 (section 17's) -- all frozen before; nothing new, nothing fitted, no dataset branch, no weight",
        "code_verbatim (scratchpad/_l1c_h3bal.py, rr_compress + beam_with)": code_verbatim,
        "code_verbatim (scratchpad/_l1c_h3bal.py, build_patches)": patch_verbatim,
    },
    "the_one_change_vs_section_17": "the frontier compression at every hop is the section-21 round-robin instead of the pinned global order; nothing else (no wider patch, no H3-specific "
                                    "scoring, no hop-3 ordering rule, no second candidate for the hop-3 slots)",
    "arms": {
        "reference (recomputed; asserted equal to the frozen section-16 / 17 / 21 records)": ["L1", "SAFE", "H2_PATCH1", "H3_PATCH1", "BALANCED_H2_PATCH1"],
        "new": {"BALANCED_H3_PATCH1": "the ONLY decided-on arm: the section-16 / 17 patch over the balanced depth-3 beam"},
        "not_run": ["a wider patch (WIDE_H2 / WIDE_H3: more than one block replaced) -- needs a separate ruling", "any H3-specific scoring or hop-3 ordering rule",
                    "depth 4", "BALANCED_H3 over the QMAX_F0 ranking (a composition; not authorised)", "any change of the patch, budget, beam width, SAFE or fusion",
                    "a second compression rule of any kind (coverage-then-score is CLOSED by section 25)"],
    },
    "seen_numbers_DEV_A (section 17 and 21 records; the reference arms of this test)": seen,
    "section17_why_diagnostic_read (pinned frontier, hop-3 group)": why,
    "outcomes_read": {"H3_PATCH1": {"verdict": sum17["verdict"], "rule": sum17["rule"]}, "BALANCED_H2_PATCH1": {"verdict": sum21["verdict"], "label": sum21["label"]},
                      "COVSCORE_H2_PATCH1": {"OUTCOME": sum25["OUTCOME"], "frontier_for_item_3": sum25["frontier_for_item_3"]}},
    "predictions_stated_now": {
        "arithmetic (gold-free, from the dry run)": "the balanced depth-3 beam holds %s novel nodes per query (hop 1 / 2 / 3 = %s) against a patch of %s nodes; hop-3 nodes are admitted "
                                                   "only into the capacity the depth-2 novel nodes leave: %s hop-3 novel nodes admitted per query on average, %s cut; %s of 1,002 queries "
                                                   "admit no hop-3 node at all; the depth-3 patch is truncated on %s queries (depth 2: %s)." % (
                                                       cap["novel_total_depth3_mean"], cap["novel_per_hop_mean (1 / 2 / 3)"], cap["patch_size_mean"], cap["hop3_novel_admitted_per_query_mean"],
                                                       cap["hop3_novel_cut_per_query_mean"], cap["queries_admitting_no_hop3_node"], cap["queries_truncated_depth3"], cap["queries_truncated_depth2"]),
        "mechanism": "hops 1-2 of the new arm are BALANCED_H2_PATCH1's beam, so hop-1 / hop-2 coverage can differ from BALANCED_H2_PATCH1 only through fill-position displacement by the "
                     "hop-3 nodes (section 17 saw 0 such flips on metaqa, 2 on musique). Hop 3 differs from H3_PATCH1 through (a) which hop-2 parents are expanded (the balanced hop-2 "
                     "frontier represents ~24 hop-1 parents instead of ~19 and keeps 650 vs 536 missing hop-2 gold nodes on SAFE's hop-2 failures -- the only quantity that is genuinely "
                     "open at depth 3 is which hop-3 candidates that frontier scores) and (b) the round-robin at hop 3 itself (~1 child per hop-2 parent over ~95 parents; the hop-3 "
                     "Jaccard vs the pinned beam is %s on the dry run). Section 17's hop-3 failures miss 16-17 gold nodes each (2,597 over 159 failures; 1,673 never scored within depth 3 "
                     "under the pinned frontier; 33 failures with every missing node scored) and the patch admits ~%s hop-3 nodes per query, so a hop-3 ALL-gold repair needs a failure "
                     "whose missing nodes are few, all reached at depth 3 and all inside that capacity." % (d3["jaccard_kept_vs_pinned_mean (hop1 / hop2 / hop3)"][2], cap["hop3_novel_admitted_per_query_mean"]),
        "direction": "FAIL_NOT_BETTER is the most likely outcome (hop-3 ALL vs BALANCED_H2_PATCH1 within +0 / +3 on each metaqa cache, far from the +gain at p < 0.01 the rule needs); "
                     "FAIL_WORSE is unlikely (hops 1-2 unchanged up to fill displacement; text caches within noise of BALANCED_H2_PATCH1 and H2_PATCH1); PASS_H3 would require the "
                     "balanced hop-2 frontier to expose the hop-3 gold siblings that the pinned frontier never scored AND the one-block capacity to hold them -- the arithmetic above "
                     "makes this improbable. The informative output of this test is the hop-3 reach diagnostic under the balanced frontier (never scored / scored-but-cut / visited), "
                     "which decides whether the hop-3 residual is FRONTIER (fixable by order) or WIDTH / CAPACITY (closed unless the user reopens width).",
    },
    "population": {c: "served %s cache rows; DEV_A = sha1(query_id)[:8] & 1 == 0 exactly as _l1x90_core; hop labels as present in the cache" % c for c in CACHES}
                  | {"DEV_B": "never read", "TEST": "never read", "WebQSP / 2Wiki / HotpotQA": "not used"},
    "statistics": {"test": "exact two-sided McNemar on paired DEV_A 0/1 ALL vectors (_l1x90_core.mcnemar): gained / lost / p",
                   "alpha_gain": 0.01, "alpha_regression_and_safety": 0.05,
                   "labels": {"GAIN": "gained > lost and p < 0.01", "LOSS": "lost > gained and p < 0.01", "NEUTRAL": "otherwise"},
                   "decision_comparisons": "BALANCED_H3_PATCH1 vs BALANCED_H2_PATCH1 (hop 3 and hop 1 / hop 2 on the two metaqa caches; ALL on the three text caches) and, for text safety "
                                           "only, vs H2_PATCH1 and vs SAFE; everything else (vs H3_PATCH1 = the frontier effect at depth 3, vs L1, per hop on text, flips, beam statistics, "
                                           "the depth-3 reach diagnostic) is descriptive"},
    "decision_rule_pre_registered": {
        "metaqa_required": "hop-3 ALL of BALANCED_H3_PATCH1 vs BALANCED_H2_PATCH1 with gained > lost and p < 0.01 on BOTH metaqa and metaqa_phg",
        "no_regression_required": "hop-1 and hop-2 vs BALANCED_H2_PATCH1 on both metaqa caches: not (lost > gained and p < 0.05)",
        "text_safety_required": "squad, squad_phg, musique: ALL vs BALANCED_H2_PATCH1, vs H2_PATCH1 and vs SAFE: not (lost > gained and p < 0.05)",
        "OUTCOME": {"PASS_H3": "metaqa_required AND no_regression_required AND text_safety_required",
                    "FAIL_WORSE": "not PASS and (a LOSS vs BALANCED_H2_PATCH1 on a metaqa cache at hop 3 or on ALL (p < 0.01) OR a hop-1 / hop-2 regression OR a text-safety violation)",
                    "FAIL_NOT_BETTER": "otherwise (neither the required gain nor a loss)"},
        "what_follows": {
            "PASS_H3": "BALANCED_H3_PATCH1 = the current best experimental successor (MECHANISM_CONFIRMED_ON_DEV_A, pre-registered; not promoted; NOT transfer-confirmed); "
                       "BALANCED_H2_PATCH1 keeps its section-21 record; H2_PATCH1 stays the frozen WebQSP transfer reference exactly as section 16; the depth-3 arm is a candidate for the "
                       "multi-hop text transfer populations only after a ruling; no wider patch, no depth 4 without a ruling",
            "FAIL_WORSE": "depth stays CLOSED also under the balanced frontier (the tenth ruling's 'H3 frozen' stands); BALANCED_H2_PATCH1 keeps the successor role; the diagnostic's "
                          "cause (frontier / width / capacity) is reported; a wider patch needs a separate ruling and is not run",
            "FAIL_NOT_BETTER": "depth stays CLOSED also under the balanced frontier (the tenth ruling's 'H3 frozen' stands); BALANCED_H2_PATCH1 keeps the successor role; the diagnostic's "
                               "cause (frontier / width / capacity) is reported; a wider patch needs a separate ruling and is not run",
        },
        "no_promotion": "nothing here changes the served L1 or any frozen record; sections 16-25 are unchanged; the WebQSP primary transfer remains frozen H2_PATCH1 exactly as section 16",
    },
    "also_recorded (descriptive, never decided on)": [
        "BALANCED_H3_PATCH1 vs H3_PATCH1 (the frontier effect at depth 3: hop 3, ALL, ANY, hop-2 + hop-3 pooled), vs H2_PATCH1, vs SAFE, vs L1",
        "flips vs BALANCED_H2_PATCH1, vs H3_PATCH1 and vs H2_PATCH1 with their cause (entering via hop-3 / hop-1-2 positions; newly visited / visited by both beams but cut by the reference "
        "patch; no longer visited / beyond the patch capacity; fill under the reference)",
        "the depth-3 reach diagnostic on SAFE failures per hop group (metaqa) / all (text) for the pinned and the balanced depth-3 beams: missing gold visited (first at hop 1 / 2 / 3), "
        "scored-but-cut (first at hop 1 / 2 / 3), never scored within depth 3, gold-bearing hop-1 / hop-2 parents and how many keep a gold child, failures with every missing node scored / "
        "visited, covered / not covered (beyond capacity / dropped with the removed block)",
        "gold-free depth-3 beam statistics: beam sizes, truncation, rounds used at hop 3, Jaccard vs pinned, distinct parents, largest-parent share, transition counts, wall clock",
        "the gold-free patch-capacity arithmetic (novel per hop, hop-3 admitted / cut, queries admitting no hop-3 node)",
    ],
    "forbidden": [
        "reading DEV_B or TEST; using WebQSP, 2Wiki or HotpotQA",
        "any wider patch, any H3-specific scoring, any hop-3 ordering rule, any second compression rule, any k_local / lambda / weight, any dataset branch, any change of the patch, budget, "
        "beam width, depth, SAFE, F6, RRF or hub definition",
        "editing any CONTRACT_FILE or pinned module; editing the two new modules after this record (their shas are pinned below and asserted at run time)",
        "signalling the foreign process; touching the WebQSP watcher; writing under data/",
        "tuning after seeing numbers; changing a label, clause or threshold after a number is seen; re-running a cache after seeing its number",
    ],
    "code": {
        "new_modules": {n: pin(os.path.join(HERE, n)) for n in ("_l1c_h3bal.py", "_l1c_h3bal_summary.py")},
        "usage": ["python -u scratchpad/_l1c_h3bal.py <cache> for metaqa, metaqa_phg, squad, squad_phg, musique (chain: scratchpad/_l1c_h3bal_chain.sh)", "python -u scratchpad/_l1c_h3bal_summary.py"],
        "imports_unchanged": {**{n: pin(os.path.join(HERE, n)) for n in ("_l1g_core.py", "_l1s_core.py", "_l1x90_core.py", "_ta_prepartition.py", "_l1ps_router.py", "_l1kb_core.py",
                                                                        "_l1c_microl3.py", "_l1c_h2patch.py", "_l1c_h2balanced.py", "_l1c_h3patch.py")},
                              **{"src/l1_canonical/%s.py" % n: pin(os.path.join(X.REPO, "src", "l1_canonical", n + ".py")) for n in ("adapter", "hypergraph")}},
        "pins_asserted_at_run_time": "the runner asserts the pinned module shas, this record's new-module / import / cache / record shas and write-once outputs; the new arm is computed only "
                                     "after the reference arms are reproduced against the section-16 / 17 / 21 records",
    },
    "records_read_only": {**{"h2patch_A_%s.json" % c: pin(os.path.join(OUT, "h2patch_A_%s.json" % c)) for c in CACHES},
                          **{"h3patch_A_%s.json" % c: pin(os.path.join(OUT, "h3patch_A_%s.json" % c)) for c in CACHES},
                          **{"h2balanced_A_%s.json" % c: pin(os.path.join(OUT, "h2balanced_A_%s.json" % c)) for c in CACHES},
                          **{"h3patch_why_A_%s.json" % c: pin(os.path.join(OUT, "h3patch_why_A_%s.json" % c)) for c in ("metaqa", "metaqa_phg")},
                          **{os.path.basename(SAFE_RECORD[c]): pin(os.path.join(X.REPO, SAFE_RECORD[c])) for c in CACHES},
                          "h3patch_A_SUMMARY.json": pin(os.path.join(OUT, "h3patch_A_SUMMARY.json")), "h2balanced_A_SUMMARY.json": pin(os.path.join(OUT, "h2balanced_A_SUMMARY.json")),
                          "h2patch_A_SUMMARY.json": pin(os.path.join(OUT, "h2patch_A_SUMMARY.json")), "covscore_SUMMARY.json": pin(os.path.join(OUT, "covscore_SUMMARY.json")),
                          "PREREGISTRATION_H3_PATCH1_DEV_A.json": pin(os.path.join(OUT, "PREREGISTRATION_H3_PATCH1_DEV_A.json")),
                          "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json": pin(os.path.join(OUT, "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json")),
                          "PREREGISTRATION_H2_PATCH1_DEV_A.json": pin(os.path.join(OUT, "PREREGISTRATION_H2_PATCH1_DEV_A.json")),
                          "PREREGISTRATION_COVSCORE_H2_PATCH1.json": pin(os.path.join(OUT, "PREREGISTRATION_COVSCORE_H2_PATCH1.json"))},
    "caches_read_only": {n: pin(X.CACHES[n]) for n in CACHES},
    "outputs": {"per_cache": "results/L1_COVPART/h3bal_A_<cache>.{json,log} (write-once)", "summary": "results/L1_COVPART/h3bal_SUMMARY.{json,log} (write-once)"},
}
S.wj(FP, rec)
print("written", os.path.relpath(FP, X.REPO), sha_file(FP), os.path.getsize(FP), rec["utc"])
