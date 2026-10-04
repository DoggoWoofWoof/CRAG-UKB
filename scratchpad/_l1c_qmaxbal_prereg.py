"""Write-once pre-registration for QMAX_BALANCED_H2_PATCH1 (fourteenth ruling, item 1, 2026-09-16): the ONE composition
QMAX_F0 block ranking -> SAFE -> BALANCED_H2_PATCH1 on the five DEV_A caches.  Written after the identity-only dry run and before any
coverage number of any composed arm exists.

    python -u scratchpad/_l1c_qmaxbal_prereg.py   -> results/L1_COVPART/PREREGISTRATION_QMAX_BALANCED_H2_PATCH1.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402
import _l1g_core  # noqa: E402,F401  (registers the squad_phg cache)
import _l1c_br2_lib as L  # noqa: E402  (sha_file / pin helpers only)

X = S.X
OUT = os.path.join(X.REPO, "results", "L1_COVPART")
GEOM = os.path.join(X.REPO, "results", "L1_GEOM")
FP = os.path.join(OUT, "PREREGISTRATION_QMAX_BALANCED_H2_PATCH1.json")
assert not os.path.exists(FP), "write-once"
assert not any(f.startswith("qmaxbal") for f in os.listdir(OUT)), "a QMAX_BALANCED_H2_PATCH1 result exists already"
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
sha_file, pin = L.sha_file, L.pin
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json", "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json", "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}

pre_geom = json.load(open(os.path.join(GEOM, "PREREGISTRATION_DEV_B.json"), encoding="utf-8"))
conf_geom = json.load(open(os.path.join(GEOM, "DEV_B_CONFIRMATION.json"), encoding="utf-8"))
pre21 = json.load(open(os.path.join(OUT, "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json"), encoding="utf-8"))
sum21 = json.load(open(os.path.join(OUT, "h2balanced_A_SUMMARY.json"), encoding="utf-8"))
sum16 = json.load(open(os.path.join(OUT, "h2patch_A_SUMMARY.json"), encoding="utf-8"))
r21 = {c: json.load(open(os.path.join(OUT, "h2balanced_A_%s.json" % c), encoding="utf-8")) for c in CACHES}
for c in CACHES:
    assert r21[c]["code"]["sha256"] == pre21["code"]["this_module"]["sha256"] == sha_file(os.path.join(HERE, "_l1c_h2balanced.py"))

seen = {}
for c in CACHES:
    a = r21[c]["arms"]
    q = pre_geom["recorded_DEV_A"][c]["AGGREGATION"]
    b = conf_geom["caches"][c]["arms"]["AGGREGATION"]
    seen[c] = {"n_DEV_A": r21[c]["n_DEV_A"],
               "reference_chain_DEV_A (h2balanced_A_%s.json)" % c: {
                   "L1": a["L1"]["ALL"], "SAFE": a["SAFE"]["ALL"], "H2_PATCH1": a["H2_PATCH1"]["ALL"], "BALANCED_H2_PATCH1": a["BALANCED_H2_PATCH1"]["ALL"],
                   "BALANCED_vs_SAFE": a["BALANCED_H2_PATCH1"]["ALL_vs_SAFE"], "BALANCED_vs_H2_PATCH1": a["BALANCED_H2_PATCH1"]["ALL_vs_H2_PATCH1"],
                   "H2_PATCH1_vs_SAFE": a["H2_PATCH1"]["ALL_vs_SAFE"],
                   "per_hop_BALANCED": {h: e["ALL"] for h, e in a["BALANCED_H2_PATCH1"]["per_hop"].items()}, "per_hop_L1": {h: e["ALL"] for h, e in a["L1"]["per_hop"].items()}},
               "QMAX_F0_DEV_A (results/L1_GEOM/PREREGISTRATION_DEV_B.json recorded_DEV_A AGGREGATION; vs BASE)": q,
               "QMAX_F0_DEV_B_and_full_dev_CONTEXT_ONLY (results/L1_GEOM/DEV_B_CONFIRMATION.json; DEV_B read twice already; NOT used here)": {"B": b["B"]["ALL"], "B_gained_lost": [b["B"]["gained"], b["B"]["lost"]], "full": b["full"]["ALL"]}}

RULING = (
    "there are three real frontiers left. I would stop inventing new partitioners and focus on these. ... 1. Combine the two mechanisms that solve different "
    "problems ... the next universal candidate ... `QMAX_F0 ranking -> SAFE -> BALANCED_H2_PATCH1`. Same algorithm everywhere. No training. No relation "
    "dictionaries. No embedding modification. ... Do one composition, not a factorial of D2d/QMAX/interleave/etc. I'd use the cheaper `QMAX_F0` formulation "
    "because it retained most of D2d's MuSiQue gain and does not need the five directional matrix scores. The scientific question is simply: Does stronger "
    "semantic node->block aggregation and structurally balanced dynamic patching stack? If yes, that's probably your real universal L1+ candidate. "
    "... 2. [coverage-then-score frontier: a separate pre-registration] ... 3. [H3 successor only after H2 compression is fixed] ... 4. [transfer: WebQSP KB, "
    "2Wiki / Hotpot multi-hop text, SQuAD non-regression] ... MuSiQue specifically needs a different kind of help ... better scoring of evidence already present, "
    "not more structural reach ... We have over-read DEV_A. So don't use MetaQA/MuSiQue indefinitely to select every next rule. ... Everything else we've "
    "explored -- new hard partitions, PATH2/3, vertex-cut variants, relation offsets, mass-normalized voting, triplet geometry, latent beams -- is now lower "
    "priority or closed. Target: strong semantic aggregation + safe static compression + diversity-preserving tiny graph refinement + query-local node patch; "
    "no training, no dataset-specific logic, no embedding modification.")

rec = {
    "RECORD": "PREREGISTRATION",
    "name": "QMAX_BALANCED_H2_PATCH1",
    "STATUS": "PREREGISTERED_COMPOSITION_TEST_DEV_A",
    "written_before_any_run": True,
    "written_before": ["any coverage (ALL / ANY) number of QMAX_SAFE or QMAX_BALANCED_H2_PATCH1 on any cache", "any paired comparison of a composed arm", "any per-hop number of a composed arm",
                       "any number of any kind for a composed arm on metaqa_phg, squad, squad_phg or musique"],
    "what_was_run_before_this_record": {
        "dry_run": "python -u scratchpad/_l1c_qmaxbal.py metaqa --dry (55 s): asserted F0(Cd, Cs) == the served base_rank (first 200 blocks, every row); QMAX_F0 on DEV_A == the L1_GEOM record "
                   "(0.6327, +3/-10 vs BASE 0.6397; hops 0.911 / 0.4797 / 0.5181); canonical SAFE == the L1_REPLAY record (0.6407 / 0.6397); the pinned beam with candidate recording == the exec'd "
                   "pinned beam; the balanced beam reproduces every deterministic figure of h2balanced_A_metaqa.json; |QMAX_SAFE| = 50 on every DEV_A row; challengers identical under both rankings. "
                   "Gold-free identities printed: QMAX P50 shares 36.45 of 50 blocks with the BASE P50 (mean); QMAX_SAFE shares 38.68 of 50 with SAFE, is identical to SAFE for 0 of 1,002 queries, "
                   "keeps the same weakest admitted block for 57, admitted-six overlap 2.19; challengers admitted from outside the chain's own P50: 4.59 (BASE) / 4.92 (QMAX) of 6; nominal exposure "
                   "5,023.0 / 5,023.1 / 5,028.3 / 5,027.3 nodes (L1 / SAFE / QMAX_F0 / QMAX_SAFE). No coverage number of any composed arm; no record.",
        "coverage_of_the_composed_arms": "NEVER computed anywhere before this record",
    },
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ruling_2026-09-16_fourteenth_verbatim_essentials": RULING,
    "question": "Does stronger semantic node -> block aggregation (QMAX_F0: the exhaustive dense block max as a third channel of the frozen RRF; DEV_B-confirmed as the "
                "aggregation component of D2d on musique, neutral on metaqa / squad) and structurally balanced dynamic patching (BALANCED_H2_PATCH1: SAFE minus its weakest admitted "
                "block + one query-local block of the balanced beam's unexposed visited nodes; DEV_A post-hoc confirmed on metaqa, text-safe) STACK -- i.e. does ONE algorithm, "
                "QMAX_F0 ranking -> SAFE -> BALANCED_H2_PATCH1, carry both regime gains (musique's ranking gain and metaqa's hop-2 structural gain) without conflict?",
    "the_one_composition": {
        "ranking": "QMAX_F0 = _l1g_core.F0 (the frozen RRF, K0 = 60, dense tie-break) over [Cd, Cs, Cmax]; Cd / Cs = _l1g_core.block_channel of the served dense / SPLADE top-100 ids "
                   "through the legacy own + directed-out-neighbour table (= the served BASE's two channels); Cmax = _l1g_candidate.dense_blockmax_rank: score(B) = max_{v in B} q . e_v "
                   "over all actual nodes (canonical fp16 embeddings -> fp32, unit query), stable descending order; pinned code path, asserted to reproduce the recorded DEV_A numbers",
        "entry_into_SAFE": "the first TOPP = 200 blocks of the QMAX_F0 ranking replace base_rank in the replay-cache view read by _l1ps_router.build_cache and _l1kb_core.contexts: "
                           "protected = its first 44 blocks, boundary = its blocks 45-50, canonical position cpos = its positions; the structural challengers (s_node = the frozen "
                           "seed-based directional expansion, seeds = the frozen top-5 RRF nodes; independent of the block ranking -- asserted identical under both rankings) and the "
                           "retrieval challengers (ret_rrf) are unchanged; F6 admission (RRF over cpos / spos / rpos, B = 6, ties by cpos then block id) unchanged; QMAX_SAFE = protected "
                           "+ the six admitted; weakest = the sixth admitted",
        "patch": "section 16 verbatim on QMAX_SAFE: minus the weakest admitted block, plus one query-local block of the same size filled with the balanced beam's visited nodes that the 49 "
                 "kept blocks do not expose (first-visit order: hop 1 then hop 2), topped up from the removed block (ret_rrf position, then node id); matched unique-node exposure to QMAX_SAFE",
        "beam": "the pinned uL3 beam (frozen admissibility, transition geometry, RRF fused order, BEAM = 100, DEPTH = 2, seeds = frozen top-5 RRF nodes) with the section-21 balanced "
                "compression (per-parent rank round-robin over the frozen fused order); identical for the reference and the composed chain (the beam never reads the block ranking); "
                "the section-21 beam figures are asserted reproduced",
        "constants": "K0 = 60, K_LOCK = 100, P_MAIN = 50, TOPP = 200, B = 6, M_struct = 64, M_ret = 32, S4, F6, BEAM = 100, DEPTH = 2 -- all frozen before; nothing new, nothing fitted, "
                     "no dataset branch, no weight; the same algorithm on every cache",
        "cost_note": "Cmax needs one exhaustive query x node similarity pass (nq x N fp32: 1,998 x 43,234 on metaqa, 2,000 x 117,534 on musique); the L1_GEOM lane showed the pooled "
                     "variant (dense u SPLADE top-1000) indistinguishable from the exact one -- the exact form is used here because it is the recorded, pinned arm",
    },
    "arms": {
        "reference_chain (recomputed; asserted equal to the frozen section-16 / 21 records)": ["L1", "SAFE", "H2_PATCH1", "BALANCED_H2_PATCH1"],
        "composed_chain": {"QMAX_F0": "P50 of the QMAX_F0 ranking (the L1_GEOM AGGREGATION arm; asserted equal to its record); attribution stage, parent 1",
                           "QMAX_SAFE": "the frozen F6 swap over the QMAX_F0 ranking; attribution stage",
                           "QMAX_BALANCED_H2_PATCH1": "PRIMARY -- the composition; the ONLY decided-on arm"},
        "not_run": ["QMAX_H2_PATCH1 (pinned-beam patch over QMAX_SAFE) -- would make the test a 2 x 2 factorial", "D2d / directional compositions", "interleave fusions", "any re-weighting",
                    "any change of the compression rule inside this test (item 2 of the ruling is a separate pre-registration)"],
        "parents": {"parent 1": "QMAX_F0 (P50; the aggregation alone)", "parent 2": "BALANCED_H2_PATCH1 (the structural chain alone, over the served BASE ranking)"},
    },
    "seen_numbers_DEV_A": seen,
    "section21_and_16_outcomes_read": {"BALANCED_H2_PATCH1": {"verdict": sum21["verdict"], "label": sum21["label"]}, "H2_PATCH1": {"verdict": sum16.get("verdict"), "label": sum16.get("label")}},
    "predictions_stated_now": {
        "musique": "QMAX_F0 0.7319 already exceeds BALANCED 0.7269 and H2_PATCH1 0.7329; the reference chain adds +5.2 (SAFE) + 1.5 (patch) on top of BASE 0.6596. If the SAFE swap and the "
                   "patch recover queries other than the ones the block max already ranks, the composite lands above 0.75; if they recover the same second-hop paragraphs, it stays near 0.73-0.74 "
                   "(NOT_STACKED on this cache). Prediction: partial redundancy -- composite 0.74-0.77, GAIN vs BALANCED, NEUTRAL-to-GAIN vs QMAX_F0",
        "metaqa / metaqa_phg": "QMAX_F0 is neutral vs BASE (0.6327 / 0.6747 vs 0.6397 / 0.6727) and shares 36.5 of 50 P50 blocks with it; the patch mechanism does not read the ranking, so the "
                               "composite should land within about +/-1 pt of BALANCED (0.7764 / 0.7984): GAIN vs QMAX_F0 (~ +140 / -5), NEUTRAL vs BALANCED. A LOSS vs BALANCED would mean "
                               "the QMAX ranking's protected set drops hop-2 blocks that SAFE kept (a CONFLICT)",
        "squad / squad_phg": "ceiling (0.99); composite NEUTRAL vs both parents",
    },
    "population": {c: "served %s cache rows; DEV_A = sha1(query_id)[:8] & 1 == 0 exactly as _l1x90_core; hop labels as present in the cache" % c for c in CACHES}
                  | {"DEV_B": "never read (read twice by earlier lanes; the QMAX_F0 DEV_B numbers above are context only)", "TEST": "never read", "WebQSP / 2Wiki / HotpotQA": "not used"},
    "statistics": {"test": "exact two-sided McNemar on paired DEV_A 0/1 ALL vectors (_l1x90_core.mcnemar): gained / lost / p", "alpha": 0.01,
                   "labels": {"GAIN": "gained > lost and p < 0.01", "LOSS": "lost > gained and p < 0.01", "NEUTRAL": "otherwise"},
                   "decision_comparisons": "ten in total: PRIMARY vs BALANCED_H2_PATCH1 and PRIMARY vs QMAX_F0 on each of the five caches; everything else (vs H2_PATCH1 / SAFE / L1 / "
                                           "QMAX_SAFE, stage increments, per hop, flips, exposure) is descriptive"},
    "decision_rule_pre_registered": {
        "per_cache_cell": "CONFLICT if PRIMARY is LOSS vs either parent; STACKS if GAIN vs both; CARRIES if GAIN vs one and NEUTRAL vs the other; NEUTRAL otherwise",
        "(a) no_CONFLICT": "PRIMARY vs BALANCED_H2_PATCH1 != LOSS AND PRIMARY vs QMAX_F0 != LOSS on ALL five caches (the universal-candidate condition: one algorithm, never worse than either parent)",
        "(b) musique_GAIN_over_BALANCED": "PRIMARY vs BALANCED_H2_PATCH1 = GAIN on musique (the aggregation's ranking gain shows through the structural chain)",
        "(c) metaqa_GAIN_over_QMAX": "PRIMARY vs QMAX_F0 = GAIN on BOTH metaqa and metaqa_phg (the structural hop-2 gain shows through the aggregation ranking)",
        "OUTCOME": {"PASS_STACKS": "(a) AND (b) AND (c)", "FAIL_CONFLICT": "not (a)", "FAIL_NOT_STACKED": "(a) but not ((b) AND (c))"},
        "secondary_flags (reported, never decided on)": {
            "ADDITIVE_ON_MUSIQUE": "PRIMARY vs QMAX_F0 = GAIN on musique (SAFE + patch add on top of the aggregation)",
            "ADDITIVE_ON_METAQA": "PRIMARY vs BALANCED_H2_PATCH1 = GAIN on both metaqa caches (the aggregation adds on top of the structural chain)",
            "SQUAD_NON_REGRESSION": "no LOSS vs either parent on squad / squad_phg (already inside (a); reported for the ceiling population)"},
        "what_follows": {
            "PASS_STACKS": "QMAX_BALANCED_H2_PATCH1 = the universal L1+ candidate of the fourteenth ruling: status MECHANISM_CONFIRMED_ON_DEV_A (composition; not promoted; NOT transfer-confirmed). "
                           "Roles: H2_PATCH1 stays the frozen WebQSP transfer reference (section 16, unchanged); the composite is the candidate for the multi-hop text transfer populations "
                           "(2Wiki / HotpotQA) when their substrates exist and a resource ruling allows it. Item 2 (coverage-then-score frontier) is pre-registered separately as a successor "
                           "of the compression component; the composite's compression stays the section-21 rule until item 2 earns its own verdict.",
            "FAIL_CONFLICT": "the two mechanisms interfere (the QMAX ranking's protected set or admission drops blocks the structural chain needs, or the patch destroys the aggregation's "
                             "ranking gain): the composition is recorded as NOT a candidate; the parents keep their frozen roles; the flip attribution says which stage conflicts; no re-weighting, "
                             "no second composition without a new ruling; item 2 proceeds on the reference chain",
            "FAIL_NOT_STACKED": "no conflict, but a required GAIN is missing (typically: the SAFE swap / patch recover the same musique queries the block max already ranks -- redundancy): the "
                                "composite is recorded as 'carries at most one regime gain'; the parents keep their frozen roles; the ruling's MuSiQue diagnosis (ranking / aggregation, not "
                                "reach) is then confirmed from the other side; item 2 proceeds on the reference chain",
        },
        "no_promotion": "nothing here changes the served L1 or any frozen record; sections 16-23 are unchanged; the WebQSP primary transfer remains frozen H2_PATCH1 exactly as section 16",
    },
    "forbidden": [
        "reading DEV_B or TEST; using WebQSP, 2Wiki or HotpotQA",
        "any second composition (D2d, interleave, re-weighted RRF, QMAX_H2_PATCH1, BALANCED over any other ranking) inside this test",
        "changing the compression rule, the patch rule, the beam, the F6 admission, TOPP, K0 / K_LOCK / P_MAIN, B, M_struct, M_ret, the RRF or the hub definition",
        "editing any CONTRACT_FILE or pinned module; editing the two new modules after this record (their shas are pinned below and asserted at run time)",
        "signalling the foreign process; touching the WebQSP watcher; writing under data/",
        "tuning after seeing numbers; changing a label, cell, clause or threshold after a number is seen; re-running a cache after seeing its number",
    ],
    "code": {
        "new_modules": {n: pin(os.path.join(HERE, n)) for n in ("_l1c_qmaxbal.py", "_l1c_qmaxbal_summary.py")},
        "usage": ["python -u scratchpad/_l1c_qmaxbal.py <cache> for metaqa, metaqa_phg, squad, squad_phg, musique (chain: scratchpad/_l1c_qmaxbal_chain.sh)", "python -u scratchpad/_l1c_qmaxbal_summary.py"],
        "imports_unchanged": {**{n: pin(os.path.join(HERE, n)) for n in ("_l1g_candidate.py", "_l1g_core.py", "_l1s_core.py", "_l1x90_core.py", "_ta_prepartition.py", "_l1ps_router.py", "_l1kb_core.py",
                                                                        "_l1c_microl3.py", "_l1c_h2patch.py", "_l1c_h2balanced.py")},
                              **{"src/l1_canonical/%s.py" % n: pin(os.path.join(X.REPO, "src", "l1_canonical", n + ".py")) for n in ("adapter", "hypergraph")}},
        "pins_asserted_at_run_time": "the runner asserts the pinned module shas, the L1_GEOM pre-registration's code and cache shas, this record's new-module / import / cache / record shas, "
                                     "and write-once outputs; the composed arms are computed only after the reference chain and QMAX_F0 are reproduced",
    },
    "records_read_only": {**{"h2patch_A_%s.json" % c: pin(os.path.join(OUT, "h2patch_A_%s.json" % c)) for c in CACHES},
                          **{"h2balanced_A_%s.json" % c: pin(os.path.join(OUT, "h2balanced_A_%s.json" % c)) for c in CACHES},
                          **{os.path.basename(SAFE_RECORD[c]): pin(os.path.join(X.REPO, SAFE_RECORD[c])) for c in CACHES},
                          "h2balanced_A_SUMMARY.json": pin(os.path.join(OUT, "h2balanced_A_SUMMARY.json")), "h2patch_A_SUMMARY.json": pin(os.path.join(OUT, "h2patch_A_SUMMARY.json")),
                          "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json": pin(os.path.join(OUT, "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json")),
                          "PREREGISTRATION_H2_PATCH1_DEV_A.json": pin(os.path.join(OUT, "PREREGISTRATION_H2_PATCH1_DEV_A.json")),
                          "L1_GEOM/PREREGISTRATION_DEV_B.json": pin(os.path.join(GEOM, "PREREGISTRATION_DEV_B.json")), "L1_GEOM/DEV_B_CONFIRMATION.json": pin(os.path.join(GEOM, "DEV_B_CONFIRMATION.json"))},
    "caches_read_only": {n: pin(X.CACHES[n]) for n in CACHES},
    "outputs": {"per_cache": "results/L1_COVPART/qmaxbal_A_<cache>.{json,log} (write-once)", "summary": "results/L1_COVPART/qmaxbal_SUMMARY.{json,log} (write-once)"},
}
S.wj(FP, rec)
print("written", os.path.relpath(FP, X.REPO), sha_file(FP), os.path.getsize(FP), rec["utc"])
