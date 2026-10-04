"""Write-once pre-registration for R2_LEGACY_ROUTER (thirteenth ruling, 2026-09-16): the frozen section-22 bounded-overlap substrate
(memberships unchanged) served through the canonical frozen router (own + directed-out-neighbour vote spread), same P50 / same RRF,
gated against each dataset's canonical hard BASE on DEV_A.  Written before any ranking or coverage number of this arm exists.

    python -u scratchpad/_l1c_r2router_prereg.py   -> results/L1_COVPART/PREREGISTRATION_R2_LEGACY_ROUTER.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402
import _l1g_core  # noqa: E402,F401  (registers the squad_phg cache)
import _l1c_vcut_lib as V  # noqa: E402
import _l1c_br2_lib as L  # noqa: E402

X = S.X
OUT = os.path.join(X.REPO, "results", "L1_COVPART")
PDIR = os.path.join(OUT, "parts")
FP = os.path.join(OUT, "PREREGISTRATION_R2_LEGACY_ROUTER.json")
assert not os.path.exists(FP), "write-once"
DS = ("metaqa", "squad", "musique")
assert not any(f.startswith("r2router") for f in os.listdir(OUT)), "an R2_LEGACY_ROUTER result exists already"
sha_file, pin = L.sha_file, L.pin

pre22 = json.load(open(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json"), encoding="utf-8"))
why = json.load(open(os.path.join(OUT, "br2_why_A_metaqa.json"), encoding="utf-8"))
sum22 = json.load(open(os.path.join(OUT, "br2_SUMMARY.json"), encoding="utf-8"))

graphs, substrates = {}, {}
for ds in DS:
    Gr = V.Graph(ds)
    g22 = pre22["graphs_frozen"][ds]
    assert Gr.keys_sha == g22["struct_keys_sha256"] and int(Gr.N) == g22["N"] and int(Gr.C) == 100 and L.k_of(Gr) == g22["k_R2"]
    graphs[ds] = {"N": int(Gr.N), "struct_edges": int(Gr.E), "k_f": int(Gr.k_f), "C": int(Gr.C), "k_R2": L.k_of(Gr), "hubs": int(Gr.hub.sum()), "degree_0": int((Gr.deg == 0).sum()),
                  "mean_struct_degree": round(float(Gr.deg.mean()), 2), "struct_keys_sha256": Gr.keys_sha}
    fp = L.r2_path(ds, Gr)
    rec = json.load(open(fp[:-4] + ".json", encoding="utf-8"))
    assert rec["output"]["sha256"] == sha_file(fp) and rec["representation"]["k"] == L.k_of(Gr)
    substrates[ds] = {**pin(fp), "record": pin(fp[:-4] + ".json"), "k": L.k_of(Gr), "RF": rec["representation"]["RF"], "read_only": "memberships (home, alt) read verbatim; nothing rebuilt, no alternate re-assigned"}
    del Gr

seen_router = why["tables"]["R2_k864 + legacy out-neighbour spread (REACH ONLY; not a cell; the natural follow-up, not evaluated)"]
seen_control = why["tables"]["OWNERS_ONLY_k864 + legacy out-neighbour spread (REACH ONLY; not a cell)"]
seen_hard = why["tables"]["HARD_LEGACY (served BASE: own block + directed out-neighbour blocks; k 432)"]
seen_r2 = why["tables"]["R2_k864 (this ruling: home + one alternate; votes hub -> home only; served through both blocks)"]
seen_v1 = why["tables"]["V1_k1004 (section 19 vertex-cut: votes hub -> home only; served through ALL incident-edge blocks)"]

RULING = (
    "Yes -- one bounded-substrate + frozen 1-hop-routing experiment is worth running. BOUNDED_VERTEXCUT_R2 failed for a very specific reason: you removed the "
    "mechanism that gives hub answers enough routing reach. The post-hoc reach diagnostic is strong enough to justify exactly one new preregistered arm: R2 "
    "geometry / memberships stay frozen + the canonical frozen own + directed-out-neighbour vote spread + same P50 / same RRF. Call it something like "
    "R2_LEGACY_ROUTER. The hypothesis is not 'R2 itself is better.' It is: Can bounded overlap reduce replication/exposure while the existing universal router "
    "supplies the hub reach that R2 cannot encode with lambda <= 2? That is scientifically clean because partitioning and routing are separate mechanisms. The "
    "important comparisons should be: hard canonical BASE vs R2_LEGACY_ROUTER on MetaQA, SQuAD and MuSiQue, with: ALL-gold@P50, actual unique-node exposure, "
    "matched-exposure secondary, hub-gold reach, non-hub reach, votes/hit. The gate should require no significant text regression and a significant MetaQA gain or "
    "exposure-efficiency improvement. Run no R3/R4 or alternate routing if it fails. The post-hoc fact that R2 + legacy spread has reach 0.807, above both hard "
    "0.713 and V1 0.801, is enough to justify that single test. It does not justify claiming it will improve ALL-gold ranking. ... Do not replace frozen H2_PATCH1 "
    "with BALANCED_H2_PATCH1 for the WebQSP primary transfer. ... WEBQSP PRIMARY: frozen H2_PATCH1 exactly as section 16; BALANCED_H2_PATCH1 remains: LEADING "
    "SUCCESSOR / MECHANISM_CONFIRMED_ON_DEV_A_POSTHOC / NOT YET TRANSFER-CONFIRMED. ... I would therefore freeze these roles now: H2_PATCH1 = transfer reference; "
    "BALANCED_H2_PATCH1 = current best experimental successor; H3_PATCH1 = closed; depth not current bottleneck; BOUNDED_VERTEXCUT_R2 = partition-only hypothesis "
    "rejected; R2 + canonical router = one justified new partition/routing hypothesis. And crucially, don't combine BALANCED + R2-router yet. We finally have "
    "individually interpretable mechanisms; keep them separate until each earns transfer evidence.")

rec = {
    "RECORD": "PREREGISTRATION",
    "name": "R2_LEGACY_ROUTER",
    "STATUS": "POSTHOC_PARTITION_ROUTING_MECHANISM_TEST_DEV_A",
    "written_before_any_run": True,
    "written_before": ["any fused ranking through the router table on any dataset", "any coverage (ALL / ANY) number of this arm", "any exposure or matched-budget number of this arm",
                       "any number of any kind for this arm on squad or musique"],
    "what_was_run_before_this_record": {
        "seen_metaqa_reach_only (br2_why_A_metaqa.json, written after section 22's cells)": {
            "R2 + legacy out-neighbour spread (the router table of this arm)": {"reached_all_gold_rate": seen_router["reached_all_gold_rate"], "hub_gold_reached": seen_router["gold_node_reached_rate_A_by_hubness"]["hub"],
                                                                                 "non_hub_gold_reached": seen_router["gold_node_reached_rate_A_by_hubness"]["non_hub"], "votes_per_hit": seen_router["votes_per_hit_mean (distinct blocks)"],
                                                                                 "blocks_with_evidence_per_query": seen_router["blocks_with_evidence_per_query_mean"]},
            "OWNERS_ONLY + legacy spread (the R = 1 control of this arm)": {"reached_all_gold_rate": seen_control["reached_all_gold_rate"], "hub_gold_reached": seen_control["gold_node_reached_rate_A_by_hubness"]["hub"],
                                                                            "non_hub_gold_reached": seen_control["gold_node_reached_rate_A_by_hubness"]["non_hub"], "votes_per_hit": seen_control["votes_per_hit_mean (distinct blocks)"]},
            "HARD_LEGACY (gate)": {"reached_all_gold_rate": seen_hard["reached_all_gold_rate"], "hub_gold_reached": seen_hard["gold_node_reached_rate_A_by_hubness"]["hub"], "non_hub_gold_reached": seen_hard["gold_node_reached_rate_A_by_hubness"]["non_hub"], "votes_per_hit": seen_hard["votes_per_hit_mean (distinct blocks)"]},
            "R2 membership-only vote (section 22)": {"reached_all_gold_rate": seen_r2["reached_all_gold_rate"], "hub_gold_reached": seen_r2["gold_node_reached_rate_A_by_hubness"]["hub"], "non_hub_gold_reached": seen_r2["gold_node_reached_rate_A_by_hubness"]["non_hub"]},
            "V1 (section 19)": {"reached_all_gold_rate": seen_v1["reached_all_gold_rate"], "lambda_mean_gold_nodes_A": seen_v1["lambda_mean_gold_nodes_A (membership)"]},
            "coverage_seen_for_the_ladder (ALL DEV_A, metaqa)": why["coverage_seen (from the pre-registered records, for the ladder; ALL DEV_A)"]},
        "dry_run_before_this_record": "python -u scratchpad/_l1c_r2router.py metaqa --dry: built the tables, asserted that the spread rule applied to the hard own-block table reproduces the served legacy vote table exactly (109,790 entries), asserted that the four reach tables reproduce the seen br2_why numbers above; no ranking, no coverage, no text-set number, no record",
        "coverage_of_this_arm": "NEVER computed anywhere before this record (reach only; coverage <= reach)"},
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ruling_2026-09-16_thirteenth_verbatim_essentials": RULING,
    "roles_frozen_by_the_thirteenth_ruling": {
        "H2_PATCH1": "transfer reference (section 16, frozen; the WebQSP PRIMARY transfer runs it exactly as section 16 -- not BALANCED)",
        "BALANCED_H2_PATCH1": "current best experimental successor: LEADING SUCCESSOR / MECHANISM_CONFIRMED_ON_DEV_A_POSTHOC / NOT YET TRANSFER-CONFIRMED; not run on WebQSP in the same confirmation pass; to be tested later on another untouched, ideally KB-like, population",
        "H3_PATCH1": "closed; depth is not the current bottleneck",
        "BOUNDED_VERTEXCUT_R2": "partition-only hypothesis rejected (section 22)",
        "R2 + canonical router (this arm)": "the one justified new partition / routing hypothesis",
        "composition": "BALANCED + R2-router NOT combined; every mechanism stays separate until it earns transfer evidence",
    },
    "question": "Can bounded overlap (the frozen R = 2 substrate: every node in <= 2 blocks, cap 100, RF 1.53 / 1.61 / 1.53) reduce replication / unique-node exposure while the existing universal router (the canonical own + directed-out-neighbour vote spread) supplies the hub reach that a lambda <= 2 membership rule cannot encode? Partitioning and routing are separate mechanisms: section 22 removed both the unbounded replication and the router; this arm restores the router on the bounded substrate.",
    "hypothesis_is_not": "'R2 itself is better'; and the seen reach 0.807 (> hard 0.713, > V1 0.801) justifies this single test, NOT a claim that ALL-gold ranking improves (reach is an upper bound on coverage; the router also spreads votes to unrelated blocks: 423 blocks with evidence per query out of 864)",
    "relation_to_earlier_sections": {
        "section_19": "V1's metaqa gain (0.640 -> 0.678) was MEMBERSHIP replication concentrated on hub gold (mean gold node in 43.7 blocks); V1 frozen 'MetaQA mechanism positive / universal replacement NOT_SUPPORTED', never composed, untouched here",
        "section_22": "R2 (membership-only vote, hub -> home) = NOT_SUPPORTED: text SAFE + EFFICIENT (69-75 % of the hard exposure) but metaqa 0.640 -> 0.333 because it has neither spread (reach 0.535; hub gold 0.627; 1.7 votes/hit); its substrate, membership rule, serving rule, exposure and matched-budget contracts are reused VERBATIM; its R2_P50 cell is recomputed here and asserted equal to the record",
        "the_served_hard_BASE": "reaches every hub gold through its VOTE spread (2.5-3.4 votes/hit; own-block-only vote gives 0.18 coverage on metaqa): the router IS the hard BASE's mechanism; this arm applies exactly that router to the bounded memberships",
        "what_is_new": "ONLY the vote table: spread_table(mem_vc) = own vote set (hub -> H(v), non-hub -> M(v)) union the vote sets of the directed STRUCT out-neighbours (deduped) -- _l1c_br2_why.spread_table, the function whose reach was seen; asserted at run time to reproduce _l1s_core.Data.legacy_mem exactly when applied to a hard partition",
    },
    "what_this_is": "a post-hoc partition / routing mechanism test on DEV_A of metaqa, squad and musique: does the bounded substrate under the canonical router match or beat each dataset's served canonical hard BASE in ALL-gold at P50, at what unique-node exposure, and at the matched budget. DEV_A has been read many times in this lane; DEV_B and TEST are never read; nothing is promoted; the served L1 is unchanged; nothing is composed with BALANCED_H2_PATCH1 / H2_PATCH1 / SAFE.",
    "constants": {"R": L.R_MAX, "C": 100, "k_per_dataset": {ds: graphs[ds]["k_R2"] for ds in DS}, "K0": S.K0, "K_LOCK": S.K_LOCK, "P_MAIN": S.P_MAIN, "alpha": 0.01,
                  "hub": "deg + 1 > C (frozen definition; a hub's own vote set is {H(v)}, as in sections 19 / 22; the rule is applied to the hit and to each of its out-neighbours alike)",
                  "no_new_constant": "no weight, no per-dataset constant, no local K; the router has no parameter (it is the served BASE's static node -> blocks map applied to a multi-membership table)"},
    "substrate_frozen": "parts/<ds>__BR2_k<k>__R2.npz of section 22 (home, alt): read verbatim, sha-pinned below, asserted at run time; the owner PHG partitions, degree-0 placements and alternates are NOT rebuilt or changed",
    "router_verbatim": {
        "vote_set_per_node": "V(v) = {H(v)} if hub(v) else M(v)   (section 22's mem_vc; M(v) = {H(v)} or {H(v), alt(v)})",
        "router_table": "T(h) = V(h) union  U_{t in N_out(h)} V(t)   over the directed STRUCT out-neighbours N_out(h) of the canonical adapter's struct_csr(directed=True); deduped; static; query-free",
        "identity_on_a_hard_partition": "with V(v) = {hard(v)} the table T equals _l1s_core.Data.legacy_mem (the served BASE's map) entry for entry -- asserted at run time on the gate partition",
        "routing": "PR_d / PR_s = _ta_prepartition.partition_ranking of the served dense / SPLADE top-K_LOCK ids through T, _l1s_core.rrf_ranks fusion, P50 = the first P_MAIN fused blocks (via _l1c_vcut_replay.fused_rank, imported unchanged)",
        "serving": "gold g served iff M(g) meets the selection (mem_all; _l1c_vcut_replay.served); node-level ALL / ANY",
        "exposure": "unique exposure = |union of the selected blocks' node sets| (_l1c_vcut_replay.unique_exposure); matched prefixes vs the gate's per-query P50 exposure (_l1c_vcut_replay.matched_prefixes)",
        "control_R1": "the same router on the owners-only partition (k = 2 k_f, degree-0 placed, no alternates): V(v) = {H(v)} for every node, home serving, owner blocks for exposure -- isolates the alternates' value under the router",
    },
    "population": {ds: "served %s cache rows; DEV_A = sha1(query_id)[:8] & 1 == 0 exactly as _l1x90_core; hop labels as present in the cache" % ds for ds in DS} | {"DEV_B": "never read", "TEST": "never read", "WebQSP": "not used (its primary transfer is frozen H2_PATCH1 exactly as section 16)"},
    "baselines_gate": {
        "metaqa": {"HARD_MTK_BASE (GATE)": "served `metaqa` cache = canonical H4_SK Mt-KaHyPar partition (k_f = 432), served P50 through the frozen legacy vote table", "HARD_PHG_BASE (contrast only)": "served `metaqa_phg` cache; reported, no verdict"},
        "squad": {"HARD_MTK_BASE (GATE)": "served `squad` cache (k_f = 202)", "HARD_PHG_BASE (contrast only)": "served `squad_phg` cache"},
        "musique": {"HARD_PHG_BASE (GATE)": "served `musique` cache = LOWMEM PHG_C1 partition (k_f = 1,175), the dataset's only served hard BASE"},
        "matched_budget_target": "per query, the GATE baseline's P50 exposure (sum of its selected hard block sizes = unique nodes)",
        "node_level_check": "the node-level ALL through each served hard partition's own-block table must equal the served block-level base_all (asserted)",
    },
    "cells_per_dataset": {
        "R2ROUTER_P50 (PRIMARY)": "50 fused blocks per query through the router table; ALL / ANY (per hop where labelled); unique and nominal exposure; McNemar vs the GATE (and vs the contrast, no verdict); exposure flag EXPOSURE_BELOW_HARD iff the DEV_A mean unique exposure < the gate's DEV_A mean P50 exposure; unique / hard ratio reported",
        "R2ROUTER_MATCHED_LE (SECONDARY)": "per query, the largest fused prefix whose unique exposure <= the GATE's P50 exposure of the same query; McNemar vs the GATE",
        "R2ROUTER_MATCHED_GE (informational)": "the smallest prefix with unique exposure >= the hard exposure; overshoot reported",
        "R2_MEMBERSHIP_VOTE_P50 (section 22 reproduced)": "section 22's R2_P50 recomputed (membership-only vote), asserted equal to br2_replay_A_<ds>.json; R2ROUTER_P50 vs it = the router's value on the same substrate (informational)",
        "OWNERS_LEGACY_P50 / OWNERS_LEGACY_MATCHED_LE (R = 1 control)": "the owners-only partition under the same router; R2ROUTER vs it = the alternates' value under the router (informational)",
        "reach diagnostics": "for the gate table, the section-22 table, the router table and the control: reached-all-gold rate (per hop), hub-gold and non-hub-gold reached rates, votes per hit per channel, blocks with evidence per query (the seen metaqa numbers are asserted reproduced; squad / musique are new)",
        "reach decomposition of the primary cell's failures": "UNREACHED (some gold node in no block with evidence) vs REACHED_WEAK (reached but not selected), worst gold block position, hub / degree-0 gold among the failures; failure overlap with the gate for the primary and the secondary cell",
        "ladder": "read-only numbers from the frozen records (section 22 cells; section 19 V1 on metaqa) beside this record's cells; no recompute of V1",
    },
    "statistics": {"test": "exact two-sided McNemar on paired DEV_A 0/1 ALL vectors (_l1x90_core.mcnemar): gained / lost / p", "alpha": 0.01,
                   "labels": {"GAIN": "gained > lost and p < 0.01", "LOSS": "lost > gained and p < 0.01", "NEUTRAL": "otherwise"},
                   "multiplicity_note": "five decision comparisons in total: metaqa R2ROUTER_P50 vs gate (the gain / efficiency clause) and, for each text dataset, R2ROUTER_P50 and R2ROUTER_MATCHED_LE vs gate (the safety clause); everything else is descriptive"},
    "decision_rule_pre_registered": {
        "TEXT_SAFE(ds)": "R2ROUTER_P50 vs the GATE != LOSS AND R2ROUTER_MATCHED_LE vs the GATE != LOSS (squad and musique must both be SAFE: 'no significant text regression' at P50 and at the matched budget)",
        "METAQA_GAIN": "R2ROUTER_P50 vs HARD_MTK_BASE = GAIN ('a significant MetaQA gain')",
        "METAQA_EFFICIENT": "R2ROUTER_P50 vs HARD_MTK_BASE != LOSS AND DEV_A mean unique exposure < the hard DEV_A mean P50 exposure ('exposure-efficiency improvement': the same ALL-gold at fewer unique nodes; the lane's EFFICIENT label of sections 20 / 22). Stated now: the exposure inequality is expected by construction (|B_p| <= 100, so 50 blocks expose <= 5,000 unique nodes vs 5,023 / 5,089 / 5,117 for the hard P50; section 22 measured 81 % on metaqa), so this clause is decided by the coverage side -- the router has to recover the whole 30.6-pt P50 deficit of section 22 on the same substrate for it to fire",
        "METAQA_MATCHED_GAIN (informational sub-label, not a clause)": "R2ROUTER_MATCHED_LE vs HARD_MTK_BASE = GAIN (a significant gain at the equal budget)",
        "OUTCOME": {"PASS_GAIN": "TEXT_SAFE(squad) AND TEXT_SAFE(musique) AND METAQA_GAIN",
                    "PASS_EFFICIENCY": "TEXT_SAFE(squad) AND TEXT_SAFE(musique) AND not METAQA_GAIN AND METAQA_EFFICIENT",
                    "FAIL": "otherwise (a text LOSS in either cell of either text dataset, or metaqa LOSS at P50, or metaqa not below the hard exposure)"},
        "what_follows": {
            "PASS_GAIN": "post-hoc DEV_A evidence that the bounded substrate under the canonical router beats the served hard BASE in ALL-gold ranking at lower exposure with the text sets safe; status MECHANISM_CONFIRMED_ON_DEV_A (POSTHOC), not promoted; no composition with BALANCED_H2_PATCH1 / H2_PATCH1 / SAFE (the ruling keeps the mechanisms separate until each earns transfer evidence); any transfer test needs its own ruling and an untouched population",
            "PASS_EFFICIENCY": "post-hoc DEV_A evidence for the hypothesis as stated -- bounded overlap reduces exposure while the router supplies the reach -- WITHOUT a ranking gain; status MECHANISM_CONFIRMED_ON_DEV_A (POSTHOC; efficiency only), not promoted; the same no-composition rule; the ranking claim is explicitly NOT made",
            "FAIL": "the ruling's letter: 'Run no R3/R4 or alternate routing if it fails' -- no R = 3 / 4, no second alternate, no undirected / 2-hop / weighted / hub-rule-changed spread, no other router on the bounded substrate; the bounded-substrate line closes in this lane with sections 22 + 23 as its record; V1 stays frozen as ruled",
        },
        "no_promotion": "nothing here changes the served L1 or any frozen record; sections 16-22 are unchanged; the WebQSP primary transfer remains frozen H2_PATCH1 exactly as section 16",
    },
    "forbidden": [
        "reading DEV_B or TEST; using WebQSP",
        "rebuilding or changing the R2 substrate (owners, degree-0 placement, alternates), R, C, the k rule, K0 / K_LOCK / P_MAIN, the RRF, the hub definition, the serving rule, the exposure or matched-budget rule",
        "any other router: undirected spread, 2-hop spread, weighted votes, hub rule changed, own-vote-only, KNN spread; any R = 3 / 4; any second alternate",
        "selecting more than 50 blocks except in the labelled matched-budget cells",
        "any SAFE / H2_PATCH1 / BALANCED_H2_PATCH1 / H3 / V1 composition (the ruling: keep the mechanisms separate until each earns transfer evidence)",
        "editing any CONTRACT_FILE or pinned module; editing _l1c_vcut_replay.py, _l1c_vcut_lib.py, _l1c_br2_lib.py, _l1c_br2_why.py; editing the two new modules after this record (their shas are pinned below and asserted at run time)",
        "signalling the foreign process; touching the WebQSP watcher; writing under data/",
        "tuning after seeing numbers; changing a label, cell, clause or threshold after a number is seen; re-running a dataset after seeing its number",
    ],
    "code": {
        "new_modules": {n: pin(os.path.join(HERE, n)) for n in ("_l1c_r2router.py", "_l1c_r2router_summary.py")},
        "usage": ["python -u scratchpad/_l1c_r2router.py metaqa", "python -u scratchpad/_l1c_r2router.py squad", "python -u scratchpad/_l1c_r2router.py musique", "python -u scratchpad/_l1c_r2router_summary.py"],
        "imports_unchanged": {**{n: pin(os.path.join(HERE, n)) for n in ("_l1s_core.py", "_l1x90_core.py", "_ta_prepartition.py", "_l1g_core.py", "_l1c_vcut_lib.py", "_l1c_vcut_replay.py", "_l1c_br2_lib.py", "_l1c_br2_why.py")},
                              **{"src/l1_canonical/%s.py" % n: pin(os.path.join(X.REPO, "src", "l1_canonical", n + ".py")) for n in ("adapter", "hypergraph")}},
    },
    "graphs_frozen": graphs,
    "substrates_frozen": substrates,
    "records_read_only": {n: pin(os.path.join(OUT, n)) for n in ("br2_replay_A_metaqa.json", "br2_replay_A_squad.json", "br2_replay_A_musique.json", "br2_SUMMARY.json", "br2_why_A_metaqa.json",
                                                              "vcut_replay_A_metaqa.json", "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json")},
    "section22_outcome_read": {"OUTCOME": sum22["OUTCOME"], "per_dataset_VERDICT": {ds: sum22["per_dataset"][ds]["VERDICT"] for ds in DS},
                               "R2_P50_ALL": {ds: sum22["per_dataset"][ds]["R2_P50_ALL"] for ds in DS}, "gate_ALL": {ds: sum22["per_dataset"][ds]["gate_ALL"] for ds in DS},
                               "R2_P50_unique_over_hard": {ds: round(sum22["per_dataset"][ds]["R2_P50_unique"] / sum22["per_dataset"][ds]["gate_exposure"], 4) for ds in DS}},
    "caches_read_only": {n: pin(X.CACHES[n]) for n in ("metaqa", "metaqa_phg", "squad", "squad_phg", "musique")},
}
S.wj(FP, rec)
print("written", os.path.relpath(FP, X.REPO), sha_file(FP), os.path.getsize(FP), rec["utc"])
