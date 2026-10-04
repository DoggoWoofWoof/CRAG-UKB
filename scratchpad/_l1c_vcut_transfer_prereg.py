"""Write-once pre-registration for STRUCT_VERTEXCUT_V1_TRANSFER (eleventh ruling, 2026-09-15): the vertex-cut BASE replay of
section 19 carried unchanged to squad and musique with ONE universal degree-0 rule, gated against each dataset's canonical hard BASE.
Written before any degree-0 placement, any capacity repair on text and any text replay number exists (the text k-grid / k* runs of the
already pre-registered section-18 chain may be in flight; they produce structural numbers only).

    python -u scratchpad/_l1c_vcut_transfer_prereg.py   -> results/L1_COVPART/PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json
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
FP = os.path.join(OUT, "PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json")
assert not os.path.exists(FP), "write-once"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pin(p):
    return {"path": os.path.relpath(p, X.REPO).replace("\\", "/"), "sha256": sha_file(p), "bytes": os.path.getsize(p)}


for ds in ("squad", "musique"):
    assert not os.path.exists(os.path.join(OUT, "vcut_transfer_A_%s.json" % ds))
    assert not any(f.startswith("%s__STRUCT_VCUT_V1_k" % ds) and "CAP" in f for f in os.listdir(os.path.join(OUT, "parts")))

RULING = (
    "STRUCT_VERTEXCUT_V1_REPLAY should remain MECHANISM_POSITIVE_AT_LOWER_EXPOSURE, not promoted yet. But I would absolutely keep it alive. "
    "... 1. Do not combine VCUT with H2_PATCH1 yet. First establish whether VCUT is universal rather than a MetaQA-specific structural win. "
    "2. Do not touch H2 width/depth yet. Frozen H2_PATCH1 still needs WebQSP transfer. 3. Run VCUT transfer on SQuAD and MuSiQue next, with "
    "one universal handling rule for structurally isolated nodes frozen before seeing results. 4. Leave the WebQSP watcher alone. Do not kill "
    "the recurring m3b_run.py. ... For the text VCUT extension, the only design problem we need to solve first is degree-0 nodes. I would not "
    "invent dataset-specific behavior. Use one universal rule: Nodes with STRUCT incidence participate normally in vertex-cut. A node with zero "
    "STRUCT degree receives exactly one non-overlapping home determined by its nearest frozen KNN neighbor that already has a VCUT membership; "
    "use that neighbor's deterministic home. If no such anchored neighbor exists, place it in the currently smallest VCUT block, canonical-node-"
    "order tie break. That is query-independent, training-free, uses the already-frozen KNN substrate, and most importantly does not create fake "
    "structural overlap for isolated nodes. Then the universal representation is: M(v) = VCUT memberships(v) if d_STRUCT(v) > 0; "
    "{H_KNN-anchor(v)} if d_STRUCT(v) = 0. Hubs still vote only through deterministic H(v). That should be preregistered once and applied "
    "unchanged to SQuAD and MuSiQue. What would make VCUT worth carrying forward? I would not require it to beat SAFE yet, because section 19 is "
    "comparing a replacement for the BASE routing substrate. The clean gate should be against each dataset's canonical hard BASE: no significant "
    "ALL-gold loss on SQuAD or MuSiQue; unique-node exposure reported and preferably <= hard P50 exposure; then report gain/loss separately at "
    "matched exposure. If it passes that, we have evidence for a universal statement: Overlap-native partitioning can replace generic 1-hop "
    "block expansion with a more exposure-efficient static routing representation. Then it becomes worth composing with SAFE and eventually "
    "H2_PATCH1. ... So my immediate ruling is: freeze H2_PATCH1; continue WebQSP watcher unchanged; preregister one universal degree-0 rule and "
    "run VCUT BASE transfer on SQuAD + MuSiQue. No composition yet.")

rec = {
    "RECORD": "PREREGISTRATION",
    "name": "STRUCT_VERTEXCUT_V1_TRANSFER",
    "STATUS": "POSTHOC_STRUCTURAL_MECHANISM_TRANSFER_DEV_A",
    "written_before_any_run": True,
    "written_before": ["any degree-0 placement", "any capacity repair on a text vertex-cut", "any text replay number", "any CAP file for squad or musique"],
    "in_flight_when_written": "the section-18 chain (_l1c_vcut_kstar.py squad) had started; it only produces the pre-registered k-grid / k* structural numbers (RF, block sizes) and no replay or serving number",
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ruling_2026-09-15_eleventh_verbatim_essentials": RULING,
    "relation_to_sections_18_19": {
        "section_18": "STRUCT_VERTEXCUT_V1 = NOT_MATERIAL by its pre-registered letter; unchanged. Its construction chain (dual incidence hypergraph, PHG CONNECTIVITY, k-grid {k_f, 2k_f, 3k_f}, k* fixed point, deterministic capacity repair at C) was pre-registered for metaqa, squad and musique ('squad and musique are secondary curves') and is applied to the text corpora here UNCHANGED",
        "section_19": "STRUCT_VERTEXCUT_V1_REPLAY (metaqa DEV_A) = MECHANISM_POSITIVE_AT_LOWER_EXPOSURE; kept, not promoted; its representation, routing, serving, exposure and matched-budget contracts are reused here VERBATIM (the module's functions are imported, not re-implemented)",
        "this_record": "adds exactly one thing -- the universal degree-0 rule of the ruling with its reading fixed below -- and the per-dataset gate of the ruling; nothing else is new",
    },
    "what_this_is": "a post-hoc structural-mechanism transfer test on DEV_A of squad and musique: does the vertex-cut BASE (static overlap decided by the partitioner, no 1-hop vote expansion) replace each dataset's served canonical hard BASE without a significant ALL-gold loss, at what unique-node exposure, and what happens at the matched exposure. DEV_A has been read many times in this lane, so the status is POSTHOC; DEV_B and TEST are never read; nothing is promoted; the served L1 is unchanged.",
    "construction_frozen_per_dataset": {
        "chain": "python -u scratchpad/_l1c_vcut_kstar.py <ds>  (unchanged section-18 script: _l1c_vcut_build.py <ds> <k> -> _l1c_phg.py <ds> STRUCT_VCUT_V1_k<k> with the validated PHG driver, NP = 4, IMBALANCE_TOL 1.03, CONNECTIVITY, non-empty repair) for k in {k_f, 2k_f, 3k_f}, then k_1 = round(RF(2k_f) N / C), k_{i+1} = round(RF(k_i) N / C) until |mean |B_p| - C| <= 0.05 C or three k* runs -> vcut_kstar_<ds>.json",
        "k_f_and_C": {"squad": {"N": 20233, "k_f": 202, "C": 100}, "musique": {"N": 117534, "k_f": 1175, "C": 100}},
        "capacity_repair": "the section-18 deterministic rule, _l1c_vcut_lib.capacity_repair(G, z, k*, C = 100), applied ONCE to the k* cell over the STRUCT edges only -> parts/<ds>__STRUCT_VCUT_V1_k<k*>__PHG_con__CAP100.npy (+ a .CAP100.json repair record); written by the new repair-only wrapper _l1c_vcut_cap.py <ds>, which calls the pinned library function exactly as _l1c_vcut_metrics.py does after its structural cells (the expensive structural metrics are not needed for this gate and may be run later; they gate nothing here)",
        "k_star_target_unchanged": "the k* target counts STRUCT endpoint memberships only (RF N / k = C); degree-0 nodes are NOT part of the k* target and are added to blocks afterwards by the rule below -- block sizes after placement are reported, and no re-balancing follows (the ruling: nodes with STRUCT incidence participate normally in vertex-cut)",
        "no_dataset_specific_step": "one code path; the dataset name only selects the frozen substrate, the served caches and the output file names",
    },
    "degree_0_rule_universal": {
        "ruling_verbatim": "Nodes with STRUCT incidence participate normally in vertex-cut. A node with zero STRUCT degree receives exactly one non-overlapping home determined by its nearest frozen KNN neighbor that already has a VCUT membership; use that neighbor's deterministic home. If no such anchored neighbor exists, place it in the currently smallest VCUT block, canonical-node-order tie break.",
        "when": "after the capacity repair of the k* cell; the repair, the eviction rule and the k* search are never re-run afterwards (no re-repair, no re-balancing, no eviction of a placed node)",
        "anchored_node": "a node u with d_STRUCT(u) > 0 (it holds >= 1 vertex-cut membership by construction); its deterministic home H(u) = argmax_p #{incident STRUCT edges of u assigned to p}, ties broken by the smallest block id -- the section-19 definition, computed by the pinned _l1c_vcut_replay.memberships",
        "frozen_KNN_neighbourhood": "data/final_canonical/<ds>/graph/knn.npz rows (src, dst, weight float32) read through the frozen adapter CanonicalDataset.family('knn'): undirected, one row per unordered pair (storage census: 0 reverse rows), weight = exact cosine similarity of the frozen dense vectors (higher = nearer). KNN neighbours of v = {u : (v, u) or (u, v) is a row}. No residualisation is involved: the adapter subtracts KNN pairs that are also STRUCT edges, which cannot involve a degree-0 node",
        "nearest_anchored_neighbour": "a(v) = argmax over the anchored KNN neighbours of v of the row weight; ties (equal float32 weight) -> the smallest node id; the home of v is H(a(v)) (the anchor's deterministic home, NOT its full membership set)",
        "no_chaining_fixed_reading": "'already has a VCUT membership' = holds a membership from the vertex-cut itself, i.e. d_STRUCT > 0. A degree-0 node never anchors another degree-0 node: the anchored pass is therefore order-independent and every home is a function of the frozen graph alone. (The number of fallback nodes that would have been anchored under a chaining reading is reported as a descriptive count; no second cell.)",
        "fallback": "a degree-0 node with no anchored KNN neighbour (including a node with KNN neighbours that are all degree-0) is placed in the currently smallest block: blocks are sized by ALL nodes they hold at that moment (STRUCT endpoint sets after the repair + every degree-0 node placed so far); all anchored placements are applied before the first fallback; fallbacks are processed in ascending canonical node id and each placement updates the sizes; ties -> the smallest block id",
        "representation_after_placement": "M(v) = vertex-cut memberships (blocks holding an edge of v) if d_STRUCT(v) > 0; M(v) = {H_KNN-anchor(v)} (or the fallback block) if d_STRUCT(v) = 0 -- exactly one non-overlapping home, never a halo. H(v) for a degree-0 node = that block. Hubs (deg + 1 > C, frozen definition) vote only through H(v); a degree-0 node is a non-hub with a single membership, so it votes to and is served through its one block. B_p is extended by the placed nodes for serving and exposure",
        "degree_0_counts_frozen_facts": {"squad": 6349, "musique": 9858, "note": "every degree-0 node of both corpora has >= 1 KNN row (checked on the frozen substrate before this record; whether the neighbour is anchored is decided by the rule at run time)"},
        "reported": ["anchored / fallback counts", "block sizes after placement (mean / max, blocks > C, blocks > 1.5 C)", "RF after placement", "gold nodes that are degree-0 (fraction, queries with >= 1)", "served hits (dense / SPLADE top-100) that are degree-0", "fallback nodes with a placed degree-0 KNN neighbour (chaining-reading count)"],
    },
    "representation_routing_serving_exposure": {
        "identical_to_section_19": "M(v), H(v), halo, hub rule, mem_vc (non-hub -> M(v); hub -> {H(v)}), PR_d / PR_s = _ta_prepartition.partition_ranking over the served dense / SPLADE top-K_LOCK ids of the replay cache's own rows, _l1s_core.rrf_ranks fusion, P50 = the first P_MAIN blocks, served(g) iff M(g) meets the selection, node-level ALL / ANY, unique exposure = |union B_p| over the selection, matched prefixes -- all via the imported, pinned _l1c_vcut_replay functions",
        "constants_unchanged": {"K0": 60, "K_LOCK": 100, "P_MAIN": 50, "RRF": "frozen", "hub": "deg + 1 > C"},
        "no_added_halo": "no 1-hop / out-neighbour expansion of any kind; the served hard BASE keeps its frozen legacy table (own block + directed STRUCT out-neighbours' blocks) exactly as served",
        "node_level_check": "the node-level ALL through the hard partition's own-block table must equal the served block-level base_all on every served cache used (asserted, as in section 19)",
    },
    "population": {
        "squad": "data/l1_canonical/squad/replay_cache.npz rows; DEV_A = sha1(query_id)[:8] & 1 == 0 exactly as _l1x90_core; no hop labels (one gold paragraph per query: ALL = ANY)",
        "musique": "data/l1_lowmem/musique/replay_cache__LOWMEM__PHG_C1_con.npz rows; DEV_A as above; hop labels as present in the cache (reported per hop and pooled hop 2+3 where both exist)",
        "DEV_B": "never read",
        "TEST": "never read",
    },
    "baselines_gate": {
        "squad": {"HARD_MTK_BASE (GATE)": "the served `squad` cache = canonical H4_SK Mt-KaHyPar partition (k_f = 202), served P50 through the frozen legacy vote table -- the dataset's canonical hard BASE",
                  "HARD_PHG_BASE (contrast only)": "the served `squad_phg` cache (LOWMEM PHG partition); reported, no verdict"},
        "musique": {"HARD_PHG_BASE (GATE)": "the served `musique` cache = LOWMEM PHG_C1 partition (k_f = 1,175), the dataset's only served / canonical hard BASE (Mt-KaHyPar RESOURCE_INFEASIBLE_LOCAL); served P50 through the frozen legacy vote table"},
        "matched_budget_target": "per query, the GATE baseline's P50 exposure (sum of its selected hard block sizes = unique nodes)",
    },
    "cells_per_dataset": {
        "VCUT_P50 (primary)": "50 fused vertex-cut blocks per query; ALL / ANY (/ per hop on musique); unique exposure (and nominal) reported; McNemar vs the GATE baseline (and vs the contrast baseline on squad, no verdict)",
        "VCUT_MATCHED_LE (secondary)": "per query, the LARGEST fused prefix whose unique exposure <= the GATE baseline's P50 exposure of the same query (never exceeds the hard budget); McNemar vs the GATE baseline",
        "VCUT_MATCHED_GE (informational only)": "the SMALLEST fused prefix whose unique exposure >= the hard exposure (overshoot reported); not a decision cell",
        "VCUT_HOME_ONLY_VOTE (labelled diagnostic)": "every node votes only to H(v); serving unchanged; P50; not a decision cell",
        "HARD_OWN_BLOCK_VOTE (labelled diagnostic)": "the GATE hard partition voted through its own-block table instead of the legacy table; P50; not a decision cell",
        "reach_decomposition (diagnostic)": "as section 19 (UNREACHED / REACHED_WEAK, worst gold position, hub gold, degree-0 gold)",
    },
    "statistics": {"test": "exact two-sided McNemar on paired DEV_A 0/1 ALL vectors (_l1x90_core.mcnemar): gained / lost / p", "alpha": 0.01,
                   "multiplicity_note": "two decision cells per dataset (primary, secondary), two datasets, each at alpha = 0.01; every other number is descriptive"},
    "decision_rule_pre_registered": {
        "per_dataset_primary_label": {"LOSS": "VCUT_P50 vs GATE on DEV_A (ALL, all queries): lost > gained AND p < 0.01",
                                      "GAIN": "gained > lost AND p < 0.01",
                                      "NEUTRAL": "otherwise"},
        "per_dataset_exposure_flag": {"EXPOSURE_LE_HARD": "mean unique exposure of VCUT_P50 over DEV_A <= mean P50 exposure of the GATE baseline", "EXPOSURE_ABOVE_HARD": "otherwise"},
        "per_dataset_matched_label": "VCUT_MATCHED_LE vs GATE labelled GAIN / NEUTRAL / LOSS by the same rule; reported separately, as the ruling asks ('then report gain/loss separately at matched exposure')",
        "per_dataset_verdict": "<primary label>_<exposure flag>; the ruling's pass condition for a dataset = primary label != LOSS ('no significant ALL-gold loss'), exposure preferably LE_HARD",
        "universal_statement": {
            "statement": "Overlap-native partitioning can replace generic 1-hop block expansion with a more exposure-efficient static routing representation.",
            "SUPPORTED": "metaqa section 19 = MECHANISM_POSITIVE_AT_LOWER_EXPOSURE (already recorded) AND squad primary != LOSS AND musique primary != LOSS AND both text datasets EXPOSURE_LE_HARD",
            "PARTIALLY_SUPPORTED": "no LOSS on either text dataset, but at least one EXPOSURE_ABOVE_HARD",
            "NOT_SUPPORTED": "LOSS on squad or musique",
        },
        "what_follows": "SUPPORTED / PARTIALLY_SUPPORTED = evidence for the statement, recorded as post-hoc DEV_A evidence; composition with SAFE and later H2_PATCH1 needs a further ruling and is NOT run here. NOT_SUPPORTED = the vertex-cut BASE is a metaqa-specific structural win (or the degree-0 rule is the failure point -- the degree-0 diagnostics say which); still no rule change without a ruling",
        "no_promotion": "nothing here changes the served L1 or any frozen record; the section-19 verdict is unchanged",
    },
    "forbidden": [
        "reading DEV_B or TEST",
        "any change to K0 / K_LOCK / P_MAIN / the RRF / the hub definition / the home tie-break / the anchored tie-break / the fallback order",
        "a second degree-0 rule or variant, a chaining pass, a KNN-weight threshold, a dataset-specific branch",
        "re-repairing, re-balancing or re-partitioning after the degree-0 placement; re-running k* with another target; replaying any k other than the CAP k* cell",
        "selecting more than 50 blocks except in the labelled matched-budget cells",
        "any SAFE / H2_PATCH1 / H3 / beam combination (the ruling: 'No composition yet')",
        "any halo or out-neighbour expansion added to the vertex-cut memberships",
        "editing any CONTRACT_FILE or pinned module; editing _l1c_vcut_replay.py or _l1c_vcut_lib.py",
        "changing NP or the PHG parameters for the text runs; signalling the foreign process; touching the WebQSP watcher",
        "writing under data/",
        "tuning after seeing numbers; changing a label or threshold after a number is seen",
    ],
    "code": {
        "repair_wrapper": "scratchpad/_l1c_vcut_cap.py <ds> (new; pinned by sha in the result) -> parts/<ds>__STRUCT_VCUT_V1_k<k*>__PHG_con__CAP100.{npy,json}",
        "transfer": "scratchpad/_l1c_vcut_transfer.py <cache> (new; pinned by sha in the result; imports _l1c_vcut_replay and _l1c_vcut_lib unchanged) -> results/L1_COVPART/vcut_transfer_A_<ds>.json + .log",
        "imports_unchanged": {n: pin(os.path.join(HERE, n)) for n in ("_l1s_core.py", "_l1x90_core.py", "_ta_prepartition.py", "_l1g_core.py", "_l1c_vcut_lib.py",
                                                                      "_l1c_vcut_replay.py", "_l1c_vcut_build.py", "_l1c_phg.py", "_l1c_vcut_kstar.py")},
        "section_18_and_19_preregistrations": {n: pin(os.path.join(OUT, n)) for n in ("PREREGISTRATION_STRUCT_VERTEXCUT_V1.json", "PREREGISTRATION_STRUCT_VERTEXCUT_V1_REPLAY.json")},
        "section_19_result_unchanged": pin(os.path.join(OUT, "vcut_replay_A_metaqa.json")),
    },
    "caches_read_only": {n: pin(X.CACHES[n]) for n in ("squad", "squad_phg", "musique")},
    "knn_read_only": {ds: pin(os.path.join(X.REPO, "data", "final_canonical", ds, "graph", "knn.npz")) for ds in ("squad", "musique")},
}
S.wj(FP, rec)
print("written", os.path.relpath(FP, X.REPO), sha_file(FP), os.path.getsize(FP), rec["utc"])
