"""FBX_SCALE declaration addendum 1 (write-once): the user's partitioner ruling (2026-09-30) and the WebQSP calibration it orders.

The declaration HOST_STAGE_DECLARATION__FBX_SCALE__v1.json is never edited; a later stage is a dated addendum.  This one records the ruling
verbatim, the ladder and the acceptance rule BEFORE any candidate map is scored, and names the WebQSP inputs that move to the host.

Usage: python scratchpad/_fbx_scale_addendum1.py            (refuses to overwrite)
"""
import glob
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
OUT = "results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_1.json"
BASE = "results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1.json"
KGRID = [100, 250, 500, 1000, 2000, 5000, 25928]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    refs = {}
    for K in KGRID:
        rj = "results/L1_HOST/parts/webqsp__H4_SK_k%d__PHG_con.RUN.json" % K
        J = json.load(io.open(rj, encoding="utf-8"))
        refs[str(K)] = {"map": J["output"]["file"], "map_sha256": J["output"]["sha256"], "run_json_sha256": sha(rj),
                        "blocks": K, "size_max": J["balance"]["size_max"], "size_min": J["balance"]["size_min"]}
    rec = {
        "stage": "FBX_SCALE / addendum 1: the partitioner ruling and the WebQSP calibration (stage 3a)",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S+05:30", time.localtime()),
        "extends": {BASE: sha(BASE)},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism",
        "user_ruling (verbatim, chat, 2026-09-30, answer to 'which route to finish FREEBASE_SCALE')": [
            "We should not decide 'LP because it fits memory' first. We should define the target as: minimum-cost partitioning method that preserves retrieval recall -- and let the WebQSP calibration choose the substrate.",
            "Which cheapest partitioner gives us essentially the same query recall/routing quality? ... choose by recall first, then minimize cost among the recall-preserving candidates.",
            "Candidates, cheapest to more expensive: 1. hash / balanced hash; 2. streaming graph partitioning (LDG/FENNEL-style); 3. deterministic label propagation / balanced LP; 4. two-level structural partitioning; 5. PHG (reference; infeasible on this host at Freebase scale).",
            "Use WebQSP PHG as the reference: for each cheap candidate run the same frozen query/router workload. At each relevant (B_P, B_N) measure R_ALL of the method against R_ALL of PHG. Also measure ANY-gold, shards contacted, B_P/K, edge cut, shard imbalance, construction RAM, time, disk, query-time routing latency. A Pareto table is cleaner than a scalar.",
            "Sequential elimination with early stopping at the first recall-preserving method: Hash -> Streaming -> LP -> Hierarchical PHG.",
            "Don't optimize for edge cut alone: the primary selection criterion is ALL-gold recall under matched (B_P, B_N); among recall-equivalent methods choose the cheapest.",
            "Since this is development/calibration, I wouldn't create another massive preregistration exercise. Practically look at absolute recall difference, paired gains/losses, and whether losses persist over the B_P x B_N curve rather than one point. Consistently within roughly 1 point and no systematic failure mode = recall-preserving; 3-5 points lower = keep searching.",
            "Then run that one method at K = 1k, 2k, 5k, 10k, 20k plus hash as the structural-control baseline."
        ],
        "standing_rulings_kept": ["IR_L1 node localization -> ES partition routing -> KNEE_SDIV (alpha 0.75 KSCALE rule, REPORT 36-37) unchanged", "no new mechanism tuning on Freebase",
                                  "TEST never read; split B sealed; no encoder training, no LLM", "nothing under data/final_canonical is written",
                                  "every host job under scratchpad/_host_yield.py; foreign jobs never touched; mpr > crag > jigsaw"],
        "stage_3a_calibration (WebQSP; laptop or host; no Freebase compute)": {
            "population": "the 786 split-A rows of REPORT 44 (results/L3_DEV/l3w_population_webqsp__v1.json, sha prefix 7e67c9b1); the same rows the reference maps' L1 numbers are on",
            "router": "the frozen IR_L1 of REPORT 44.1 (FLAT dense+SPLADE RRF, one-hop LOC over STRUCT_out U STRUCT_in U KNN U NER, RRF K0 60, ACT 200), ES shard ranking by first occurrence in the router order, "
                      "served list = the first n = min(B_N, contacted mass) nodes of the L1 order inside the contacted shards; B_N in {100, 250, 500, 1000, 2000, 5000}. Only the node -> block table changes between arms.",
            "reference": {"maps": "the seven WebQSP PHG maps (H4_SK hypergraph, Zoltan-PHG NP 4, PHG_NONEMPTY_REPAIR_V1)", "K_grid": KGRID, "pins": refs,
                          "regression_gate": "before any candidate is scored the new harness must reproduce REPORT 44.2's L1 ALL counts on PHG_k25928 (370 / 507 / 605 / 676 / 710 / 727 at B_N 100 / 250 / 500 / 1000 / 2000 / 5000) and the UNROUTED row (370 / 507 / 602 / 668 / 705 / 730), and the position array POS__PHG_k25928__L1 of l3w_webqsp__v1.npz"},
            "rungs (in this order; a rung is implemented and scored only if every cheaper rung failed the gate)": {
                "R1 HASH": "balanced hash placement: nodes ordered by splitmix64(position xor seed), block = rank mod K (block sizes differ by at most 1); seeds 0, 1, 2; the structural-control baseline, also run on Freebase",
                "R2 LDG": "Linear Deterministic Greedy, one pass in position order: place v in argmax_i |N(v) n P_i| (1 - |P_i| / C) with hard capacity C = ceil(1.03 N / K) (the PHG contract's balance bound); a vertex with no placed neighbour, or whose touched blocks are full, goes to the least-loaded block; ties -> smaller load, then smaller id. No parameter, no randomness.",
                "R3 LP": "size-constrained label propagation initialised from R2, at most 5 asynchronous sweeps in position order, a move only into a block with room (same C), a move only if it strictly increases the neighbour count; no randomness (declared here, implemented only if R2 fails)",
                "R4 TWO_LEVEL": "cheap coarse split into super-blocks then PHG inside each (declared here, implemented only if R3 fails; needs its own addendum for the super-block rule)"},
            "graph the candidates read": {"S": "the undirected STRUCT pair set (data/l1_canonical/webqsp/keys.npz, key STRUCT): the only family Freebase has, so the GATE reads this arm",
                                          "SK": "STRUCT U KNN, the reference hypergraph's edge families: a diagnostic that separates 'the partitioner is worse' from 'the KNN family is missing'"},
            "measures": {
                "matched B_P (PRIMARY)": "per row j and K, the number of shards contacted = the PHG reference's B_P(q_j, K) (its own KNEE_SDIV on the PHG K 100 map, alpha 0.75); the candidate contacts that many shards in ITS ES order; ALL-gold and ANY-gold counts of 786 at each B_N; paired gained / lost against the PHG cell (descriptive)",
                "own B_P": "the candidate's own KNEE_SDIV on ITS K 100 map and the alpha 0.75 table (the frozen policy); ALL / ANY counts, B_P, B_P / K, contacted mass, contacted shards",
                "map": "block-size distribution (min, mean, max, load skew = max / mean), cut fraction of the STRUCT pairs, mean number of distinct blocks in a closed neighbourhood (the hypergraph objective's per-anchor connectivity), construction wall seconds, peak RSS, output bytes",
                "query time": "routing latency per row (ES order + B_P + contacted set)"},
            "acceptance (fixed BEFORE any candidate is scored)": {
                "gate_cells": "K in {100, 250, 500} (blocks of 25.9k / 10.4k / 5.2k nodes: the nearest WebQSP has to Freebase's 15k-302k nodes per block at K 20k-1k); K 1000-25928 are reported, not gated",
                "recall_preserving": "at every gate K and every B_N in the curve, the candidate's matched-B_P ALL-gold count on graph S is at most 7 rows (0.009 of 786, 'roughly 1 point') below the PHG cell's",
                "failed": "at any gate K and B_N the count is 24 or more rows (3 points) below the PHG cell's, or the frozen policy fails to route (B_P = K on more than half the rows)",
                "borderline": "everything else: reported and put to the user; the next rung is implemented only if the user rules it or if the borderline rung is not the cheapest that passes",
                "choice": "the first rung that is recall-preserving (cheapest first); among methods that preserve recall the cheaper by (peak RAM, wall time), the frozen fan-out B_P / K reported beside them. No fan-out threshold is invented here."
            },
            "limits declared": [
                "WebQSP is 116 times smaller than Freebase: its blocks at the gate K have 5k-26k nodes against Freebase's 15k-302k; the calibration says which method preserves the WebQSP reference behaviour, not that it does so at 302M nodes",
                "the population is a development population already used by REPORT 44; no held-out row is read",
                "the reference maps are H4_SK PHG; no PHG map on the STRUCT-only hypergraph exists, so the S-arm gate is conservative by any KNN-family effect (the SK arm diagnoses it)"]},
        "stage_3b_freebase (not started; needs the stage-3a verdict and its own dated addendum)": [
            "the chosen partitioner (a C / numba streaming pass over the served CSR, memory-mapped) at K in {1000, 2000, 5000, 10000, 20000} plus the balanced-hash control; record build time, peak RSS / disk, block sizes, load skew, cut fraction, mean closed-neighbourhood connectivity",
            "then the routing / fan-out measurements on the WebQSP population mapped through bridge/webqsp_positions.npy (a systems proxy, not IR_L1): B_P, B_P / K, contacted mass, bytes and nodes touched, latency"],
        "inputs_that_move_to_the_host": {
            "why": "the calibration's router evidence needs the frozen dense + SPLADE node vectors of WebQSP (8.5 GB) and the laptop has about 0.7 GB free RAM (another project is running on it); the host holds no WebQSP tree",
            "how": "python <rx> push --inputs <glob> --force  (the declared rx channel; no external service); read-only under ws/data/final_canonical/webqsp",
            "files": ["data/final_canonical/webqsp/DATASET.json", "data/final_canonical/webqsp/embeddings/**", "data/final_canonical/webqsp/graph/**", "data/final_canonical/webqsp/queries/**",
                      "data/final_canonical/webqsp/retrieval_cache/**", "data/final_canonical/webqsp/v1/**", "data/l1_canonical/webqsp/query_index.npz"],
            "excluded": ["data/final_canonical/webqsp/_acquisition/**", "data/final_canonical/webqsp/nodes.jsonl (the L1 calibration reads no node text)"]},
        "refusals_kept": ["no candidate is tuned on the calibration (every parameter above is fixed here)", "no Freebase partition is built before the stage-3a verdict",
                          "no held-out row; no L3 arm in this calibration (L1 only)"],
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
