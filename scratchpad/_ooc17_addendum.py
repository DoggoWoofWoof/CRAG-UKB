"""FBX_SCALE declaration addendum 17 (write-once): the OUT-OF-CORE implementation of the V-cycle top VCQ512L4N4FM2 and its Freebase STRUCT-only systems measurement, declared BEFORE any engine code runs.

Addendum 16 (REPORT §59) found the V-cycle arm Q512L4N4 (37,560 clusters, net cap 4, 12.16x fewer pins) RECALL_PRESERVING on WebQSP (max loss 7 rows, ON the line, F = 8) and set it as the reported Track B top for a
Freebase STRUCT-only systems measurement; decision-table row 1 sends the next step here: build the streaming implementation, regress it bit for bit against the WebQSP records, run the measurement.
This addendum fixes (i) the stages of the engine, (ii) the regression every stage must pass before the next one starts, (iii) the memory / disk budgets, (iv) what the Freebase run reports and the stop rules.
It changes no gate and no earlier record; it reads no gold label anywhere.   Usage: python scratchpad/_ooc17_addendum.py   (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_17.json"
A = [FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % i for i in ("1", "2", "3", "4", "5", "6", "6A", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16")]
REFS = [FB + "/CALIB_EVAL_WEBQSP__vc16_v1.json", FB + "/vc/webqsp__VC_Q512L4N4_fm2_top_k25.D1.json", FB + "/vc/webqsp__VC_Q512L4N4_fm2_top_k25.npy", FB + "/ml2/COARSEN__M2_W512.json",
        FB + "/FBX_GRAPH_STATS__v1.json"]
CODE = ["scratchpad/_vc.py", "scratchpad/_vc_ref2.c", "scratchpad/_ml2_coarsen.py", "scratchpad/_ml2_agg.c", "scratchpad/_ml_coarsen.py", "scratchpad/_h2lt.py", "scratchpad/_h2l.py", "scratchpad/_fbx_stats.py",
        "scratchpad/_fbx_part.py", "scratchpad/_host_yield.py", "src/l1_canonical/hypergraph.py", "src/l1_lowmem/phg_driver/phg_driver_vw.c"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    st = json.load(io.open(FB + "/FBX_GRAPH_STATS__v1.json", encoding="utf-8"))["stats"]
    h = st["hist_log2_bin0_is_zero_binb_is_2^(b-1)_to_2^b-1"]["und"]
    deg1, deg23 = int(h[1]), int(h[2])
    lo = 2 * deg1 + 3 * deg23
    hi = 2 * deg1 + 4 * deg23
    rec = {
        "stage": "FBX_SCALE / addendum 17: the out-of-core V-cycle engine for VCQ512L4N4FM2 (STRUCT-only) and its Freebase systems measurement, declared before any engine code runs",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {p: sha(p) for p in A},
        "basis": {"records": {p: sha(p) for p in REFS},
                  "addendum 16 decision table, row 1": "an arm >= 10.9x that is RECALL_PRESERVING is the reported Track B top; the next addendum builds the streaming implementation, regresses it bit for bit against these records "
                                                       "and runs the STRUCT-only Freebase measurement; the SK requirement (15.4x) stays open",
                  "arm": "VCQ512L4N4FM2 = the stored M2_W512 ladder level 4 (37,560 clusters on WebQSP), nets of more than 4 pins dropped on the top hypergraph before the quotient, Zoltan-PHG NP 4 / IMBALANCE_TOL 1.01 on the "
                         "vertex-weighted surrogate at k = S = 25, then FM2 (gain-cache Fiduccia-Mattheyses, 6 passes, stop after 500 non-improving moves, per-block cap floor(1.01 N / 25)) at every ladder level 4..0 "
                         "on the exact quotient of the FULL top hypergraph",
                  "caveat carried from REPORT §59.4/59.5": "the verdict is ON the pass line (7 rows, F = 8), one top per arm, the nested top's own replicate floor is unmeasured; the engine reproduces THIS top, it does not strengthen the verdict"},
        "freebase_facts_used": {"N": st["nodes"], "anchors_deg_ge_1": st["anchors_deg_ge_1"], "unique_undirected_pairs_F": st["unique_undirected_pairs_F"], "STRUCT_only_pins": st["H4_STRUCT_only"]["pins_precap"],
                                "und_degree_1_anchors": deg1, "und_degree_2_or_3_anchors": deg23, "share_of_anchors_with_degree_1": round(deg1 / st["anchors_deg_ge_1"], 4),
                                "pins_in_nets_of_at_most_4_pins_before_any_quotient": "%d .. %d (exact split of degrees 2 / 3 is not in the histogram)" % (lo, hi),
                                "reading": "the net-cap-4 surrogate holds ~0.41e9 pins BEFORE the ladder quotient, above the 0.39e9 budget; the ladder's open-twin / identical-net merging is what must bring it under, and WebQSP's 12.16x is "
                                           "not assumed to transfer: the census (stage E1) measures it exactly"},
        "engine": {
            "class": "semi-external multilevel (ruling 1A): O(V) arrays in RAM, pins streamed from / mmap'd on disk, every stage resumable; level 0 of the STRUCT-only hypergraph is IMPLICIT (net u = {u} U N(u), "
                     "weight max(1, rint(1000 / |N(u)|)), read from the symmetric adjacency), so no copy of the 4.25e9 pins is written",
            "stages": [
                {"id": "E1", "what": "streaming symmetric adjacency (per chunk: unique(out U in) minus self, as scratchpad/_fbx_stats.py) + open-twin contraction (level 1) + the LEVEL-1 CENSUS: V_1, M_1, P_1 of the exact quotient, "
                                    "and the pins of the cap-4 surrogate quotient at level 1; hashes + exact neighbour-list check",
                 "regression": "WebQSP (family S only and the SK pair): cluster map == stored ml2/clusters__M2_W512_L1.npy, quotient stats == the COARSEN record's level-1 row"},
                {"id": "E2", "what": "streaming EXACT quotient: map pins, sort + dedupe inside a net, drop nets of < 2 pins, merge identical nets (size + two independent 64-bit hashes + a third hash, buckets on disk), weights summed, "
                                    "surviving nets in their original order",
                 "regression": "bit-identical eptr / eidx / ew / vw to scratchpad/_ml_coarsen.quotient on random hypergraphs and on WebQSP levels 1..4 (host qcache)"},
                {"id": "E3", "what": "ladder aggregation passes L2..L4: _ml2_agg.c made 64-bit-pin and mmap-based (same algorithm, same order (vertex weight, id), same arithmetic -ffp-contract=off) + a disk-backed transpose",
                 "regression": "cluster maps == stored L2, L3, L4 maps bit for bit (WebQSP); the C == plain-Python reference tests of _ml2_coarsen._test"},
                {"id": "E4", "what": "surrogate builder (nets > 4 pins dropped, quotient at level 4) written as the hypergraph shards the frozen vertex-weighted Zoltan driver reads; the Zoltan top is NOT reimplemented",
                 "regression": "WebQSP surrogate shards byte-identical to the in-memory ones; the same Zoltan top map (sha 4c043f0c... after the V-cycle) is reproduced end to end"},
                {"id": "E5", "what": "semi-external FM2: the algorithm, tie rules and move order of scratchpad/_vc_ref2.c unchanged (a different data structure, not a different method): pins mmap'd, the gain cache B / R / D and the "
                                    "net-block counts phi held in compact / sparse form (uint8 / uint16 with an overflow table, or per-vertex sparse lists) so that levels 1 and 0 fit the RAM budget",
                 "regression": "labels_out identical to _vc_ref2.c at every WebQSP level (4 .. 0, mode 1) and on random hypergraphs; km1 before / after identical",
                 "fallback": "if bit-identity cannot be kept inside the budgets, the FM2 pass at the offending levels is replaced only through a NEW addendum that re-gates the changed arm on WebQSP (no silent substitution)"},
                {"id": "E6", "what": "V-cycle driver over E2..E5 (project, refine at every level 4..0) writing the 25-way top map",
                 "regression": "the WebQSP top map is byte-identical to results/FREEBASE_SCALE/vc/webqsp__VC_Q512L4N4_fm2_top_k25.npy (sha 4c043f0c...) and the D1 trajectory (km1_in / km1_out per level, moves) is equal"},
                {"id": "E7", "what": "the Freebase STRUCT-only measurement: E1..E6 on the 301,977,131-node / 4,253,534,391-pin hypergraph at S = 25 (the frozen split rule for anchors above round(N / 25) neighbours, streamed)",
                 "reports": ["V_l, M_l, P_l of every level 1..4 and the cumulative cluster-map sizes", "surrogate pins (cap 4, level 4) against the 0.39e9 budget and the Zoltan peak RSS / wall", "per stage: wall, CPU, peak RSS, peak disk, bytes read / written",
                             "the V-cycle KM1 trajectory (km1 in / out per level) and the final KM1 relative to the surrogate's projection", "block sizes, validity under the frozen rule"],
                 "not_claimed": "no recall at Freebase scale (no Freebase gold; the WebQSP bridge proxy is a later addendum), no K-way sub-partition, no SK arm"}],
            "order": "E1 -> E2 -> E3 -> E4 -> E5 -> E6 are developed and regressed on WebQSP (and on random hypergraphs) before E7; E1's census may be run on Freebase as soon as E1 passes its regression (it is a measurement, no partition)"},
        "budgets": {"ram": "peak RSS summed over all processes of one stage <= 70 GB on the host (the shared VM has ~81.6 GB; the earlier 73.4 GB Zoltan budget); a stage reserves what it measured, not more",
                    "disk": "<= 55 GB of crag scratch on the host at any time (78 GB free on 2026-10-03, shared with other users' jobs); transients (transposes, bucket files, superseded levels) are deleted as soon as consumed, "
                            "the peak is recorded",
                    "scheduling": "every job runs under scratchpad/_host_yield.py, resumable at chunk / pass granularity (checkpoint + stale-checkpoint assertion), yields to mpr, requests measured resources only",
                    "wall": "reported, not gated"},
        "stop_rules": ["a stage that fails its WebQSP regression blocks the next stage; the failure is reported with the first differing element, nothing is relaxed",
                       "E7 stops and reports (user's call) if the measured Freebase surrogate (cap 4, level 4) exceeds 0.45e9 pins (the 0.39e9 budget + 15 %): the 12.16x reduction did not transfer on structure",
                       "E7 stops and reports if any stage exceeds the RAM or disk budget above at its measured bytes per pin; no stage is run with a larger reservation to push through",
                       "no stage reads a gold label; no earlier record is edited (supersession only)"],
        "not_done_here": ["any claim of recall at Freebase scale", "any K-way sub-partition of a Freebase top", "the SK arm (15.4x requirement), KNN families", "a second Zoltan seed of the L4N4 top (the replicate floor): a separate addendum if the user wants it",
                          "any change to the frozen driver, the ladder (M2, W 512), the gate or the arm"],
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
