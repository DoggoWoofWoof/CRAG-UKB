"""FBX_SCALE declaration addendum 7 (write-once): Track B, the two-level (hierarchical) PHG on WebQSP, declared BEFORE any H2L arm is partitioned or scored.

The user's ruling of 2026-09-30 (late): stop the M1-M4 search; architecture = modest M2 coarsening (about 2x) -> out-of-core PHG -> projection + refinement, verified on WebQSP against the
in-memory result.  There is no disk-backed Zoltan: the installed Zoltan PHG holds its hypergraph in RAM (measured ~129 B/pin, NP 4).  The one structure that bounds memory by construction is a
NESTED partition: a top S-way partition, then one independent K/S-way partition per top block on the block's net-split hypergraph; peak memory is the largest block's pins (blocks are loaded from
disk one at a time).  This addendum asks the prior question on WebQSP, where the flat in-memory PHG_SK is the reference: is the nested partition RECALL-PRESERVING under the addendum-1 gate?
The top level is left flat and in memory here (WebQSP fits); making the top level out-of-core (M2 coarsening + projection + refinement) is addendum 8 and starts only from this result.
Usage: python scratchpad/_h2l_addendum7.py   (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_7.json"
A = [FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % i for i in ("1", "2", "3", "4", "5", "6", "6A")]
CODE = ["scratchpad/_h2l.py", "scratchpad/_h2l_eval.py", "scratchpad/_ml2_eval.py", "scratchpad/_fbx_calib.py", "scratchpad/_l1h_host.py", "src/l1_canonical/hypergraph.py", "src/l1_lowmem/phg.py",
        "src/l1_lowmem/phg_repair.py", "src/l1_lowmem/freight.py", "src/l1_lowmem/phg_driver/phg_driver.c"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    rec = {
        "stage": "FBX_SCALE / addendum 7: two-level (hierarchical) PHG on WebQSP (Track B), declared before any H2L arm is partitioned or scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {p: sha(p) for p in A},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism",
        "question": "does replacing the flat K-way PHG by a nested partition (top S-way PHG, then independent K/S-way PHGs on the net-split blocks) preserve ALL-gold recall under the addendum-1 gate?  "
                    "What it buys at scale: the K-way stage's memory becomes the largest block's pins (about (pins + split-net pins) / S) instead of all pins.  What it does not buy: the top S-way stage still reads the whole hypergraph (addendum 8).",
        "not_claimed": ["no bit-for-bit reproduction of the flat partition: a different algorithm cannot reproduce it and none is expected; 'equivalent' means the addendum-1 gate, stated below",
                        "nothing about Freebase feasibility: the top level is flat and in memory here; the per-block memory and time constants are measured for the extrapolation only",
                        "no M2 coarsening, no projection, no refinement in this addendum"],
        "arms": {
            "reference": "PHG_SK = the frozen flat WebQSP maps on H4_SK at K in {100, 250, 500} (Zoltan PHG, NP 4); their cells must reproduce the earlier EVAL bit for bit (control)",
            "H2L5, H2L25": "S = 5 and S = 25 top blocks; both divide every gate K (K/S = 20/50/100 and 4/10/20 blocks per top block); these two S bracket the Freebase ratio (pins / S fitting RAM) on the WebQSP scale",
            "TOP": "the frozen H4_SK hypergraph rule with k = S (cap = round(N/S); nothing else changed) partitioned by the frozen driver (NP 4) into S blocks with every frozen parameter unchanged EXCEPT IMBALANCE_TOL = 1.01 (declared: the top level keeps slack so the K-way bound 1.03 is still reachable below it); the frozen validity rule applies (largest top block <= ceil(1.03 N / S), else PARTITION_INVALID, no map); a top block within 1.01 N / S leaves every sub-problem a tolerance of at least 1.03 / 1.01 = 1.0198",
            "SUB": "for each top block b: the K-way hypergraph H4_SK_K (the frozen rule, verified to have the frozen content digest, CONTROL) restricted to the block's vertices; every net keeps only its pins inside the block, nets left with < 2 pins are dropped, every kept net keeps its weight; vertex weights unit; partitioned by the frozen driver (NP 4) into K/S blocks with IMBALANCE_TOL = floor_4dp(1.03 (N/K) (K/S) / n_b) where n_b is the block's vertex count, so Zoltan's own bound max-part <= tol x n_b/(K/S) equals the contract bound 1.03 N/K; a block with n_b > 1.03 N/S has no tolerance >= 1 and the cell is PARTITION_INVALID",
            "combination": "global block id = b x (K/S) + local id; the K-way map is validated by the frozen rule (length, ids in range, largest block <= ceil(1.03 N / K)); empty blocks are repaired by the frozen PHG_NONEMPTY_REPAIR_V1 on the K-way hypergraph; a map that violates the size bound is REFUSED (PARTITION_INVALID, no map written, never relaxed or replaced)",
            "identity_used": "K-way KM1 of the combined map = KM1 of the top partition on the K-way nets + the sum of the sub-problems' KM1 (net splitting preserves connectivity exactly); checked in scratchpad/_h2l.py TEST"},
        "gates": {
            "end_to_end (the only accepting gate)": "addendum 1's verdict rule, unchanged, via scratchpad/_h2l_eval.py (scratchpad/_ml2_eval.py's arithmetic, the 786-row population, matched-B_P comparison): rows below PHG_SK's ALL-gold count at matched B_P over every B_N in {100, 250, 500, 1000, 2000, 5000}, at each of K in {100, 250, 500}: at most 7 of 786 = RECALL_PRESERVING, at least 24 = FAILED, else BORDERLINE; also FAILED if B_P = K on more than half the rows",
            "diagnostics (not gating)": "KM1 of the combined map on H4_SK_K versus the flat PHG's KM1 (ratio); per-sub-problem pins, nets dropped by splitting, seconds and peak RSS per rank; the top-level KM1 share; sizes of the K blocks (min/max)"},
        "decision_table": {
            "both H2L5 and H2L25 RECALL_PRESERVING at K 100, 250, 500": "the nested decomposition is accepted as the K-way stage of the out-of-core partitioner; addendum 8 then declares the top-level (M2 coarsening ~2x -> top PHG on the coarse hypergraph -> projection + refinement) arms against the in-memory H2L top map",
            "exactly one of the two RECALL_PRESERVING": "reported; the passing S bounds the design (S is the number of independently loadable blocks); the failing S is not used; addendum 8 proceeds with the passing S",
            "neither RECALL_PRESERVING (FAILED or BORDERLINE)": "reported with the tables; BORDERLINE goes to the user; FAILED rejects the nested decomposition at these S (the flat K-way stage cannot be cut into independent blocks without losing recall) and the out-of-core design must change (no relaxation of the gate, no re-run with other S after seeing the result)",
            "PARTITION_INVALID cells": "counted as not scored (as addendum 1); a K-way cell that is invalid at both S is reported, never relaxed"},
        "measured_constants_carried_in": {"zoltan_bytes_per_pin_rss": "~129 (WebQSP H4_SK, NP 4, 29.57M pins)", "zoltan_seconds": "partition ~187 s + load 50-95 s per K at 29.57M pins, roughly flat in K",
                                           "freebase_h4_sk_pins": "~6e9 (STRUCT 4.25e9 + KNN ~1.8e9), ~775 GB Zoltan flat, ~390 GB after a 2x M2 coarsening; host 81.6 GB usable RAM, 285 GB disk"},
        "not_done_here": ["the top-level surrogate (M2 coarsening + OOC top PHG + projection + refinement): addendum 8, declared only after this result",
                          "any Freebase partition", "K above 500 (the WebQSP gate K set is 100/250/500 by addendum 1)"],
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
