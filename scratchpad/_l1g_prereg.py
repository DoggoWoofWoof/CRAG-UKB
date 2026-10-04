"""Pre-registration of the DEV_B confirmation for the L1_GEOM lane (write-once; run BEFORE any DEV_B number of a
candidate arm is read).  Freezes: the candidate (module _l1g_candidate.py), the code that evaluates it, the populations,
the DEV_A numbers the confirmation must reproduce (recomputed here from the candidate module itself), the decision rule,
and the disclosure that DEV_B was already read once by the L1_STATIC confirmation (a different candidate family)."""
import hashlib
import io
import json
import os
import shutil
import sys
import time

import _l1g_candidate as CAND
import _l1g_core as G
import _l1x90_core as X

OUT = G.OUT
PRE = os.path.join(OUT, "PREREGISTRATION_DEV_B.json")
if os.path.exists(PRE):
    sys.exit("REFUSED: %s exists (write-once; supersede by hash, never edit)" % PRE)
SNAP = os.path.join(OUT, "code_snapshot")
os.makedirs(SNAP, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ["_l1g_candidate.py", "_l1g_core.py", "_l1g_devB_confirm.py", "_l1s_core.py", "_l1x90_core.py", "_ta_prepartition.py"]
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


code = {}
for fn in CODE:
    p = os.path.join(HERE, fn)
    code[fn] = sha(p)
    shutil.copyfile(p, os.path.join(SNAP, fn))
caches = {name: {"path": os.path.relpath(X.CACHES[name], X.REPO), "partition": G.PARTITION_OF[name], "sha256": sha(X.CACHES[name])} for name in CACHES}
freeze = os.path.join(X.REPO, "data", "final_canonical", "CANONICAL_FREEZE.json")
devA = {}
ladder_check = {}
for name in CACHES:
    D = G.Data(name, dense_fp32=True)
    ranks, ch, gdiag = CAND.candidate_ranks(D)
    devA[name] = {"BASE": D.r_base["ALL_split"]["A"]["ALL"], "n_A": int(D.A.sum()), "geometry_diag": {k: round(v, 4) for k, v in gdiag.items()}}
    for arm, key in CAND.ARMS.items():
        out, _ = D.eval_rank(ranks[key], key, quiet=True)
        a = out["A"]
        devA[name][arm] = {"key": key, "ALL": a["ALL"], "gained": a["gained"], "lost": a["lost"], "p": a["p"], "per_hop": a["per_hop"]}
    # cross-check against the DEV_A ladder record (same arms under their ladder names)
    lad = json.load(io.open(os.path.join(OUT, "ladder_A_%s.json" % name), encoding="utf-8"))["arms"]
    lad.update(json.load(io.open(os.path.join(OUT, "dir_A_%s.json" % name), encoding="utf-8"))["arms"])
    cpath = os.path.join(OUT, "combo_A_%s.json" % name)
    if os.path.exists(cpath):
        lad.update(json.load(io.open(cpath, encoding="utf-8"))["arms"])
    names = {"G2_IL": "G G2 extrapolated 2q - mu | + BASE | F1c interleave (round robin)",
             "D2d": "D2d BASE + D1d (frozen RRF, 3 ch)",
             "IL": "F F1c interleave (round robin)", "G2_F0": "G G2 extrapolated 2q - mu | + BASE | F0 frozen RRF",
             "QMAX_F0": "QMAX F0(Cd,Cs,Cmax)"}
    ladder_check[name] = {}
    for arm, key in CAND.ARMS.items():
        if names[key] not in lad:                      # QMAX_F0 has DEV_A records only where the controls / combos ran
            ladder_check[name][arm] = "no prior DEV_A record (first computed here)"
            continue
        same = abs(lad[names[key]]["ALL"] - devA[name][arm]["ALL"]) < 1e-9 and lad[names[key]]["gained"] == devA[name][arm]["gained"] and lad[names[key]]["lost"] == devA[name][arm]["lost"]
        ladder_check[name][arm] = bool(same)
        if not same:
            sys.exit("ABORT: candidate module does not reproduce the DEV_A ladder for %s %s (%s vs %s)" % (name, arm, devA[name][arm], lad[names[key]]))
    G.log("  DEV_A %-10s BASE %.4f  PRIMARY %.4f (+%d/-%d p=%.1e)  SECONDARY %.4f (+%d/-%d p=%.1e)  FUSION %.4f (+%d/-%d)  GEOMETRY %.4f (+%d/-%d)" % (
        name, devA[name]["BASE"], devA[name]["PRIMARY"]["ALL"], devA[name]["PRIMARY"]["gained"], devA[name]["PRIMARY"]["lost"], devA[name]["PRIMARY"]["p"],
        devA[name]["SECONDARY"]["ALL"], devA[name]["SECONDARY"]["gained"], devA[name]["SECONDARY"]["lost"], devA[name]["SECONDARY"]["p"],
        devA[name]["FUSION"]["ALL"], devA[name]["FUSION"]["gained"], devA[name]["FUSION"]["lost"],
        devA[name]["GEOMETRY"]["ALL"], devA[name]["GEOMETRY"]["gained"], devA[name]["GEOMETRY"]["lost"]))
    del D, ranks, ch
rec = {
    "record": "PREREGISTRATION_DEV_B",
    "lane": "L1_GEOM",
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "research_question": "Under the fixed P50 budget and the frozen H4_SK / PHG partitions, does a parameter-free query-time signal "
                         "derived from q and its frozen SEED_K=5 seeds (PRIMARY: the directional block signal max_{v in B} "
                         "cos(q - e_s, e_v - e_s) over actual nodes, RRF over seeds, fused with the Dense/SPLADE block ranks by the "
                         "frozen RRF; SECONDARY: one extra retrieval point 2q - mu under a parameter-free interleave) raise "
                         "P(all gold evidence inside L1's 50 blocks) on a held-out population, without a significant loss on any "
                         "corpus / partition?",
    "candidate": {"module": "_l1g_candidate.py", "definition": CAND.__doc__, "arms": dict(CAND.ARMS), "point": CAND.POINT,
                  "frozen_numerics": {"K0": G.K0, "K_LOCK": G.K_LOCK, "P_MAIN": G.P_MAIN, "SEED_K": 5,
                                      "partition_ranking": "_ta_prepartition.partition_ranking (imported, unchanged)",
                                      "rrf": "_ta_prepartition.rrf_partitions (imported, unchanged)",
                                      "pinv_rcond": "not used by any candidate arm (2q - mu and the directional cosines need no pseudo-inverse)"},
                  "inputs": "served dense top-100 and served SPLADE top-100 node hits and the frozen top-5 seeds (replay caches), "
                            "the canonical dense embeddings (exact scoring of all nodes for the directional signal; exact retrieval "
                            "of one extra point), the frozen partition of each cache, the legacy node -> blocks table; nothing is "
                            "walked at query time, no parameter is fitted, no dataset rule"},
    "populations": {"exploration": "DEV_A = replay-cache dev rows with sha1(query_id)[:8] & 1 == 0 (the whole L1_GEOM factorial read DEV_A only)",
                    "confirmation": "DEV_B = the complementary rows (sha1(query_id)[:8] & 1 == 1)",
                    "DEV_B_reuse_disclosure": "DEV_B was read ONCE before, by results/L1_STATIC/DEV_B_CONFIRMATION.json (2026-09-14) for a "
                                              "different candidate family (multi-radius static tables, shelved). No L1_GEOM arm has read DEV_B. "
                                              "Because this is the second look at the same held-out rows, the gain threshold is tightened to p < 0.005.",
                    "caches": caches, "test_split": "never read"},
    "recorded_DEV_A": devA,
    "ladder_reproduced": ladder_check,
    "decision_rule": {
        "primary_passes_iff": [
            "on EACH of the five DEV_B populations (metaqa MtK, metaqa PHG, squad MtK, squad PHG, musique PHG): NOT (lost > gained and exact McNemar p < 0.05)   [no significant loss anywhere]",
            "on AT LEAST ONE of the five DEV_B populations: gained > lost and exact McNemar p < 0.005    [a significant gain; stricter than L1_STATIC's 0.01 because DEV_B is re-used]"],
        "prediction_from_DEV_A": "PRIMARY: large gain on musique (PHG), neutral on metaqa (both partitions) and squad (both partitions); "
                                 "SECONDARY: gain on musique, neutral elsewhere",
        "secondary": "evaluated in the same run under the same rule; adopted only if the PRIMARY fails; reported regardless",
        "decomposition_arms": "FUSION and GEOMETRY are reported on DEV_B for the factorial (how much from fusion, how much from the point); never decided on",
        "controls": "results/L1_GEOM/dircontrol_A_*.json (reversed / random direction, exhaustive dense block max, size ranking, pooled variant) were read on DEV_A only, before this record",
        "reproduction_guard": "the confirmation run must reproduce the recorded DEV_A ALL / gained / lost of every arm exactly, else it aborts",
        "per_hop": "reported, never used for the decision",
        "full_dev": "A u B numbers reported after the DEV_B decision, for the record only",
        "no_other_arm": "no other point, fusion, channel count or partition variant is evaluated on DEV_B",
        "one_shot": "the confirmation script refuses to run if DEV_B_CONFIRMATION.json exists",
    },
    "code_sha256": code,
    "code_snapshot_dir": os.path.relpath(SNAP, X.REPO),
    "canonical_freeze_sha256": sha(freeze) if os.path.exists(freeze) else None,
    "adoption_note": "a PASS is a lane result on the canonical replay caches; adopting the candidate as the served L1 selector changes "
                     "the L1 contract (PRIMARY: 5 exact directional scorings of the node matrix per query, i.e. the cost of 5 extra exact "
                     "dense retrievals, plus a third RRF channel; SECONDARY: one extra exact dense retrieval + the interleave) and needs "
                     "a ruling -- no cache is rebuilt",
}
G.S.wj(PRE, rec)
print("wrote", os.path.relpath(PRE, X.REPO))
print("record sha256", sha(PRE))
for k, v in code.items():
    print("  %-22s %s" % (k, v))
for k, v in caches.items():
    print("  cache %-11s %s" % (k, v["sha256"]))
